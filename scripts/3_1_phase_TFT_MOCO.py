#!/usr/bin/env python
# coding: utf-8

# # MC-STGAT-IDS — Phase 3 TFT (MoCo variant)
# 
# Identical to the BYOL TFT notebook except:
# - `CACHE_BUILD_SSL = ['moco']` — builds MoCo frozen caches only
# - All run calls use `ssl='moco'`
# - Cache files are written as `frozen_cache_moco_L*.pt`
# - Reports and checkpoints go to `phase3_{kind}_moco/`
# 
# **Requires:** SSL MoCo encoder already trained at `{dataset_dir}/ssl_checkpoints_moco/ssl_encoder_best.pth`.

# # Cell 1 — Imports

# In[69]:


import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import gc, json, pickle, contextlib, time, warnings, random, copy, hashlib, re, math
from collections import Counter, defaultdict
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
from torch_geometric.data import Batch
from torch_geometric.nn import GATConv, GlobalAttention, global_mean_pool
from torch.cuda.amp import GradScaler
from sklearn.metrics import (f1_score, confusion_matrix,
                             roc_auc_score, average_precision_score)
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark        = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
torch.set_float32_matmul_precision('high')
print('Imports ready.')


# # Cell 2 — Global configuration

# In[70]:


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Architecture
HIDDEN_DIM      = 128
NUM_HEADS       = 4
NUM_LAYERS      = 3
TFT_HIDDEN_DIM  = 384
TFT_NUM_HEADS   = 2
TFT_DROPOUT     = 0.10

# Sequence
SEQUENCE_LENGTH = 10
SEQUENCE_STRIDE = 1

# Training
BATCH_SIZE       = 512
EVAL_BATCH_SIZE  = 32    # eval uses a smaller batch than training
GRAD_ACCUM_STEPS = 1
LEARNING_RATE    = 1e-3
WEIGHT_DECAY     = 1e-3
EPOCHS           = 500
PATIENCE         = 100
WARMUP_EPOCHS    = 10
ENCODER_LR_MULT  = 0.15
MAIN_SELECTION_METRIC = 'roc'

# Lexicographic selection priority (first wins; ties fall through)
SELECTION_METRIC_PRIORITY = ['roc', 'macro_f1_merged', 'pr', 'f1']
SELECTION_METRIC_TOL      = 1e-4

# Loss
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0
POS_WEIGHT  = 3
W_MULTI     = 0.5

# Graph caps (per dataset, matches SSL stage)
MAX_NODES = 128
MAX_EDGES = 20000
MAX_EDGES_PER_DATASET = {'gotham': 2000, 'nf_ton': 10000, 'nf_bot': 15000}

# Gap-aware sequences
ENFORCE_MAX_WINDOW_GAP   = True
DEFAULT_MAX_WINDOW_GAP_S = 60

# Threshold rule
USE_PRIOR_MATCHED_THRESHOLD = False

# Few-shot
FRACTIONS              = [0.05, 0.25, 0.50, 1.00]
FEWSHOT_EPOCHS         = 40
FEWSHOT_WARMUP         = 5
FEWSHOT_WARMUP_SMALL   = 2
FEWSHOT_PATIENCE_SMALL = 6
SMALL_LABEL_THRESHOLD  = 300

# Ensemble
ENSEMBLE_SEEDS_DEFAULT = [1337, 2024, 7]
BOOTSTRAP_N    = 1000
BOOTSTRAP_SEED = 123

# Checkpoint cadence
CKPT_EVERY_N_STEPS   = 512
KEEP_LAST_N_PARTIALS = 5

# Misc
NUM_WORKERS = 0
PREFETCH_FACTOR = 2
SEED = 42
DATASET_THRESHOLDS = {'gotham': 0, 'nf_ton': 0, 'nf_bot': 0}
BLACKLIST_CLASSES = {'Unknown','unknown','UNKNOWN'}
USE_BALANCED_SAMPLER_PER_DATASET = {'gotham': True, 'nf_bot': False, 'nf_ton': True}
AUTOCAST_DTYPE = 'bf16'
SKIP_NONFINITE_STEPS = True
SANITIZE_GRAPH_EMB = True
SANITIZE_FROZEN_CACHE = True
LR_WARMUP_ENABLED = True
USE_TORCH_COMPILE = False

torch.manual_seed(SEED); np.random.seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
print(f'Device: {DEVICE}')
print(f'Selection priority: {SELECTION_METRIC_PRIORITY}')


# # Cell 2b — override_config

# In[71]:


class _OverrideConfig:
    def __init__(self, **overrides):
        self.overrides = overrides; self._saved = {}; self._added = []
    def __enter__(self):
        g = globals()
        for k, v in self.overrides.items():
            if k in g: self._saved[k] = g[k]
            else: self._added.append(k)
            g[k] = v
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        g = globals()
        for k, v in self._saved.items(): g[k] = v
        for k in self._added: g.pop(k, None)
        return False
def override_config(**overrides): return _OverrideConfig(**overrides)
print('override_config ready.')


# # Cell 3 — Dataset registry + autocast + sanitizer

# In[72]:


DATASET_CONFIG = {
    'gotham': 'Gotham_processed_features_10s/graphs',
    'nf_bot': 'NFBoTIoT_v3_10s/graphs',
    'nf_ton': 'NFToNIoT_v3_10s/graphs',
}
def autocast_ctx():
    if DEVICE.type != 'cuda' or AUTOCAST_DTYPE == 'off':
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if AUTOCAST_DTYPE == 'bf16' else torch.float16
    try: return torch.amp.autocast(device_type='cuda', dtype=dtype)
    except Exception:
        if dtype is torch.bfloat16: return contextlib.nullcontext()
        return torch.cuda.amp.autocast(enabled=True)
def sanitize_tensor(t, nan=0.0, posinf=1e4, neginf=-1e4):
    if t is None: return None
    if not torch.isfinite(t).all():
        t = torch.nan_to_num(t, nan=nan, posinf=posinf, neginf=neginf)
    return t
print(f'autocast ready (dtype={AUTOCAST_DTYPE})')


# # Cell 4 — Graph utilities + dominant_family + gap-aware starts

# In[73]:


def cap_graph(g, max_nodes=MAX_NODES, max_edges=None):
    if max_edges is None: max_edges = MAX_EDGES
    g = g.clone(); num_nodes = g.num_nodes
    if g.edge_index.size(1) > max_edges:
        keep = torch.randperm(g.edge_index.size(1))[:max_edges]
        g.edge_index = g.edge_index[:, keep]
        if g.edge_attr is not None and g.edge_attr.size(0) > 0:
            if g.edge_attr.dim() > 2:
                g.edge_attr = g.edge_attr.view(-1, g.edge_attr.size(-1))
            g.edge_attr = g.edge_attr[keep]
    if num_nodes > max_nodes:
        deg = torch.zeros(num_nodes, dtype=torch.long)
        deg.scatter_add_(0, g.edge_index[0], torch.ones(g.edge_index.size(1), dtype=torch.long))
        kn = torch.topk(deg, max_nodes).indices
        mask = torch.zeros(num_nodes, dtype=torch.bool); mask[kn] = True
        s, d = g.edge_index; keep_e = mask[s] & mask[d]
        g.edge_index = g.edge_index[:, keep_e]
        if g.edge_attr is not None and g.edge_attr.size(0) > 0:
            g.edge_attr = g.edge_attr[keep_e]
        remap = torch.zeros(num_nodes, dtype=torch.long)
        remap[mask] = torch.arange(max_nodes, dtype=torch.long)
        g.edge_index = remap[g.edge_index]; g.x = g.x[mask]
    return g

def dominant_family(g, attack_threshold=0.15,
                    blacklist=('Unknown','unknown','UNKNOWN')):
    names = getattr(g, 'edge_attack_names', None)
    if not names: return 'Benign'
    c = Counter(names); total = sum(c.values())
    atk = {k: v for k, v in c.items() if k != 'Benign' and k not in blacklist}
    if not atk: return 'Benign'
    if sum(atk.values()) / max(total, 1) < attack_threshold: return 'Benign'
    return max(atk, key=atk.get)

def build_gap_aware_starts(graphs, seq_len, stride, max_gap_sec,
                           enforce=True, verbose=True):
    n = len(graphs)
    if n < seq_len: return np.array([], dtype=np.int64)
    stride_only = np.arange(0, n - seq_len + 1, stride, dtype=np.int64)
    if not enforce: return stride_only
    if not hasattr(graphs[0], 'window_start_unix'): return stride_only
    times = np.array([float(g.window_start_unix) for g in graphs], dtype=np.float64)
    gaps = np.diff(times)
    ok_adj = (gaps <= float(max_gap_sec)).astype(np.int32)
    cum = np.concatenate([[0], np.cumsum(ok_adj)])
    window_sum = cum[stride_only + seq_len - 1] - cum[stride_only]
    keep = window_sum == (seq_len - 1)
    kept = stride_only[keep]
    if verbose: print(f'  [gap] candidates {len(stride_only)} kept {len(kept)}')
    if len(kept) < 10 and len(stride_only) >= 10: return stride_only
    return kept
def _sanitize_graph_attrs(g):
    """Convert any numpy scalar attributes on a Data object to Python scalars.
    Call after cap_graph in _prepare_ctx."""
    for k, v in list(g):
        if isinstance(v, np.generic):        # numpy scalar
            setattr(g, k, v.item())
    return g
print('Graph utilities ready.')


# # Cell 5 — Model classes

# In[74]:


class GATEncoder(nn.Module):
    def __init__(self, in_ch, hid, heads, layers, edge_dim):
        super().__init__()
        self.convs = nn.ModuleList(); self.bns = nn.ModuleList()
        self.convs.append(GATConv(in_ch, hid, heads=heads, edge_dim=edge_dim, concat=True, dropout=0.0))
        self.bns.append(nn.BatchNorm1d(hid * heads))
        for _ in range(layers - 2):
            self.convs.append(GATConv(hid*heads, hid, heads=heads, edge_dim=edge_dim, concat=True, dropout=0.0))
            self.bns.append(nn.BatchNorm1d(hid * heads))
        self.convs.append(GATConv(hid*heads, hid, heads=1, edge_dim=edge_dim, concat=False, dropout=0.0))
        self.bns.append(nn.BatchNorm1d(hid))
        self.edge_dim = edge_dim
        self.pool_gate = nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, 1))
    def forward(self, x, ei, ea=None, b=None, return_nodes=False):
        for i, c in enumerate(self.convs):
            x = c(x, ei, edge_attr=ea) if (ea is not None and self.edge_dim > 0) else c(x, ei)
            x = F.relu(self.bns[i](x))
        h = GlobalAttention(self.pool_gate)(x, b)
        return (x, h) if return_nodes else h

class GatedResidualNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.fc1  = nn.Linear(input_dim, hidden_dim); self.elu = nn.ELU()
        self.fc2  = nn.Linear(hidden_dim, output_dim); self.drop = nn.Dropout(dropout)
        self.gate = nn.Linear(hidden_dim, output_dim); self.norm = nn.LayerNorm(output_dim)
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None
    def forward(self, x):
        z = self.elu(self.fc1(x)); h = self.drop(self.fc2(z))
        g = torch.sigmoid(self.gate(z)); h = g * h
        if self.skip is not None: x = self.skip(x)
        return self.norm(x + h)

class VariableSelectionNetwork(nn.Module):
    def __init__(self, var_dims, hidden_dim, dropout=0.1):
        super().__init__()
        self.num_vars = len(var_dims)
        self.var_grns = nn.ModuleList(
            [GatedResidualNetwork(d, hidden_dim, hidden_dim, dropout) for d in var_dims])
        self.weight_grn = GatedResidualNetwork(sum(var_dims), hidden_dim, self.num_vars, dropout)
    def forward(self, vars_list):
        processed = [grn(v) for grn, v in zip(self.var_grns, vars_list)]
        flat = torch.cat(vars_list, dim=-1)
        w = F.softmax(self.weight_grn(flat), dim=-1)
        stacked = torch.stack(processed, dim=2)
        return (stacked * w.unsqueeze(-1)).sum(dim=2), w

class InterpretableMHA(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1, init_temp=1.0):
        super().__init__()
        self.h = hidden_dim; self.nh = num_heads; self.hd = hidden_dim // num_heads
        assert self.hd * num_heads == hidden_dim
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.o = nn.Linear(hidden_dim, hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.register_buffer('log_temp', torch.tensor(float(np.log(init_temp))))
        self.min_temp = 0.05; self.max_temp = 100.0
    def forward_last(self, x):
        B, T, _ = x.shape
        q_last = self.q(x[:, -1:, :]).view(B, 1, self.nh, self.hd).transpose(1, 2)
        k_all  = self.k(x).view(B, T, self.nh, self.hd).transpose(1, 2)
        v_all  = self.v(x).view(B, T, self.nh, self.hd).transpose(1, 2)
        temp   = torch.exp(self.log_temp).clamp(self.min_temp, self.max_temp)
        scale  = 1.0 / ((self.hd ** 0.5) * temp)
        s = torch.matmul(q_last, k_all.transpose(-2, -1)) * scale
        a = self.drop(F.softmax(s, dim=-1))
        c = torch.matmul(a, v_all).transpose(1, 2).contiguous().view(B, 1, self.h)
        return self.o(c).squeeze(1), a.squeeze(2).mean(dim=1)

class TemporalFusionModuleV2(nn.Module):
    """
    Dense Spatio-Temporal Highway Network (Chain + All-to-Last Highway):
    1. Sequential Chain: 2-layer Bidirectional GRU linking all 10 temporal steps.
    2. Attention Context: Interpretable MHA over chain states preserving attention interfaces.
    3. Dense Highway Extractions:
       - h_chain_last: final sequential state (2 * hidden_dim)
       - h_snap_raw: un-smoothed instantaneous snapshot (graph_emb[:, -1, :])
       - h_mean_pool: global background average across sequence
       - h_max_pool: peak burst anomaly across sequence
       - h_velocity: attack velocity vector (graph_emb[:, -1, :] - graph_emb[:, 0, :])
       - h_attn: attended context
    4. Gated Residual Highway Mixer: fuses multi-scale paths into hidden_dim output representation.
    """
    def __init__(self, graph_emb_dim, struct_dim, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.proj_graph  = nn.Linear(graph_emb_dim, hidden_dim)
        self.proj_struct = nn.Linear(struct_dim, hidden_dim)
        self.vsn  = VariableSelectionNetwork([hidden_dim, hidden_dim], hidden_dim, dropout)
        
        # 1. The Sequential Chain (BiGRU connecting all 10 steps)
        self.chain = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        # 2. Interpretable Attention Context (for XAI & visualization)
        self.attn = InterpretableMHA(hidden_dim * 2, num_heads, dropout)
        
        # 3. Dense All-to-Last Highway Mixer:
        # Components: h_chain_last (2*hid) + h_snap_raw (graph_emb_dim) + h_mean (2*hid) +
        #             h_max (2*hid) + h_velocity (graph_emb_dim) + h_attn (2*hid)
        dense_mix_dim = (hidden_dim * 2) + graph_emb_dim + (hidden_dim * 2) + (hidden_dim * 2) + graph_emb_dim + (hidden_dim * 2)
        self.highway_grn1 = GatedResidualNetwork(dense_mix_dim, hidden_dim * 2, hidden_dim, dropout)
        self.highway_grn2 = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, graph_emb, struct_feats):
        v_g = self.proj_graph(graph_emb); v_s = self.proj_struct(struct_feats)
        selected, vsn_w = self.vsn([v_g, v_s])
        
        # 1. The Sequential Chain
        chain_out, _ = self.chain(selected) # [B, T, 2*hid]
        chain_norm = F.layer_norm(chain_out, (chain_out.shape[-1],))
        
        # 2. Attention Context (preserves attention maps for visualization)
        attn_last, attn_w = self.attn.forward_last(chain_norm)
        
        # 3. Dense Highway Extractions to Last Layer
        h_chain_last = chain_out[:, -1, :]                           # [B, 2*hid]
        h_snap_raw   = graph_emb[:, -1, :]                          # [B, graph_dim] Zero-latency bypass
        h_mean_pool  = chain_out.mean(dim=1)                        # [B, 2*hid] Background baseline
        h_max_pool   = chain_out.max(dim=1)[0]                      # [B, 2*hid] Peak burst anomaly
        h_velocity   = graph_emb[:, -1, :] - graph_emb[:, 0, :]     # [B, graph_dim] Attack velocity
        h_attn       = attn_last                                    # [B, 2*hid] Attended context
        
        # 4. Dense All-to-Last Highway Mixing
        dense_all = torch.cat([h_chain_last, h_snap_raw, h_mean_pool, h_max_pool, h_velocity, h_attn], dim=-1)
        out = self.highway_grn1(dense_all)
        out = self.highway_grn2(out)
        out = self.norm(out)
        return out, attn_w, vsn_w

class MC_STGAT_IDS(nn.Module):
    def __init__(self, graph_emb_dim, struct_dim, tft_hidden, num_classes, num_heads=4, dropout=0.1):
        super().__init__()
        self.temporal = TemporalFusionModuleV2(graph_emb_dim, struct_dim, tft_hidden, num_heads, dropout)
        self.binary_head = nn.Sequential(
            nn.Linear(tft_hidden, tft_hidden // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(tft_hidden // 2, 1))
        self.multi_head = nn.Sequential(
            nn.Linear(tft_hidden, tft_hidden // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(tft_hidden // 2, num_classes))
    def forward(self, graph_emb, struct_feats):
        temporal_emb, attn_w, vsn_w = self.temporal(graph_emb, struct_feats)
        return (self.binary_head(temporal_emb).squeeze(-1),
                self.multi_head(temporal_emb), attn_w, vsn_w)

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=1.0):
        super().__init__()
        self.alpha = alpha; self.gamma = gamma
        self.register_buffer('pos_weight', torch.tensor(float(pos_weight)))
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none', pos_weight=self.pos_weight)
        p = torch.sigmoid(logits).clamp(1e-6, 1.0 - 1e-6)
        p_t = p * targets + (1 - p) * (1 - targets)
        return (self.alpha * (1 - p_t) ** self.gamma * bce).mean()

class CombinedLoss(nn.Module):
    def __init__(self, w_multi=0.5, pos_weight=1.5, focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.w_multi = w_multi
        self.binary_loss = BinaryFocalLoss(focal_alpha, focal_gamma, pos_weight)
        self.focal_alpha = focal_alpha; self.focal_gamma = focal_gamma
    def forward(self, bin_logits, multi_logits, y_bin, y_multi):
        l_bin = self.binary_loss(bin_logits, y_bin)
        logp = F.log_softmax(multi_logits, dim=-1)
        logp_t = logp.gather(1, y_multi.unsqueeze(1)).squeeze(1).clamp_min(-20.0)
        p_t = logp_t.exp().clamp(1e-6, 1.0 - 1e-6)
        l_multi = (-self.focal_alpha * (1 - p_t) ** self.focal_gamma * logp_t).mean()
        return l_bin + self.w_multi * l_multi, l_bin.item(), l_multi.item()
print('Model classes ready.')


# # Cell 6 — Sequence dataset + collate

# In[75]:


class GraphSequenceDataset(Dataset):
    def __init__(self, graphs, starts, seq_len, y_bin_seq, y_multi_seq, struct_feats):
        self.graphs = graphs; self.starts = np.asarray(starts); self.seq_len = seq_len
        self.y_bin_seq = y_bin_seq; self.y_multi_seq = y_multi_seq; self.struct = struct_feats
    def __len__(self): return len(self.starts)
    def __getitem__(self, i):
        s = int(self.starts[i]); e = s + self.seq_len
        return (self.graphs[s:e], self.struct[s:e], self.y_bin_seq[i], self.y_multi_seq[i])

def collate_graph_sequences(batch):
    seq_graphs_list, seq_struct_list, y_bin_list, y_multi_list = zip(*batch)
    B = len(seq_graphs_list); T = len(seq_graphs_list[0])
    flat_graphs = []
    for seq in seq_graphs_list: flat_graphs.extend(seq)
    # Skip custom attributes that PyG's default collator can't handle.
    # 'edge_attack_names'  -> list of strings
    # 'window_start_unix'  -> scalar (possibly numpy)
    flat_batch = Batch.from_data_list(
        flat_graphs,
        exclude_keys=['edge_attack_names', 'window_start_unix'])
    struct_tensor = torch.stack(
        [torch.as_tensor(s, dtype=torch.float32) for s in seq_struct_list], dim=0)
    return (flat_batch, B, T, struct_tensor,
            torch.tensor([float(v) for v in y_bin_list],  dtype=torch.float32),
            torch.tensor([int(v)   for v in y_multi_list], dtype=torch.long))
print('Sequence dataset ready.')


# # Cell 7 — Structural feature precompute

# In[76]:


def structural_features(g):
    x  = g.x.numpy()
    ea = g.edge_attr.numpy() if g.edge_attr is not None else np.zeros((0, 0))
    ei = g.edge_index.numpy()
    N = x.shape[0]; E = ei.shape[1]
    deg = np.bincount(ei[0], minlength=N) if E > 0 else np.zeros(N)
    base = np.array([float(N), float(E),
                     float(deg.mean()) if N > 0 else 0.0,
                     float(np.log1p(E))], dtype=np.float32)
    nm = x.mean(0) if N > 0 else np.zeros(x.shape[1], dtype=np.float32)
    ns = x.std(0)  if N > 0 else np.zeros(x.shape[1], dtype=np.float32)
    em = ea.mean(0) if E > 0 and ea.ndim == 2 else np.zeros(
        ea.shape[-1] if ea.ndim == 2 else 0, dtype=np.float32)
    return np.concatenate([base, nm, ns, em]).astype(np.float32)

def precompute_struct(graphs):
    return torch.tensor(np.stack([structural_features(g) for g in graphs]),
                        dtype=torch.float32)
print('Structural features ready.')


# # Cell 7b — Logging, checkpoint, metric helpers

# In[77]:


def _signatures():
    enc_sig = hashlib.md5("|".join([f"hid={HIDDEN_DIM}", f"heads={NUM_HEADS}", f"layers={NUM_LAYERS}"]).encode()).hexdigest()[:10]
    tft_sig = hashlib.md5("|".join([f"enc={enc_sig}", f"tft_hid={TFT_HIDDEN_DIM}", f"tft_heads={TFT_NUM_HEADS}", f"seq={SEQUENCE_LENGTH}"]).encode()).hexdigest()[:10]
    return enc_sig, tft_sig
def _config_signature():
    e, t = _signatures(); return f"{e}_{t}"
def _fmt_time(s):
    if s < 60: return f"{s:5.1f}s"
    m, ss = divmod(s, 60); return f"{int(m):02d}m{ss:04.1f}s"
def _bar(v, width=20, vmin=0.5, vmax=1.0):
    if v is None or (isinstance(v, float) and np.isnan(v)): return " " * width
    f = float(np.clip((v - vmin) / max(vmax - vmin, 1e-9), 0.0, 1.0))
    n = int(round(f * width)); return "#" * n + "." * (width - n)
def _print_epoch_header(tag, total_epochs):
    print(f"  {'ep':>4}  {'train':>8}  {'bin':>7}  {'multi':>7}  "
          f"{'gnorm':>7}  {'lr':>9}  |  "
          f"{'valROC':>7} {'valPR':>7} {'valF1':>7} {'valmF1':>8}  "
          f"{'ROC bar':<20}  {'time':>7}  {'pat':>3}  note")
    print("  " + "-" * 130)
def _print_epoch_row(ep, total_epochs, tr, val, gnorm, lr,
                     patience_left, best_macro_f1, elapsed, note=""):
    print(f"  {ep:>4}  {tr['total']:>8.4f}  {tr['binary']:>7.4f}  {tr['multi']:>7.4f}  "
          f"{gnorm:>7.3f}  {lr:>9.2e}  |  "
          f"{val['roc']:>7.4f} {val['pr']:>7.4f} {val['f1']:>7.4f} "
          f"{val['macro_f1_merged']:>8.4f}  {_bar(val['roc']):<20}  "
          f"{_fmt_time(elapsed):>7}  {patience_left:>3}  {note}")
def _optimal_threshold(prob, y):
    from sklearn.metrics import precision_recall_curve
    prec, rec, thr = precision_recall_curve(y, prob)
    prec, rec = prec[:-1], rec[:-1]
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    if not np.isfinite(f1).any(): return 0.5, 0.0
    b = int(np.nanargmax(f1))
    return float(thr[b]), float(f1[b])
def _prior_matched_threshold(prob, y):
    r = float(np.asarray(y).mean())
    if r <= 0 or r >= 1: return 0.5
    return float(np.quantile(prob, 1.0 - r))
def _binary_metrics_at(prob, y, thr):
    pred = (prob >= thr).astype(int)
    f1 = float(f1_score(y, pred, average='binary', zero_division=0))
    cm = confusion_matrix(y, pred, labels=[0, 1])
    cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
    tn, fp = int(cm[0, 0]), int(cm[0, 1]); fn, tp = int(cm[1, 0]), int(cm[1, 1])
    return {'f1': f1, 'fpr': float(cmn[0, 1]), 'fnr': float(cmn[1, 0]),
            'n_true_benign': tn + fp, 'n_true_attack': fn + tp,
            'n_total': tn + fp + fn + tp,
            'n_pred_benign': tn + fn, 'n_pred_attack': fp + tp,
            'tn_benign_correct': tn, 'fp_benign_wrong': fp,
            'fn_attack_missed': fn, 'tp_attack_correct': tp,
            'true_attack_rate': (fn + tp) / max(tn + fp + fn + tp, 1),
            'pred_attack_rate': (fp + tp) / max(tn + fp + fn + tp, 1)}
def _per_class_score_analysis(pb_test, test_labels_seq, class_names, thr):
    rows = []
    for cls in class_names:
        if cls == 'Benign': continue
        mask = (test_labels_seq == cls) | (test_labels_seq == 'Benign')
        y = (test_labels_seq[mask] != 'Benign').astype(int); p = pb_test[mask]
        if len(set(y)) < 2: continue
        try:
            roc = float(roc_auc_score(y, p)); pr = float(average_precision_score(y, p))
        except ValueError: continue
        pos = (y == 1)
        rows.append({'class': cls, 'n_test': int(pos.sum()),
                     'roc_vs_benign': roc, 'pr_vs_benign': pr,
                     'mean_prob_attack': float(p[pos].mean()) if pos.sum() > 0 else 0.0,
                     'recall_at_thr': float((p[pos] >= thr).sum() / max(int(pos.sum()), 1)),
                     'missed_at_thr': int((p[pos] < thr).sum())})
    rows.sort(key=lambda r: -r['roc_vs_benign'])
    return rows
def _print_per_class_table(rows, thr, title):
    if not rows: print(f'  {title}: no rows'); return
    print(f'\n  {title} - thr={thr:.2f}')
    print(f"  {'class':<28} {'n':>6} {'ROC':>7} {'PR':>7} {'recall':>7}")
    for r in rows:
        print(f"  {r['class'][:28]:<28} {r['n_test']:>6} "
              f"{r['roc_vs_benign']:>7.4f} {r['pr_vs_benign']:>7.4f} "
              f"{r['recall_at_thr']:>7.3f}")
def _print_binary_confusion(m, label="binary"):
    print(f"    [{label}] attack true={m['n_true_attack']} pred={m['n_pred_attack']} "
          f"TP={m['tp_attack_correct']} FN={m['fn_attack_missed']}  "
          f"| benign true={m['n_true_benign']} pred={m['n_pred_benign']} "
          f"TN={m['tn_benign_correct']} FP={m['fp_benign_wrong']}  "
          f"| FPR={m['fpr']:.4f} FNR={m['fnr']:.4f}")
def _partial_re(tag): return re.compile(rf"^partial_{re.escape(tag)}_e(\d+)_s(\d+)\.pth$")
def _prune_partials(ckpt_dir, tag, keep=5):
    if not os.path.isdir(ckpt_dir): return
    pat = _partial_re(tag)
    m = [f for f in os.listdir(ckpt_dir) if pat.match(f)]
    m.sort(key=lambda f: tuple(int(x) for x in pat.match(f).groups()))
    for f in (m if keep <= 0 else m[:-keep]):
        try: os.remove(os.path.join(ckpt_dir, f))
        except OSError: pass
def _encode_sequence(encoder, flat_batch, B, T, device, frozen):
    ctx = torch.no_grad() if frozen else contextlib.nullcontext()
    with ctx:
        flat_batch = flat_batch.to(device, non_blocking=True)
        out = encoder(flat_batch.x, flat_batch.edge_index, flat_batch.edge_attr, flat_batch.batch)
        h = out[-1] if isinstance(out, tuple) else out
        h = h.view(B, T, -1).float()
        if SANITIZE_GRAPH_EMB: h = sanitize_tensor(h)
        return h
def _val_metrics(pb_val, pm_val, yb_val, ym_val, thr=0.5,
                 train_visible_idx=None, class_names=None, dataset=None):
    y = yb_val.astype(int); pb = np.asarray(pb_val, dtype=np.float64)
    if not np.isfinite(pb).all(): pb = np.nan_to_num(pb, 0.0, 1.0, 0.0)
    try: roc = float(roc_auc_score(y, pb))
    except ValueError: roc = 0.5
    try: pr = float(average_precision_score(y, pb))
    except ValueError: pr = 0.0
    f1b = float(f1_score(y, (pb >= thr).astype(int), average='binary', zero_division=0))
    ym_pred = pm_val.argmax(-1)
    mf1 = float(f1_score(ym_val, ym_pred, average='macro', zero_division=0))
    wf1 = float(f1_score(ym_val, ym_pred, average='weighted', zero_division=0))
    return {'roc': roc, 'pr': pr, 'f1': f1b, 'macro_f1': mf1, 'macro_f1_merged': mf1,
            'weighted_f1': wf1, 'macro_f1_visible': mf1}
print('Logging helpers ready.  config_sig =', _config_signature())


# # Cell 7c — Frozen embedding cache (memory-adaptive)

# In[78]:


_FROZEN_EMB_CACHE = {}
def _frozen_emb_key(ds, ssl):
    enc_path = os.path.join(DATASET_CONFIG[ds], f"ssl_checkpoints_{ssl}",
                            "ssl_encoder_best.pth")
    try:
        mt = os.path.getmtime(enc_path); sz = os.path.getsize(enc_path)
    except OSError:
        mt, sz = -1.0, -1
    return (ds, ssl, int(SEQUENCE_LENGTH), int(SEQUENCE_STRIDE),
            bool(ENFORCE_MAX_WINDOW_GAP), int(MAX_NODES),
            int(MAX_EDGES_PER_DATASET.get(ds, MAX_EDGES)),
            int(HIDDEN_DIM), int(NUM_HEADS), int(NUM_LAYERS),
            float(DATASET_THRESHOLDS.get(ds, 0.0)), str(AUTOCAST_DTYPE),
            bool(SANITIZE_FROZEN_CACHE), float(mt), int(sz))

def _encode_graphs_adaptive(enc, graphs, initial_bs=32, min_bs=1,
                            verbose=True, tag=''):
    n = len(graphs)
    if n == 0: return torch.zeros(0, 0), initial_bs
    embs = [None] * n
    cur_bs = max(1, int(initial_bs)); i = 0
    while i < n:
        chunk = graphs[i:i + cur_bs]
        try:
            b = Batch.from_data_list(chunk).to(DEVICE)
            with torch.inference_mode():
                out = enc(b.x, b.edge_index, b.edge_attr, b.batch)
            h = out[-1] if isinstance(out, tuple) else out
            h = h.float()
            if SANITIZE_GRAPH_EMB: h = sanitize_tensor(h)
            h_cpu = h.cpu()
            for j in range(len(chunk)): embs[i + j] = h_cpu[j]
            i += len(chunk)
            del b, out, h, h_cpu
            if DEVICE.type == 'cuda': torch.cuda.empty_cache()
        except torch.OutOfMemoryError:
            if DEVICE.type == 'cuda': torch.cuda.empty_cache()
            if cur_bs <= min_bs:
                raise RuntimeError(f'{tag}: OOM at bs={cur_bs} (min_bs={min_bs})')
            cur_bs = max(min_bs, cur_bs // 2)
            if verbose: print(f'    [{tag}] OOM; retry bs={cur_bs}')
    if verbose: print(f'    [{tag}] encoded {n} graphs (final bs={cur_bs})')
    return torch.stack([e for e in embs], 0), cur_bs

def precache_frozen_embeddings(ds, ssl, class_names=None, label_to_idx=None,
                               force=False, verbose=True, initial_bs=32, min_bs=1):
    key = _frozen_emb_key(ds, ssl)
    if not force and key in _FROZEN_EMB_CACHE:
        if verbose: print(f'[frozen-emb] {ds}/{ssl}: memory HIT')
        return _FROZEN_EMB_CACHE[key]
    ENC_FILE = os.path.join(DATASET_CONFIG[ds], f"ssl_checkpoints_{ssl}", "ssl_encoder_best.pth")
    if not os.path.isfile(ENC_FILE):
        print(f'[frozen-emb] {ds}/{ssl}: no SSL encoder at {ENC_FILE}'); return None
    if class_names is None or label_to_idx is None:
        raise ValueError('class_names and label_to_idx required')
    t0 = time.time(); DATA_DIR = DATASET_CONFIG[ds]
    with open(os.path.join(DATA_DIR, 'norm_stats.json')) as f: ns = json.load(f)
    N_CONT = ns['n_continuous_edge_feats']
    nmean = torch.tensor(ns['node_mean'], dtype=torch.float); nstd = torch.tensor(ns['node_std'], dtype=torch.float)
    emean = torch.tensor(ns['edge_mean'], dtype=torch.float); estd = torch.tensor(ns['edge_std'], dtype=torch.float)
    MWG = float(ns.get('max_window_gap_sec', DEFAULT_MAX_WINDOW_GAP_S))
    cap = MAX_EDGES_PER_DATASET.get(ds, MAX_EDGES)
    def _load(s):
        with open(os.path.join(DATA_DIR, f'{s}_graphs.pkl'), 'rb') as f: return pickle.load(f)
    tg = [_sanitize_graph_attrs(cap_graph(g, max_edges=cap)) for g in _load('train')]
    vg = [_sanitize_graph_attrs(cap_graph(g, max_edges=cap)) for g in _load('val')]
    eg = [_sanitize_graph_attrs(cap_graph(g, max_edges=cap)) for g in _load('test')]
    def _norm(graphs):
        for g in graphs:
            g.x = torch.clamp((g.x - nmean) / (nstd + 1e-8), -10, 10)
            if g.edge_attr is not None and g.edge_attr.size(0) > 0:
                if g.edge_attr.dim() > 2: g.edge_attr = g.edge_attr.view(-1, g.edge_attr.size(-1))
                c, p = g.edge_attr[:, :N_CONT], g.edge_attr[:, N_CONT:]
                c = torch.clamp((c - emean) / (estd + 1e-8), -10, 10)
                g.edge_attr = torch.cat([c, p], dim=1)
            if g.edge_attr is None or g.edge_attr.size(0) == 0:
                g.edge_attr = torch.zeros((g.edge_index.size(1), N_CONT), dtype=torch.float)
            g.x = torch.nan_to_num(g.x, 0.0, 0.0, 0.0); g.edge_attr = torch.nan_to_num(g.edge_attr, 0.0, 0.0, 0.0)
    _norm(tg); _norm(vg); _norm(eg)
    ck = torch.load(ENC_FILE, map_location=DEVICE)
    enc = GATEncoder(ck['node_dim'], ck['hidden_dim'], ck['num_heads'], ck['num_layers'], ck['edge_dim']).to(DEVICE)
    enc.load_state_dict(ck['encoder_state_dict']); enc.eval()
    for p in enc.parameters(): p.requires_grad = False
    if verbose: print(f'[frozen-emb] {ds}/{ssl}: initial_bs={initial_bs} max_edges={cap}')
    splits = {}
    for name, graphs in [('train', tg), ('val', vg), ('test', eg)]:
        starts = build_gap_aware_starts(graphs, SEQUENCE_LENGTH, SEQUENCE_STRIDE, MWG, enforce=ENFORCE_MAX_WINDOW_GAP, verbose=False)
        all_embs, used_bs = _encode_graphs_adaptive(enc, graphs, initial_bs=initial_bs, min_bs=min_bs, verbose=verbose, tag=name)
        if SANITIZE_FROZEN_CACHE: all_embs = all_embs.clamp_(-1e4, 1e4)
        all_embs_h = all_embs.half()
        struct_all = precompute_struct(graphs)
        labels_per_graph = np.array([dominant_family(g, DATASET_THRESHOLDS[ds]) for g in graphs])
        labels_seq = np.array([labels_per_graph[s + SEQUENCE_LENGTH - 1] for s in starts])
        yb_seq = (labels_seq != 'Benign').astype(np.float32)
        ym_seq = np.array([label_to_idx[c] for c in labels_seq], dtype=np.int64)
        items = []
        for k, s in enumerate(starts):
            items.append((all_embs_h[s:s + SEQUENCE_LENGTH].clone(),
                          struct_all[s:s + SEQUENCE_LENGTH].clone(),
                          float(yb_seq[k]), int(ym_seq[k])))
        splits[name] = items
        if verbose: print(f'    [{name}] {len(items)} sequences (bs used={used_bs})')
        del all_embs, all_embs_h, struct_all; gc.collect()
        if DEVICE.type == 'cuda': torch.cuda.empty_cache()
    del enc; gc.collect()
    if DEVICE.type == 'cuda': torch.cuda.empty_cache()
    _FROZEN_EMB_CACHE[key] = splits
    if verbose: print(f'[frozen-emb] {ds}/{ssl} ready in {_fmt_time(time.time() - t0)}')
    return splits

def clear_frozen_emb_cache(ds=None, ssl=None):
    if ds is None: _FROZEN_EMB_CACHE.clear(); print('[frozen-emb] cleared'); return
    v = [k for k in _FROZEN_EMB_CACHE if k[0] == ds and (ssl is None or k[1] == ssl)]
    for k in v: del _FROZEN_EMB_CACHE[k]
    print(f'[frozen-emb] cleared {len(v)} entries')
print('Cell 7c ready.')


# # Cell 7d — Persistent frozen cache

# In[79]:


def _cache_disk_path(ds, ssl):
    DATA_DIR = DATASET_CONFIG[ds]
    return os.path.join(DATA_DIR,
        f"frozen_cache_{ssl}_L{SEQUENCE_LENGTH}_S{SEQUENCE_STRIDE}_E{MAX_EDGES_PER_DATASET.get(ds, MAX_EDGES)}.pt")
def _cache_signature(ds, ssl):
    enc_path = os.path.join(DATASET_CONFIG[ds], f"ssl_checkpoints_{ssl}", "ssl_encoder_best.pth")
    try: enc_mt = os.path.getmtime(enc_path); enc_sz = os.path.getsize(enc_path)
    except OSError: enc_mt, enc_sz = -1.0, -1
    return {'ds': ds, 'ssl': ssl, 'seq_len': int(SEQUENCE_LENGTH), 'seq_stride': int(SEQUENCE_STRIDE),
            'enforce_gap': bool(ENFORCE_MAX_WINDOW_GAP), 'max_edges': int(MAX_EDGES_PER_DATASET.get(ds, MAX_EDGES)),
            'max_nodes': int(MAX_NODES), 'hidden': int(HIDDEN_DIM), 'heads': int(NUM_HEADS), 'layers': int(NUM_LAYERS),
            'attack_thr': float(DATASET_THRESHOLDS.get(ds, 0.0)), 'autocast': str(AUTOCAST_DTYPE),
            'enc_mtime': float(enc_mt), 'enc_size': int(enc_sz)}
def _labels_for_cache(ds):
    DATA_DIR = DATASET_CONFIG[ds]; atk = DATASET_THRESHOLDS[ds]; found = set()
    for split in ('train', 'val', 'test'):
        with open(os.path.join(DATA_DIR, f'{split}_graphs.pkl'), 'rb') as f:
            for g in pickle.load(f): found.add(dominant_family(g, atk))
    class_names = sorted(found); return class_names, {c: i for i, c in enumerate(class_names)}
def save_frozen_cache(ds, ssl, splits, verbose=True):
    path = _cache_disk_path(ds, ssl)
    torch.save({'splits': splits, 'signature': _cache_signature(ds, ssl)}, path)
    if verbose: print(f'[frozen-cache] saved {path} ({os.path.getsize(path)/1e6:.1f} MB)')
    return path
def load_frozen_cache(ds, ssl, verbose=True):
    path = _cache_disk_path(ds, ssl)
    if not os.path.isfile(path):
        if verbose: print(f'[frozen-cache] {ds}/{ssl}: no cache file'); return None
    try: payload = torch.load(path, map_location='cpu')
    except Exception as e: print(f'[frozen-cache] {ds}/{ssl}: cannot read ({e})'); return None
    if payload.get('signature', {}) != _cache_signature(ds, ssl):
        if verbose: print(f'[frozen-cache] {ds}/{ssl}: signature mismatch (stale)')
        return None
    if verbose:
        s = payload['splits']
        print(f"[frozen-cache] {ds}/{ssl}: loaded train={len(s['train'])} val={len(s['val'])} test={len(s['test'])}")
    return payload['splits']
def build_frozen_cache(ds, ssl, force=False, verbose=True, class_names=None, label_to_idx=None, initial_bs=None, min_bs=1):
    key = _frozen_emb_key(ds, ssl)
    if not force and key in _FROZEN_EMB_CACHE:
        if verbose: print(f'[frozen-cache] {ds}/{ssl}: memory HIT')
        return _FROZEN_EMB_CACHE[key]
    if not force:
        splits = load_frozen_cache(ds, ssl, verbose=verbose)
        if splits is not None: _FROZEN_EMB_CACHE[key] = splits; return splits
    if class_names is None or label_to_idx is None:
        class_names, label_to_idx = _labels_for_cache(ds)
    if initial_bs is None: initial_bs = CACHE_INITIAL_BS.get(ds, 16)
    splits = precache_frozen_embeddings(ds, ssl, class_names=class_names, label_to_idx=label_to_idx,
                                        force=True, verbose=verbose, initial_bs=initial_bs, min_bs=min_bs)
    if splits is not None:
        save_frozen_cache(ds, ssl, splits, verbose=verbose)
        _FROZEN_EMB_CACHE[key] = splits
    return splits
def cache_status(verbose=True):
    rows = []
    for ds in DATASET_CONFIG:
        for ssl in ('byol', 'moco'):
            path = _cache_disk_path(ds, ssl)
            if not os.path.isfile(path): rows.append((ds, ssl, 'MISSING', None)); continue
            try: p = torch.load(path, map_location='cpu')
            except Exception: rows.append((ds, ssl, 'CORRUPT', None)); continue
            state = 'VALID' if p.get('signature', {}) == _cache_signature(ds, ssl) else 'STALE'
            rows.append((ds, ssl, state, len(p['splits']['train'])))
    if verbose:
        print(f"{'dataset':<10} {'ssl':<6} {'state':<10} {'n_train':>8}")
        print('-' * 40)
        for ds, ssl, st, n in rows:
            print(f"{ds:<10} {ssl:<6} {st:<10} {(n if n is not None else '-'):>8}")
    return rows
print('Cell 7d ready.')


# # Cell 8 — Shared training helpers

# In[80]:


class _NoScaler:
    def scale(self, x): return x
    def unscale_(self, opt): pass
    def step(self, opt): opt.step()
    def update(self): pass
    def get_scale(self): return 1.0
def _make_scaler():
    if DEVICE.type != 'cuda': return _NoScaler()
    return GradScaler() if AUTOCAST_DTYPE == 'fp16' else _NoScaler()
def _build_lr_lambda(warmup_epochs, total_epochs):
    def fn(ep):
        if not LR_WARMUP_ENABLED or warmup_epochs <= 0:
            prog = ep / max(1, total_epochs - 1); return 0.5 * (1.0 + np.cos(np.pi * prog))
        if ep < warmup_epochs: return float(ep + 1) / float(warmup_epochs)
        prog = (ep - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * prog))
    return fn
def _balanced_accuracy_threshold_np(prob, y):
    from sklearn.metrics import roc_curve
    prob = np.asarray(prob); y = np.asarray(y)
    if len(set(y)) < 2: return 0.5, 0.0
    fpr, tpr, thr = roc_curve(y, prob); ba = 0.5 * (tpr + (1.0 - fpr)); idx = int(np.argmax(ba))
    return float(thr[idx]), float(ba[idx])
def _save_attention(attn_dir, attn_test, ym_test, class_names):
    os.makedirs(attn_dir, exist_ok=True)
    if attn_test.ndim == 3: attn_test = attn_test[:, -1, :]
    assert attn_test.ndim == 2
    mean_attn = attn_test.mean(axis=0)
    per_class = np.zeros((len(class_names), attn_test.shape[1]), dtype=np.float32)
    counts = np.zeros(len(class_names), dtype=int)
    for i in range(len(class_names)):
        mask = (ym_test == i); counts[i] = int(mask.sum())
        if mask.any(): per_class[i] = attn_test[mask].mean(axis=0)
    np.save(os.path.join(attn_dir, "mean_attention.npy"), mean_attn)
    np.save(os.path.join(attn_dir, "per_class_attention.npy"), per_class)
    np.save(os.path.join(attn_dir, "all_test_attn.npy"), attn_test)
    with open(os.path.join(attn_dir, "class_names.json"), 'w') as f:
        json.dump({'class_names': class_names, 'class_counts': counts.tolist()}, f, indent=2)
def _save_partial(ckpt_dir, tag, epoch, step, enc, tft, finetune):
    if not CKPT_EVERY_N_STEPS > 0: return
    p = {'tag': tag, 'epoch': int(epoch), 'step': int(step),
         'tft_state_dict': {k: v.detach().cpu() for k, v in tft.state_dict().items()}}
    if finetune and enc is not None:
        p['encoder_state_dict'] = {k: v.detach().cpu() for k, v in enc.state_dict().items()}
    torch.save(p, os.path.join(ckpt_dir, f"partial_{tag}_e{epoch}_s{step}.pth"))
    _prune_partials(ckpt_dir, tag, keep=KEEP_LAST_N_PARTIALS)
def _save_best(ckpt_dir, tag, epoch, enc, tft, metrics, finetune):
    p = {'tag': tag, 'epoch': int(epoch), 'metrics': metrics,
         'tft_state_dict': {k: v.detach().cpu() for k, v in tft.state_dict().items()}}
    if finetune and enc is not None:
        p['encoder_state_dict'] = {k: v.detach().cpu() for k, v in enc.state_dict().items()}
    torch.save(p, os.path.join(ckpt_dir, f"best_{tag}.pth"))
    
def _train_epoch_frozen(tft, data, opt, crit, scaler, epoch, ckpt_dir, tag, finetune):
    tft.train()
    tot = torch.zeros((), device=DEVICE); bn = 0.0; mn = 0.0
    gt = torch.zeros((), device=DEVICE); n = 0; gb = 0; n_skip = 0
    opt.zero_grad(set_to_none=True)
    pbar = tqdm(enumerate(data), total=len(data), desc=f"ep {epoch}", leave=False)
    for step, item in pbar:
        if step % 10 == 0:
            torch.cuda.empty_cache()
        emb, struct_seq, y_bin, y_multi = item
        ge = emb.to(DEVICE, non_blocking=True).float()
        struct_seq = struct_seq.to(DEVICE, non_blocking=True)
        if SANITIZE_GRAPH_EMB: ge = sanitize_tensor(ge)
        struct_seq = sanitize_tensor(struct_seq)
        y_bin = y_bin.to(DEVICE); y_multi = y_multi.to(DEVICE)
        with autocast_ctx():
            bl, ml, _, _ = tft(ge, struct_seq)
            loss, lb, lm = crit(bl, ml, y_bin, y_multi)
            loss = loss / GRAD_ACCUM_STEPS
        if SKIP_NONFINITE_STEPS and not torch.isfinite(loss):
            n_skip += 1; opt.zero_grad(set_to_none=True); continue
        scaler.scale(loss).backward()
        if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(data):
            scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_(list(tft.parameters()), 5.0)
            if not torch.isfinite(gn):
                n_skip += 1; opt.zero_grad(set_to_none=True); scaler.update(); continue
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            gt += gn.detach(); gb += 1
        tot += loss.detach() * GRAD_ACCUM_STEPS
        bn += lb if torch.is_tensor(lb) else float(lb)
        mn += lm if torch.is_tensor(lm) else float(lm)
        n += 1
        pbar.set_postfix({'loss': f"{float(tot)/max(n,1):.4f}", 'skip': n_skip})

        # ---- partial save: cadence OR last batch of epoch ----
        _do_partial = False
        if ckpt_dir is not None and CKPT_EVERY_N_STEPS > 0:
            is_cadence = ((step + 1) % CKPT_EVERY_N_STEPS == 0)
            is_last    = ((step + 1) == len(data))
            _do_partial = is_cadence or is_last
        if _do_partial:
            _save_partial(ckpt_dir, tag, epoch, step + 1,
                          None, tft, finetune)    # ← None for encoder in frozen mode

    pbar.close()
    d = max(n, 1)
    return {'total': float(tot)/d, 'binary': bn/d, 'multi': mn/d,
            'gnorm': float(gt)/max(gb, 1), 'n_skipped': int(n_skip)}
    
def _train_epoch_finetune(enc, tft, data, opt, crit, scaler, epoch, ckpt_dir, tag, finetune):
    enc.train(); tft.train()
    tot = torch.zeros((), device=DEVICE); bn = 0.0; mn = 0.0
    gt = torch.zeros((), device=DEVICE); n = 0; gb = 0; n_skip = 0
    opt.zero_grad(set_to_none=True)
    pbar = tqdm(enumerate(data), total=len(data), desc=f"ep {epoch}", leave=False)
    for step, item in pbar:
        # --- adaptive VRAM pressure monitor ---
        if DEVICE.type == 'cuda' and (step % 10 == 0):
            total     = torch.cuda.get_device_properties(0).total_memory
            reserved  = torch.cuda.memory_reserved()
            allocated = torch.cuda.memory_allocated()
            headroom  = total - reserved
            cache_gap = reserved - allocated
            if headroom < 1.0 * 1024**3 and cache_gap > 0.5 * 1024**3:
                torch.cuda.empty_cache()
                # if step % 50 == 0:
                #     print(f"    [vram] ep={epoch} step={step}  "
                #         f"reserved={reserved/1e9:.2f}GB  "
                #         f"alloc={allocated/1e9:.2f}GB  "
                #         f"freed={cache_gap/1e9:.2f}GB")

        fb, B, T, struct_seq, y_bin, y_multi = item
        struct_seq = struct_seq.to(DEVICE, non_blocking=True)
        y_bin = y_bin.to(DEVICE); y_multi = y_multi.to(DEVICE)
        with autocast_ctx():
            ge = _encode_sequence(enc, fb, B, T, DEVICE, frozen=False)
            if SANITIZE_GRAPH_EMB: ge = sanitize_tensor(ge)
            struct_seq = sanitize_tensor(struct_seq)
            bl, ml, _, _ = tft(ge, struct_seq)
            loss, lb, lm = crit(bl, ml, y_bin, y_multi)
            loss = loss / GRAD_ACCUM_STEPS
        if SKIP_NONFINITE_STEPS and not torch.isfinite(loss):
            n_skip += 1; opt.zero_grad(set_to_none=True); continue
        scaler.scale(loss).backward()
        if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(data):
            scaler.unscale_(opt)
            params = list(enc.parameters()) + list(tft.parameters())
            gn = torch.nn.utils.clip_grad_norm_(params, 5.0)
            if not torch.isfinite(gn):
                n_skip += 1; opt.zero_grad(set_to_none=True); scaler.update(); continue
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            gt += gn.detach(); gb += 1
        tot += loss.detach() * GRAD_ACCUM_STEPS
        bn += lb if torch.is_tensor(lb) else float(lb)
        mn += lm if torch.is_tensor(lm) else float(lm)
        n += 1
        pbar.set_postfix({'loss': f"{float(tot)/max(n,1):.4f}", 'skip': n_skip})
        _do_partial = False
        if ckpt_dir is not None and CKPT_EVERY_N_STEPS > 0:
            is_last = (step + 1) == len(data)
            is_cadence = ((step + 1) % CKPT_EVERY_N_STEPS == 0)
            _do_partial = is_cadence or is_last
        if _do_partial:
            _save_partial(ckpt_dir, tag, epoch, step + 1, enc, tft, finetune)
    pbar.close()
    d = max(n, 1)
    return {'total': float(tot)/d, 'binary': bn/d, 'multi': mn/d,
            'gnorm': float(gt)/max(gb, 1), 'n_skipped': int(n_skip)}
def _evaluate(enc, tft, data, collect_attn=False):
    torch.cuda.empty_cache()
    enc.eval(); tft.eval()
    bp, mp, yb, ym, aa = [], [], [], [], []
    with torch.no_grad():
        for item in data:
            if len(item) == 4:
                emb, struct_seq, y_bin, y_multi = item
                ge = emb.to(DEVICE, non_blocking=True).float()
                struct_seq = struct_seq.to(DEVICE, non_blocking=True)
                if SANITIZE_GRAPH_EMB: ge = sanitize_tensor(ge)
                struct_seq = sanitize_tensor(struct_seq)
            else:
                fb, B, T, struct_seq, y_bin, y_multi = item
                struct_seq = struct_seq.to(DEVICE, non_blocking=True)
                ge = _encode_sequence(enc, fb, B, T, DEVICE, frozen=True)
                if SANITIZE_GRAPH_EMB: ge = sanitize_tensor(ge)
                struct_seq = sanitize_tensor(struct_seq)
            bl, ml, aw, _ = tft(ge, struct_seq)
            bp.append(np.nan_to_num(torch.sigmoid(bl.float()).cpu().numpy(), 0.0, 1.0, 0.0))
            mp.append(np.nan_to_num(F.softmax(ml.float(), -1).cpu().numpy(), 1.0/max(ml.shape[-1],1), 1.0, 0.0))
            yb.append(y_bin.numpy()); ym.append(y_multi.numpy())
            if collect_attn:
                a = aw.float()
                if a.dim() == 3: a = a[:, -1, :] if a.size(1) == a.size(2) else a.mean(1)
                elif a.dim() == 4: a = a[:, 0, -1, :]
                aa.append(a.cpu().numpy())
    out = (np.concatenate(bp), np.concatenate(mp), np.concatenate(yb), np.concatenate(ym))
    return out + (np.concatenate(aa),) if collect_attn else out
print('Cell 8 ready — shared training helpers.')


# # Cell 9 — Dataset prep + single-seed trainer (lexicographic selection)

# In[81]:


def _make_dirs(ds, ssl, kind):
    base = os.path.join(DATASET_CONFIG[ds], f"phase3_{kind}_{ssl}")
    dirs = {'root': base, 'attn': os.path.join(base, 'attention'), 'partial': os.path.join(base, 'partial_checkpoints')}
    for d in dirs.values(): os.makedirs(d, exist_ok=True)
    dirs['ckpt'] = os.path.join(base, 'mc_stgat_ids.pth')
    dirs['report'] = os.path.join(base, 'phase3_report.json')
    return dirs

def _prepare_ctx(ds, ssl, finetune, use_cache=True, verbose=True):
    DATA_DIR = DATASET_CONFIG[ds]
    ENC_FILE = os.path.join(DATA_DIR, f"ssl_checkpoints_{ssl}", "ssl_encoder_best.pth")
    if not os.path.isfile(ENC_FILE): raise FileNotFoundError(f"no SSL encoder at {ENC_FILE}")
    cap = MAX_EDGES_PER_DATASET.get(ds, MAX_EDGES)
    def _load(s):
        with open(os.path.join(DATA_DIR, f'{s}_graphs.pkl'), 'rb') as f: return pickle.load(f)
    tg = [cap_graph(g, max_edges=cap) for g in _load('train')]
    vg = [cap_graph(g, max_edges=cap) for g in _load('val')]
    eg = [cap_graph(g, max_edges=cap) for g in _load('test')]
    with open(os.path.join(DATA_DIR, 'norm_stats.json')) as f: ns = json.load(f)
    N_CONT = ns['n_continuous_edge_feats']
    nmean = torch.tensor(ns['node_mean'], dtype=torch.float); nstd = torch.tensor(ns['node_std'], dtype=torch.float)
    emean = torch.tensor(ns['edge_mean'], dtype=torch.float); estd = torch.tensor(ns['edge_std'], dtype=torch.float)
    MWG = float(ns.get('max_window_gap_sec', DEFAULT_MAX_WINDOW_GAP_S))
    def _norm(graphs):
        for g in graphs:
            g.x = torch.clamp((g.x - nmean) / (nstd + 1e-8), -10, 10)
            if g.edge_attr is not None and g.edge_attr.size(0) > 0:
                if g.edge_attr.dim() > 2: g.edge_attr = g.edge_attr.view(-1, g.edge_attr.size(-1))
                c, p = g.edge_attr[:, :N_CONT], g.edge_attr[:, N_CONT:]
                c = torch.clamp((c - emean) / (estd + 1e-8), -10, 10)
                g.edge_attr = torch.cat([c, p], dim=1)
            if g.edge_attr is None or g.edge_attr.size(0) == 0:
                g.edge_attr = torch.zeros((g.edge_index.size(1), N_CONT), dtype=torch.float)
            g.x = torch.nan_to_num(g.x, 0.0, 0.0, 0.0); g.edge_attr = torch.nan_to_num(g.edge_attr, 0.0, 0.0, 0.0)
    _norm(tg); _norm(vg); _norm(eg)
    atk = DATASET_THRESHOLDS[ds]
    tl = np.array([dominant_family(g, atk) for g in tg])
    vl = np.array([dominant_family(g, atk) for g in vg])
    el = np.array([dominant_family(g, atk) for g in eg])
    class_names = sorted(set(tl.tolist()) | set(vl.tolist()) | set(el.tolist()))
    label_to_idx = {c: i for i, c in enumerate(class_names)}
    NUM_CLASSES = len(class_names)
    tb = (tl != 'Benign').astype(np.float32); vb = (vl != 'Benign').astype(np.float32); eb = (el != 'Benign').astype(np.float32)
    tm = np.array([label_to_idx[c] for c in tl], np.int64)
    vm = np.array([label_to_idx[c] for c in vl], np.int64)
    em = np.array([label_to_idx[c] for c in el], np.int64)
    tvis = np.unique(tm); tvis_fams = sorted({(class_names[i], ds) for i in tvis})
    ts_tr = build_gap_aware_starts(tg, SEQUENCE_LENGTH, SEQUENCE_STRIDE, MWG, enforce=ENFORCE_MAX_WINDOW_GAP, verbose=False)
    ts_va = build_gap_aware_starts(vg, SEQUENCE_LENGTH, SEQUENCE_STRIDE, MWG, enforce=ENFORCE_MAX_WINDOW_GAP, verbose=False)
    ts_te = build_gap_aware_starts(eg, SEQUENCE_LENGTH, SEQUENCE_STRIDE, MWG, enforce=ENFORCE_MAX_WINDOW_GAP, verbose=False)
    def _seq(y, s): return np.array([y[i + SEQUENCE_LENGTH - 1] for i in s])
    tr_yb, va_yb, te_yb = _seq(tb, ts_tr), _seq(vb, ts_va), _seq(eb, ts_te)
    tr_ym, va_ym, te_ym = _seq(tm, ts_tr), _seq(vm, ts_va), _seq(em, ts_te)
    te_lab = _seq(el, ts_te)
    str_tr = precompute_struct(tg); str_va = precompute_struct(vg); str_te = precompute_struct(eg)
    NUM_STRUCT = str_tr.size(1)
    def _mk(graphs, starts, yb, ym, struct, shuffle=True, sampler=None, batch_size=None):
        d = GraphSequenceDataset(graphs, starts, SEQUENCE_LENGTH, yb, ym, struct)
        kw = dict(batch_size=batch_size if batch_size is not None else BATCH_SIZE,
                collate_fn=collate_graph_sequences,
                num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == 'cuda'))
        return DataLoader(d, sampler=sampler, **kw) if sampler is not None \
            else DataLoader(d, shuffle=shuffle, **kw)
    if USE_BALANCED_SAMPLER_PER_DATASET.get(ds, True):
        nb = max(int((tr_yb == 0).sum()), 1); na = max(int((tr_yb == 1).sum()), 1)
        w = np.where(tr_yb == 0, 1.0/nb, 1.0/na); w = w / w.sum() * len(tr_yb)
        samp = WeightedRandomSampler(torch.tensor(w, dtype=torch.double), len(tr_yb), replacement=True)
        train_loader = _mk(tg, ts_tr, tr_yb, tr_ym, str_tr, sampler=samp)
    else:
        train_loader = _mk(tg, ts_tr, tr_yb, tr_ym, str_tr, shuffle=True)
    # Eval loaders use a smaller batch size because frozen mode
    # streams raw graphs through the encoder at final eval time.
    val_loader  = _mk(vg, ts_va, va_yb, va_ym, str_va, shuffle=False,
                    batch_size=EVAL_BATCH_SIZE)
    test_loader = _mk(eg, ts_te, te_yb, te_ym, str_te, shuffle=False,
                    batch_size=EVAL_BATCH_SIZE)
    ck = torch.load(ENC_FILE, map_location=DEVICE)
    def _fresh():
        e = GATEncoder(ck['node_dim'], ck['hidden_dim'], ck['num_heads'], ck['num_layers'], ck['edge_dim']).to(DEVICE)
        e.load_state_dict(ck['encoder_state_dict'])
        if not finetune:
            e.eval()
            for p in e.parameters(): p.requires_grad = False
        return e
    cache = None
    if use_cache and not finetune:
        cache = build_frozen_cache(ds, ssl, force=False, verbose=verbose, class_names=class_names, label_to_idx=label_to_idx)
    if verbose:
        print(f'  loaded: train={len(tg)} val={len(vg)} test={len(eg)}')
        print(f'  sequences: train={len(ts_tr)} val={len(ts_va)} test={len(ts_te)}')
        print(f'  classes={NUM_CLASSES} struct_dim={NUM_STRUCT} mode={"FINETUNE" if finetune else "FROZEN"}')
    return dict(encoder_ckpt=ck, fresh_encoder=_fresh, NUM_CLASSES=NUM_CLASSES, NUM_STRUCT=NUM_STRUCT,
                class_names=class_names, label_to_idx=label_to_idx, train_visible_idx=tvis, train_visible_fams=tvis_fams,
                train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                tr_yb=tr_yb, va_yb=va_yb, te_yb=te_yb, tr_ym=tr_ym, va_ym=va_ym, te_ym=te_ym, te_lab=te_lab,
                cache=cache, MWG=MWG, DATA_DIR=DATA_DIR,
                split_sizes=dict(train=len(tg), val=len(vg), test=len(eg)),
                seq_counts=dict(train=len(ts_tr), val=len(ts_va), test=len(ts_te)))
def _eval_batches(ctx, split, finetune):
    """Return batches for evaluation. For frozen mode, use cached embeddings
    (no encoder forward). For fine-tune, use the streaming loader."""
    if not finetune and ctx.get('cache') is not None:
        return _rebatch(ctx['cache'][split], None, EVAL_BATCH_SIZE)
    return ctx[f'{split}_loader']

def _rebatch(flat_cache, order=None, bs=None):
    if bs is None: bs = BATCH_SIZE
    if order is None: order = range(len(flat_cache))
    order = list(order); batches = []
    for i in range(0, len(order), bs):
        chunk = [flat_cache[j] for j in order[i:i+bs]]
        embs    = torch.stack([c[0] for c in chunk], 0)
        structs = torch.stack([c[1] for c in chunk], 0)
        ybs     = torch.tensor([float(c[2]) for c in chunk], dtype=torch.float32)
        yms     = torch.tensor([int(c[3])   for c in chunk], dtype=torch.long)
        batches.append((embs, structs, ybs, yms))
    return batches

def _lexicographic_improved(val_now, val_best, priority, tol):
    for m in priority:
        cur = float(val_now.get(m, float('-inf')))
        bst = float(val_best.get(m, float('-inf')))
        if not np.isfinite(cur): continue
        if not np.isfinite(bst): return True
        if cur > bst + tol: return True
        if cur < bst - tol: return False
    return False

def _train_single_seed(ds, ssl, finetune, seed, ctx, ckpt_dir, tag, epochs=None, warmup=None, patience=None, verbose=True):
    if epochs   is None: epochs   = EPOCHS
    if warmup   is None: warmup   = WARMUP_EPOCHS
    if patience is None: patience = PATIENCE
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    enc = ctx['fresh_encoder']()
    tft = MC_STGAT_IDS(ctx['encoder_ckpt']['hidden_dim'], ctx['NUM_STRUCT'], TFT_HIDDEN_DIM, ctx['NUM_CLASSES'], TFT_NUM_HEADS, TFT_DROPOUT).to(DEVICE)
    decay_params, no_decay_params = [], []
    for name, param in tft.named_parameters():
        if not param.requires_grad: continue
        if any(k in name for k in ['attn.q', 'attn.k', 'bias', 'norm']):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    groups = []
    if finetune:
        groups.append({'params': enc.parameters(),
                       'lr': LEARNING_RATE * ENCODER_LR_MULT,
                       'weight_decay': WEIGHT_DECAY})
    groups.append({'params': decay_params, 'lr': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY})
    groups.append({'params': no_decay_params, 'lr': LEARNING_RATE, 'weight_decay': 0.0})
    opt    = torch.optim.Adam(groups)
    sched  = torch.optim.lr_scheduler.LambdaLR(opt, _build_lr_lambda(warmup, epochs))
    scaler = _make_scaler()
    crit   = CombinedLoss(w_multi=W_MULTI, pos_weight=POS_WEIGHT, focal_alpha=FOCAL_ALPHA, focal_gamma=FOCAL_GAMMA).to(DEVICE)
    priority = list(SELECTION_METRIC_PRIORITY); tol = float(SELECTION_METRIC_TOL)
    frozen_train = frozen_val = None; balanced_w = None
    if not finetune and ctx['cache'] is not None:
        frozen_train = ctx['cache']['train']; frozen_val = ctx['cache']['val']
        if USE_BALANCED_SAMPLER_PER_DATASET.get(ds, True):
            yb_arr = np.array([float(c[2]) for c in frozen_train])
            n_b = max(int((yb_arr == 0).sum()), 1); n_a = max(int((yb_arr == 1).sum()), 1)
            w = np.where(yb_arr == 0, 1.0/n_b, 1.0/n_a).astype(np.float64); balanced_w = w / w.sum()
            if verbose: print(f'  [frozen-data] balanced sampler ON (benign={n_b} attack={n_a})')
    def _epoch_train_batches(epoch_seed):
        if frozen_train is None: return ctx['train_loader']
        rng_ep = np.random.default_rng(epoch_seed); n = len(frozen_train)
        if balanced_w is not None: order = rng_ep.choice(n, size=n, replace=True, p=balanced_w)
        else: order = rng_ep.permutation(n)
        return _rebatch(frozen_train, order=order.tolist(), bs=BATCH_SIZE)
    data_val = _rebatch(frozen_val, None, BATCH_SIZE) if frozen_val is not None else ctx['val_loader']
    best_val = {}; best_score = -1.0; best_sec = -1.0; best_epoch = 0
    best_enc = None; best_tft = None; no_imp = 0; best_reason = 'init'
    if verbose:
        print(f'  selection priority = {priority}  tol={tol}')
        _print_epoch_header(tag, epochs)
    t0 = time.time()
    for ep in range(1, epochs + 1):
        if DEVICE.type == 'cuda': torch.cuda.empty_cache()
        ep_t0 = time.time()
        if finetune:
            tr = _train_epoch_finetune(enc, tft, ctx['train_loader'], opt, crit, scaler, ep, ckpt_dir, tag, finetune)
        else:
            ep_seed = int(hashlib.md5(f'{seed}|{ep}|{ds}|{ssl}'.encode()).hexdigest()[:8], 16)
            data_train_ep = _epoch_train_batches(ep_seed)
            tr = _train_epoch_frozen(tft, data_train_ep, opt, crit, scaler, ep, ckpt_dir, tag, finetune)
        pbv, pmv, ybv, ymv = _evaluate(enc, tft, data_val)
        val = _val_metrics(pbv, pmv, ybv, ymv, thr=0.5, train_visible_idx=ctx['train_visible_idx'], class_names=ctx['class_names'], dataset=ds)
        sched.step(); cur_lr = opt.param_groups[-1]['lr']; dt = time.time() - ep_t0
        warmed = ep >= warmup
        if warmed:
            improved = _lexicographic_improved(val, best_val, priority, tol)
            if improved:
                reason = 'init'
                for m in priority:
                    cur = float(val.get(m, float('-inf'))); bst = float(best_val.get(m, float('-inf')))
                    if not np.isfinite(bst) or cur > bst + tol: reason = m; break
                best_reason = reason
        else: improved = False
        if improved:
            best_val = {k: float(v) for k, v in val.items() if isinstance(v, (int, float))}
            best_score = float(val.get(MAIN_SELECTION_METRIC, best_score))
            best_sec = float(val.get('macro_f1_merged', best_sec))
            best_epoch = ep
            best_enc = {k: v.clone() for k, v in enc.state_dict().items()}
            best_tft = {k: v.clone() for k, v in tft.state_dict().items()}
            no_imp = 0
            _save_best(ckpt_dir, tag, ep, enc, tft,
                       {**{f'val_{m}': float(val.get(m, float('nan'))) for m in priority},
                        'val_roc': float(val['roc']), 'val_pr': float(val['pr']),
                        'val_f1': float(val['f1']),
                        'val_macro_f1_merged': float(val['macro_f1_merged']),
                        'selection_reason': reason, 'selection_priority': priority}, finetune)
        else:
            if warmed: no_imp += 1
        pl = max(patience - no_imp, 0)
        note = f'*best ({best_reason})' if improved else ('  (warmup)' if not warmed else '')
        if tr.get('n_skipped', 0) > 0: note += f"  skip={tr['n_skipped']}"
        if verbose: _print_epoch_row(ep, epochs, tr, val, tr['gnorm'], cur_lr, pl, best_score, dt, note)
        if warmed and no_imp >= patience:
            if verbose: print(f'  early stop at ep {ep}')
            break
    if best_enc is None:
        best_enc = {k: v.clone() for k, v in enc.state_dict().items()}
        best_tft = {k: v.clone() for k, v in tft.state_dict().items()}
        best_score = float(val[MAIN_SELECTION_METRIC])
    enc.load_state_dict(best_enc); tft.load_state_dict(best_tft)
    if verbose:
        print(f'  -> best @ ep {best_epoch} (reason={best_reason},  ' + '  '.join(f'{m}={best_val.get(m, float("nan")):.4f}' for m in priority) + f')  | time {_fmt_time(time.time() - t0)}')
    return enc, tft, best_score, best_epoch
print('Cell 9 ready.')


# # Cell 10 — Three orchestrators: run_single / run_ensemble / run_fewshot

# In[82]:


def run_single(ds, ssl, finetune, force=False):
    kind = 'finetune_single' if finetune else 'frozen_single'
    dirs = _make_dirs(ds, ssl, kind)
    print(f"\n{'='*70}\nSINGLE  {ds}  ssl={ssl.upper()}  mode={'FINETUNE' if finetune else 'FROZEN'}\n{'='*70}")
    if (not force) and os.path.isfile(dirs['report']):
        print('[SKIP] existing single-seed report')
        with open(dirs['report']) as f: return json.load(f)
    ctx = _prepare_ctx(ds, ssl, finetune)
    enc, tft, vscore, bep = _train_single_seed(ds, ssl, finetune, SEED, ctx, dirs['partial'], tag=f"{ds}_{ssl}_single")
    pb, pm, yb, ym, attn = _evaluate(enc, tft,
                                    _eval_batches(ctx, 'test', finetune),
                                    collect_attn=True)
    pbv, _, ybv, _ = _evaluate(enc, tft, _eval_batches(ctx, 'val', finetune))
    yb_i = yb.astype(int)
    thr_f1, _  = _optimal_threshold(pbv, ybv.astype(int))
    thr_bal, _ = _balanced_accuracy_threshold_np(pbv, ybv.astype(int))
    if USE_PRIOR_MATCHED_THRESHOLD:
        thr = _prior_matched_threshold(pbv, ybv.astype(int))
    else:
        thr = thr_f1
    m = _binary_metrics_at(pb, yb_i, thr)
    roc = float(roc_auc_score(yb_i, pb))
    pr  = float(average_precision_score(yb_i, pb))
    ym_pred = pm.argmax(-1)
    mf1 = float(f1_score(ym, ym_pred, average='macro', zero_division=0))
    wf1 = float(f1_score(ym, ym_pred, average='weighted', zero_division=0))
    print(f"\n  TEST ROC={roc:.4f} PR={pr:.4f} F1={m['f1']:.4f} FPR={m['fpr']:.4f} FNR={m['fnr']:.4f}")
    print(f'  MULTI macro-F1={mf1:.4f} weighted-F1={wf1:.4f}')
    _print_binary_confusion(m, label='test')
    pc = _per_class_score_analysis(pb, ctx['te_lab'], ctx['class_names'], thr)
    _print_per_class_table(pc, thr, 'test per-class')
    _save_attention(dirs['attn'], attn, ym, ctx['class_names'])
    torch.save({'encoder_state_dict': {k: v.cpu() for k, v in enc.state_dict().items()},
                'tft_state_dict': {k: v.cpu() for k, v in tft.state_dict().items()},
                'class_names': ctx['class_names'], 'label_to_idx': ctx['label_to_idx'],
                'train_visible_fams': ctx['train_visible_fams'],
                'sequence_length': SEQUENCE_LENGTH, 'num_classes': ctx['NUM_CLASSES'],
                'num_struct_feats': ctx['NUM_STRUCT'], 'finetune': finetune,
                'ssl_method': ssl, 'dataset': ds, 'threshold': float(thr)}, dirs['ckpt'])
    report = {'stage': 'single', 'dataset': ds, 'ssl': ssl, 'finetune': finetune,
              'seed': SEED, 'val_selection_score': float(vscore), 'best_epoch': int(bep),
              'thresholds': {'f1_opt': float(thr_f1), 'balanced': float(thr_bal), 'used': float(thr)},
              'test': {'roc': roc, 'pr': pr, 'f1': float(m['f1']), 'fpr': float(m['fpr']), 'fnr': float(m['fnr']),
                       'macro_f1': mf1, 'weighted_f1': wf1,
                       'n_true_attack': int(m['n_true_attack']), 'n_pred_attack': int(m['n_pred_attack']),
                       'tn': int(m['tn_benign_correct']), 'fp': int(m['fp_benign_wrong']),
                       'fn': int(m['fn_attack_missed']), 'tp': int(m['tp_attack_correct'])},
              'per_class': pc, 'split_sizes': ctx['split_sizes'], 'seq_counts': ctx['seq_counts']}
    with open(dirs['report'], 'w') as f: json.dump(report, f, indent=2)
    print(f"  saved {dirs['report']}")
    del enc, tft, ctx; gc.collect()
    if DEVICE.type == 'cuda': torch.cuda.empty_cache()
    return report

def run_ensemble(ds, ssl, finetune, seeds=None, force=False):
    if seeds is None: seeds = list(ENSEMBLE_SEEDS_DEFAULT)
    kind = 'finetune_ensemble' if finetune else 'frozen_ensemble'
    dirs = _make_dirs(ds, ssl, kind)
    print(f"\n{'='*70}\nENSEMBLE  {ds}  ssl={ssl.upper()}  seeds={seeds}  mode={'FINETUNE' if finetune else 'FROZEN'}\n{'='*70}")
    if (not force) and os.path.isfile(dirs['report']):
        print('[SKIP] existing ensemble report')
        with open(dirs['report']) as f: return json.load(f)
    ctx = _prepare_ctx(ds, ssl, finetune)
    per_pb, per_pm, per_metrics = [], [], []
    yb_last = ym_last = None
    for s in seeds:
        print(f'\n  -- seed {s} --')
        enc, tft, vscore, bep = _train_single_seed(ds, ssl, finetune, s, ctx, dirs['partial'], tag=f"{ds}_{ssl}_ens{s}")
        pb, pm, yb, ym = _evaluate(enc, tft, _eval_batches(ctx, 'test', finetune))
        yb_last, ym_last = yb, ym; per_pb.append(pb); per_pm.append(pm)
        roc_s = float(roc_auc_score(yb.astype(int), pb))
        mf1_s = float(f1_score(ym, pm.argmax(-1), average='macro', zero_division=0))
        per_metrics.append({'seed': int(s), 'val_score': float(vscore), 'test_roc': roc_s, 'test_macro_f1': mf1_s})
        print(f'    seed {s}: val={vscore:.4f}  testROC={roc_s:.4f}  mF1={mf1_s:.4f}')
        del enc, tft; gc.collect()
        if DEVICE.type == 'cuda': torch.cuda.empty_cache()
    pb_ens = np.mean(per_pb, axis=0); pm_ens = np.mean(per_pm, axis=0)
    yb_i = yb_last.astype(int)
    thr_f1, _ = _optimal_threshold(pb_ens, yb_i); thr_bal, _ = _balanced_accuracy_threshold_np(pb_ens, yb_i)
    if USE_PRIOR_MATCHED_THRESHOLD:
        thr = _prior_matched_threshold(pbv, ybv.astype(int))
    else:
        thr = thr_f1
    m = _binary_metrics_at(pb_ens, yb_i, thr)
    roc = float(roc_auc_score(yb_i, pb_ens)); pr = float(average_precision_score(yb_i, pb_ens))
    ym_pred = pm_ens.argmax(-1)
    mf1 = float(f1_score(ym_last, ym_pred, average='macro', zero_division=0))
    wf1 = float(f1_score(ym_last, ym_pred, average='weighted', zero_division=0))
    rng = np.random.default_rng(BOOTSTRAP_SEED); n = len(yb_i); boot = np.empty(BOOTSTRAP_N)
    for b in range(BOOTSTRAP_N):
        idx = rng.integers(0, n, size=n); boot[b] = roc_auc_score(yb_i[idx], pb_ens[idx])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"\n  ENSEMBLE: ROC={roc:.4f} [{lo:.4f},{hi:.4f}]  F1={m['f1']:.4f}  FPR={m['fpr']:.4f}  FNR={m['fnr']:.4f}  mF1={mf1:.4f}")
    pc = _per_class_score_analysis(pb_ens, ctx['te_lab'], ctx['class_names'], thr)
    report = {'stage': 'ensemble', 'dataset': ds, 'ssl': ssl, 'finetune': finetune,
            'seeds': list(map(int, seeds)), 'per_seed': per_metrics,
            'thresholds': {'f1_opt': float(thr_f1), 'balanced': float(thr_bal),
                            'used': float(thr)},
            'test': {'roc': roc, 'pr': pr, 'f1': float(m['f1']),
                    'fpr': float(m['fpr']), 'fnr': float(m['fnr']),
                    'macro_f1': mf1, 'weighted_f1': wf1,
                    'roc_ci95': [float(lo), float(hi)],
                    # ---- added: confusion-matrix counts ----
                    'n_true_attack':  int(m['n_true_attack']),
                    'n_pred_attack':  int(m['n_pred_attack']),
                    'tn': int(m['tn_benign_correct']),
                    'fp': int(m['fp_benign_wrong']),
                    'fn': int(m['fn_attack_missed']),
                    'tp': int(m['tp_attack_correct'])},
            'per_class': pc, 'split_sizes': ctx['split_sizes'],
            'seq_counts': ctx['seq_counts']}
    with open(dirs['report'], 'w') as f: json.dump(report, f, indent=2)
    print(f"  saved {dirs['report']}")
    del ctx; gc.collect()
    if DEVICE.type == 'cuda': torch.cuda.empty_cache()
    return report

def run_fewshot(ds, ssl, finetune, fractions=None, force=False):
    if fractions is None: fractions = FRACTIONS
    kind = 'finetune_fewshot' if finetune else 'frozen_fewshot'
    dirs = _make_dirs(ds, ssl, kind)
    print(f"\n{'='*70}\nFEW-SHOT  {ds}  ssl={ssl.upper()}  fractions={fractions}\n{'='*70}")
    ctx = _prepare_ctx(ds, ssl, finetune)
    n_seq = ctx['seq_counts']['train']
    rng = np.random.default_rng(SEED); perm = rng.permutation(n_seq)
    results = {}
    for frac in fractions:
        fpath = os.path.join(dirs['root'], f'fewshot_{int(frac*100):03d}.json')
        if (not force) and os.path.isfile(fpath):
            with open(fpath) as f: results[frac] = json.load(f)
            print(f'  [skip] frac={frac:.0%} cached'); continue
        n_use = max(20, int(frac * n_seq)); idx = perm[:n_use]
        print(f'\n  frac={frac:.0%}  n={n_use}')
        torch.manual_seed(SEED); np.random.seed(SEED)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
        enc = ctx['fresh_encoder']()
        tft = MC_STGAT_IDS(ctx['encoder_ckpt']['hidden_dim'], ctx['NUM_STRUCT'], TFT_HIDDEN_DIM, ctx['NUM_CLASSES'], TFT_NUM_HEADS, TFT_DROPOUT).to(DEVICE)
        decay_params, no_decay_params = [], []
        for name, param in tft.named_parameters():
            if not param.requires_grad: continue
            if any(k in name for k in ['attn.q', 'attn.k', 'bias', 'norm']):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        groups = []
        if finetune:
            groups.append({'params': enc.parameters(), 'lr': LEARNING_RATE * ENCODER_LR_MULT, 'weight_decay': WEIGHT_DECAY})
        groups.append({'params': decay_params, 'lr': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY})
        groups.append({'params': no_decay_params, 'lr': LEARNING_RATE, 'weight_decay': 0.0})
        opt = torch.optim.Adam(groups)
        fu_ep = FEWSHOT_EPOCHS
        fu_wu = FEWSHOT_WARMUP_SMALL if n_use < SMALL_LABEL_THRESHOLD else FEWSHOT_WARMUP
        fu_pa = FEWSHOT_PATIENCE_SMALL if n_use < SMALL_LABEL_THRESHOLD else PATIENCE
        sched = torch.optim.lr_scheduler.LambdaLR(opt, _build_lr_lambda(fu_wu, fu_ep))
        scaler = _make_scaler()
        crit = CombinedLoss(w_multi=W_MULTI, pos_weight=POS_WEIGHT, focal_alpha=FOCAL_ALPHA, focal_gamma=FOCAL_GAMMA).to(DEVICE)
        if not finetune and ctx['cache'] is not None:
            data_tr = _rebatch(ctx['cache']['train'], order=idx.tolist(), bs=BATCH_SIZE)
        else:
            base_ds = GraphSequenceDataset(ctx['train_loader'].dataset.graphs, ctx['train_loader'].dataset.starts,
                                            SEQUENCE_LENGTH, ctx['tr_yb'], ctx['tr_ym'], ctx['train_loader'].dataset.struct)
            data_tr = DataLoader(Subset(base_ds, idx.tolist()), batch_size=BATCH_SIZE, shuffle=True,
                                 collate_fn=collate_graph_sequences, num_workers=NUM_WORKERS,
                                 pin_memory=(DEVICE.type == 'cuda'))
        best_score = -1.0; best_ep = 0; no_imp = 0; best_enc = None; best_tft = None
        tag = f"{ds}_{ssl}_fs{int(frac*100):03d}"
        _print_epoch_header(tag, fu_ep)
        for ep in range(1, fu_ep + 1):
            if DEVICE.type == 'cuda': torch.cuda.empty_cache()
            ep_t0 = time.time()
            if finetune:
                tr = _train_epoch_finetune(enc, tft, data_tr, opt, crit, scaler, ep, dirs['partial'], tag, finetune)
            else:
                tr = _train_epoch_frozen(tft, data_tr, opt, crit, scaler, ep, dirs['partial'], tag, finetune)
            pbv, pmv, ybv, ymv = _evaluate(enc, tft, _eval_batches(ctx, 'val', finetune))
            val = _val_metrics(pbv, pmv, ybv, ymv, thr=0.5, train_visible_idx=ctx['train_visible_idx'], class_names=ctx['class_names'], dataset=ds)
            sched.step(); cur_lr = opt.param_groups[-1]['lr']; dt = time.time() - ep_t0
            score = val[MAIN_SELECTION_METRIC]; warmed = ep >= fu_wu
            imp = warmed and (score > best_score + 1e-4)
            if imp:
                best_score = score; best_ep = ep
                best_enc = {k: v.clone() for k, v in enc.state_dict().items()}
                best_tft = {k: v.clone() for k, v in tft.state_dict().items()}; no_imp = 0
            else:
                if warmed: no_imp += 1
            _print_epoch_row(ep, fu_ep, tr, val, tr['gnorm'], cur_lr, max(fu_pa - no_imp, 0), best_score, dt, '*best' if imp else '')
            if warmed and no_imp >= fu_pa:
                print(f'  early stop at ep {ep}'); break
        if best_enc is None:
            best_enc = {k: v.clone() for k, v in enc.state_dict().items()}
            best_tft = {k: v.clone() for k, v in tft.state_dict().items()}
            best_score = float(val[MAIN_SELECTION_METRIC])
        enc.load_state_dict(best_enc); tft.load_state_dict(best_tft)
        pbt, pmt, ybt, ymt = _evaluate(enc, tft, _eval_batches(ctx, 'test', finetune))  
        ybt_i = ybt.astype(int); troc = float(roc_auc_score(ybt_i, pbt))
        thr_f, _ = _optimal_threshold(pbt, ybt_i)
        m_fs = _binary_metrics_at(pbt, ybt_i, thr_f)
        tmf1 = float(f1_score(ymt, pmt.argmax(-1), average='macro', zero_division=0))
        results[frac] = {'frac': float(frac), 'n_train': int(n_use), 'val_selection_score': float(best_score),
                         'best_epoch': int(best_ep), 'test_roc': troc, 'test_f1': float(m_fs['f1']),
                         'test_fpr': float(m_fs['fpr']), 'test_fnr': float(m_fs['fnr']),
                         'test_macro_f1_merged': tmf1, 'test_threshold': float(thr_f)}
        print(f'  frac {frac:.0%} -> ROC {troc:.4f}  F1 {m_fs["f1"]:.4f}  mF1 {tmf1:.4f}')
        with open(fpath, 'w') as f: json.dump(results[frac], f, indent=2)
        del enc, tft, opt, sched, scaler; gc.collect()
        if DEVICE.type == 'cuda': torch.cuda.empty_cache()
    return results
print('Cell 10 ready — three orchestrators.')


# # Cell 10b — Build MoCo frozen caches (run once)

# In[83]:


CACHE_BUILD_DS  = ['gotham', 'nf_bot', 'nf_ton']
CACHE_BUILD_SSL = ['moco']       
FORCE_REBUILD   = False
CACHE_INITIAL_BS = {'gotham': 64, 'nf_ton': 32, 'nf_bot': 16}
CACHE_MIN_BS = 1

print('Building MoCo frozen embedding caches...')
print()
for ds in CACHE_BUILD_DS:
    for ssl in CACHE_BUILD_SSL:
        try:
            print(f'--- {ds} / {ssl} ---')
            build_frozen_cache(ds, ssl, force=FORCE_REBUILD, verbose=True,
                               class_names=None, label_to_idx=None)
            print()
        except FileNotFoundError as e:
            print(f'  [skip] {ds}/{ssl}: {e}\n')
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f'  [FAIL] {ds}/{ssl}: {e}\n')
print('Cache build done. Current status:')
cache_status()


# # Cell 11 — Run calls (MoCo)

# In[ ]:


_ = run_single('gotham', 'moco', finetune=False, force=True)
_ = run_single('nf_bot', 'moco', finetune=False, force=True)
_ = run_single('nf_ton', 'moco', finetune=False, force=True)


# In[ ]:


_ = run_ensemble('gotham', 'moco', finetune=False, force=True)
_ = run_ensemble('nf_bot', 'moco', finetune=False, force=True)
_ = run_ensemble('nf_ton', 'moco', finetune=False, force=True)


# In[ ]:


# with _OverrideConfig(BATCH_SIZE = 16, EVAL_BATCH_SIZE = 8, EPOCHS= 100, PATIENCE= 30):
#     _ = run_single('gotham', 'moco', finetune=True, force=True)
#     _ = run_single('nf_bot', 'moco', finetune=True, force=True)
#     _ = run_single('nf_ton', 'moco', finetune=True, force=True)
print('[BYPASSED] Finetuning skipped per user instruction.')


# In[ ]:


# -------- MoCo (when SSL MoCo checkpoints exist) --------
_ = run_fewshot('gotham', 'moco', finetune=False, force=True)
_ = run_fewshot('nf_bot', 'moco', finetune=False, force=True)
_ = run_fewshot('nf_ton', 'moco', finetune=False, force=True)


# # Cell 12 — Cross-config summary (BYOL + MoCo)

# In[86]:


def _load_all_reports():
    out = {}
    for ds in DATASET_CONFIG:
        root = DATASET_CONFIG[ds]
        if not os.path.isdir(root): continue
        for entry in sorted(os.listdir(root)):
            if not entry.startswith('phase3_'): continue
            rp = os.path.join(root, entry, 'phase3_report.json')
            if not os.path.isfile(rp): continue
            try:
                with open(rp) as f: r = json.load(f)
            except Exception: continue
            key = (ds, r.get('ssl', '?'), r.get('stage', '?'), 'finetune' if r.get('finetune') else 'frozen')
            out[key] = r
    return out
REPORTS = _load_all_reports()
print(f'[summary] {len(REPORTS)} report(s):')
for k in sorted(REPORTS): print(f'  {k}')
print(f"\n{'='*110}")
print(f"{'dataset':<8} {'ssl':<6} {'stage':<10} {'mode':<10} {'ROC':>8} {'PR':>8} {'F1':>8} {'FPR':>8} {'FNR':>8} {'mF1':>8}")
print('-'*110)
for k in sorted(REPORTS):
    r = REPORTS[k]; t = r.get('test', {})
    f = lambda key: float(t.get(key, float('nan')))
    print(f"{k[0]:<8} {k[1]:<6} {k[2]:<10} {k[3]:<10} "
          f"{f('roc'):>8.4f} {f('pr'):>8.4f} {f('f1'):>8.4f} "
          f"{f('fpr'):>8.4f} {f('fnr'):>8.4f} {f('macro_f1'):>8.4f}")


# TFT — Attention visualization

# In[ ]:


# =====================================================================
# Cell 13 — TFT Attention Visualization Suite (MoCo)
#
# Loads attention weights saved by _save_attention() during TFT evaluation
# and generates:
#   1. Mean Temporal Attention Profile & Cohort Dispersion Boxplot
#   2. Per-Class Temporal Attention Heatmap (filtered to active classes)
#   3. Multi-Dataset Comparison Overview
# =====================================================================
import os, json, glob
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except Exception:
    pass

# Flags & Directories
SHOW_PLOTS = False
SAVE_PLOTS = True
PLOT_ROOT  = 'phase3_plots'
PLOT_DPI   = 160

def _attn_out_dir(ds, ssl, mode):
    p = os.path.join(PLOT_ROOT, ssl.upper(), ds, mode)
    os.makedirs(p, exist_ok=True)
    return p

def load_tft_attention(ds, ssl='moco', mode='finetune_single'):
    """Load precomputed test attention matrices."""
    adir = os.path.join(DATASET_CONFIG[ds], f'phase3_{mode}_{ssl}', 'attention')
    if not os.path.isdir(adir):
        return None
    try:
        mean_a = np.load(os.path.join(adir, 'mean_attention.npy'))
        per_cls = np.load(os.path.join(adir, 'per_class_attention.npy'))
        all_a  = np.load(os.path.join(adir, 'all_test_attn.npy'))
    except Exception as e:
        print(f"  [attn] error loading from {adir}: {e}")
        return None
    cpath = os.path.join(adir, 'class_names.json')
    if os.path.isfile(cpath):
        meta = json.load(open(cpath))
        names  = meta.get('class_names', [f"c{i}" for i in range(per_cls.shape[0])])
        counts = meta.get('class_counts', [0] * per_cls.shape[0])
    else:
        names  = [f"c{i}" for i in range(per_cls.shape[0])]
        counts = [0] * per_cls.shape[0]
    return mean_a, per_cls, all_a, names, counts

def plot_tft_attention(ds, ssl='moco', mode='finetune_single', show=SHOW_PLOTS, save=SAVE_PLOTS):
    res = load_tft_attention(ds, ssl, mode)
    if res is None:
        print(f"  [WARN] No attention data found for {ds}/{ssl}/{mode}")
        return
    mean_a, per_cls, all_a, names, counts = res
    out_dir = _attn_out_dir(ds, ssl, mode)
    T = len(mean_a)
    col_labels = [f"t-{T-1-i}" if i < T-1 else "t (query)" for i in range(T)]

    # ---------------- 1. Mean Attention + Cohort Dispersion ----------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Bar chart with highlighted query position
    colors = ['#2b5c8f'] * (T - 1) + ['#d62728']
    bars = ax1.bar(np.arange(T), mean_a, color=colors, edgecolor='black', alpha=0.85)
    for b, v in zip(bars, mean_a):
        ax1.text(b.get_x() + b.get_width()/2, v + max(mean_a)*0.02,
                f"{v:.3f}", ha='center', fontsize=9, fontweight='bold')
    ax1.set_xticks(np.arange(T))
    ax1.set_xticklabels(col_labels, fontsize=10, rotation=25)
    ax1.set_ylabel("Mean Attention Weight", fontsize=11, fontweight='bold')
    ax1.set_title(f"Mean Temporal Attention ({ds.upper()} — {ssl.upper()} — {mode})", fontsize=12, fontweight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.6)

    # Boxplot distribution
    bp = ax2.boxplot(all_a, positions=np.arange(T), patch_artist=True,
                     boxprops=dict(facecolor='#aec7e8', color='black'),
                     medianprops=dict(color='red', linewidth=1.5))
    ax2.set_xticks(np.arange(T))
    ax2.set_xticklabels(col_labels, fontsize=10, rotation=25)
    ax2.set_ylabel("Attention Weight Distribution", fontsize=11, fontweight='bold')
    ax2.set_title(f"Cohort Attention Dispersion (Test N={len(all_a)})", fontsize=12, fontweight='bold')
    ax2.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    if save:
        p = os.path.join(out_dir, 'attention_mean_and_dispersion.png')
        fig.savefig(p, dpi=PLOT_DPI, bbox_inches='tight')
        print(f"  [saved] {p}")
    if show: plt.show()
    else: plt.close(fig)

    # ---------------- 2. Per-Class Heatmap (Active classes only) ----------------
    active_idx = [i for i, c in enumerate(counts) if c > 0]
    if len(active_idx) > 0:
        act_cls = per_cls[active_idx]
        act_names = [f"{names[i]} (n={counts[i]})" for i in active_idx]
        df_cls = pd.DataFrame(act_cls, index=act_names, columns=col_labels)

        fig_h, ax_h = plt.subplots(figsize=(11, max(3.5, 0.45 * len(act_names) + 1.5)))
        sns.heatmap(df_cls, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax_h,
                    cbar_kws={'label': 'Attention Weight'}, linewidths=0.5)
        ax_h.set_title(f"Per-Class Temporal Attention Heatmap ({ds.upper()} — {ssl.upper()} — {mode})",
                       fontsize=12, fontweight='bold', pad=12)
        ax_h.set_xlabel("Sequence Snapshot Position (Oldest -> Query)", fontsize=11, fontweight='bold')
        ax_h.set_ylabel("Traffic & Attack Family", fontsize=11, fontweight='bold')

        plt.tight_layout()
        if save:
            p = os.path.join(out_dir, 'attention_per_class_heatmap.png')
            fig_h.savefig(p, dpi=PLOT_DPI, bbox_inches='tight')
            print(f"  [saved] {p}")
        if show: plt.show()
        else: plt.close(fig_h)

# =====================================================================
# Generate Attention Plots for all Available MoCo Models
# =====================================================================
_DATASETS = ['gotham', 'nf_bot', 'nf_ton']
_MODES    = ['frozen_single', 'finetune_single']

for ds in _DATASETS:
    for mode in _MODES:
        adir = os.path.join(DATASET_CONFIG.get(ds, ''), f'phase3_{mode}_moco', 'attention')
        if os.path.isdir(adir):
            print(f"\n{'='*60}\nRendering Attention Plots: {ds} | MOCO | {mode}\n{'='*60}")
            plot_tft_attention(ds, ssl='moco', mode=mode, show=SHOW_PLOTS, save=True)


# TFT — SSL-style plots for every phase-3 config

# In[ ]:


import os, gc, json, glob
from collections import Counter
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (roc_curve, precision_recall_curve,
                             confusion_matrix, classification_report,
                             average_precision_score, roc_auc_score,
                             f1_score)
try:
    from umap import UMAP
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False
    print("[plot] umap-learn not installed — t-SNE only")

# ---------------------------------------------------------------
# Configurable flags
# ---------------------------------------------------------------
SHOW_PLOTS    = False          # <-- set True to display figures inline
PLOT_DPI      = 140
PLOT_ROOT     = 'phase3_plots' # will be split into BYOL/ and MOCO/
SEED          = globals().get('SEED', 42)
MAX_POINTS    = 3000           # embedding subsample size

# ---------------------------------------------------------------
# Output path helper — phase3_plots/<SSL>/<dataset>/<mode>/
# ---------------------------------------------------------------
def _tft_out_dir(ds, ssl, mode):
    ssl_dir = ssl.upper()      # 'BYOL' or 'MOCO'
    p = os.path.join(PLOT_ROOT, ssl_dir, ds, mode)
    os.makedirs(p, exist_ok=True)
    return p

def _save_fig(fig, path, show=None):
    if show is None: show = SHOW_PLOTS
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches='tight')
    if show: plt.show()
    plt.close(fig)
    print(f"[plot] saved {path}")

# ---------------------------------------------------------------
# Forward pass that also returns the temporal embedding
# ---------------------------------------------------------------
@torch.no_grad()
def _tft_forward_full(tft, ge, ss):
    te, _, _ = tft.temporal(ge, ss)
    return (torch.sigmoid(tft.binary_head(te).squeeze(-1).float()),
            F.softmax(tft.multi_head(te).float(), -1),
            te.float())

@torch.no_grad()
def _collect_frozen(tft, batches):
    tft.eval()
    pb, pm, em, yb, ym = [], [], [], [], []
    for e, s, ybs, yms in batches:
        b, m, h = _tft_forward_full(tft, e.to(DEVICE).float(), s.to(DEVICE))
        pb.append(b.cpu().numpy()); pm.append(m.cpu().numpy())
        em.append(h.cpu().numpy())
        yb.append(ybs.numpy()); ym.append(yms.numpy())
    return dict(p_bin=np.concatenate(pb), p_multi=np.concatenate(pm),
                emb=np.concatenate(em),
                y_bin=np.concatenate(yb), y_multi=np.concatenate(ym))

@torch.no_grad()
def _collect_finetune(enc, tft, loader):
    enc.eval(); tft.eval()
    pb, pm, em, yb, ym = [], [], [], [], []
    for fb, B, T, ss, ybs, yms in loader:
        ss = ss.to(DEVICE); b = fb.to(DEVICE)
        h = enc(b.x, b.edge_index, b.edge_attr, b.batch)
        if isinstance(h, tuple): h = h[-1]
        bo, mo, te = _tft_forward_full(tft, h.view(B, T, -1).float(), ss)
        pb.append(bo.cpu().numpy()); pm.append(mo.cpu().numpy())
        em.append(te.cpu().numpy())
        yb.append(ybs.numpy()); ym.append(yms.numpy())
    return dict(p_bin=np.concatenate(pb), p_multi=np.concatenate(pm),
                emb=np.concatenate(em),
                y_bin=np.concatenate(yb), y_multi=np.concatenate(ym))

# ---------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------
def _read_meta(state):
    return dict(
        num_classes=int(state['multi_head.3.weight'].shape[0]),
        num_struct =int(state['temporal.proj_struct.weight'].shape[1]),
        graph_dim  =int(state['temporal.proj_graph.weight'].shape[1]),
        tft_hidden =int(state['temporal.proj_graph.weight'].shape[0]))

def _build_tft(state):
    m = _read_meta(state)
    t = MC_STGAT_IDS(m['graph_dim'], m['num_struct'], m['tft_hidden'],
                     m['num_classes'], TFT_NUM_HEADS, TFT_DROPOUT).to(DEVICE)
    t.load_state_dict(state); t.eval()
    return t

def _load_enc(ssl_meta, state):
    e = GATEncoder(ssl_meta['node_dim'], ssl_meta['hidden_dim'],
                   ssl_meta['num_heads'], ssl_meta['num_layers'],
                   ssl_meta['edge_dim']).to(DEVICE)
    e.load_state_dict(state); e.eval()
    return e

def _load_ssl_meta(ds, ssl):
    p = os.path.join(DATASET_CONFIG[ds], f'ssl_checkpoints_{ssl}',
                     'ssl_encoder_best.pth')
    return torch.load(p, map_location='cpu') if os.path.isfile(p) else None

# ---------------------------------------------------------------
# Per-config inference (single or ensemble)
# ---------------------------------------------------------------
def _infer_config(ds, ssl, mode):
    ddir = DATASET_CONFIG[ds]
    sub  = f'phase3_{mode}_{ssl}'
    ft   = 'finetune' in mode
    ctx  = _prepare_ctx(ds, ssl, ft, verbose=False)
    if ctx is None: return None
    ssl_meta = _load_ssl_meta(ds, ssl)

    def _infer_one(state, encoder_state=None):
        tft = _build_tft(state)
        if ft and encoder_state is not None:
            enc = _load_enc(ssl_meta, encoder_state)
            out = _collect_finetune(enc, tft, ctx['test_loader'])
            del enc
        else:
            batches = _eval_batches(ctx, 'test', ft)
            if not isinstance(batches, list):
                print(f"  [skip] {ds}/{ssl}/{mode}: no frozen cache")
                del tft
                return None
            out = _collect_frozen(tft, batches)
        del tft
        return out

    if mode.endswith('_single'):
        ck = os.path.join(ddir, sub, 'mc_stgat_ids.pth')
        if not os.path.isfile(ck): return None
        payload = torch.load(ck, map_location='cpu')
        out = _infer_one(payload['tft_state_dict'],
                         payload.get('encoder_state_dict'))
    elif mode.endswith('_ensemble'):
        ck_dir = os.path.join(ddir, sub, 'partial_checkpoints')
        seeds  = sorted(glob.glob(os.path.join(ck_dir, 'best_*_ens*.pth')))
        if not seeds: return None
        pb_l, pm_l, em_l = [], [], []
        yb_last = ym_last = None
        for f in seeds:
            payload = torch.load(f, map_location='cpu')
            o = _infer_one(payload['tft_state_dict'],
                           payload.get('encoder_state_dict'))
            if o is None: return None
            pb_l.append(o['p_bin']); pm_l.append(o['p_multi'])
            em_l.append(o['emb'])
            yb_last, ym_last = o['y_bin'], o['y_multi']
        out = dict(p_bin=np.mean(pb_l, axis=0),
                   p_multi=np.mean(pm_l, axis=0),
                   emb=np.mean(em_l, axis=0),
                   y_bin=yb_last, y_multi=ym_last)
    else:
        return None

    if out is None: return None

    # family labels for t-SNE per-family
    if ft:
        fam_labels = ctx['te_lab']
    else:
        inv = {v: k for k, v in ctx['label_to_idx'].items()}
        ym = out['y_multi'].astype(int)
        fam_labels = np.array([inv.get(int(y), 'Benign') for y in ym])
    out['fam'] = fam_labels
    out['ctx'] = ctx

    # threshold from phase3_report
    rp = os.path.join(ddir, sub, 'phase3_report.json')
    thr = 0.5
    if os.path.isfile(rp):
        rep = json.load(open(rp))
        thr = float(rep.get('thresholds', {}).get('used', 0.5))
    out['thr'] = thr
    return out

# ---------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------
def _plot_roc_pr(o, ds, ssl, mode, out_dir, tag):
    y = o['y_bin'].astype(int); p = o['p_bin']
    fpr, tpr, _ = roc_curve(y, p)
    roc = roc_auc_score(y, p)
    prec, rec, _ = precision_recall_curve(y, p)
    pr = average_precision_score(y, p)
    base = float(y.mean())

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, lw=2, label=f"ROC-AUC = {roc:.4f}")
    ax.plot([0,1],[0,1],'k--',alpha=.4)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title(f"{ds}/{ssl.upper()}/{mode} — ROC")
    ax.legend(); ax.grid(alpha=.3)
    _save_fig(fig, os.path.join(out_dir, f"{tag}_roc.png"))

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(rec, prec, lw=2, label=f"PR-AUC = {pr:.4f}")
    ax.axhline(base, color='k', ls='--', alpha=.4,
               label=f"baseline (prevalence = {base:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"{ds}/{ssl.upper()}/{mode} — Precision-Recall")
    ax.legend(); ax.grid(alpha=.3)
    _save_fig(fig, os.path.join(out_dir, f"{tag}_pr.png"))

def _plot_confusion_rates(o, ds, ssl, mode, out_dir, tag):
    y = o['y_bin'].astype(int); p = o['p_bin']; thr = o['thr']
    pred = (p >= thr).astype(int)
    cm = confusion_matrix(y, pred, labels=[0,1])
    rec_b = cm[0,0] / max(cm[0].sum(), 1)
    rec_a = cm[1,1] / max(cm[1].sum(), 1)
    fpr   = cm[0,1] / max(cm[0].sum(), 1)
    fnr   = cm[1,0] / max(cm[1].sum(), 1)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
    im = ax.imshow(cmn, cmap='Blues', vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]}\n({cmn[i,j]:.2%})",
                    ha='center', va='center', fontsize=11,
                    color='white' if cmn[i,j] > 0.5 else 'black')
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(['Benign','Attack']); ax.set_yticklabels(['Benign','Attack'])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{ds}/{ssl.upper()}/{mode} — confusion (thr={thr:.3f})")
    plt.colorbar(im, ax=ax, fraction=0.045)
    _save_fig(fig, os.path.join(out_dir, f"{tag}_confusion.png"))

    fig, ax = plt.subplots(figsize=(7, 5))
    labels = ['Benign recall','Attack recall','FPR','FNR']
    vals   = [rec_b, rec_a, fpr, fnr]
    colors = ['#4C72B0','#C44E52','#DD8452','#55A868']
    bars = ax.bar(labels, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.3f}",
                ha='center', va='bottom', fontsize=10)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Rate")
    ax.set_title(f"{ds}/{ssl.upper()}/{mode} — class recall / error rates")
    ax.grid(alpha=.3, axis='y')
    _save_fig(fig, os.path.join(out_dir, f"{tag}_class_rates.png"))

    rep = classification_report(y, pred,
                                target_names=['Benign','Attack'],
                                digits=4, zero_division=0)
    txt = os.path.join(out_dir, f"{tag}_classification_report.txt")
    with open(txt, 'w') as f:
        f.write(f"{ds}/{ssl.upper()}/{mode}\nthreshold = {thr:.4f}\n\n{rep}")
    print(f"[plot] saved {txt}")

def _plot_embeddings(o, ds, ssl, mode, out_dir, tag, max_points=MAX_POINTS):
    X_raw = np.nan_to_num(o['emb'], nan=0.0, posinf=100.0, neginf=-100.0)
    y_bin = o['y_bin'].astype(int); fam = o['fam']
    rng = np.random.default_rng(SEED)

    def _sub(y):
        idx0 = np.where(y == 0)[0]; idx1 = np.where(y == 1)[0]
        k = min(max_points // 2, len(idx0), len(idx1))
        if k == 0: return np.arange(len(y))
        s0 = rng.choice(idx0, k, replace=False)
        s1 = rng.choice(idx1, k, replace=False)
        return np.concatenate([s0, s1])

    sub = _sub(y_bin)
    X_s = X_raw[sub]; y_s = y_bin[sub]; f_s = fam[sub]
    X = X_s / np.clip(np.linalg.norm(X_s, axis=1, keepdims=True), 1e-8, None)
    n_pca = min(30, X.shape[1]-1, X.shape[0]-1)
    Xp = PCA(n_components=n_pca, random_state=SEED).fit_transform(X)
    Xp = Xp / np.clip(np.linalg.norm(Xp, axis=1, keepdims=True), 1e-8, None)
    perp = min(15, max(5, (len(Xp)-1)//3))
    tsne = TSNE(n_components=2, perplexity=perp, init='pca',
                learning_rate='auto', metric='cosine',
                random_state=SEED).fit_transform(Xp)

    # t-SNE binary
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for c, lbl, col in [(0,'Benign','#1f77b4'), (1,'Attack','#d62728')]:
        m = y_s == c
        ax.scatter(tsne[m,0], tsne[m,1], s=12, c=col, alpha=0.6,
                   label=f"{lbl} ({m.sum()})", edgecolors='none')
    ax.set_title(f"{ds}/{ssl.upper()}/{mode} — t-SNE binary")
    ax.legend(fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    _save_fig(fig, os.path.join(out_dir, f"{tag}_tsne_binary.png"))

    # t-SNE per family
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    uniq = sorted(set(f_s.tolist()))
    cmap = plt.get_cmap('tab20')
    for i, f in enumerate(uniq):
        m = f_s == f
        ax.scatter(tsne[m,0], tsne[m,1], s=12, color=cmap(i % 20),
                   alpha=0.65, label=f"{f} ({m.sum()})", edgecolors='none')
    ax.set_title(f"{ds}/{ssl.upper()}/{mode} — t-SNE per dominant family")
    ax.legend(fontsize=8, ncol=2, loc='best')
    ax.set_xticks([]); ax.set_yticks([])
    _save_fig(fig, os.path.join(out_dir, f"{tag}_tsne_family.png"))

    # UMAP binary
    if HAS_UMAP:
        try:
            um = UMAP(n_components=2, n_neighbors=50, min_dist=0.3,
                      metric='cosine', random_state=SEED).fit_transform(Xp)
            fig, ax = plt.subplots(figsize=(7.5, 6.5))
            for c, lbl, col in [(0,'Benign','#1f77b4'), (1,'Attack','#d62728')]:
                m = y_s == c
                ax.scatter(um[m,0], um[m,1], s=12, c=col, alpha=0.6,
                           label=f"{lbl} ({m.sum()})", edgecolors='none')
            ax.set_title(f"{ds}/{ssl.upper()}/{mode} — UMAP")
            ax.legend(fontsize=9); ax.set_xticks([]); ax.set_yticks([])
            _save_fig(fig, os.path.join(out_dir, f"{tag}_umap_binary.png"))
        except Exception as e:
            print(f"[plot] UMAP failed ({e})")

    # LDA-1
    if len(set(y_s.tolist())) == 2:
        try:
            Xs = StandardScaler().fit_transform(X_s)
            lda1 = LinearDiscriminantAnalysis(n_components=1).fit_transform(Xs, y_s)
            jit = rng.normal(0, 0.02, size=(len(lda1), 1))
            l2d = np.hstack([lda1, jit])
            fig, ax = plt.subplots(figsize=(8.5, 4.5))
            for c, lbl, col in [(0,'Benign','#1f77b4'), (1,'Attack','#d62728')]:
                m = y_s == c
                ax.scatter(l2d[m,0], l2d[m,1], s=14, c=col, alpha=0.55,
                           label=f"{lbl} ({m.sum()})", edgecolors='none')
            ax.set_title(f"{ds}/{ssl.upper()}/{mode} — supervised LDA-1")
            ax.set_yticks([]); ax.set_xlabel("LDA axis 1")
            ax.legend(fontsize=9)
            _save_fig(fig, os.path.join(out_dir, f"{tag}_lda.png"))
        except Exception as e:
            print(f"[plot] LDA failed ({e})")

# ---------------------------------------------------------------
# Driver
# ---------------------------------------------------------------
print(f"\n[plot] TFT SSL-style plots — SHOW_PLOTS={SHOW_PLOTS}")
print(f"[plot] output root: {os.path.abspath(PLOT_ROOT)}/<BYOL|MOCO>/<dataset>/<mode>/\n")

_DATASETS = globals().get('DATASETS_TO_RUN', ['gotham','nf_bot','nf_ton'])
_SSL_LIST = ['byol', 'moco']
_MODES    = ['frozen_single', 'finetune_single',
             'frozen_ensemble', 'finetune_ensemble']

SUMMARY = []
for ds in _DATASETS:
    if not os.path.isdir(DATASET_CONFIG[ds]): continue
    for ssl in _SSL_LIST:
        for mode in _MODES:
            sub = f'phase3_{mode}_{ssl}'
            rp  = os.path.join(DATASET_CONFIG[ds], sub, 'phase3_report.json')
            if not os.path.isfile(rp): continue
            out_dir = _tft_out_dir(ds, ssl, mode)
            tag     = f"tft_{ds}_{ssl}_{mode}"
            print(f"\n[plot] ==== {ds} / {ssl.upper()} / {mode} ====")
            try:
                o = _infer_config(ds, ssl, mode)
                if o is None:
                    print(f"  [skip] no inference result"); continue
                _plot_roc_pr(o, ds, ssl, mode, out_dir, tag)
                _plot_confusion_rates(o, ds, ssl, mode, out_dir, tag)
                _plot_embeddings(o, ds, ssl, mode, out_dir, tag)
                y = o['y_bin'].astype(int)
                SUMMARY.append(dict(
                    ds=ds, ssl=ssl, mode=mode,
                    roc=float(roc_auc_score(y, o['p_bin'])),
                    pr =float(average_precision_score(y, o['p_bin'])),
                    f1 =float(f1_score(y, (o['p_bin'] >= o['thr']).astype(int),
                                       zero_division=0))))
                del o
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"  [FAIL] {ds}/{ssl}/{mode}: {e}")
            gc.collect()
            if DEVICE.type == 'cuda': torch.cuda.empty_cache()

# ---------------------------------------------------------------
# Cross-config summary per SSL
# ---------------------------------------------------------------
for ssl in _SSL_LIST:
    rows = [r for r in SUMMARY if r['ssl'] == ssl]
    if not rows: continue
    labels = [f"{r['ds']}\n{r['mode']}" for r in rows]
    x = np.arange(len(rows)); w = 0.25
    fig, ax = plt.subplots(figsize=(max(10, 1.2*len(rows)), 5.5))
    for i, k in enumerate(['roc','pr','f1']):
        vals = [r[k] for r in rows]
        bars = ax.bar(x + (i-1)*w, vals, width=w, label=k.upper())
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(b.get_x()+b.get_width()/2, v+0.005, f"{v:.3f}",
                        ha='center', va='bottom', fontsize=7, rotation=90)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.12); ax.set_ylabel("Score")
    ax.set_title(f"TFT {ssl.upper()} — cross-config summary")
    ax.legend(loc='lower right', ncol=3, fontsize=9)
    ax.grid(alpha=.3, axis='y')
    out_top = os.path.join(PLOT_ROOT, ssl.upper())
    os.makedirs(out_top, exist_ok=True)
    _save_fig(fig, os.path.join(out_top, '00_cross_config.png'))

print(f"\n[plot] done — {len(SUMMARY)} configs plotted")


# In[ ]:


# Quick interactive call for single model:
# plot_tft_attention('gotham', ssl='moco', mode='finetune_single')
plot_tft_attention('gotham', ssl='moco', mode='finetune_single')

