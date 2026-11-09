import os
import sys
import time
import json
import copy
import gc
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PLOT_DPI = 200

# ----------------- Model Architecture -----------------
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
        assert self.hd * num_heads == hidden_dim, f"hidden_dim {hidden_dim} not divisible by num_heads {num_heads}"
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
    def __init__(self, graph_emb_dim, struct_dim, tft_hidden, num_classes, num_heads=2, dropout=0.1):
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
    def __init__(self, w_multi=0.5, pos_weight=3.0, focal_alpha=0.25, focal_gamma=2.0):
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

# ----------------- Training & Eval Helpers -----------------
def rebatch(flat_cache, order=None, bs=512):
    if order is None: order = range(len(flat_cache))
    batches = []
    order = list(order)
    for i in range(0, len(order), bs):
        chunk = [flat_cache[j] for j in order[i:i + bs]]
        embs    = torch.stack([c[0] for c in chunk], 0)
        structs = torch.stack([c[1] for c in chunk], 0)
        ybs     = torch.tensor([float(c[2]) for c in chunk], dtype=torch.float32)
        yms     = torch.tensor([int(c[3])   for c in chunk], dtype=torch.long)
        batches.append((embs, structs, ybs, yms))
    return batches

def evaluate_model(tft, batches, collect_attn=False):
    tft.eval()
    bp, mp, yb, ym, aa = [], [], [], [], []
    with torch.no_grad():
        for emb, struct_seq, y_bin, y_multi in batches:
            ge = emb.to(DEVICE, non_blocking=True).float()
            struct_seq = struct_seq.to(DEVICE, non_blocking=True)
            bl, ml, attn_w, _ = tft(ge, struct_seq)
            bp.append(np.nan_to_num(torch.sigmoid(bl.float()).cpu().numpy(), 0.0, 1.0, 0.0))
            mp.append(np.nan_to_num(F.softmax(ml.float(), -1).cpu().numpy(),
                                    1.0/max(ml.shape[-1], 1), 1.0, 0.0))
            yb.append(y_bin.numpy()); ym.append(y_multi.numpy())
            if collect_attn:
                a = attn_w.float()
                if a.dim() == 3: a = a[:, -1, :] if a.size(1) == a.size(2) else a.mean(1)
                elif a.dim() == 4: a = a[:, 0, -1, :]
                aa.append(a.cpu().numpy())
    out = (np.concatenate(bp), np.concatenate(mp),
           np.concatenate(yb), np.concatenate(ym))
    return out + (np.concatenate(aa),) if collect_attn else out

def compute_val_metrics(pb_val, pm_val, yb_val, ym_val, thr=0.5):
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
    return {'roc': roc, 'pr': pr, 'f1': f1b, 'macro_f1_merged': mf1, 'weighted_f1': wf1}

def optimal_threshold(prob, y):
    from sklearn.metrics import precision_recall_curve
    prec, rec, thr = precision_recall_curve(y, prob)
    prec, rec = prec[:-1], rec[:-1]
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    if not np.isfinite(f1).any(): return 0.5, 0.0
    b = int(np.nanargmax(f1))
    return float(thr[b]), float(f1[b])

def binary_metrics_at(prob, y, thr):
    pred = (prob >= thr).astype(int)
    f1 = float(f1_score(y, pred, average='binary', zero_division=0))
    cm = confusion_matrix(y, pred, labels=[0, 1])
    cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
    tn, fp = int(cm[0, 0]), int(cm[0, 1]); fn, tp = int(cm[1, 0]), int(cm[1, 1])
    return {'f1': f1, 'fpr': float(cmn[0, 1]), 'fnr': float(cmn[1, 0]),
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
            'n_true_attack': fn + tp, 'n_true_benign': tn + fp,
            'n_pred_attack': fp + tp, 'n_pred_benign': tn + fn}

def per_class_score_analysis(pb_test, test_labels_seq, class_names, thr):
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
        recall = float((p[pos] >= thr).sum() / max(int(pos.sum()), 1))
        rows.append({'class': cls, 'n_test': int(pos.sum()),
                     'roc_vs_benign': roc, 'pr_vs_benign': pr,
                     'mean_prob_attack': float(p[pos].mean()) if pos.sum() > 0 else 0.0,
                     'recall_at_thr': recall,
                     'missed_at_thr': int((p[pos] < thr).sum())})
    rows.sort(key=lambda r: -r['roc_vs_benign'])
    return rows

def lexicographic_improved(val_now, val_best, priority, tol=1e-4):
    for m in priority:
        cur = float(val_now.get(m, float('-inf')))
        bst = float(val_best.get(m, float('-inf')))
        if not np.isfinite(cur): continue
        if not np.isfinite(bst): return True
        if cur > bst + tol: return True
        if cur < bst - tol: return False
    return False

def build_lr_lambda(warmup_epochs, total_epochs):
    def fn(ep):
        if ep < warmup_epochs:
            return float(ep + 1) / float(max(1, warmup_epochs))
        prog = (ep - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return 0.5 * (1.0 + np.cos(np.pi * prog))
    return fn

def save_attention_data(attn_dir, attn_test, ym_test, class_names):
    os.makedirs(attn_dir, exist_ok=True)
    if attn_test.ndim == 3: attn_test = attn_test[:, -1, :]
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

def plot_attention_figures(attn_dir, out_plot_dir, ds, ssl, mode):
    os.makedirs(out_plot_dir, exist_ok=True)
    mean_a_path = os.path.join(attn_dir, "mean_attention.npy")
    per_c_path = os.path.join(attn_dir, "per_class_attention.npy")
    all_a_path = os.path.join(attn_dir, "all_test_attn.npy")
    meta_path = os.path.join(attn_dir, "class_names.json")

    if not (os.path.isfile(mean_a_path) and os.path.isfile(all_a_path) and os.path.isfile(meta_path)):
        return

    mean_a = np.load(mean_a_path)
    per_cls = np.load(per_c_path)
    all_a = np.load(all_a_path)
    with open(meta_path) as f:
        meta = json.load(f)
    names = meta['class_names']
    counts = meta['class_counts']

    T = len(mean_a)
    col_labels = [f"t-{T-1-i}" if i < T-1 else "t (query)" for i in range(T)]

    # 1. Mean Attention + Cohort Dispersion Boxplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
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

    bp = ax2.boxplot(all_a, positions=np.arange(T), patch_artist=True,
                     boxprops=dict(facecolor='#aec7e8', color='black'),
                     medianprops=dict(color='red', linewidth=1.5))
    ax2.set_xticks(np.arange(T))
    ax2.set_xticklabels(col_labels, fontsize=10, rotation=25)
    ax2.set_ylabel("Attention Weight Distribution", fontsize=11, fontweight='bold')
    ax2.set_title(f"Cohort Attention Dispersion (Test N={len(all_a)})", fontsize=12, fontweight='bold')
    ax2.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    p1 = os.path.join(out_plot_dir, 'attention_mean_and_dispersion.png')
    fig.savefig(p1, dpi=PLOT_DPI, bbox_inches='tight')
    plt.close(fig)

    # 2. Per-Class Heatmap
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
        p2 = os.path.join(out_plot_dir, 'attention_per_class_heatmap.png')
        fig_h.savefig(p2, dpi=PLOT_DPI, bbox_inches='tight')
        plt.close(fig_h)

# ----------------- Training Pipeline -----------------
def train_seed(splits, class_names, hparams, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    graph_emb_dim = splits['train'][0][0].shape[-1]
    struct_dim = splits['train'][0][1].shape[-1]
    num_classes = len(class_names)

    tft = MC_STGAT_IDS(
        graph_emb_dim=graph_emb_dim,
        struct_dim=struct_dim,
        tft_hidden=hparams['tft_hidden'],
        num_classes=num_classes,
        num_heads=hparams['num_heads'],
        dropout=hparams['dropout']
    ).to(DEVICE)

    decay_params, no_decay_params = [], []
    for name, param in tft.named_parameters():
        if not param.requires_grad: continue
        if any(k in name for k in ['attn.q', 'attn.k', 'bias', 'norm']):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    groups = [
        {'params': decay_params, 'lr': hparams['lr'], 'weight_decay': hparams['weight_decay']},
        {'params': no_decay_params, 'lr': hparams['lr'], 'weight_decay': 0.0}
    ]
    opt = torch.optim.Adam(groups)
    epochs = hparams.get('epochs', 100)
    warmup = hparams.get('warmup', 5)
    patience = hparams.get('patience', 25)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, build_lr_lambda(warmup, epochs))

    crit = CombinedLoss(
        w_multi=hparams['w_multi'],
        pos_weight=hparams['pos_weight'],
        focal_alpha=hparams['focal_alpha'],
        focal_gamma=hparams['focal_gamma']
    ).to(DEVICE)

    bs = hparams.get('batch_size', 512)
    use_balanced = hparams.get('use_balanced_sampler', False)
    train_data = splits['train']
    val_data = splits['val']
    test_data = splits['test']

    balanced_w = None
    if use_balanced:
        yb_arr = np.array([float(c[2]) for c in train_data])
        n_b = max(int((yb_arr == 0).sum()), 1); n_a = max(int((yb_arr == 1).sum()), 1)
        w = np.where(yb_arr == 0, 1.0/n_b, 1.0/n_a).astype(np.float64)
        balanced_w = w / w.sum()

    val_batches = rebatch(val_data, bs=bs)
    test_batches = rebatch(test_data, bs=bs)

    priority = hparams.get('priority', ['roc', 'macro_f1_merged', 'pr', 'f1'])
    tol = 1e-4

    best_val = {}
    best_tft_weights = None
    best_epoch = 0
    no_imp = 0

    t0 = time.time()
    for ep in range(1, epochs + 1):
        tft.train()
        n = len(train_data)
        if balanced_w is not None:
            rng_ep = np.random.default_rng(seed + ep)
            order = rng_ep.choice(n, size=n, replace=True, p=balanced_w)
        else:
            rng_ep = np.random.default_rng(seed + ep)
            order = rng_ep.permutation(n)
        train_batches = rebatch(train_data, order=order.tolist(), bs=bs)

        for emb, struct_seq, y_bin, y_multi in train_batches:
            ge = emb.to(DEVICE, non_blocking=True).float()
            struct_seq = struct_seq.to(DEVICE, non_blocking=True)
            y_bin = y_bin.to(DEVICE); y_multi = y_multi.to(DEVICE)

            bl, ml, _, _ = tft(ge, struct_seq)
            loss, _, _ = crit(bl, ml, y_bin, y_multi)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tft.parameters(), 5.0)
            opt.step()

        sched.step()

        pbv, pmv, ybv, ymv = evaluate_model(tft, val_batches)
        val_m = compute_val_metrics(pbv, pmv, ybv, ymv, thr=0.5)

        warmed = ep >= warmup
        if warmed:
            if lexicographic_improved(val_m, best_val, priority, tol):
                best_val = {k: float(v) for k, v in val_m.items()}
                best_tft_weights = {k: v.cpu().clone() for k, v in tft.state_dict().items()}
                best_epoch = ep
                no_imp = 0
            else:
                no_imp += 1
                if no_imp >= patience:
                    break
        else:
            no_imp = 0

    elapsed = time.time() - t0

    if best_tft_weights is not None:
        tft.load_state_dict({k: v.to(DEVICE) for k, v in best_tft_weights.items()})

    pbv, pmv, ybv, ymv = evaluate_model(tft, val_batches)
    pbt, pmt, ybt, ymt, attn = evaluate_model(tft, test_batches, collect_attn=True)

    return {
        'tft_state_dict': {k: v.cpu() for k, v in tft.state_dict().items()},
        'val_preds': (pbv, pmv, ybv, ymv),
        'test_preds': (pbt, pmt, ybt, ymt),
        'attn': attn,
        'best_epoch': best_epoch,
        'total_epochs': ep,
        'elapsed': elapsed,
        'best_val_metrics': best_val
    }

# ----------------- Single & Ensemble Production Runs -----------------
def execute_production_single(ds, ssl, splits, class_names, hparams, dirs, seed=42):
    print(f"\n[{ssl.upper()} SINGLE] {ds} (seed={seed})...")
    res = train_seed(splits, class_names, hparams, seed)
    pbv, _, ybv, _ = res['val_preds']
    thr_f1, _ = optimal_threshold(pbv, ybv.astype(int))

    pbt, pmt, ybt, ymt = res['test_preds']
    ybt_i = ybt.astype(int)
    roc = float(roc_auc_score(ybt_i, pbt))
    pr  = float(average_precision_score(ybt_i, pbt))
    ym_pred = pmt.argmax(-1)
    macro_f1 = float(f1_score(ymt, ym_pred, average='macro', zero_division=0))
    weighted_f1 = float(f1_score(ymt, ym_pred, average='weighted', zero_division=0))

    m = binary_metrics_at(pbt, ybt_i, thr_f1)
    test_labels_seq = np.array([class_names[y] for y in ymt])
    per_class = per_class_score_analysis(pbt, test_labels_seq, class_names, thr_f1)

    # Save attention
    save_attention_data(dirs['attn'], res['attn'], ymt, class_names)
    plot_attention_figures(dirs['attn'], dirs['plot'], ds, ssl, 'frozen_single')

    # Save checkpoint
    torch.save({
        'tft_state_dict': res['tft_state_dict'],
        'class_names': class_names,
        'sequence_length': 10,
        'num_classes': len(class_names),
        'finetune': False,
        'ssl_method': ssl,
        'dataset': ds,
        'threshold': float(thr_f1)
    }, dirs['ckpt'])

    # Save phase3_report.json
    report = {
        'stage': 'single', 'dataset': ds, 'ssl': ssl, 'finetune': False,
        'seed': seed, 'val_selection_score': float(res['best_val_metrics'].get('roc', 0)),
        'best_epoch': int(res['best_epoch']),
        'thresholds': {'f1_opt': float(thr_f1), 'balanced': float(thr_f1), 'used': float(thr_f1)},
        'test': {
            'roc': roc, 'pr': pr, 'f1': float(m['f1']),
            'fpr': float(m['fpr']), 'fnr': float(m['fnr']),
            'macro_f1': macro_f1, 'weighted_f1': weighted_f1,
            'tn': int(m['tn']), 'fp': int(m['fp']),
            'fn': int(m['fn']), 'tp': int(m['tp']),
            'n_true_attack': int(m['n_true_attack']), 'n_pred_attack': int(m['n_pred_attack'])
        },
        'per_class': per_class,
        'split_sizes': {'train': len(splits['train']), 'val': len(splits['val']), 'test': len(splits['test'])},
        'seq_counts': {'train': len(splits['train']), 'val': len(splits['val']), 'test': len(splits['test'])}
    }
    with open(dirs['report'], 'w') as f:
        json.dump(report, f, indent=2)

    print(f"  -> Done in {res['elapsed']:.1f}s | TEST: ROC={roc:.4f} PR={pr:.4f} F1={m['f1']:.4f} FPR={m['fpr']:.4f} FNR={m['fnr']:.4f} mF1={macro_f1:.4f}")
    return report

def execute_production_ensemble(ds, ssl, splits, class_names, hparams, dirs, seeds=[1337, 2024, 7]):
    print(f"\n[{ssl.upper()} ENSEMBLE] {ds} (seeds={seeds})...")
    val_pbs, val_pms = [], []
    test_pbs, test_pms = [], []
    test_attns = []
    yb_val = ym_val = yb_test = ym_test = None
    per_metrics = []
    total_time = 0.0

    for s in seeds:
        res = train_seed(splits, class_names, hparams, s)
        total_time += res['elapsed']
        pbv, pmv, ybv, ymv = res['val_preds']
        pbt, pmt, ybt, ymt = res['test_preds']
        val_pbs.append(pbv); val_pms.append(pmv)
        test_pbs.append(pbt); test_pms.append(pmt)
        test_attns.append(res['attn'])
        yb_val, ym_val = ybv, ymv
        yb_test, ym_test = ybt, ymt
        roc_s = float(roc_auc_score(ybt.astype(int), pbt))
        mf1_s = float(f1_score(ymt, pmt.argmax(-1), average='macro', zero_division=0))
        per_metrics.append({'seed': int(s), 'val_score': float(res['best_val_metrics'].get('roc', 0)),
                            'test_roc': roc_s, 'test_macro_f1': mf1_s})

    pb_val_ens = np.mean(val_pbs, axis=0)
    pb_test_ens = np.mean(test_pbs, axis=0)
    pm_test_ens = np.mean(test_pms, axis=0)
    attn_ens = np.mean(test_attns, axis=0)

    thr_f1, _ = optimal_threshold(pb_val_ens, yb_val.astype(int))
    ybt_i = yb_test.astype(int)
    roc = float(roc_auc_score(ybt_i, pb_test_ens))
    pr  = float(average_precision_score(ybt_i, pb_test_ens))
    ym_pred = pm_test_ens.argmax(-1)
    macro_f1 = float(f1_score(ym_test, ym_pred, average='macro', zero_division=0))
    weighted_f1 = float(f1_score(ym_test, ym_pred, average='weighted', zero_division=0))

    m = binary_metrics_at(pb_test_ens, ybt_i, thr_f1)
    test_labels_seq = np.array([class_names[y] for y in ym_test])
    per_class = per_class_score_analysis(pb_test_ens, test_labels_seq, class_names, thr_f1)

    # Bootstrap 95% CI
    rng = np.random.default_rng(123)
    n = len(ybt_i); boot = np.empty(1000)
    for b in range(1000):
        idx = rng.integers(0, n, size=n)
        try: boot[b] = roc_auc_score(ybt_i[idx], pb_test_ens[idx])
        except ValueError: boot[b] = roc
    lo, hi = np.percentile(boot, [2.5, 97.5])

    # Save attention
    save_attention_data(dirs['attn'], attn_ens, ym_test, class_names)
    plot_attention_figures(dirs['attn'], dirs['plot'], ds, ssl, 'frozen_ensemble')

    # Save phase3_report.json
    report = {
        'stage': 'ensemble', 'dataset': ds, 'ssl': ssl, 'finetune': False,
        'seeds': list(map(int, seeds)), 'per_seed': per_metrics,
        'thresholds': {'f1_opt': float(thr_f1), 'balanced': float(thr_f1), 'used': float(thr_f1)},
        'test': {
            'roc': roc, 'pr': pr, 'f1': float(m['f1']),
            'fpr': float(m['fpr']), 'fnr': float(m['fnr']),
            'macro_f1': macro_f1, 'weighted_f1': weighted_f1,
            'roc_ci95': [float(lo), float(hi)],
            'tn': int(m['tn']), 'fp': int(m['fp']),
            'fn': int(m['fn']), 'tp': int(m['tp']),
            'n_true_attack': int(m['n_true_attack']), 'n_pred_attack': int(m['n_pred_attack'])
        },
        'per_class': per_class,
        'split_sizes': {'train': len(splits['train']), 'val': len(splits['val']), 'test': len(splits['test'])},
        'seq_counts': {'train': len(splits['train']), 'val': len(splits['val']), 'test': len(splits['test'])}
    }
    with open(dirs['report'], 'w') as f:
        json.dump(report, f, indent=2)

    print(f"  -> Done in {total_time:.1f}s | ENSEMBLE: ROC={roc:.4f} [{lo:.4f},{hi:.4f}] PR={pr:.4f} F1={m['f1']:.4f} FPR={m['fpr']:.4f} FNR={m['fnr']:.4f} mF1={macro_f1:.4f}")
    return report

# ----------------- Main Orchestration -----------------
def make_dirs(ds_root, ssl, mode):
    base = os.path.join(ds_root, f"phase3_{mode}_{ssl}")
    attn = os.path.join(base, "attention")
    plot = os.path.join("phase3_plots", ssl.upper(), os.path.basename(os.path.dirname(ds_root)), mode)
    os.makedirs(base, exist_ok=True)
    os.makedirs(attn, exist_ok=True)
    os.makedirs(plot, exist_ok=True)
    return {
        'base': base,
        'attn': attn,
        'plot': plot,
        'ckpt': os.path.join(base, "mc_stgat_ids.pth"),
        'report': os.path.join(base, "phase3_report.json")
    }

def main():
    datasets = {
        'gotham': {
            'root': 'Gotham_processed_features_10s/graphs',
            'classes': ['Benign', 'CoAP Amplification', 'Merlin C&C Communication', 'Merlin TCP Flooding',
                        'Merlin UDP Flooding', 'Mirai C&C Communication', 'Mirai GRE Flooding',
                        'Mirai TCP Flooding', 'Mirai UDP Flooding', 'TCP Scan', 'Telnet Brute Force', 'UDP Scan'],
            'use_balanced_sampler': True
        },
        'nf_bot': {
            'root': 'NFBoTIoT_v3_10s/graphs',
            'classes': ['Benign', 'DDoS', 'DoS', 'Reconnaissance', 'Theft'],
            'use_balanced_sampler': False
        },
        'nf_ton': {
            'root': 'NFToNIoT_v3_10s/graphs',
            'classes': ['Backdoor', 'Benign', 'ddos', 'dos', 'injection', 'mitm', 'password', 'ransomware', 'scanning', 'xss'],
            'use_balanced_sampler': True
        }
    }

    base_hparams = {
        'tft_hidden': 384, 'num_heads': 2, 'dropout': 0.10, 'lr': 1e-3,
        'weight_decay': 1e-3, 'w_multi': 0.5, 'pos_weight': 3.0,
        'focal_alpha': 0.25, 'focal_gamma': 2.0, 'batch_size': 512,
        'priority': ['roc', 'macro_f1_merged', 'pr', 'f1'],
        'epochs': 100, 'patience': 25, 'warmup': 5
    }

    print("=" * 115)
    print("STEP 1: RUNNING MOCO TFT PIPELINE (SINGLE + ENSEMBLE ACROSS ALL DATASETS)")
    print("=" * 115)

    all_reports = {}

    for ds_name in ['gotham', 'nf_bot', 'nf_ton']:
        d_info = datasets[ds_name]
        cache_file = os.path.join(d_info['root'], f"frozen_cache_moco_L10_S1_{'E2000' if ds_name=='gotham' else ('E15000' if ds_name=='nf_bot' else 'E10000')}.pt")
        print(f"\n>>> Processing {ds_name.upper()} MOCO from {cache_file}...")
        splits = torch.load(cache_file, map_location='cpu')['splits']
        hp = copy.deepcopy(base_hparams)
        hp['use_balanced_sampler'] = d_info['use_balanced_sampler']

        # Single
        d_single = make_dirs(d_info['root'], 'moco', 'frozen_single')
        rep_s = execute_production_single(ds_name, 'moco', splits, d_info['classes'], hp, d_single, seed=42)

        # Ensemble
        d_ens = make_dirs(d_info['root'], 'moco', 'frozen_ensemble')
        rep_e = execute_production_ensemble(ds_name, 'moco', splits, d_info['classes'], hp, d_ens, seeds=[1337, 2024, 7])

        all_reports[(ds_name, 'moco', 'single')] = rep_s
        all_reports[(ds_name, 'moco', 'ensemble')] = rep_e

    print("\n" + "=" * 115)
    print("STEP 2: RUNNING BYOL TFT PIPELINE (SINGLE + ENSEMBLE ACROSS ALL DATASETS)")
    print("=" * 115)

    for ds_name in ['gotham', 'nf_bot', 'nf_ton']:
        d_info = datasets[ds_name]
        cache_file = os.path.join(d_info['root'], f"frozen_cache_byol_L10_S1_{'E2000' if ds_name=='gotham' else ('E15000' if ds_name=='nf_bot' else 'E10000')}.pt")
        print(f"\n>>> Processing {ds_name.upper()} BYOL from {cache_file}...")
        splits = torch.load(cache_file, map_location='cpu')['splits']
        hp = copy.deepcopy(base_hparams)
        hp['use_balanced_sampler'] = d_info['use_balanced_sampler']

        # Single
        d_single = make_dirs(d_info['root'], 'byol', 'frozen_single')
        rep_s = execute_production_single(ds_name, 'byol', splits, d_info['classes'], hp, d_single, seed=42)

        # Ensemble
        d_ens = make_dirs(d_info['root'], 'byol', 'frozen_ensemble')
        rep_e = execute_production_ensemble(ds_name, 'byol', splits, d_info['classes'], hp, d_ens, seeds=[1337, 2024, 7])

        all_reports[(ds_name, 'byol', 'single')] = rep_s
        all_reports[(ds_name, 'byol', 'ensemble')] = rep_e

    # Save to phase3_all_reports.json
    unified_reports = {}
    for (ds, ssl, mode), r in all_reports.items():
        if ds not in unified_reports: unified_reports[ds] = {}
        if ssl not in unified_reports[ds]: unified_reports[ds][ssl] = {}
        unified_reports[ds][ssl][mode] = r
    with open('phase3_all_reports.json', 'w') as f:
        json.dump(unified_reports, f, indent=2)
    print(f"\n[OK] Updated phase3_all_reports.json with all production runs.")

    print("\n" + "=" * 115)
    print("ALL PRODUCTION RUNS FINISHED SUCCESSFULLY!")
    print("=" * 115)

if __name__ == '__main__':
    main()
