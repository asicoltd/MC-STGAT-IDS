# 1\. Assigned Thesis Title

_"Momentum-Contrastive Spatio-Temporal Graph Attention Networks for Explainable IoT Intrusion Detection"_

**Short working title for code/repo: MC-STGAT-IDS**

Acceptable alternate (if the committee wants something shorter): _"A Hybrid Self-Supervised Graph Attention and Temporal Fusion Framework for IoT Network Intrusion Detection."_

# 2\. Why this project is a good fit for you

You are not starting from zero on any of the three pillars this thesis needs:

- Self-supervised pretraining - you already have hands-on SimCLR (Brain Tumor Classification) and exposure to BYOL/MoCo. The core novelty of this thesis is porting that momentum-contrastive idea from images to network-flow graphs.
- Graph learning - your Multi-hop GAT work in the forecasting track transfers almost directly; here the "signal" is host/flow communication instead of a sensor network.
- Temporal modeling - your GRU/TFT experience becomes the module that captures how an attack unfolds across time, not just within one traffic snapshot.

RL (Dueling DQN/PPO) and NLP/RAG are kept as optional Phase-5 extensions - only pursue them if Phases 1-4 finish on schedule. Do not let them delay your core contribution.

# 3\. The research gap (what's already been done, and what you're adding)

Three papers define the current state of the art you must position against and beat:

- Anomal-E (Caville et al., 2022) - self-supervised E-GraphSAGE encoder trained via edge reconstruction and real-vs-corrupted graph discrimination, then a separate unsupervised anomaly detector (PCA/Isolation Forest) on top.
- PPT-GNN (Van Langendonck et al., 2025) - a pretrained spatio-temporal GNN using sliding time windows and self-supervised link prediction as the pretext task, built on E-GraphSAGE/E-ResGAT backbones; strong on few-shot fine-tuning to new networks.
- GraphIDS (Guerra et al., NeurIPS 2025) - an inductive GNN (E-GraphSAGE) combined with a Transformer masked-autoencoder that reconstructs flow embeddings and flags high reconstruction error as anomalous.

## The gap

All three rely on GraphSAGE-style backbones and either masked-autoencoder or link-prediction pretext tasks. None of them use a momentum/twin-network contrastive objective (BYOL/MoCo-style), none use multi-hop attention (GAT) as the encoder, and none pair the graph encoder with an interpretable temporal-attention fusion module (TFT-style). That combination is your contribution:

1. Contrastive SSL pretraining using an online/target (EMA) network pair over augmented graph views - not masked reconstruction, not link prediction.
2. A Multi-hop GAT encoder instead of GraphSAGE, so the model can learn which neighborhood hops matter for a given flow.
3. A temporal fusion layer with interpretable attention over a sequence of graph snapshots, giving you both accuracy and an attention map you can show in your defense as an explanation.

# 4\. Assigned datasets (both released within the last ~18 months)

You will use two datasets so you can also report cross-dataset generalization (a weakness reviewers always flag in this literature).

## Dataset A - Gotham Dataset 2025 (Cardiff University + Toshiba, Feb 2025)

- 78 emulated IoT devices across 4 network segments (smart home, industrial, wearables, appliances), traffic on MQTT/CoAP/RTSP.
- Captured per-device (distributed, non-IID) - good for the host-communication-graph view.
- Attacks: DoS, Telnet brute force, network scanning, CoAP amplification, Mirai/Merlin C2 stages.
- ~23 extracted features per record, PCAP + labeled CSV both provided.
- Access: Zenodo, DOI 10.5281/zenodo.14502760 (also indexed on Kaggle). Paper: arXiv 2502.03134.

## Dataset B - BCCC-IoT-MQTT-IDS-2025 (York University BCCC, MQTTFlowLyzer paper, 2026)

- Protocol-aware MQTT flow-level dataset combining MQTTset, MQTT-IoT-IDS2020, and DoS/DDoS-MQTT-IoT.
- 404 raw features (378 after preprocessing) capturing session behavior and message-level interaction - richer semantics than packet/TCP-only datasets.
- Attacks: brute-force authentication, malformed messages, SlowITe flooding, scanning, DoS/DDoS variants.
- Explicitly designed by its authors to support attention-based and LLM-assisted intrusion detection - convenient if you reach the Phase-5 explainability extension.
- Access: request form on the BCCC datasets page (Cybersecurity Datasets - Intelligence-led Security section); cite the MQTTFlowLyzer paper (Kouhi & Lashkari, Journal of Supercomputing, 2026).

**Rule for you:** build and validate the full pipeline on Dataset A first (it is public and instantly downloadable). Only bring in Dataset B once Phase 3 is working, to test generalization and to avoid a request-approval delay blocking your early milestones.

# 5\. Step-by-step execution plan

| **Phase**    | **Weeks**                  | **Task**                                                                                                                                  | **Deliverable**                                                                                 |
| ------------ | -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| 0            | 1-2                        | Read Anomal-E, PPT-GNN, GraphIDS in full; reproduce one baseline (e.g., plain GCN or GraphSAGE) end-to-end on Dataset A                   | 2-page lit summary + working baseline notebook                                                  |
| 1            | 3-4                        | Graph construction pipeline for Dataset A (see §6a)                                                                                       | Cleaned graph-snapshot dataset, train/val/test split with temporal (not random) partitioning    |
| 2            | 5-8                        | Implement momentum-contrastive SSL pretraining (§6b) on unlabeled snapshots                                                               | Pretrained encoder checkpoint + embedding visualization (t-SNE/UMAP of normal vs. attack flows) |
| 3            | 9-12                       | Add temporal fusion module (§6c) + fine-tune classifier (§6d)                                                                             | Full MC-STGAT-IDS model, results vs. baselines on Dataset A                                     |
| 4            | 13-14                      | Repeat graph construction + evaluation on Dataset B; report cross-dataset generalization                                                  | Comparison table across both datasets                                                           |
| 5 (optional) | 15-17                      | RL adaptive-alerting layer and/or LLM/RAG explanation layer                                                                               | Working demo of at least one extension                                                          |
| 6            | throughout, finalize 18-20 | Thesis writing (use the latex-paper-writer structure - Introduction → Related Work → Methodology → Experiments → Discussion → Conclusion) | Full thesis draft                                                                               |

Send me your Phase-0 lit summary and baseline notebook before starting Phase 1 - I want to check your graph construction assumptions before you build on top of them.

# 6\. Detailed hybrid model build guide

## 6a. Graph construction

- Nodes: hosts (IP/device ID). Edges: flows between them, directed, one edge per flow record.
- Edge features: the dataset's native flow statistics (packet counts, byte counts, duration, protocol, inter-arrival stats).
- Snapshotting: bin flows into fixed-length time windows (start with 60-second windows; tune later) to produce a sequence of graphs - this sequence feeds the temporal module in §6c. Do not use fully random edge sampling across the whole capture; that causes the temporal leakage PPT-GNN explicitly warns about. Split train/val/test by time, not randomly.

## 6b. Stage 1 - Momentum-contrastive SSL pretraining

- Two augmented views per graph snapshot: edge dropout, feature masking, and node/subgraph sampling (mirror the augmentation families you already used conceptually in SimCLR, adapted to graphs).
- Online network: Multi-hop GAT encoder (start with 3 hops, 4 attention heads, hidden dim 128) + projection head.
- Target network: identical architecture, weights updated via EMA of the online network (BYOL-style) - this avoids needing negative pairs, which is an advantage on imbalanced traffic where "negatives" are hard to define safely.
- Loss: cosine similarity loss between online prediction and target projection (BYOL loss). If you want a MoCo-style alternative to compare, add a queue of negative embeddings and use InfoNCE - worth one ablation row in your results table.
- Train this stage on unlabeled data only (simulates the realistic setting: labels are scarce, traffic is not).

## 6c. Stage 2 - Temporal fusion

- Feed the sequence of per-snapshot graph embeddings (pooled from the Stage-1 encoder) into a TFT-style layer: variable selection network + interpretable multi-head temporal attention over the last N snapshots (start with N=10).
- This is what gives you the interpretability angle for your defense: you can plot which past snapshots the model attended to when it flagged an attack.

## 6d. Stage 3 - Fine-tuning and classification

- Attach a classification head (2-3 layer MLP) on top of the fused temporal-graph embedding.
- Use focal loss, not plain cross-entropy - both datasets are heavily imbalanced toward benign traffic.
- Fine-tune end-to-end with a small labeled subset first (mimic PPT-GNN's few-shot protocol) to report label-efficiency, then with the full label set for your headline numbers.

## 6e. (Optional, Phase 5) RL adaptive alerting

- Treat the fine-tuned model's anomaly score as state input to a Dueling DQN or PPO agent whose action is "alert" vs. "suppress" vs. "escalate," with a reward that penalizes false positives and false negatives asymmetrically (false negatives cost more). This targets a real operational pain point - alert fatigue - that none of the three baseline papers address.

## 6f. (Optional, Phase 5) LLM/RAG explainability

- For each flagged anomaly, retrieve relevant context from a small vector store of MITRE ATT&CK technique descriptions and generate a short natural-language explanation grounding the alert in a known technique. Dataset B's authors designed their feature set with exactly this kind of downstream use in mind, so pair this extension with Dataset B.

# 7\. Baselines and metrics

**Baselines to implement:** plain GCN, GraphSAGE (Anomal-E-style unsupervised variant), a supervised GAT without SSL pretraining, and - if time allows - your own re-implementation of PPT-GNN's link-prediction pretext task as an SSL ablation (same encoder, different pretext task, isolates your contrastive objective as the variable).

**Metrics:** macro-F1, PR-AUC (not ROC-AUC alone - both datasets are imbalanced), per-attack-class recall, false-positive rate, and few-shot fine-tuning curves (performance vs. % of labels used).

# 8\. Compute and tools

- PyTorch Geometric, single Colab/Kaggle T4 or P100 GPU is sufficient - these are flow-level tabular graphs, not raw packet/image-scale data, so you don't need heavier compute.
- Track experiments with Weights & Biases so your ablation table (SSL objective × encoder × temporal module) is easy to assemble for the thesis.

# 9\. Target venues (for the eventual paper spun out of this thesis)

IEEE Access, Computers & Security, or Journal of Network and Computer Applications are realistic first targets given the novelty scope. We'll pick the final one once your Dataset-A results are in.

# 10\. What I need from you this week

1. Confirm you can access Dataset A (Zenodo) and have requested Dataset B from BCCC.
2. Read the three baseline papers and send me a 2-page summary comparing their pretext tasks and encoders.
3. Get a plain GCN or GraphSAGE baseline running on Dataset A - don't touch the SSL/temporal machinery until that baseline is working end-to-end.

_Come to me with questions on graph construction before you commit to a design - that decision is the hardest one to undo later._
