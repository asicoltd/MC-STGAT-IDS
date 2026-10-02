# Datasets Used in MC-STGAT-IDS Thesis

## Dataset Status Overview

| Dataset | Original Plan | Actual Use | Status |
|---|---|---|---|
| Gotham Dataset 2025 | Primary dataset | Used | ✅ Available |
| BCCC-IoT-MQTT-IDS-2025 | Secondary dataset | Not used | ❌ Access unavailable |
| NF-ToN-IoT-v3 | Replacement secondary dataset | Used | ✅ Available |
| NF-BoT-IoT-v3 | Replacement secondary dataset | Tertiary dataset | ✅ Available |

---

## Dataset Details

### Dataset A: Gotham Dataset 2025 (Primary Dataset)

**Access:** [Zenodo DOI 10.5281/zenodo.14502760](https://zenodo.org/records/14502760) (also indexed on Kaggle)

**Publication:** arXiv 2502.03134

**Description:**
- 78 emulated IoT devices across 4 network segments (smart home, industrial, wearables, appliances)
- Traffic includes MQTT/CoAP/RTSP protocols
- Captured per-device (distributed, non-IID) — suitable for host-communication-graph view
- PCAP + labeled CSV provided

**Key Statistics (from notebook exploration):**
- **Total packets:** 35,134,281 across 78 CSV files
- **Total flows after packet→flow aggregation:** 7,919,548
- **Graphs built (10 s windows):** 3,960
- **Unique source IPs:** 108
- **Unique destination IPs:** 4,116
- **Unique (src,dst) pairs:** 20,178

**Attack Types Present (original counts):**
- Mirai UDP Flooding (8,897,895 packets)
- Mirai TCP Flooding (6,548,173 packets)
- Mirai GRE Flooding (5,911,401 packets)
- TCP Scan (737,764 packets)
- CoAP Amplification (274,837 packets)
- Telnet Brute Force (227,649 packets)
- Merlin TCP Flooding (120,000 packets)
- Merlin ICMP Flooding (57,580 packets)
- Merlin UDP Flooding (29,996 packets)
- Merlin C&C Communication (29,356 packets)
- Ingress Tool Transfer (21,587 packets)
- File Download (7,196 packets)
- UDP Scan (4,242 packets)
- Mirai C&C Communication (1,074 packets)
- C&C Communication (528 packets)
- Reporting (450 packets)
- Unknown (7,670 packets)
- **Benign:** 12,256,883 packets

**Protocol Distribution (original):**
| Protocol | Count |
|---|---|
| UDP (17) | 20,659,109 |
| TCP (6) | 8,481,367 |
| GRE (47) | 5,911,401 |
| ICMP (1) | 82,404 |

**Features Used for Graph Construction:**
- **Edge features (NetFlow v9-compatible):** `n_pkts`, `n_bytes`, `dur`, `pkt_size_mean`, `flag_syn`, `flag_ack`, `flag_fin`, `flag_rst`, `flag_psh`, `proto`
- **Node features:** `out_pkts`, `out_bytes`, `out_mean_pkt_size`, `out_syn_ratio`, `out_ack_ratio`, `out_rst_ratio`, `out_fin_ratio`
- **Window size:** 10 seconds
- **Max nodes per graph:** 128
- **Max edges per graph:** 20,000

**Role in Thesis:**
- Primary dataset for developing and validating the full MC-STGAT-IDS pipeline
- Used for graph construction, SSL pretraining, temporal fusion, and classification fine-tuning
- All baseline comparisons performed on this dataset

---

### Dataset B: NF-ToN-IoT-v3 (Secondary Dataset)

**Access:** `NF-ToN-IoT-V3/NF-ToN-IoT-v3.csv`

**Description:**
- Flow-level IoT intrusion detection dataset
- Contains network flow statistics rather than raw packet captures
- Raw file has 55 columns; a NetFlow v9-compatible subset is used for graph construction

**Key Statistics (from notebook exploration):**
- **Total records loaded:** 27,520,260 flows
- **Graphs built (10 s windows):** 33,435
- **Unique source IPs:** 23,079
- **Unique destination IPs:** 6,868
- **Unique (src,dst) pairs:** 31,645

**Attack Distribution (from notebook):**
| Attack Type | Count |
|---|---|
| Benign | 16,792,214 |
| ddos | 4,141,256 |
| xss | 2,834,435 |
| password | 1,594,777 |
| scanning | 1,358,977 |
| injection | 381,777 |
| dos | 203,456 |
| Backdoor | 203,384 |
| mitm | 6,013 |
| ransomware | 3,971 |

**Binary Label Distribution:**
| Label | Count |
|---|---|
| 1 (Attack) | 10,728,046 |
| 0 (Benign) | 16,792,214 |

**Features Used for Graph Construction (NetFlow v9-compatible subset):**
- Network layer: `IPV4_SRC_ADDR`, `IPV4_DST_ADDR`, `PROTOCOL`, `L4_SRC_PORT`, `L4_DST_PORT`
- Traffic volume: `IN_BYTES`, `OUT_BYTES`, `IN_PKTS`, `OUT_PKTS`
- Duration: `FLOW_DURATION_MILLISECONDS`
- TCP: `TCP_FLAGS`, `TCP_WIN_MAX_IN`, `TCP_WIN_MAX_OUT`
- Labels: `Attack` (multiclass), used to derive `label`

**Role in Thesis:**
- Secondary dataset for evaluating cross-dataset generalization
- Tests whether MC-STGAT-IDS trained on Gotham Dataset can transfer to a different IoT traffic distribution
- Provides richer flow-level features for ablation studies

---

### Dataset C: NF-BoT-IoT-v3 (Tertiary Dataset)

**Access:** `NF-BoT-IoT-v3/NF-BoT-IoT-v3.csv`

**Description:**
- Flow-level IoT intrusion detection dataset (NetFlow v3 version)
- Same 55-column schema as NF-ToN-IoT-v3
- Used as an additional dataset to test generalization across different IoT attack families

**Key Statistics (from notebook exploration):**
- **Total records loaded:** 16,933,808 flows
- **Graphs built (10 s windows):** 7,387
- **Unique source IPs:** 32,381 (node rows)

**Attack Distribution (from notebook):**
| Attack Type | Count |
|---|---|
| DoS | 8,034,190 |
| DDoS | 7,150,882 |
| Reconnaissance | 1,695,132 |
| Benign | 51,989 |
| Theft | 1,615 |

**Binary Label Distribution:**
| Label | Count |
|---|---|
| 1 (Attack) | 16,881,819 |
| 0 (Benign) | 51,989 |

**Features Used for Graph Construction (NetFlow v9-compatible subset):**
- Same as NF-ToN-IoT-v3 (network layer, traffic volume, duration, TCP flags, windows)
- Labels: `Attack` (multiclass), used to derive `label`

**Role in Thesis:**
- Tertiary dataset for additional cross-dataset generalization
- Provides a different attack distribution (DoS/DDoS/Reconnaissance/Theft) compared to NF-ToN-IoT-v3
- Strengthens robustness evaluation of the MC-STGAT-IDS framework

---

### Dataset D: BCCC-IoT-MQTT-IDS-2025 (Not Used)

**Status:** ❌ Access unavailable during implementation period

**Original Plan:**
- Protocol-aware MQTT flow-level dataset combining MQTTset, MQTT-IoT-IDS2020, and DoS/DDoS-MQTT-IoT
- 404 raw features (378 after preprocessing) capturing session behavior and message-level interaction
- Explicitly designed to support attention-based and LLM-assisted intrusion detection
- Attacks: brute-force authentication, malformed messages, SlowITe flooding, scanning, DoS/DDoS variants
- Authors: Kouhi & Lashkari, Journal of Supercomputing, 2026

**Reason for Non-Use:**
- Access required a request form through BCCC datasets page
- Approval/access not granted within the project timeline
- The dataset request process would have delayed early milestones

---

## Research Decision Log

### Decision 1: BCCC-IoT-MQTT-IDS-2025 → NF-ToN-IoT-v3

**Date:** Phase 0

**Decision:** Replace BCCC-IoT-MQTT-IDS-2025 with NF-ToN-IoT-v3 as the secondary dataset.

**Reasoning:**
- BCCC-IoT-MQTT-IDS-2025 could not be accessed during the implementation period due to access request delays
- NF-ToN-IoT-v3 is publicly available and can be immediately used
- NF-ToN-IoT-v3 provides flow-level IoT traffic data comparable to the primary dataset's format

**Objective Preserved:**
- Evaluate the MC-STGAT-IDS framework on an additional IoT intrusion-detection dataset
- Assess the model's generalization beyond the primary dataset (Gotham Dataset 2025)

**Effect on Architecture:** None. Both datasets are graph-structured with flow-based features, requiring the same graph construction pipeline (nodes = hosts, edges = flows).

**Effect on Experiments:**
- Dataset-specific preprocessing adapted to NF-ToN-IoT-v3's 55-column structure
- Graph construction uses the same node/edge methodology:
  - Nodes: IP addresses (`IPV4_SRC_ADDR`, `IPV4_DST_ADDR`)
  - Edges: flows between (src, dst, protocol, src_port, dst_port)
  - Edge features: traffic volume, timing, packet distribution, TCP flags
- Class distribution differs, requiring appropriate handling of class imbalance (focal loss remains suitable)
- Evaluation procedures and metrics unchanged

---

### Decision 2: Addition of NF-BoT-IoT-v3 as Tertiary Dataset

**Date:** Phase 0

**Decision:** Add NF-BoT-IoT-v3 as an additional dataset for cross-dataset generalization.

**Reasoning:**
- NF-BoT-IoT-v3 is publicly available and uses the same 55-column NetFlow v3 schema as NF-ToN-IoT-v3
- Provides a different attack distribution (DoS, DDoS, Reconnaissance, Theft) that complements NF-ToN-IoT-v3
- Allows testing of the framework on a broader range of IoT attack families without additional preprocessing complexity

**Effect on Architecture:** None. The same graph construction pipeline is reused.

**Effect on Experiments:**
- Additional cross-dataset evaluation results reported
- Graph construction follows the identical node/edge methodology
- Class imbalance handled with focal loss

---

### Decision 3: Real Timestamps for NF-ToN-IoT-v3 and NF-BoT-IoT-v3

**Date:** Phase 0

**Decision:** Use the provided `FLOW_START_MILLISECONDS` column as real timestamps for window creation.

**Reasoning:**
- Both NF-ToN-IoT-v3 and NF-BoT-IoT-v3 include `FLOW_START_MILLISECONDS`, which provides a real temporal ordering
- The original plan to synthesize timestamps is no longer necessary
- Real timestamps enable more faithful temporal fusion (Stage 2)

**Implementation:**
- `window_start = pd.to_datetime(FLOW_START_MILLISECONDS, unit='ms', utc=True).dt.floor('10s')`
- No synthetic timestamp generation is performed

**Effect on Architecture:** None. The temporal fusion module treats these as 10-second windows regardless of timestamp source.

**Effect on Experiments:**
- Results reflect real temporal dynamics for NF-ToN-IoT-v3 and NF-BoT-IoT-v3
- Cross-dataset generalization evaluation remains valid

---

### Decision 4: Global Chronological Split (70/15/15)

**Date:** Phase 0

**Decision:** Use a single global chronological split across all graphs, rather than a per-class split.

**Reasoning:**
- Prevents temporal leakage that could occur if validation/test windows were interleaved with training windows
- Ensures train < val < test in wall-clock order
- Classes confined to a narrow time range may have zero examples in some splits; this is reported rather than reshuffled

**Implementation:**
- Sort all graphs by `window_start`
- Cut the sorted timeline at 70% (train), 15% (val), 15% (test)
- Persist `window_start_unix` on every PyG Data object for downstream gap filtering

**Effect on Experiments:**
- Split protocol documented as `global_chronological_70_15_15`
- Class coverage per split is reported in the notebook output

---

### Decision 5: NetFlow v9-Compatible Feature Set

**Date:** Phase 0

**Decision:** Restrict edge and node features to fields derivable from NetFlow v9 records.

**Reasoning:**
- Aligns with PPT-GNN / GraphIDS literature
- Ensures features are available in both packet-level (Gotham) and flow-level (NF-v3) datasets after aggregation
- Simplifies cross-dataset transfer

**Dropped Features:**
- Edge: `pkt_size_std`, `ttl_mean`, `tos_mean`, `win_size_mean`
- Node: `out_mean_ttl`

**Final Feature Set:**
- **Edge:** `n_pkts`, `n_bytes`, `dur`, `pkt_size_mean`, `flag_syn`, `flag_ack`, `flag_fin`, `flag_rst`, `flag_psh`, `proto`
- **Node:** `out_pkts`, `out_bytes`, `out_mean_pkt_size`, `out_syn_ratio`, `out_ack_ratio`, `out_rst_ratio`, `out_fin_ratio`

**Effect on Architecture:** None. The reduced feature set is used consistently across all datasets.

---

## Preprocessing Summary

### Gotham Dataset 2025 Preprocessing

| Step | Description |
|---|---|
| CSV loading | 78 files loaded in chunks, only necessary columns kept |
| Timestamp conversion | `frame.time` → UTC datetime |
| Window creation | `window_start = frame.time.dt.floor('10s')` |
| TCP flag decoding | `tcp.flags` hex string → integer bitmask → 5 binary flags |
| Port extraction | TCP/UDP source and destination ports combined |
| Packet → flow aggregation | Group by `(window_start, src_ip_id, dst_ip_id, src_port, dst_port, ip.proto)` |
| Flow features computed | `n_pkts`, `n_bytes`, `dur`, `pkt_size_mean`, TCP flags (max) |
| Graph construction | 3,960 graphs built with 10 s windows, max 128 nodes / 20,000 edges |
| Split | Global chronological 70/15/15 |
| Normalisation | Computed from TRAIN split only |
| Export | `train_graphs.pkl`, `val_graphs.pkl`, `test_graphs.pkl`, `norm_stats.json` |

### NF-ToN-IoT-v3 Preprocessing

| Step | Description |
|---|---|
| CSV loading | Chunked loading (2M rows per chunk), 27,520,260 rows total |
| Timestamp conversion | `FLOW_START_MILLISECONDS` → UTC datetime |
| Window creation | `window_start = timestamp.dt.floor('10s')` |
| IP encoding | `IPV4_SRC_ADDR`, `IPV4_DST_ADDR` → integer IDs |
| Port/protocol extraction | `L4_SRC_PORT`, `L4_DST_PORT`, `PROTOCOL` |
| Flow feature aggregation | `n_bytes`, `n_pkts`, `dur`, `pkt_size_mean`, TCP flags decoded |
| Graph construction | 33,435 graphs built with 10 s windows |
| Split | Global chronological 70/15/15 |
| Normalisation | Computed from TRAIN split only |
| Export | `train_graphs.pkl`, `val_graphs.pkl`, `test_graphs.pkl`, `norm_stats.json` |

### NF-BoT-IoT-v3 Preprocessing

| Step | Description |
|---|---|
| CSV loading | Chunked loading (2M rows per chunk), 16,933,808 rows total |
| Timestamp conversion | `FLOW_START_MILLISECONDS` → UTC datetime |
| Window creation | `window_start = timestamp.dt.floor('10s')` |
| IP encoding | `IPV4_SRC_ADDR`, `IPV4_DST_ADDR` → integer IDs |
| Port/protocol extraction | `L4_SRC_PORT`, `L4_DST_PORT`, `PROTOCOL` |
| Flow feature aggregation | `n_bytes`, `n_pkts`, `dur`, `pkt_size_mean`, TCP flags decoded |
| Graph construction | 7,387 graphs built with 10 s windows |
| Split | Global chronological 70/15/15 |
| Normalisation | Computed from TRAIN split only |
| Export | `train_graphs.pkl`, `val_graphs.pkl`, `test_graphs.pkl`, `norm_stats.json` |

---

## References

1. **Gotham Dataset 2025** (Cardiff University + Toshiba, Feb 2025)
   - DOI: 10.5281/zenodo.14502760
   - Paper: arXiv 2502.03134
   - 78 emulated IoT devices, 23 features, 35M packets

2. **NF-ToN-IoT-v3**
   - Flow-level IoT intrusion dataset
   - 55 columns, 27.5M flows
   - Binary + multiclass labels (10 attack types)

3. **NF-BoT-IoT-v3**
   - Flow-level IoT intrusion dataset
   - 55 columns, 16.9M flows
   - Binary + multiclass labels (DoS, DDoS, Reconnaissance, Theft, Benign)

4. **BCCC-IoT-MQTT-IDS-2025** (Not used)
   - Kouhi & Lashkari, Journal of Supercomputing, 2026
   - MQTTFlowLyzer dataset
   - Access unavailable during implementation