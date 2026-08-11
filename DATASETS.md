# Datasets Used in MC-STGAT-IDS Thesis

## Dataset Status Overview

| Dataset | Original Plan | Actual Use | Status |
|---|---|---|---|
| Gotham Dataset 2025 | Primary dataset | Used | ✅ Available |
| BCCC-IoT-MQTT-IDS-2025 | Secondary dataset | Not used | ❌ Access unavailable |
| NF-ToN-IoT-v2 | Not originally specified | Secondary/replacement dataset | ✅ Used |

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

**Key Statistics (from exploration):**
- **Total records:** 35,134,281 packets across 78 CSV files
- **Time range:** 2025-01-14 18:39:55 to 2025-01-25 19:46:06 (~11 days)
- **Unique source IPs:** 108
- **Unique destination IPs:** 4,116
- **Unique (src,dst) pairs:** 20,178

**Attack Types Present:**
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

**Protocol Distribution:**
| Protocol | Count |
|---|---|
| UDP (17) | 20,659,109 |
| TCP (6) | 8,481,367 |
| GRE (47) | 5,911,401 |
| ICMP (1) | 82,404 |

**Available Features (23 columns):**
- `frame.time`, `frame.len`, `frame.protocols`
- `eth.src`, `eth.dst`
- `ip.src`, `ip.dst`, `ip.flags`, `ip.ttl`, `ip.proto`, `ip.checksum`, `ip.tos`
- `tcp.srcport`, `tcp.dstport`, `tcp.flags`, `tcp.window_size_value`, `tcp.window_size_scalefactor`, `tcp.checksum`, `tcp.options`, `tcp.pdu.size`
- `udp.srcport`, `udp.dstport`
- `label`, `source_file`

**Role in Thesis:**
- Primary dataset for developing and validating the full MC-STGAT-IDS pipeline
- Used for graph construction, SSL pretraining, temporal fusion, and classification fine-tuning
- All baseline comparisons performed on this dataset

---

### Dataset B: NF-ToN-IoT-v2 (Replacement Secondary Dataset)

**Access:** `NF-ToN-IoT-v2/NF-ToN-IoT-v2.csv`

**Original Reference:** NF-ToN-IoT-v2 (Network Flow-based ToN-IoT version 2)

**Description:**
- Flow-level IoT intrusion detection dataset
- Contains network flow statistics rather than raw packet captures
- Rich set of 45 features capturing flow behavior

**Key Statistics (from exploration):**
- **Total records:** 16,940,496 flows
- **Unique source IPs:** 23,079
- **Unique destination IPs:** 6,868
- **Unique (src,dst) pairs:** 31,645

**Attack Distribution:**
| Attack Type | Count |
|---|---|
| Benign | 6,099,469 |
| scanning | 3,781,419 |
| xss | 2,455,020 |
| ddos | 2,026,234 |
| password | 1,153,323 |
| dos | 712,609 |
| injection | 684,465 |
| backdoor | 16,809 |
| mitm | 7,723 |
| ransomware | 3,425 |

**Binary Label Distribution:**
| Label | Count |
|---|---|
| 1 (Attack) | 10,841,027 |
| 0 (Benign) | 6,099,469 |

**Protocol Distribution:**
| Protocol | Count |
|---|---|
| TCP (6) | 14,427,073 |
| UDP (17) | 2,491,336 |
| ICMP (1) | 17,933 |
| Others | < 5,000 |

**Available Features (45 columns):**
- Network layer: `IPV4_SRC_ADDR`, `IPV4_DST_ADDR`, `PROTOCOL`, `L4_SRC_PORT`, `L4_DST_PORT`
- Traffic volume: `IN_BYTES`, `IN_PKTS`, `OUT_BYTES`, `OUT_PKTS`
- Duration: `FLOW_DURATION_MILLISECONDS`, `DURATION_IN`, `DURATION_OUT`
- Packet length: `MIN_IP_PKT_LEN`, `MAX_IP_PKT_LEN`, `LONGEST_FLOW_PKT`, `SHORTEST_FLOW_PKT`
- Throughput: `SRC_TO_DST_AVG_THROUGHPUT`, `DST_TO_SRC_AVG_THROUGHPUT`
- Retransmission: `RETRANSMITTED_IN_BYTES`, `RETRANSMITTED_IN_PKTS`, `RETRANSMITTED_OUT_BYTES`, `RETRANSMITTED_OUT_PKTS`
- TCP: `TCP_FLAGS`, `CLIENT_TCP_FLAGS`, `SERVER_TCP_FLAGS`, `TCP_WIN_MAX_IN`, `TCP_WIN_MAX_OUT`
- Packet size distribution: `NUM_PKTS_UP_TO_128_BYTES`, `NUM_PKTS_128_TO_256_BYTES`, `NUM_PKTS_256_TO_512_BYTES`, `NUM_PKTS_512_TO_1024_BYTES`, `NUM_PKTS_1024_TO_1514_BYTES`
- Protocol-specific: `L7_PROTO`, `ICMP_TYPE`, `ICMP_IPV4_TYPE`, `DNS_QUERY_ID`, `DNS_QUERY_TYPE`, `DNS_TTL_ANSWER`, `FTP_COMMAND_RET_CODE`
- Labels: `Label` (binary), `Attack` (multiclass)

**Role in Thesis:**
- Secondary dataset for evaluating cross-dataset generalization
- Tests whether MC-STGAT-IDS trained on Gotham Dataset can transfer to a different IoT traffic distribution
- Provides richer flow-level features for ablation studies

---

### Dataset C: BCCC-IoT-MQTT-IDS-2025 (Not Used)

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

### Decision 1: BCCC-IoT-MQTT-IDS-2025 → NF-ToN-IoT-v2

**Date:** Phase 0

**Decision:** Replace BCCC-IoT-MQTT-IDS-2025 with NF-ToN-IoT-v2 as the secondary dataset.

**Reasoning:**
- BCCC-IoT-MQTT-IDS-2025 could not be accessed during the implementation period due to access request delays
- NF-ToN-IoT-v2 is publicly available and can be immediately used
- NF-ToN-IoT-v2 provides flow-level IoT traffic data comparable to the primary dataset's format

**Objective Preserved:**
- Evaluate the MC-STGAT-IDS framework on an additional IoT intrusion-detection dataset
- Assess the model's generalization beyond the primary dataset (Gotham Dataset 2025)

**Effect on Architecture:** None. Both datasets are graph-structured with flow-based features, requiring the same graph construction pipeline (nodes = hosts, edges = flows).

**Effect on Experiments:**
- Dataset-specific preprocessing adapted to NF-ToN-IoT-v2's 45-column structure
- Graph construction uses the same node/edge methodology:
  - Nodes: IP addresses (`IPV4_SRC_ADDR`, `IPV4_DST_ADDR`)
  - Edges: flows between (src, dst, protocol, src_port, dst_port)
  - Edge features: traffic volume, timing, packet distribution, TCP flags
- Class distribution differs, requiring appropriate handling of class imbalance (focal loss remains suitable)
- Evaluation procedures and metrics unchanged

---

### Decision 2: Synthetic Time Windows for NF-ToN-IoT-v2

**Date:** Phase 0

**Decision:** Create synthetic timestamps for NF-ToN-IoT-v2 based on row ordering rather than using real timestamps.

**Reasoning:**
- NF-ToN-IoT-v2 does not include real timestamp data
- The original dataset is organized as flows without explicit temporal ordering
- To maintain the temporal-fusion architecture (Stage 2), a temporal sequence is required

**Implementation:**
- `synthetic_time = '2024-01-01' + index (in seconds)`
- `window_start = synthetic_time.floor('60s')`
- Maintains the flow sequence order as a proxy for temporal progression

**Effect on Architecture:** None. The temporal fusion module treats these as 60-second windows regardless of whether timestamps are real or synthetic.

**Effect on Experiments:**
- Results on NF-ToN-IoT-v2 reflect sequential pattern learning rather than wall-clock temporal dynamics
- Cross-dataset generalization evaluation remains valid as the model learns structural patterns independent of absolute time

---

## Preprocessing Summary

### Gotham Dataset 2025 Preprocessing

| Step | Description |
|---|---|
| Missing value handling | Critical columns (frame.time, ip.src, ip.dst, label, ip.proto, frame.len) have no missing values |
| Port column filling | `tcp.srcport`, `tcp.dstport`, `udp.srcport`, `udp.dstport` filled with 0 |
| Window creation | `window_start = frame.time.floor('60s')` |
| Protocol mapping | `proto_name` mapping for TCP(6), UDP(17), ICMP(1), IGMP(2), OSPF(89) |
| Data export | Saved to `combined_raw_with_windows.parquet` (35M rows) |

### NF-ToN-IoT-v2 Preprocessing

| Step | Description |
|---|---|
| Missing value handling | None found in any column |
| Synthetic timestamp | `synthetic_time = '2024-01-01' + index (seconds)` |
| Window creation | `window_start = synthetic_time.floor('60s')` |
| Data export | In-memory only (16.9M rows) |

---

## References

1. **Gotham Dataset 2025** (Cardiff University + Toshiba, Feb 2025)
   - DOI: 10.5281/zenodo.14502760
   - Paper: arXiv 2502.03134
   - 78 emulated IoT devices, 23 features, 35M packets

2. **NF-ToN-IoT-v2**
   - Flow-level IoT intrusion dataset
   - 45 features, 16.9M flows
   - Binary + multiclass labels (10 attack types)

3. **BCCC-IoT-MQTT-IDS-2025** (Not used)
   - Kouhi & Lashkari, Journal of Supercomputing, 2026
   - MQTTFlowLyzer dataset
   - Access unavailable during implementation
