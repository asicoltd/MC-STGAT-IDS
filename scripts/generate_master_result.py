#!/usr/bin/env python3
"""
generate_master_result.py

Consolidates all experimental results across the MC-STGAT-IDS pipeline into a single
unified 'master_result.json' file:
  1. Baseline models (MLP, GCN, GAT, GraphSAGE, E-GraphSAGE, Anomal-E) across 3 datasets
  2. SSL pretraining & Linear Probes (BYOL, MoCo) across 3 datasets
  3. TFT Spatio-Temporal Fusion (frozen single, frozen ensemble, finetune single)
  4. Decision Threshold sweeps and comparison tables (winner, F1-optimal, balanced, all)
  5. Learning Rate sweeps and diagnostics (dataset statistics, topology, split protocol)
  6. Probing analyses (per-family, multi-class)
  7. Consolidated 42-row summary table and merged flat representation

Usage:
    python generate_master_result.py [--compare-with master_result_backup.json]
"""

import os
import sys
import json
import datetime
from collections import OrderedDict

# Configuration paths
DATASET_CONFIG = {
    'gotham': 'Gotham_processed_features_10s/graphs',
    'nf_bot': 'NFBoTIoT_v3_10s/graphs',
    'nf_ton': 'NFToNIoT_v3_10s/graphs',
}
DATASET_ORDER = ['gotham', 'nf_bot', 'nf_ton']
BASELINE_ORDER = ['mlp', 'gcn', 'gat', 'sage', 'egs', 'anomal_e']
SSL_ORDER = ['byol', 'moco']
TFT_MODES = [
    'finetune_single_byol', 'frozen_ensemble_byol', 'frozen_single_byol',
    'finetune_single_moco', 'frozen_ensemble_moco', 'frozen_single_moco'
]

def load_json_safe(path):
    if os.path.isfile(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"  [WARN] Failed to read {path}: {e}")
            return None
    return None

def build_master_result():
    print("=" * 80)
    print("BUILDING UNIFIED master_result.json")
    print("=" * 80)

    files_loaded = []
    files_missing = []

    def track(name, obj):
        if obj is not None:
            files_loaded.append(name)
        else:
            files_missing.append(name)
        return obj

    # 1. Baseline Benchmark
    print("1. Loading Baseline Models...")
    baseline_path = os.path.join('baseline_checkpoints_tftparity', 'baseline_benchmark.json')
    if not os.path.isfile(baseline_path):
        baseline_path = 'baseline_benchmark.json'
    raw_baseline = track("baseline", load_json_safe(baseline_path))

    # 2. SSL Pretraining Summaries
    print("2. Loading SSL Pretraining Summaries...")
    ssl_byol = track("ssl_byol", load_json_safe('pure_byol_ssl_summary.json'))
    ssl_moco = track("ssl_moco", load_json_safe('pure_moco_ssl_summary.json'))
    ssl_byol_univ = track("ssl_byol_univ", load_json_safe('pure_byol_univ_ssl_summary.json'))
    ssl_moco_univ = track("ssl_moco_univ", load_json_safe('pure_moco_univ_ssl_summary.json'))

    # 3. TFT Phase 3 Reports
    print("3. Loading Phase 3 TFT Reports from disk...")
    tft_dict = {ds: {'byol': {}, 'moco': {}} for ds in DATASET_ORDER}
    tft_report_count = 0

    for ds in DATASET_ORDER:
        root = DATASET_CONFIG.get(ds, '')
        if not os.path.isdir(root):
            continue
        for entry in sorted(os.listdir(root)):
            if not entry.startswith('phase3_'):
                continue
            rep_path = os.path.join(root, entry, 'phase3_report.json')
            if not os.path.isfile(rep_path):
                continue
            rep_data = load_json_safe(rep_path)
            if not rep_data:
                continue

            mode = entry.replace('phase3_', '')
            ssl = rep_data.get('ssl', 'moco' if 'moco' in mode else 'byol')
            if ssl in tft_dict[ds]:
                tft_dict[ds][ssl][mode] = rep_data
                tft_report_count += 1
                print(f"   -> Loaded {ds:7s} | {ssl:4s} | {mode}")

    # 4. Threshold Comparisons
    print("4. Loading Threshold Sweeps...")
    th_winner = track("threshold_winner", load_json_safe('threshold_comparison_winner.json'))
    th_f1opt  = track("threshold_f1opt",  load_json_safe('threshold_comparison_f1opt.json'))
    th_bal    = track("threshold_balanced", load_json_safe('threshold_comparison_balanced.json'))
    th_all    = track("threshold_all", load_json_safe('threshold_comparison_all.json'))

    # 5. LR Sweeps
    print("5. Loading LR Sweeps & Diagnostics...")
    lr_all   = track("lr_sweep_all", load_json_safe('lr_sweep_all_datasets.json'))
    lr_nfbot = track("lr_sweep_nfbot", load_json_safe('hparam_search_summary.json'))
    lr_best  = track("lr_best_final", load_json_safe('final_best_lr_results.json'))
    lr_reeval = track("re_eval_threshold", load_json_safe('re_eval_threshold_sweep.json'))

    # 6. Diagnostics
    diag_ds    = track("dataset_diag", load_json_safe('dataset_diagnostic.json'))
    diag_topo  = track("dataset_topology", load_json_safe('dataset_topology_diagnostic.json'))
    split_diag = track("split_diag", load_json_safe('split_diagnostic_results.json'))
    split_fin  = track("split_final", load_json_safe('split_final_report.json'))

    # 7. Probes
    pf_byol  = track("per_family_byol", load_json_safe('per_family_probe_results_byol.json'))
    pf_moco  = track("per_family_moco", load_json_safe('per_family_probe_results_moco.json'))
    pf_plain = track("per_family_plain", load_json_safe('per_family_probe_results.json'))
    mc_probe = track("multiclass_probe", load_json_safe('multiclass_probe_results.json'))

    # 7b. Tier 1 Controlled Experiments (Ablations & Transfer)
    print("5b. Loading Tier 1 Controlled Experiments (Ablations, Transfer, XAI)...")
    exp1_snapshot = track("tier1_exp1_snapshot", load_json_safe('tier1_experiment1_snapshot_results.json'))
    exp2_shuffle  = track("tier1_exp2_shuffle",  load_json_safe('tier1_experiment2_shuffle_results.json'))
    exp3_transfer = track("tier1_exp3_transfer", load_json_safe('tier1_experiment3_transfer_matrix.json'))
    exp4_ig_xai   = track("tier1_option_a_ig",   load_json_safe('tier1_option_a_ig_results.json'))

    # 8. Build merged_flat
    print("6. Building Normalized 'merged_flat' Dictionary...")
    merged_flat = {'baseline': {}, 'ssl': {}, 'tft': {}}

    # Baseline flat
    if raw_baseline:
        for ds in DATASET_ORDER:
            merged_flat['baseline'][ds] = {}
            for m in BASELINE_ORDER:
                if ds in raw_baseline and m in raw_baseline[ds]:
                    b = raw_baseline[ds][m]
                    merged_flat['baseline'][ds][m] = {
                        'roc': b.get('ROC', b.get('roc')),
                        'pr': b.get('PR', b.get('pr')),
                        'f1': b.get('F1', b.get('f1')),
                        'fpr': b.get('FPR', b.get('fpr')),
                        'fnr': b.get('FNR', b.get('fnr')),
                        'macro_f1': b.get('mF1_multi', b.get('macro_f1')),
                        'status': 'done'
                    }

    # SSL flat (linear probes)
    for ds in DATASET_ORDER:
        merged_flat['ssl'][ds] = {}
        for ssl, summary in [('byol', ssl_byol), ('moco', ssl_moco)]:
            if summary and ds in summary:
                s = summary[ds]
                merged_flat['ssl'][ds][ssl] = {
                    'roc': s.get('test_roc_auc'),
                    'pr': s.get('test_pr_auc'),
                    'f1': s.get('test_f1', s.get('test_f1_binary')),
                    'fpr': s.get('test_fpr'),
                    'fnr': s.get('test_fnr'),
                    'macro_f1': s.get('test_macro_f1')
                }

    # TFT flat
    for ds in DATASET_ORDER:
        merged_flat['tft'][ds] = {'byol': {}, 'moco': {}}
        for ssl in SSL_ORDER:
            for mode, rep in tft_dict[ds][ssl].items():
                t = rep.get('test', {})
                # strip mode prefix for clean keys
                short_mode = mode.replace('_byol', '').replace('_moco', '')
                merged_flat['tft'][ds][ssl][short_mode] = {
                    'roc': t.get('roc'),
                    'pr': t.get('pr'),
                    'f1': t.get('f1'),
                    'fpr': t.get('fpr'),
                    'fnr': t.get('fnr'),
                    'macro_f1': t.get('macro_f1'),
                    'best_epoch': rep.get('best_epoch'),
                    'finetune': rep.get('finetune', 'finetune' in mode)
                }

    # 9. Build summary_rows & summary_table (42 configurations)
    print("7. Generating Consolidated 42-Row Summary Table...")
    summary_rows = []

    # 18 Baseline rows
    for ds in DATASET_ORDER:
        for m in BASELINE_ORDER:
            b = merged_flat['baseline'].get(ds, {}).get(m, {})
            summary_rows.append({
                'family': 'baseline',
                'dataset': ds,
                'ssl': '-',
                'mode': m,
                'roc': b.get('roc'),
                'pr': b.get('pr'),
                'f1': b.get('f1'),
                'fpr': b.get('fpr'),
                'fnr': b.get('fnr'),
                'macro_f1': b.get('macro_f1')
            })

    # 6 SSL linear probe rows
    for ssl in SSL_ORDER:
        for ds in DATASET_ORDER:
            s = merged_flat['ssl'].get(ds, {}).get(ssl, {})
            summary_rows.append({
                'family': 'ssl',
                'dataset': ds,
                'ssl': ssl,
                'mode': 'linear_probe',
                'roc': s.get('roc'),
                'pr': s.get('pr'),
                'f1': s.get('f1'),
                'fpr': s.get('fpr'),
                'fnr': s.get('fnr'),
                'macro_f1': s.get('macro_f1')
            })

    # 18 TFT rows (ordered by dataset -> ssl -> mode)
    tft_order_modes = [
        'finetune_single',
        'frozen_ensemble',
        'frozen_single'
    ]
    for ds in DATASET_ORDER:
        for ssl in SSL_ORDER:
            for smode in tft_order_modes:
                full_mode = f"{smode}_{ssl}"
                rep = tft_dict[ds][ssl].get(full_mode)
                if rep:
                    t = rep.get('test', {})
                    summary_rows.append({
                        'family': 'tft',
                        'dataset': ds,
                        'ssl': ssl,
                        'mode': full_mode,
                        'roc': t.get('roc'),
                        'pr': t.get('pr'),
                        'f1': t.get('f1'),
                        'fpr': t.get('fpr'),
                        'fnr': t.get('fnr'),
                        'macro_f1': t.get('macro_f1'),
                        'finetune': rep.get('finetune', 'finetune' in full_mode),
                        'best_epoch': rep.get('best_epoch')
                    })

    # 10. Assemble Final Master Dictionary
    master_result = OrderedDict([
        ("meta", {
            "generated_utc": datetime.datetime.utcnow().isoformat() + "Z",
            "schema_version": "1.0",
            "files_loaded": sorted(files_loaded),
            "files_missing": sorted(files_missing),
            "n_tft_reports": tft_report_count,
            "description": "Master result file for MC-STGAT-IDS. Consolidates baseline, SSL pretraining, TFT fine-tuning, threshold comparisons, diagnostics, and LR sweeps into a single JSON."
        }),
        ("baseline", raw_baseline),
        ("ssl", {
            "byol": ssl_byol,
            "moco": ssl_moco,
            "byol_univ": ssl_byol_univ,
            "moco_univ": ssl_moco_univ
        }),
        ("tft", tft_dict),
        ("threshold_comparison", {
            "f1_optimal": th_f1opt,
            "balanced": th_bal,
            "winner": th_winner,
            "all": th_all
        }),
        ("lr_sweeps", {
            "all_datasets": lr_all,
            "nf_bot_byol": lr_nfbot,
            "final_best": lr_best,
            "re_eval": lr_reeval
        }),
        ("diagnostics", {
            "dataset": diag_ds,
            "topology": diag_topo,
            "split_search": split_diag,
            "split_final": split_fin
        }),
        ("tier1_controlled_experiments", {
            "experiment1_snapshot_ablation": exp1_snapshot,
            "experiment2_sequence_shuffle_ablation": exp2_shuffle,
            "experiment3_cross_dataset_transfer": exp3_transfer,
            "experiment4_integrated_gradients_xai": exp4_ig_xai
        }),
        ("probes", {
            "per_family": {
                "byol": pf_byol,
                "moco": pf_moco,
                "plain": pf_plain
            },
            "multiclass": mc_probe
        }),
        ("merged_flat", merged_flat),
        ("summary_table", summary_rows),
        ("summary_rows", summary_rows)
    ])

    return master_result

def compare_results(new_db, ref_path):
    if not os.path.isfile(ref_path):
        print(f"\n[INFO] Reference file '{ref_path}' not found — skipping diff comparison.")
        return

    print("\n" + "=" * 80)
    print(f"VERIFYING AND COMPARING AGAINST {ref_path}")
    print("=" * 80)

    with open(ref_path, 'r', encoding='utf-8') as f:
        old_db = json.load(f)

    # 1. Structure / Key Comparison
    print("\n[1] Section Structure Check:")
    for k in new_db.keys():
        in_old = k in old_db
        print(f"  - Section '{k}': {'MATCH' if in_old else 'NEW'}")

    # 2. Baseline Check
    b_match = (new_db['baseline'] == old_db.get('baseline'))
    print(f"\n[2] Baseline Benchmark Matrix: {'IDENTICAL MATCH (100%)' if b_match else 'DIFFERENT'}")

    # 3. SSL Check
    ssl_byol_match = (new_db['ssl']['byol'] == old_db.get('ssl', {}).get('byol'))
    ssl_moco_match = (new_db['ssl']['moco'] == old_db.get('ssl', {}).get('moco'))
    print(f"[3] SSL Pretraining Summaries: BYOL={'MATCH' if ssl_byol_match else 'DIFF'} | MoCo={'MATCH' if ssl_moco_match else 'DIFF'}")

    # 4. Diagnostics Check
    d_diag_match = (new_db['diagnostics']['dataset'] == old_db.get('diagnostics', {}).get('dataset'))
    d_topo_match = (new_db['diagnostics']['topology'] == old_db.get('diagnostics', {}).get('topology'))
    print(f"[4] Dataset Diagnostics: Dataset={'MATCH' if d_diag_match else 'DIFF'} | Topology={'MATCH' if d_topo_match else 'DIFF'}")

    # 5. Probes Check
    pr_byol = (new_db['probes']['per_family']['byol'] == old_db.get('probes', {}).get('per_family', {}).get('byol'))
    pr_moco = (new_db['probes']['per_family']['moco'] == old_db.get('probes', {}).get('per_family', {}).get('moco'))
    print(f"[5] Probing Analyses: BYOL={'MATCH' if pr_byol else 'DIFF'} | MoCo={'MATCH' if pr_moco else 'DIFF'}")

    # 6. Summary Rows Comparison
    print(f"\n[6] Summary Rows Comparison (New: {len(new_db['summary_rows'])}, Old: {len(old_db.get('summary_rows', []))} rows):")
    print(f"{'FAM':<8} {'DATASET':<8} {'SSL':<5} {'MODE':<24} | {'NEW F1':<8} {'OLD F1':<8} | {'NEW mF1':<8} {'OLD mF1':<8} | STATUS")
    print("-" * 95)

    old_map = {(r['family'], r['dataset'], r['ssl'], r['mode']): r for r in old_db.get('summary_rows', [])}
    matches = 0
    updates = 0

    for r in new_db['summary_rows']:
        key = (r['family'], r['dataset'], r['ssl'], r['mode'])
        old_r = old_map.get(key)
        nf1 = r.get('f1') or 0.0
        nmf1 = r.get('macro_f1') or 0.0
        if old_r:
            of1 = old_r.get('f1') or 0.0
            omf1 = old_r.get('macro_f1') or 0.0
            if abs(nf1 - of1) < 1e-4 and abs(nmf1 - omf1) < 1e-4:
                status = "EXACT MATCH"
                matches += 1
            else:
                status = "UPDATED (Fresh Run)"
                updates += 1
            print(f"{r['family']:<8} {r['dataset']:<8} {r['ssl']:<5} {r['mode']:<24} | {nf1:<8.4f} {of1:<8.4f} | {nmf1:<8.4f} {omf1:<8.4f} | {status}")
        else:
            print(f"{r['family']:<8} {r['dataset']:<8} {r['ssl']:<5} {r['mode']:<24} | {nf1:<8.4f} {'-':<8} | {nmf1:<8.4f} {'-':<8} | NEW ENTRY")

    print("-" * 95)
    print(f"Summary Rows Result: {matches} Exact Matches, {updates} Updated with latest run artifacts.")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate consolidated master_result.json")
    parser.add_argument("--compare-with", default="master_result_backup.json",
                        help="Path to previous master_result.json to run diff check against")
    parser.add_argument("--output", default="master_result.json",
                        help="Output path for unified master JSON")
    args = parser.parse_args()

    # Generate master result
    master = build_master_result()

    # Run comparison against reference
    if args.compare_with:
        compare_results(master, args.compare_with)

    # Save to output file
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(master, f, indent=2)
    print(f"\n[SUCCESS] Successfully written unified master result to '{args.output}' ({os.path.getsize(args.output)/1024:.1f} KB).")

if __name__ == '__main__':
    main()
