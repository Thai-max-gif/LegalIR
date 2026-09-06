#!/usr/bin/env python3
"""CLI script to aggregate OOF fold results, apply score promotion, and generate production_lock.json."""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.validation.promotion import aggregate_oof_metrics, create_production_lock


def main():
    parser = argparse.ArgumentParser(description="Select and lock production configuration from OOF results.")
    parser.add_argument("--folds-dir", type=str, default="artifacts/factory/folds", help="Path to folds directory")
    parser.add_argument("--fusion-descriptor", type=str, default="artifacts/factory/fusion/fusion_model.json", help="Path to fusion_model.json descriptor")
    parser.add_argument("--output-lock", type=str, default="artifacts/factory/production_lock.json", help="Path to output lock")
    parser.add_argument("--runtime-commit", type=str, default="a0efb25", help="Approved runtime git commit SHA")
    args = parser.parse_args()

    folds_root = Path(args.folds_dir)
    print(f"[*] Aggregating OOF metrics from {folds_root} ...")

    metrics_list = []
    for f in range(5):
        m_file = folds_root / f"fold_{f}" / "fold_metrics.json"
        if m_file.is_file():
            with open(m_file, "r", encoding="utf-8") as fp:
                metrics_list.append(json.load(fp))

    if not metrics_list:
        print("[!] No fold metrics found to aggregate.")
        sys.exit(1)

    agg = aggregate_oof_metrics(metrics_list)
    print(f"[+] Aggregate OOF Metrics: {agg}")

    # Derive fusion configuration from frozen fusion descriptor if available
    fusion_desc_p = Path(args.fusion_descriptor)
    fusion_cfg = {
        "method": "reciprocal_rank_fusion",
        "k": 60,
        "weights": {"bm25": 1.0, "bm25_pyvi": 1.0, "dense": 1.2, "exact": 2.5, "rerank": 1.8},
        "candidate_k": 150,
        "rerank_k": 50,
        "top_k": 5,
    }
    if fusion_desc_p.is_file():
        try:
            with open(fusion_desc_p, "r", encoding="utf-8") as f:
                desc = json.load(f)
            if desc.get("winning_method") == "reciprocal_rank_fusion":
                rrf_info = desc.get("rrf", {})
                fusion_cfg = {
                    "method": "reciprocal_rank_fusion",
                    "k": int(rrf_info.get("k", 60)),
                    "weights": rrf_info.get("weights", fusion_cfg["weights"]),
                    "candidate_k": 150,
                    "rerank_k": 50,
                    "top_k": 5,
                }
            elif desc.get("winning_method") == "learned_ranker":
                fusion_cfg = {
                    "method": "learned_ranker",
                    "model_file": desc.get("learned_model", {}).get("file", "fusion_model.txt"),
                    "feature_columns": desc.get("feature_columns", []),
                    "candidate_k": 150,
                    "rerank_k": 50,
                    "top_k": 5,
                }
        except Exception as e:
            print(f"[!] Warning parsing fusion descriptor: {e}")

    approved_config = {
        "fusion": fusion_cfg,
        "reranker": {
            "model_name": "BAAI/bge-reranker-v2-m3",
            "max_length": 512,
            "effective_batch_size": 16,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
        }
    }

    create_production_lock(
        output_path=args.output_lock,
        metrics=agg,
        config=approved_config,
        runtime_commit=args.runtime_commit,
    )

    print(f"[+] Successfully locked production configuration to {args.output_lock}")


if __name__ == "__main__":
    main()
