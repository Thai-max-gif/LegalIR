#!/usr/bin/env python3
"""
Kaggle Final Production Runner (Stages K0 - K9).
Executes only:
- runtime & bundle verification
- final BGE LoRA training on all 7,000 queries
- public candidate reranking
- submission validation & packaging
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.bundle.verifier import verify_production_bundle
from src.core.memory import check_memory_guard, release_memory, take_memory_snapshot, format_memory_report
from src.data.canonical import verify_canonical_dataset
from src.production.final_train import train_final_adapter
from src.production.public_rerank import rerank_and_fuse_public_predictions
from src.production.submission import package_submission, validate_submission


def main():
    parser = argparse.ArgumentParser(description="Kaggle Final Production Runner.")
    parser.add_argument("--dataset-dir", type=str, default="data/task1_canonical_v2")
    parser.add_argument("--bundle-dir", type=str, default="artifacts/bundle/production")
    parser.add_argument("--output-dir", type=str, default="artifacts/submission")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode for testing")
    args = parser.parse_args()

    print("[*] Stage K0: Verifying environment and runtime ...")
    snap = take_memory_snapshot()
    print(format_memory_report(snap, stage="K0 Environment Pre-flight"))

    dataset_p = Path(args.dataset_dir)
    bundle_p = Path(args.bundle_dir)
    out_dir = Path(args.output_dir)

    print("[*] Stage K1: Verifying canonical dataset identity ...")
    is_valid, ident, ds_errors = verify_canonical_dataset(dataset_p)
    if not is_valid:
        print(f"[!] Dataset verification FAILED: {ds_errors}")
        sys.exit(1)

    print("[*] Stage K2: Verifying immutable production bundle ...")
    b_valid, b_errors = verify_production_bundle(bundle_p)
    if not b_valid:
        print(f"[!] Bundle verification FAILED: {b_errors}")
        sys.exit(1)

    lock_p = bundle_p / "production_lock.json"
    reranker_cfg = {}
    if lock_p.is_file():
        try:
            with open(lock_p, "r", encoding="utf-8") as f:
                lock_data = json.load(f)
            reranker_cfg = lock_data.get("config", {}).get("reranker", {})
        except Exception as e:
            print(f"[!] Warning reading production lock reranker config: {e}")
            if not args.mock:
                sys.exit(1)
    elif not args.mock:
        print(f"[!] Missing required production lock at {lock_p}")
        sys.exit(1)

    print("[*] Stage K3 & K4: Training final BGE+LoRA on all queries ...")
    pairs_p = bundle_p / "final_training_pairs.parquet"
    adapter_out = out_dir / "final_adapter"
    train_report = train_final_adapter(pairs_p, adapter_out, runtime_config=reranker_cfg, mock_run=args.mock)
    if train_report.get("status") != "PASS":
        print("[!] Final training FAILED.")
        sys.exit(1)

    print("[*] Stage K6 & K7: Reranking public candidates and fusing top-5 ...")
    cands_p = bundle_p / "public_candidates.parquet"
    lock_p = bundle_p / "production_lock.json"
    evidence_p = bundle_p / "public_evidence.parquet"

    with open(dataset_p / "public-official.json", "r", encoding="utf-8") as f:
        public_dict = json.load(f)
    expected_qids = set(str(k) for k in public_dict.keys())

    predictions = rerank_and_fuse_public_predictions(
        public_candidates_path=cands_p,
        production_lock_path=lock_p,
        adapter_dir=adapter_out,
        public_evidence_path=evidence_p if evidence_p.is_file() else None,
        top_k=5,
        public_queries_dict=public_dict,
    )

    print("[*] Stage K8: Validating submission format ...")

    sub_valid, sub_errors = validate_submission(predictions, expected_qids=expected_qids, max_predictions=5)
    if not sub_valid:
        print(f"[!] Submission validation FAILED: {sub_errors}")
        sys.exit(1)

    print("[*] Stage K9: Packaging final submission.zip ...")
    json_path, zip_path = package_submission(predictions, out_dir=out_dir)
    print(f"[+] Submission successfully generated and validated at {zip_path}")
    sys.exit(0)


if __name__ == "__main__":
    main()
