#!/usr/bin/env python3
"""CLI script to verify Task 1 canonical v2 dataset identity and completeness."""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.canonical import verify_canonical_dataset


def main():
    parser = argparse.ArgumentParser(description="Verify Task 1 canonical dataset.")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="kaggle_dataset",
        help="Path to canonical dataset directory",
    )
    parser.add_argument(
        "--output-manifest",
        type=str,
        default="artifacts/factory/preflight.json",
        help="Path to save preflight verification manifest",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset_dir)
    print(f"[*] Verifying canonical dataset at {dataset_path} ...")
    is_valid, identity, errors = verify_canonical_dataset(dataset_path)

    manifest = {
        "dataset_name": identity.dataset_name,
        "dataset_version": identity.version,
        "schema_version": identity.schema_version,
        "status": "PASS" if is_valid else "FAIL",
        "counts": {
            "num_docs": identity.num_docs,
            "num_chunks": identity.num_chunks,
            "num_micro": identity.num_micro,
            "num_macro": identity.num_macro,
            "num_train_queries": identity.num_train_queries,
            "num_qrels": identity.num_qrels,
            "num_public_queries": identity.num_public_queries,
            "num_duplicate_groups": identity.num_duplicate_groups,
        },
        "errors": errors,
    }

    out_p = Path(args.output_manifest)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    if not is_valid:
        print("[!] Canonical dataset verification FAILED:")
        for err in errors:
            print(f"    - {err}")
        sys.exit(1)

    print("[+] Canonical dataset verification PASSED.")
    print(f"    - Docs: {identity.num_docs}")
    print(f"    - Chunks: {identity.num_chunks} (Micro: {identity.num_micro}, Macro: {identity.num_macro})")
    print(f"    - Train Queries: {identity.num_train_queries}, Qrels: {identity.num_qrels}")
    print(f"    - Public Queries: {identity.num_public_queries}")
    print(f"    - Duplicate Groups: {identity.num_duplicate_groups}")
    sys.exit(0)


if __name__ == "__main__":
    main()
