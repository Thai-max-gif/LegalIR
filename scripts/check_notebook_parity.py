#!/usr/bin/env python3
"""
Verify exact byte-for-byte SHA-256 parity across all distributed competition notebooks.

Authoritative specification: LEGALIR_88E1_ARCHITECTURE_REPAIR.md
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DISTRIBUTED_NOTEBOOK_PATHS: list[str] = [
    "legalir_training.ipynb",
    "kaggle_kernel_task1/legalir_training.ipynb",
    "kaggle_kernel/legalir_training.ipynb",
    "kaggle_kernel/legalqa_gpu_pipeline.ipynb",
    "notebooks/kaggle_final.ipynb",
]


def compute_sha256(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Notebook file not found: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_notebook_parity(
    root_path: Path, kaggle_path: Path
) -> tuple[bool, str, str]:
    sha_root = compute_sha256(root_path)
    sha_kaggle = compute_sha256(kaggle_path)
    return sha_root == sha_kaggle, sha_root, sha_kaggle


def check_all_notebook_parity(repo_root: Path = REPO_ROOT) -> tuple[bool, dict[str, str]]:
    """Check SHA-256 parity across all distributed Kaggle competition notebooks."""
    hashes: dict[str, str] = {}
    missing: list[str] = []

    for rel in DISTRIBUTED_NOTEBOOK_PATHS:
        nb_p = repo_root / rel
        if not nb_p.is_file():
            missing.append(rel)
        else:
            hashes[rel] = compute_sha256(nb_p)

    if missing:
        return False, {m: "MISSING" for m in missing}

    unique_hashes = set(hashes.values())
    is_valid = len(unique_hashes) == 1
    return is_valid, hashes


def main() -> int:
    parser = argparse.ArgumentParser(description="Check SHA-256 parity across all distributed LegalIR notebooks.")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT, help="Repository root path")
    args = parser.parse_args()

    is_valid, hashes = check_all_notebook_parity(args.repo_root)

    print("=================================================================")
    print("LegalIR All-Notebooks Parity Check (SHA-256 Verification)")
    print("=================================================================")
    for rel_path, digest in hashes.items():
        print(f"  • {rel_path:<45}: {digest}")
    print("=================================================================")

    if is_valid:
        print("[+] SUCCESS: All distributed competition notebooks are identical byte-for-byte.")
        return 0
    else:
        print("[-] FAILURE: Notebook SHA-256 mismatch detected!", file=sys.stderr)
        print("    Run `python scripts/generate_kaggle_notebook.py` to synchronize parity.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
