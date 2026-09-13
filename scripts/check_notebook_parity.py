#!/usr/bin/env python3
"""
Verify exact parity and zero drift across LegalIR competition notebooks.
Checks that notebooks on disk match generator output.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

CANONICAL_NOTEBOOKS: list[str] = [
    "notebooks/kaggle_t4x2_smoke.ipynb",
    "notebooks/colab_t4_smoke.ipynb",
    "notebooks/colab_a100_train.ipynb",
]


def compute_sha256(path: Path) -> str:
    """Compute SHA-256 digest of a file."""
    if not path.is_file():
        raise FileNotFoundError(f"Notebook file not found: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_notebook_parity(
    root_path: Path, kaggle_path: Path
) -> tuple[bool, str, str]:
    """Check parity between two specific notebook files."""
    sha_root = compute_sha256(root_path)
    sha_kaggle = compute_sha256(kaggle_path)
    return sha_root == sha_kaggle, sha_root, sha_kaggle


def check_all_notebook_parity(repo_root: Path = REPO_ROOT) -> tuple[bool, dict[str, str]]:
    """Check that notebooks exist and match generator output."""
    gen_script = repo_root / "scripts" / "generate_notebooks.py"
    res = subprocess.run([sys.executable, str(gen_script), "--check-drift"], capture_output=True, text=True)
    hashes = {}
    for rel in CANONICAL_NOTEBOOKS:
        nb_p = repo_root / rel
        hashes[rel] = compute_sha256(nb_p) if nb_p.is_file() else "MISSING"
    return res.returncode == 0, hashes



def main() -> int:
    parser = argparse.ArgumentParser(description="Check parity and zero-drift across LegalIR notebooks.")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT, help="Repository root path")
    args = parser.parse_args()

    gen_script = args.repo_root / "scripts" / "generate_notebooks.py"
    if not gen_script.is_file():
        print(f"[-] Generator script missing: {gen_script}", file=sys.stderr)
        return 1

    res = subprocess.run([sys.executable, str(gen_script), "--check-drift"], capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print(res.stderr, file=sys.stderr)
    return res.returncode


if __name__ == "__main__":
    sys.exit(main())

