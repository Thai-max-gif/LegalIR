#!/usr/bin/env python3
"""
Verify exact parity and zero drift across LegalIR competition notebooks.
Checks that notebooks on disk match generator output.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


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

