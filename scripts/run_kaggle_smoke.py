#!/usr/bin/env python3
"""
LegalIR Kaggle Smoke Gate Runner (Compatibility Wrapper).
Delegates directly to the authoritative scripts/gates/run_kaggle_t4x2.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import shutil

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.gates.run_kaggle_t4x2 import run_kaggle_t4x2_gate


def run_kaggle_smoke(
    dataset_dir: Path | str,
    output_dir: Path | str,
    target_sha: str = "",
    mock: bool = False,
) -> dict:
    """Backward-compatible entrypoint forwarding to authoritative Kaggle T4x2 gate."""
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    res = run_kaggle_t4x2_gate(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        expected_sha=target_sha,
        mock=mock,
    )
    # Mirror report to legacy name for backward compatibility
    legacy_report = output_dir / "kaggle_smoke_report.json"
    legacy_data = dict(res)
    legacy_data["stage"] = "B1.1_KAGGLE_T4_SMOKE"
    legacy_report.write_text(json.dumps(legacy_data, indent=2, sort_keys=True), encoding="utf-8")
    return legacy_data


def main() -> int:
    parser = argparse.ArgumentParser(description="LegalIR Kaggle Dual-T4 Smoke Gate (Wrapper)")
    parser.add_argument("--dataset-dir", type=str, default="/kaggle/input/legalir-task1-clean-data", help="Path to canonical dataset")
    parser.add_argument("--output-dir", type=str, default="artifacts/task1/gates", help="Output directory")
    parser.add_argument("--target-sha", type=str, default="", help="Expected git commit SHA")
    parser.add_argument("--mock", action="store_true", help="Run mock execution for CPU testing")
    args = parser.parse_args()

    try:
        run_kaggle_smoke(
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            target_sha=args.target_sha,
            mock=args.mock,
        )
        return 0
    except Exception as exc:
        print(f"[!] FAILED: Kaggle Smoke Gate: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
