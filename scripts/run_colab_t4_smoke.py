#!/usr/bin/env python3
"""
CLI runner for the Colab Single-T4 Contract Smoke Gate.

Authoritative specification: LEGALIR_CI_COLAB_KAGGLE_ARCHITECTURE_SPEC.md
Implementation plan: LEGALIR_CI_COLAB_KAGGLE_IMPLEMENTATION_PLAN.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.colab_smoke import ColabSmokeConfig, run_colab_t4_smoke_pipeline

DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "colab_smoke.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Execute LegalIR Colab Single-T4 GPU Contract Smoke Gate."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to official canonical v2 dataset (e.g. /content/drive/MyDrive/legalir-task1-clean-data)",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        required=True,
        help="Output working directory (e.g. /content/drive/MyDrive/legalir-smoke-runs/run_xxx)",
    )
    parser.add_argument(
        "--target-sha",
        type=str,
        default="",
        help="Target Git commit SHA (40 hex characters) to verify against CI and repo HEAD",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to colab_smoke.yaml configuration file",
    )
    parser.add_argument(
        "--allow-non-t4",
        action="store_true",
        help="Allow running on non-Tesla T4 hardware (marks result as NOT_A_T4_READINESS_GATE)",
    )
    parser.add_argument(
        "--skip-ci-check",
        action="store_true",
        help="Skip GitHub REST API CI verification (for offline / local debugging)",
    )

    args = parser.parse_args()

    if args.config.exists():
        cfg = ColabSmokeConfig.from_yaml(args.config)
    else:
        cfg = ColabSmokeConfig()

    if args.target_sha:
        cfg.target_sha = args.target_sha

    print("=================================================================")
    print("LegalIR Colab Single-T4 Contract Smoke Runner")
    print(f"  • Data Dir      : {args.data_dir}")
    print(f"  • Work Dir      : {args.work_dir}")
    print(f"  • Target SHA    : {args.target_sha or 'Not Specified'}")
    print(f"  • Allow Non-T4  : {args.allow_non_t4}")
    print(f"  • Skip CI Check : {args.skip_ci_check}")
    print("=================================================================")

    try:
        report = run_colab_t4_smoke_pipeline(
            data_dir=args.data_dir,
            work_dir=args.work_dir,
            target_sha=args.target_sha,
            config=cfg,
            skip_ci_check=args.skip_ci_check,
            allow_non_t4=args.allow_non_t4,
            use_mock_models=False,
        )
    except Exception as exc:
        print(f"\n[-] COLAB SMOKE PIPELINE ERROR: {exc}", file=sys.stderr)
        return 1

    verdict = report.get("result", "FAIL")
    print("\n=================================================================")
    print("COLAB T4 SMOKE EXECUTION SUMMARY:")
    print(f"  • Result Verdict        : {verdict}")
    print(f"  • GPU Model             : {report.get('gpu_name')}")
    print(f"  • CI Verified Green     : {report.get('ci_green')}")
    print(f"  • Dense Peak VRAM       : {report.get('dense_peak_vram_mb')} MB")
    print(f"  • Reranker Peak VRAM    : {report.get('reranker_peak_vram_mb')} MB")
    print(f"  • Optimizer Steps       : {report.get('optimizer_steps')}")
    print(f"  • Weight Param Diff     : {report.get('param_diff'):.6f}")
    learned_p = report.get("total_learned_params") or report.get("parameter_audit", {}).get("system_learned_parameters", 0)
    print(f"  • Total Learned Params  : {learned_p:,}")
    print(f"  • Prediction Validation : {report.get('prediction_validation', {}).get('valid')}")
    print(f"  • Report Path           : {args.work_dir / 'colab_smoke_report.json'}")
    print("=================================================================")

    if verdict == "PASS":
        print("[+] SUCCESS: Colab T4 Smoke PASSED. Commit is approved for Kaggle FULL run.")
        return 0
    elif verdict == "NOT_A_T4_READINESS_GATE":
        print("[!] NOTICE: Execution completed on non-T4 GPU (NOT_A_T4_READINESS_GATE).")
        return 0
    else:
        print("[-] FAILURE: Colab smoke gate failed readiness criteria.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
