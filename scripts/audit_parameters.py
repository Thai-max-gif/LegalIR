"""Audit total learned parameter count across all models used in the LegalIR system.

Enforces strict competition rule: Total learned parameters must be < 4,000,000,000 (< 4B).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.parameter_audit import (
    KNOWN_PARAM_COUNTS,
    MAX_PARAMETER_BUDGET,
    ParameterBudgetExceededError,
    audit_system_parameters,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="LegalIR Learned Parameter Budget Auditor (<4B)")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pipeline.yaml",
        help="Path to pipeline/kaggle config file to extract models from",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Explicit list of model names or paths to audit",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="parameter_audit.json",
        help="Output path for parameter_audit.json",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Audit without rewriting output JSON file (preserves git clean state)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=True,
        help="Fail hard with exit code 1 if parameter budget is exceeded",
    )
    args = parser.parse_args()

    print("=" * 65)
    print("LegalIR Task 1: Learned Parameter Budget Audit (<4B Rule)")
    print("=" * 65)

    out_json = None if args.check_only else args.output_json
    try:
        report = audit_system_parameters(
            models=args.models,
            config_path=args.config,
            output_json=out_json,
            raise_on_violation=False,
            offline_fallback=True,
        )
    except Exception as e:
        print(f"ERROR during parameter audit: {e}", file=sys.stderr)
        sys.exit(1)

    print("\nAudited Models:")
    print("-" * 65)
    for model_name, details in report["models"].items():
        params = details["parameters"]
        role = details.get("role", "model")
        is_lora = details.get("is_peft_lora", False)
        extra = " [LoRA Adapter]" if is_lora else ""
        print(f"  • {model_name:<45} : {params:>12,} ({params / 1e9:>7.4f}B) [{role}]{extra}")

    total_params = report["total_learned_parameters"]
    total_b = report["total_parameters_billions"]
    budget_limit = report["budget_limit"]
    utilization = report["budget_utilization_pct"]
    headroom = report["headroom_parameters"]
    verdict = report["verdict"]

    print("\n" + "=" * 65)
    print(f"TOTAL LEARNED PARAMETERS : {total_params:>15,} ({total_b:.4f}B)")
    print(f"COMPETITION BUDGET LIMIT : {budget_limit:>15,} (4.0000B)")
    print(f"BUDGET UTILIZATION       : {utilization:>15.2f}%")
    print(f"REMAINING HEADROOM       : {headroom:>15,} ({headroom / 1e9:.4f}B)")
    print(f"COMPLIANCE VERDICT       : {verdict:>15} ({total_b:.4f}B < 4.0B)")
    print("=" * 65)

    if args.output_json:
        print(f"\nParameter audit report exported to: {args.output_json}")

    if not report["is_compliant"]:
        print(
            f"\nCRITICAL ERROR: Total learned parameters ({total_params:,}) exceed "
            f"the strict competition budget limit ({budget_limit:,})!",
            file=sys.stderr,
        )
        if args.strict:
            sys.exit(1)
    else:
        print("\nCompliance Check: PASSED - System satisfies all competition parameter rules.")
        sys.exit(0)


if __name__ == "__main__":
    main()
