#!/usr/bin/env python3
"""
LegalIR Google Colab Training Runner (Compatibility Wrapper).
Delegates to scripts/gates/run_a100.py for production A100 runs
or scripts/gates/run_colab_t4.py for T4 single-GPU runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.gates.run_a100 import run_a100_production_gate
from scripts.gates.run_colab_t4 import run_colab_t4_gate


def run_colab_production_training(
    dataset_dir: Path | str,
    output_dir: Path | str,
    smoke_report_path: Path | None = None,
    expected_sha: str = "",
    precision: str = "bf16",
    allow_non_a100: bool = False,
    mock: bool = False,
    hf_repo: str | None = None,
    hf_token: str | None = None,
    run_mode: str = "full",
    freeze_file_path: Path | str | None = None,
    algorithm_config_path: Path | str | None = None,
    runtime_profile_path: Path | str | None = None,
    hf_allow_public_repo: bool = False,
) -> dict:
    """Execute Colab run delegating to the appropriate authoritative gate."""
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)

    if run_mode == "smoke":
        # Route to Colab T4 gate
        k_rep = smoke_report_path or (output_dir / "kaggle_t4x2_report.json")
        if not Path(k_rep).is_file():
            # Create a mock/passing fixture for self-contained smoke testing if not present
            k_rep = REPO_ROOT / "artifacts" / "task1" / "gates" / "kaggle_t4x2_report.json"
        return run_colab_t4_gate(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            kaggle_report_path=k_rep if Path(k_rep).is_file() else (output_dir / "kaggle_t4x2_report.json"),
            mock=mock or allow_non_a100,
        )
    else:
        k_cands = [
            Path(smoke_report_path) if smoke_report_path else None,
            Path("/content/kaggle_t4x2_report.json"),
            Path("/content/LegalIR/artifacts/task1/gates/kaggle_t4x2_report.json"),
            REPO_ROOT / "artifacts" / "task1" / "gates" / "kaggle_t4x2_report.json",
        ]
        k_rep = Path(smoke_report_path) if smoke_report_path else next((p for p in k_cands if p and p.is_file()), k_cands[-1])

        if mock:
            output_dir.mkdir(parents=True, exist_ok=True)
            if not Path(k_rep).is_file():
                k_rep = output_dir / "kaggle_t4x2_report.json"
                k_rep.write_text(json.dumps({"stage": "KAGGLE_T4X2", "verdict": "PASS", "git_sha": "718efb7ba4565fa5b863f05927122484f8e58c2f"}), encoding="utf-8")
        # Normalize precision: accept bfloat16/bf16, default bf16 for A100
        prec_norm = str(precision or "bf16").lower().strip()
        if prec_norm in ("bfloat16", "bf16"):
            prec_norm = "bf16"
        try:
            return run_a100_production_gate(
                dataset_dir=dataset_dir,
                output_dir=output_dir,
                expected_sha=expected_sha,
                algorithm_config_path=algorithm_config_path or (REPO_ROOT / "configs" / "algorithm" / "legalir_v2.yaml"),
                runtime_profile_path=runtime_profile_path or (REPO_ROOT / "configs" / "runtime" / "colab_a100.yaml"),
                kaggle_report_path=k_rep,
                freeze_file_path=freeze_file_path or (REPO_ROOT / "artifacts" / "task1" / "freeze" / "production_freeze.json"),
                allow_non_a100=allow_non_a100,
                mock=mock,
                hf_repo=hf_repo,
                hf_token=hf_token,
                precision=prec_norm,
                hf_allow_public_repo=hf_allow_public_repo,
            )
        finally:
            if not mock and output_dir.is_dir():
                from scripts.colab.artifacts import build_recovery_archive
                try:
                    build_recovery_archive(output_dir)
                except Exception as exc:
                    print(f"[!] Recovery archive unavailable ({type(exc).__name__}); individual artifacts remain at {output_dir}.", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description="LegalIR Colab Training Runner (Wrapper)")
    parser.add_argument("--dataset-dir", type=str, default="/content/kaggle_dataset")
    parser.add_argument("--output-dir", type=str, default="artifacts/task1/production")
    parser.add_argument("--smoke-report", type=str, default=None)
    parser.add_argument("--expected-sha", type=str, default="")
    parser.add_argument("--precision", type=str, default="bf16")
    parser.add_argument("--allow-non-a100", action="store_true")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--hf-repo", type=str, default="dangphuc2109/legalir-task1-reranker")
    parser.add_argument("--freeze-file", type=str, default=None)
    parser.add_argument("--mode", type=str, default="full", choices=["full", "smoke"])
    parser.add_argument("--hf-allow-public-repo", action="store_true", help="Operator override: allow push to existing PUBLIC HF repo")
    args = parser.parse_args()

    try:
        run_colab_production_training(
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            smoke_report_path=Path(args.smoke_report) if args.smoke_report else None,
            expected_sha=args.expected_sha,
            precision=args.precision,
            allow_non_a100=args.allow_non_a100,
            mock=args.mock,
            hf_repo=args.hf_repo,
            freeze_file_path=Path(args.freeze_file) if args.freeze_file else None,
            run_mode=args.mode,
            hf_allow_public_repo=args.hf_allow_public_repo,
        )
        return 0
    except Exception as exc:
        print(f"[!] FAILED: Colab Training Runner: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
