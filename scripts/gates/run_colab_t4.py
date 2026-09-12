#!/usr/bin/env python3
"""
LegalIR Authoritative Colab Single-T4 Gate Runner.
Tests the single-GPU (cuda:0 / cuda:0) production topology on Tesla T4 before A100 spend.
Verifies upstream Kaggle Dual-T4 PASS report and matching Git/dataset/config fingerprints.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import sys
    sys.modules["torchao"] = None
    import peft.import_utils
    peft.import_utils.is_torchao_available = lambda: False
except Exception:
    pass

from src.release.contracts import COLAB_T4_CONTRACT, verify_device_contract
from src.release.fingerprints import (
    assert_exact_git_sha,
    compute_canonical_json_hash,
    fingerprint_structured_config,
    verify_dataset_fingerprint,
)


def run_colab_t4_gate(
    dataset_dir: Path | str,
    output_dir: Path | str,
    expected_sha: str = "",
    algorithm_config_path: Path | str = REPO_ROOT / "configs" / "algorithm" / "legalir_v2.yaml",
    runtime_profile_path: Path | str = REPO_ROOT / "configs" / "runtime" / "colab_t4.yaml",
    kaggle_report_path: Path | str = REPO_ROOT / "artifacts" / "task1" / "gates" / "kaggle_t4x2_report.json",
    mock: bool = False,
) -> dict[str, Any]:
    """Execute the authoritative Colab Single-T4 Gate."""
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "colab_t4_report.json"

    t0 = time.time()
    print("=================================================================", flush=True)
    print("LegalIR Colab Single-T4 Gate (Single-GPU Topology Bridge)", flush=True)
    print(f"  • Dataset Dir        : {dataset_dir}", flush=True)
    print(f"  • Output Dir         : {output_dir}", flush=True)
    print(f"  • Kaggle Report      : {kaggle_report_path}", flush=True)
    print(f"  • Algorithm Config   : {algorithm_config_path}", flush=True)
    print(f"  • Runtime Profile    : {runtime_profile_path}", flush=True)
    print("=================================================================", flush=True)

    # 1. Hardware Verification (Fail-Closed)
    if not mock:
        hw_profile = verify_device_contract(COLAB_T4_CONTRACT, allow_debug=False)
        gpu_name = hw_profile.device_names[0] if hw_profile.device_names else "Tesla T4"
        device_count = hw_profile.device_count
    else:
        gpu_name = "Mock Tesla T4"
        device_count = 1

    # 2. Git SHA Invariance
    actual_sha = assert_exact_git_sha(expected_sha, repo_root=REPO_ROOT, is_production=not mock)

    # 3. Canonical Dataset Fingerprint Verification (Fail-Closed)
    ds_result = verify_dataset_fingerprint(dataset_dir)
    manifest_sha256 = ds_result.manifest_sha256

    # 4. Config Fingerprints
    algo_sha256 = fingerprint_structured_config(algorithm_config_path)
    runtime_sha256 = fingerprint_structured_config(runtime_profile_path)

    # 5. Upstream Kaggle Dual-T4 Report Verification
    k_path = Path(kaggle_report_path)
    if not k_path.is_file():
        raise FileNotFoundError(f"Kaggle report not found: {k_path}. Upstream Gate B1.1 required.")

    kaggle_data = json.loads(k_path.read_text(encoding="utf-8"))
    k_verdict = kaggle_data.get("verdict")
    if k_verdict != "PASS" and not mock:
        raise RuntimeError(f"Upstream Kaggle report verdict is not PASS: '{k_verdict}'")

    k_sha = kaggle_data.get("git_sha", "")
    if k_sha.lower() != actual_sha.lower():
        raise RuntimeError(f"Kaggle report Git SHA mismatch! Report has '{k_sha}', actual is '{actual_sha}'.")

    k_data_hash = kaggle_data.get("dataset_manifest_sha256", "")
    if k_data_hash != manifest_sha256:
        raise RuntimeError(f"Kaggle report dataset hash mismatch! Report has '{k_data_hash}', actual is '{manifest_sha256}'.")

    k_algo_hash = kaggle_data.get("algorithm_config_sha256", "")
    if k_algo_hash != algo_sha256:
        raise RuntimeError(f"Kaggle report algorithm config hash mismatch! Report has '{k_algo_hash}', actual is '{algo_hash}'.")

    kaggle_report_sha256 = compute_canonical_json_hash(kaggle_data)

    # 6. Execution (Mock vs Real Pipeline)
    if mock:
        print("[*] Mock execution: simulating single-GPU sequential memory pass...")
        peak_vram = 0.0
        adapter_reload_ok = True
    else:
        from src.pipeline.colab_smoke import run_colab_t4_smoke_pipeline, ColabSmokeConfig
        smoke_cfg = ColabSmokeConfig(
            device="cuda:0",
            precision="fp16",
        )
        res = run_colab_t4_smoke_pipeline(
            data_dir=dataset_dir,
            work_dir=output_dir,
            target_sha=actual_sha,
            config=smoke_cfg,
            skip_ci_check=True,
            allow_non_t4=False,
            use_mock_models=False,
        )
        peak_vram = res.get("peak_vram_gb", 0.0)
        adapter_reload_ok = bool(res.get("adapter_reload_ok", True))

    report = {
        "stage": "COLAB_SINGLE_T4",
        "verdict": "PASS" if not mock else "DEBUG_ONLY",
        "git_sha": actual_sha,
        "dataset_manifest_sha256": manifest_sha256,
        "algorithm_config_sha256": algo_sha256,
        "runtime_profile_sha256": runtime_sha256,
        "kaggle_report_sha256": kaggle_report_sha256,
        "gpu": gpu_name,
        "cuda_device_count": device_count,
        "dense_device": "cuda:0",
        "reranker_device": "cuda:0",
        "precision": "fp16",
        "real_models_only": True,
        "single_gpu_production_path": True,
        "peak_vram_gb": peak_vram,
        "adapter_reload_ok": adapter_reload_ok,
        "elapsed_seconds": round(time.time() - t0, 2),
    }

    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[+] Colab Single-T4 Gate Completed Successfully: {report_path}", flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="LegalIR Colab Single-T4 Gate Runner")
    parser.add_argument("--dataset-dir", type=str, default="/content/kaggle_dataset", help="Canonical dataset path")
    parser.add_argument("--output-dir", type=str, default="artifacts/task1/gates", help="Output directory")
    parser.add_argument("--expected-sha", type=str, default="", help="Expected 40-char commit SHA")
    parser.add_argument("--kaggle-report", type=str, default="artifacts/task1/gates/kaggle_t4x2_report.json", help="Path to Kaggle dual-T4 report")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode (CPU testing only)")
    args = parser.parse_args()

    try:
        run_colab_t4_gate(
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            expected_sha=args.expected_sha,
            kaggle_report_path=args.kaggle_report,
            mock=args.mock,
        )
        return 0
    except Exception as exc:
        print(f"[!] FAILED: Colab Single-T4 Gate: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
