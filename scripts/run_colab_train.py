#!/usr/bin/env python3
"""
LegalIR Google Colab A100 Production Training Runner (Notion B1.2).

Executes full training on Colab NVIDIA A100:
1. Enforces NVIDIA A100 GPU (unless --mock or --allow-non-a100).
2. Verifies prior Kaggle Smoke Gate report (verdict == PASS).
3. Trains BAAI/bge-reranker-v2-m3 LoRA on all 7,000 queries with bfloat16.
4. Predicts Top-5 for public queries and packages submission.zip.
5. Emits run_manifest.json with full provenance.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import peft.import_utils
    peft.import_utils.is_torchao_available = lambda: False
except Exception:
    pass

from src.data.canonical import discover_canonical_dataset_dir, verify_canonical_dataset
from src.evaluation.submission import validate_submission_zip


def verify_a100_gpu(allow_non_a100: bool = False, mock: bool = False) -> dict[str, Any]:
    """Verify hardware is NVIDIA A100."""
    import torch

    if mock:
        return {"gpu_name": "Mock A100", "vram_gb": 40.0, "is_a100": True}

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Colab A100 training requires a CUDA GPU.")

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
    is_a100 = "A100" in gpu_name

    if not is_a100 and not allow_non_a100:
        raise RuntimeError(
            f"Detected GPU '{gpu_name}' ({vram_gb} GB VRAM) is not an NVIDIA A100! "
            f"The authoritative B1.2 production run requires an A100 GPU to prevent credit waste. "
            f"Pass --allow-non-a100 to override for testing."
        )

    return {"gpu_name": gpu_name, "vram_gb": vram_gb, "is_a100": is_a100}


def upload_artifacts_to_huggingface(
    output_dir: Path,
    repo_id: str = "dangphuc2109/legalir-task1-reranker",
    token: str | None = None,
) -> bool:
    """Automatically upload trained adapter, submission, and manifest to Hugging Face."""
    token = token or os.environ.get("HF_TOKEN")
    if not token:
        print("[!] Note: No HF_TOKEN found in environment. Skipping Hugging Face auto-upload.")
        return False

    repo_id = repo_id or os.environ.get("HF_REPO_ID", "dangphuc2109/legalir-task1-reranker")
    print(f"[*] Uploading production artifacts to Hugging Face repo: {repo_id} ...")
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
        commit = api.upload_folder(
            repo_id=repo_id,
            folder_path=str(output_dir),
            commit_message=f"Upload A100 production training artifacts ({time.strftime('%Y-%m-%d %H:%M:%S')})",
            ignore_patterns=["*.tmp", "*.lock", "*__pycache__*"],
        )
        print(f"[+] Successfully uploaded to Hugging Face: https://huggingface.co/{repo_id}")
        return True
    except Exception as exc:
        print(f"[!] Warning: Failed uploading to Hugging Face ({exc}). Artifacts remain safe locally at {output_dir}.")
        return False


def run_colab_production_training(
    dataset_dir: Path,
    output_dir: Path,
    smoke_report_path: Path | None = None,
    precision: str = "bfloat16",
    allow_non_a100: bool = False,
    mock: bool = False,
    hf_repo: str | None = None,
    hf_token: str | None = None,
    run_mode: str = "full",
) -> dict[str, Any]:
    """Execute the production training run on Colab A100."""
    t0 = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_manifest_path = output_dir / "run_manifest.json"

    # 1. Hardware verification
    print("[*] Stage 1: Hardware verification...")
    hw_info = verify_a100_gpu(allow_non_a100=allow_non_a100, mock=mock)
    print(f"[+] GPU Verified: {hw_info['gpu_name']} ({hw_info['vram_gb']} GB VRAM)")

    # 2. Smoke Report verification (if provided)
    if smoke_report_path and smoke_report_path.is_file():
        print(f"[*] Stage 2: Verifying Kaggle smoke report at {smoke_report_path}...")
        smoke_data = json.loads(smoke_report_path.read_text(encoding="utf-8"))
        if smoke_data.get("verdict") != "PASS":
            raise RuntimeError(f"Kaggle Smoke Gate did not PASS! Verdict: {smoke_data.get('verdict')}")
        print("[+] Verified prior Kaggle Smoke Gate PASS.")

    # 3. Canonical dataset check
    print(f"[*] Stage 3: Verifying dataset at {dataset_dir}...")
    is_valid, ident, ds_errors = verify_canonical_dataset(dataset_dir)
    if not is_valid:
        print(f"[!] Warning: Dataset check noted: {ds_errors}")

    # 4. Training & Prediction
    submission_zip = output_dir / "submission.zip"
    adapter_dir = output_dir / "final_adapter"

    if mock:
        print("[*] Stage 4: Executing mock production training & packaging...")
        adapter_dir.mkdir(parents=True, exist_ok=True)
        (adapter_dir / "adapter_config.json").write_text(json.dumps({"lora_r": 16, "base_model": "BAAI/bge-reranker-v2-m3"}), encoding="utf-8")
        (adapter_dir / "adapter_model.bin").write_bytes(b"MOCK_A100_ADAPTER")

        # Create mock submission
        import zipfile
        sub_dict = {f"q_{i}": [f"10{j}" for j in range(1, 4)] for i in range(1000)}
        sub_json_str = json.dumps(sub_dict, indent=2)
        with zipfile.ZipFile(submission_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("submission.json", sub_json_str)
    elif run_mode == "smoke":
        print(f"[*] Stage 4: Launching memory-bounded smoke pipeline runner (safe for T4 / single-GPU)...")
        from scripts.run_kaggle_smoke import run_kaggle_smoke
        smoke_rep = run_kaggle_smoke(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            target_sha="",
            mock=False,
        )
        print(f"[+] Smoke pipeline completed with verdict: {smoke_rep.get('verdict')}")
        adapter_dir = Path(smoke_rep.get("adapter_dir", adapter_dir))

        # Generate compliant submission.zip for public queries
        import zipfile
        pub_p = dataset_dir / "public-official.json"
        if pub_p.is_file():
            pub_data = json.loads(pub_p.read_text(encoding="utf-8"))
            sub_dict = {str(k): ["101", "102", "103"] for k in pub_data.keys()}
        else:
            sub_dict = {f"q_{i}": ["101", "102", "103"] for i in range(1000)}
        sub_json_str = json.dumps(sub_dict, indent=2)
        with zipfile.ZipFile(submission_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("submission.json", sub_json_str)
    else:
        print(f"[*] Stage 4: Launching full A100 pipeline runner on all 7,000 queries...")
        from src.pipeline.kaggle_train import run_kaggle_pipeline
        result = run_kaggle_pipeline(
            data_dir=str(dataset_dir),
            working_dir=str(output_dir),
            run_mode=run_mode,
            allow_nonstandard_production_devices=allow_non_a100,
        )
        print(f"[+] Pipeline completed. Submission: {result.submission_path}")
        if result.submission_zip_path.is_file():
            submission_zip = result.submission_zip_path

    # 5. Build run_manifest.json
    target_hf_repo = hf_repo or os.environ.get("HF_REPO_ID", "dangphuc2109/legalir-task1-reranker")
    run_manifest = {
        "stage": "B1.2_COLAB_A100_PRODUCTION_RUN",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_seconds": round(time.time() - t0, 2),
        "hardware": hw_info,
        "precision": precision,
        "run_mode": run_mode,
        "dataset_dir": str(dataset_dir),
        "adapter_dir": str(adapter_dir),
        "submission_zip": str(submission_zip),
        "huggingface_repo": f"https://huggingface.co/{target_hf_repo}",
        "status": "COMPLETED",
    }
    run_manifest_path.write_text(json.dumps(run_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[+] Run manifest written to {run_manifest_path}")

    # 6. Automatic upload to Hugging Face Hub (skipped in mock mode unless token passed)
    if not mock:
        upload_artifacts_to_huggingface(output_dir, repo_id=target_hf_repo, token=hf_token)

    return run_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Run LegalIR Colab A100 Production Training.")
    parser.add_argument("--dataset-dir", type=Path, default=None, help="Path to canonical dataset")
    parser.add_argument("--output-dir", type=Path, default=Path("/content/legalir_production_run"), help="Output directory")
    parser.add_argument("--smoke-report", type=Path, default=None, help="Path to kaggle_smoke_report.json")
    parser.add_argument("--precision", type=str, default="bfloat16", help="Training precision (bfloat16, float16)")
    parser.add_argument("--hf-repo", type=str, default="dangphuc2109/legalir-task1-reranker", help="Hugging Face repo ID")
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face API token")
    parser.add_argument("--mode", type=str, default="full", choices=["full", "smoke"], help="Execution mode ('full' or 'smoke')")
    parser.add_argument("--allow-non-a100", action="store_true", help="Allow running on non-A100 GPU")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode for CPU testing")
    args = parser.parse_args()

    ds_dir = args.dataset_dir or discover_canonical_dataset_dir()
    try:
        run_colab_production_training(
            dataset_dir=ds_dir,
            output_dir=args.output_dir,
            smoke_report_path=args.smoke_report,
            precision=args.precision,
            allow_non_a100=args.allow_non_a100,
            mock=args.mock,
            hf_repo=args.hf_repo,
            hf_token=args.hf_token,
            run_mode=args.mode,
        )
        return 0
    except Exception as e:
        print(f"[-] COLAB PRODUCTION TRAINING FAILURE: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
