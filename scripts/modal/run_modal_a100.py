import os
import subprocess
from pathlib import Path

import modal

# Define the Modal App
app = modal.App("legalir-a100-production")

# Define the environment image
# Pinned to match requirements-colab.txt (verified: transformers 5.15.1 exists).
# Python 3.11 to match torch cp311 wheels. torch>=2.5 is REQUIRED:
# transformers 5.x lazy-loads model classes only when torch>=2.5 is importable
# (torch 2.1.2 passes raw `import torch` but fails the backend gate with the
# misleading "requires the PyTorch library but it was not found" error).
# PyTorch is installed with CUDA 12.4 support, which is suitable for A100.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.5.1",
        "transformers==5.15.1",
        "peft==0.20.0",
        "accelerate==1.14.0",
        "huggingface-hub==1.28.0",
        "numpy>=1.24,<3",
        "pandas>=2,<3",
        "pyarrow>=14",
        "pyyaml>=6,<7",
        "scikit-learn>=1.3,<2",
        "lightgbm>=4,<5",
        "bm25s>=0.2,<1",
        "pyvi>=0.1.1,<1",
        "sentencepiece>=0.1.99",
        "faiss-cpu>=1.7",
        "psutil>=5.9",
        "kaggle>=1.8,<3",
        "tqdm>=4.65"
    )
)

# Persistent volume: without this a 5h timeout kill loses everything
# (5-fold OOF + final LoRA has no checkpoint-resume; trainer saves only at end).
# Partial outputs are synced here on success AND on failure for forensics.
volume = modal.Volume.from_name("legalir-production", create_if_missing=True)
VOLUME_MOUNT = "/root/legalir_volume"

# 5 hours timeout (5 * 60 * 60 = 18000 seconds) to prevent excessive billing
TIMEOUT_SECONDS = 18000

@app.function(
    image=image,
    gpu="A100",
    timeout=TIMEOUT_SECONDS,
    volumes={VOLUME_MOUNT: volume},
    secrets=[
        modal.Secret.from_name("kaggle-secret"),
        modal.Secret.from_name("huggingface-secret")
    ]
)
def run_production_training(expected_sha: str, skip_colab_t4: bool = True, hf_allow_public_repo: bool = True):
    """
    Executes the A100 production training pipeline within a Modal container.
    """
    import sys
    
    # 1. Clone the repository and checkout the exact expected SHA
    repo_dir = Path("/root/LegalIR")
    if not repo_dir.exists():
        print(f"[*] Cloning repository into {repo_dir}...")
        subprocess.run(["git", "clone", "https://github.com/silent9669/LegalIR.git", str(repo_dir)], check=True)
    
    print(f"[*] Checking out exact commit: {expected_sha}")
    subprocess.run(["git", "fetch", "origin", expected_sha], cwd=repo_dir, check=False)
    res = subprocess.run(["git", "checkout", "--detach", expected_sha], cwd=repo_dir, capture_output=True, text=True)
    if res.returncode != 0:
        print("[*] Checkout fallback: unshallowing repository...")
        subprocess.run(["git", "fetch", "--unshallow", "origin"], cwd=repo_dir, check=False)
        subprocess.run(["git", "checkout", "--detach", expected_sha], cwd=repo_dir, check=True)
    
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))
    os.chdir(repo_dir)
    
    # 2. Acquire Kaggle dataset
    from scripts.colab.bootstrap import prepare_dataset, verify_launch
    
    # We must point to the verified artifact paths in the repo
    kaggle_report = repo_dir / "artifacts/task1/gates/kaggle_t4x2_report.json"
    colab_report = repo_dir / "artifacts/task1/gates/colab_t4_report.json"
    freeze_file = repo_dir / "artifacts/task1/freeze/production_freeze.json"
    
    print("[*] Verifying production launch constraints...")
    if skip_colab_t4:
        print("[!] OPERATOR OVERRIDE: Colab single-T4 gate (B1.15) skipped by request.", flush=True)
        print("[!] Kaggle dual-T4 gate remains strictly enforced.", flush=True)
    verify_launch(expected_sha, kaggle_report, colab_report if not skip_colab_t4 else None, freeze_file,
                  require_colab_t4=not skip_colab_t4)
    
    dataset_dir = Path("/root/kaggle_dataset")
    print(f"[*] Downloading and verifying canonical dataset at {dataset_dir}...")
    prepare_dataset(dataset_dir, freeze_file)
    
    # 3. Preflight Hugging Face Access
    from scripts.gates.run_a100 import preflight_huggingface_access
    hf_repo = os.environ.get("HF_REPO_ID", "dangphuc2109/legalir-task1-reranker")
    print(f"[*] Verifying Hugging Face write access to {hf_repo}...")
    if hf_allow_public_repo:
        print("[!] OPERATOR OVERRIDE: public HF repos permitted for this launch (recorded in manifest).", flush=True)
    hf_ok, hf_detail = preflight_huggingface_access(hf_repo, allow_public_repo=hf_allow_public_repo)
    if not hf_ok:
        raise RuntimeError(f"Hugging Face preflight failed: {hf_detail}")
    print(f"[+] {hf_detail}")
    
    # 4. Execute the pipeline (sync partial outputs to Volume even on failure,
    # so a 5h timeout kill is debuggable instead of a total loss).
    output_dir = Path("/root/legalir_production_run")
    output_dir.mkdir(parents=True, exist_ok=True)
    volume_dir = Path(VOLUME_MOUNT) / expected_sha
    volume_dir.mkdir(parents=True, exist_ok=True)

    from scripts.run_colab_train import run_colab_production_training

    print("[*] Starting A100 production training pipeline...")
    try:
        report = run_colab_production_training(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            smoke_report_path=kaggle_report,
            colab_t4_report_path=None if skip_colab_t4 else colab_report,
            expected_sha=expected_sha,
            precision="bf16",
            allow_non_a100=False,
            mock=False,
            hf_repo=hf_repo,
            freeze_file_path=freeze_file,
            run_mode="full",
            skip_colab_t4=skip_colab_t4,
            hf_allow_public_repo=hf_allow_public_repo,
        )
    finally:
        try:
            import shutil
            import time as _time
            snap = volume_dir / f"snapshot-{_time.strftime('%Y%m%d-%H%M%S')}"
            snap.mkdir(parents=True, exist_ok=True)
            for name in ("run_manifest.json", "training.log", "submission.zip",
                         "submission.json", "checksums.sha256", "resolved_config.yaml"):
                src = output_dir / name
                if src.is_file():
                    shutil.copy2(src, snap / name)
            # Best-effort adapter snapshot (may be absent on early failure).
            for sub in ("checkpoints/reranker_final/adapter_config.json",
                        "final_adapter/adapter_config.json"):
                src = output_dir / sub
                if src.is_file():
                    (snap / Path(sub).name).write_bytes(src.read_bytes())
                    break
            volume.commit()
            print(f"[*] Synced partial outputs to Volume: {snap}", flush=True)
        except Exception as sync_exc:
            print(f"[!] Volume sync failed: {type(sync_exc).__name__}", flush=True)

    print(f"[+] Training completed. Status: {report.get('status')} | Verdict: {report.get('verdict')}")

    # The pipeline internally handles Hugging Face artifact upload and cleanup.
    return report

@app.local_entrypoint()
def main():
    import sys
    # Try to grab the SHA from local git if we are in the repo, or from env
    expected_sha = os.environ.get("LEGALIR_COMMIT_SHA")
    if not expected_sha:
        try:
            expected_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        except Exception:
            print("Error: Could not determine expected Git SHA. Please set LEGALIR_COMMIT_SHA.", file=sys.stderr)
            sys.exit(1)
            
    print(f"[*] Dispatching A100 training job to Modal for commit: {expected_sha}")
    print("[*] This process will run remotely on an A100 GPU and automatically terminate after 5 hours max.")
    print("[*] Partial outputs sync to the 'legalir-production' Volume on success AND failure.")
    print("[*] NOTE: 5h caps duration, not spend — retries/re-runs bill extra. No checkpoint-resume:")
    print("    a timeout kill still requires a full re-run (Volume holds forensics only).")
    print("[*] Ensure you have created 'kaggle-secret' and 'huggingface-secret' in the Modal dashboard!")
    
    result = run_production_training.remote(expected_sha)
    print(f"[*] Remote job finished with result: {result}")
