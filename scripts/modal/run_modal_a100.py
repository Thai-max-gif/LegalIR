import os
import subprocess
from pathlib import Path

import modal

# Define the Modal App
app = modal.App("legalir-a100-production")

# Define the environment image
# We install Git to clone the repository and standard requirements for the pipeline.
# PyTorch is installed with CUDA 12.1 support, which is suitable for A100.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch==2.1.2",
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

# 5 hours timeout (5 * 60 * 60 = 18000 seconds) to prevent excessive billing
TIMEOUT_SECONDS = 18000

@app.function(
    image=image,
    gpu=modal.gpu.A100(),
    timeout=TIMEOUT_SECONDS,
    secrets=[
        modal.Secret.from_name("kaggle-secret"),
        modal.Secret.from_name("huggingface-secret")
    ]
)
def run_production_training(expected_sha: str):
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
    subprocess.run(["git", "fetch", "origin", expected_sha], cwd=repo_dir, check=True)
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
    verify_launch(expected_sha, kaggle_report, colab_report, freeze_file)
    
    dataset_dir = Path("/root/kaggle_dataset")
    print(f"[*] Downloading and verifying canonical dataset at {dataset_dir}...")
    prepare_dataset(dataset_dir, freeze_file)
    
    # 3. Preflight Hugging Face Access
    from scripts.gates.run_a100 import preflight_huggingface_access
    hf_repo = os.environ.get("HF_REPO_ID", "dangphuc2109/legalir-task1-reranker")
    print(f"[*] Verifying Hugging Face write access to {hf_repo}...")
    hf_ok, hf_detail = preflight_huggingface_access(hf_repo)
    if not hf_ok:
        raise RuntimeError(f"Hugging Face preflight failed: {hf_detail}")
    print(f"[+] {hf_detail}")
    
    # 4. Execute the pipeline
    output_dir = Path("/root/legalir_production_run")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    from scripts.run_colab_train import run_colab_production_training
    
    print("[*] Starting A100 production training pipeline...")
    report = run_colab_production_training(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        smoke_report_path=kaggle_report,
        colab_t4_report_path=colab_report,
        expected_sha=expected_sha,
        precision="bf16",
        allow_non_a100=False,
        mock=False,
        hf_repo=hf_repo,
        freeze_file_path=freeze_file,
        run_mode="full",
    )
    
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
    print("[*] Ensure you have created 'kaggle-secret' and 'huggingface-secret' in the Modal dashboard!")
    
    result = run_production_training.remote(expected_sha)
    print(f"[*] Remote job finished with result: {result}")
