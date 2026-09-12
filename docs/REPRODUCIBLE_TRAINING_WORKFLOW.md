# Reproducible Training Workflow (Step-by-Step Guide)

## 1. Workflow Lifecycle

```
Data Owner (Kaggle Dataset)
             │
             ▼
Local Pre-Push Gate (python scripts/verify_prepush.py)
             │
             ▼
GitHub Actions CI (PASS)
             │
             ▼
Kaggle 2×T4 Smoke Gate (notebooks/kaggle_smoke.ipynb -> PASS)
             │
             ▼
Freeze Run Tuple (Git SHA + Dataset Hash + Smoke Report)
             │
             ▼
Google Colab A100 Production Run (notebooks/colab_a100_train.ipynb)
             │
             ▼
Hugging Face Release & Codabench Submission
```

---

## 2. Stage-by-Stage Operating Instructions

### Stage 1: Local Pre-Push Verification
Before pushing any code or notebook updates to GitHub, run the local gate:
```bash
python scripts/verify_prepush.py
```
This executes:
1. Python syntax compilation across `src/` and `scripts/`.
2. Modular pytest suites (`tests/unit`, `tests/dataset`, `tests/notebook`, `tests/parity`, `tests/leakage`, `tests/memory`, `tests/integration`, `tests/release`).
3. Parameter budget audit (`scripts/audit_parameters.py` < 4B).
4. Notebook zero-drift check (`scripts/generate_notebooks.py --check-drift`).
5. Offline Kaggle pipeline smoke.
6. Git working tree hygiene.

---

### Stage 2: Kaggle 2×T4 Smoke Gate (B1.1)
1. **Open Notebook on Kaggle**:
   - URL: `https://www.kaggle.com/code/phucdangg/legalir-training` (or upload `notebooks/kaggle_smoke.ipynb`).
2. **Attach Dataset**:
   - Kaggle Dataset: `phucdangg/legalir-task1-clean-data` (attached at `/kaggle/input/datasets/phucdangg/legalir-task1-clean-data` or `/kaggle/input/legalir-task1-clean-data`).
3. **Accelerator**:
   - Set Accelerator to **GPU T4 × 2** or **GPU T4**.
4. **Click "Run All"**:
   - Execution time: ~3 minutes.
   - Mines a 50-query leakage-safe subset on the fly.
   - Runs 3 optimizer updates on `BAAI/bge-reranker-v2-m3` + LoRA.
   - Asserts finite loss, weight update delta $\Delta w > 0$, and adapter checkpoint save/reload.
   - Generates `kaggle_smoke_report.json` with verdict `"PASS"`.

---

### Stage 3: Google Colab A100 Production Training (B1.2)

You can run production training either automatically via the **Colab CLI** or manually via the **Colab Web Interface**:

#### Option A: One-Command Automated CLI Runner (Recommended)
From your local terminal, run:
```bash
./scripts/run_colab_cli.sh A100
```
This script automatically:
1. Provisions an NVIDIA A100 GPU session (`colab new -s legalir-a100-run --gpu A100`).
2. Uploads local credentials from `.env` to `/content/.env` on the VM.
3. Executes `notebooks/colab_a100_train.ipynb` with a 4-hour timeout.
4. Trains the full BGE LoRA reranker on all 7,000 queries using `torch.bfloat16`.
5. Validates and packages `submission.zip`.
6. Uploads the final adapter, logs, metrics, and `run_manifest.json` directly to your private Hugging Face model repository: `https://huggingface.co/dangphuc2109/legalir-task1-reranker`.
7. Once finished, releases the VM with `colab stop -s legalir-a100-run` to protect your compute credits.

*Tip: To test with a low-cost GPU first before using A100 credits, simply pass `T4`:*
```bash
./scripts/run_colab_cli.sh T4
```

#### Option B: Manual Web Interface
1. **Open Notebook on Google Colab**:
   - Open `notebooks/colab_a100_train.ipynb`.
2. **Select Runtime**:
   - Runtime $\rightarrow$ Change runtime type $\rightarrow$ **NVIDIA A100 GPU** (High-RAM).
3. **Configure Secrets**:
   - In the Colab left sidebar 🔑 **Secrets**, add `HF_TOKEN` with write permissions.
4. **Click "Run All"**:
   - Preflight verifies GPU is NVIDIA A100 and confirms Kaggle Smoke Gate passed.
   - Executes full training, validates submission, and publishes artifacts to Hugging Face.
