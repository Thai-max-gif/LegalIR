# LegalIR Task 1: A100 Training Guide

This repository is completely configured and verified for production execution on an NVIDIA A100 GPU. You have two options for training the model based on your infrastructure preference:

## Option 1: Modal Serverless (Recommended)
This is the safest and most automated approach. It runs entirely in the cloud, automatically provisions the A100, and caps a single attempt at 5 hours (`timeout=18000s` in `scripts/modal/run_modal_a100.py`).

> Billing note: the timeout caps duration per attempt, not total spend — retries and re-runs bill extra. Partial outputs sync to the `legalir-production` Modal Volume on success AND failure for forensics. There is no checkpoint-resume (trainer saves only at end), so a timeout kill still requires a full re-run.

1. Ensure you have the `modal` CLI installed and authenticated (`modal token new`).
2. Go to your Modal Dashboard -> **Secrets** and create two custom secrets:
   - **`kaggle-secret`**: Add `KAGGLE_USERNAME` and `KAGGLE_KEY` (or `KAGGLE_API_TOKEN` for `KGAT_` bearer tokens — code auto-promotes `KAGGLE_API_TOKEN` → `KAGGLE_KEY`)
   - **`huggingface-secret`**: Add `HF_TOKEN_WRITE` (write scope required; read-only tokens fail preflight) and `HF_TOKEN`
3. Launch the job:
   ```bash
   modal run scripts/modal/run_modal_a100.py
   ```
4. The logs will stream directly to your terminal. When finished, artifacts will automatically be pushed to your Hugging Face repository.

## Option 2: Google Colab CLI Automation
This runs the pipeline on a Google Colab VM. It utilizes local `.env` filtering to securely transfer credentials and fetches the `recovery.tar.gz` and training logs locally when finished.

1. Ensure you have the `colab` CLI tool installed (`pip install colab-cli`).
2. Ensure your local `.env` file contains your `HF_TOKEN_WRITE` (or `HF_TOKEN`), `KAGGLE_USERNAME`, and `KAGGLE_KEY` (or `KAGGLE_API_TOKEN`).
3. Launch the orchestrator:
   ```bash
   ./scripts/colab/run_colab_cli.sh A100
   ```
4. The script traps EXIT/INT/TERM to pull `recovery.tar.gz`/`training.log`/`run_manifest.json`/`submission.*` and always runs `colab stop` to release compute. If the local orchestrator dies or the network drops, `stop` is never sent — the VM burns until Google reclaims it. Production config is unified: LoRA `r=8/alpha=16`, `max_length=384`, hybrid `100/30`, RRF `k=60`, `bf16`.

## Final Review
Before launching, you can pass the `A100_Production_Review_Prompt.md` file to another Claude instance or a senior ML engineer to perform a final architectural review.
