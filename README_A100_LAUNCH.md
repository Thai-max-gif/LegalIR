# LegalIR Task 1: A100 Training Guide

This repository is completely configured and verified for production execution on an NVIDIA A100 GPU. You have two options for training the model based on your infrastructure preference:

## Option 1: Modal Serverless (Recommended)
This is the safest and most automated approach. It runs entirely in the cloud, automatically provisions the A100, and has a hard 5-hour timeout to guarantee you never exceed your $30 credit limit.

1. Ensure you have the `modal` CLI installed and authenticated (`modal token new`).
2. Go to your Modal Dashboard -> **Secrets** and create two custom secrets:
   - **`kaggle-secret`**: Add `KAGGLE_USERNAME` and `KAGGLE_KEY`
   - **`huggingface-secret`**: Add `HF_TOKEN_WRITE` and `HF_TOKEN`
3. Launch the job:
   ```bash
   modal run scripts/modal/run_modal_a100.py
   ```
4. The logs will stream directly to your terminal. When finished, artifacts will automatically be pushed to your Hugging Face repository.

## Option 2: Google Colab CLI Automation
This runs the pipeline on a Google Colab VM. It utilizes local `.env` filtering to securely transfer credentials and fetches the `recovery.tar.gz` and training logs locally when finished.

1. Ensure you have the `colab` CLI tool installed (`pip install colab-cli`).
2. Ensure your local `.env` file contains your `HF_TOKEN`, `KAGGLE_USERNAME`, and `KAGGLE_KEY`.
3. Launch the orchestrator:
   ```bash
   ./scripts/colab/run_colab_cli.sh A100
   ```
4. The script will automatically halt the VM if an error occurs to prevent credit burn.

## Final Review
Before launching, you can pass the `A100_Production_Review_Prompt.md` file to another Claude instance or a senior ML engineer to perform a final architectural review.
