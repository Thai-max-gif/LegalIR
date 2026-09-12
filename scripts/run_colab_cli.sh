#!/usr/bin/env bash
# ==============================================================================
# LegalIR Google Colab CLI Automation
# Provisions GPU session on Colab, uploads local .env, runs production training,
# and automatically exports artifacts to Hugging Face Hub.
# ==============================================================================

set -eo pipefail

GPU="${1:-A100}"
GPU_LOWER=$(echo "$GPU" | tr '[:upper:]' '[:lower:]')
SESSION="legalir-${GPU_LOWER}-run"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "================================================================="
echo "LegalIR Google Colab CLI Automation"
echo "  • Target GPU : $GPU"
echo "  • Session    : $SESSION"
echo "================================================================="

# 1. Allocate VM
echo "[1/4] Allocating Colab VM with --gpu $GPU..."
colab new -s "$SESSION" --gpu "$GPU"

# 2. Upload local .env
if [ -f ".env" ]; then
    echo "[2/4] Uploading local .env to /content/.env..."
    colab upload .env /content/.env -s "$SESSION"
else
    echo "[!] Warning: No local .env found. Secrets must be configured in Colab Secrets."
fi

# 3. Execute notebook
echo "[3/4] Executing training notebook on remote Colab VM..."
colab exec -s "$SESSION" -f notebooks/colab_a100_train.ipynb --timeout 14400

# 4. Status and Release Instructions
echo ""
echo "================================================================="
echo "[+] Training and Hugging Face upload completed!"
echo "    Artifacts repository: https://huggingface.co/dangphuc2109/legalir-task1-reranker"
echo "    To inspect logs     : colab log -s $SESSION"
echo "    To open browser UI  : colab url -s $SESSION --open"
echo "    To release the VM   : colab stop -s $SESSION"
echo "================================================================="
