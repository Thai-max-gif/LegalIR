#!/usr/bin/env bash
# ==============================================================================
# LegalIR Google Colab CLI Automation
# Supports:
#   ./scripts/colab/run_colab_cli.sh T4   -> runs notebooks/colab_t4_smoke.ipynb
#   ./scripts/colab/run_colab_cli.sh A100 -> runs notebooks/colab_a100_train.ipynb
# Guaranteed fail-closed cleanup trap to always stop VM and prevent credit burn.
# ==============================================================================

set -Eeuo pipefail

GPU_MODE="${1:-A100}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

case "$GPU_MODE" in
  T4)
    GPU="T4"
    NOTEBOOK="notebooks/colab_t4_smoke.ipynb"
    SESSION="legalir-t4-gate"
    NEW_FLAGS=("--gpu" "T4")
    ;;
  A100)
    GPU="A100"
    NOTEBOOK="notebooks/colab_a100_train.ipynb"
    SESSION="legalir-a100-production"
    NEW_FLAGS=("--gpu" "A100" "--high-mem")
    ;;
  *)
    echo "[!] Error: Invalid GPU mode '$GPU_MODE'. Usage: $0 {T4|A100}" >&2
    exit 2
    ;;
esac

echo "================================================================="
echo "LegalIR Google Colab CLI Automation"
echo "  • Target Mode : $GPU_MODE"
echo "  • Session     : $SESSION"
echo "  • Notebook    : $NOTEBOOK"
echo "================================================================="

cleanup() {
  echo ""
  echo "[*] Cleaning up: Stopping Colab session '$SESSION' to release compute..."
  colab stop -s "$SESSION" >/dev/null 2>&1 || true
  echo "[+] Colab session '$SESSION' stopped."
}
trap cleanup EXIT INT TERM

# 1. Allocate VM
echo "[1/4] Allocating Colab VM with ${NEW_FLAGS[*]}..."
colab new -s "$SESSION" "${NEW_FLAGS[@]}"

# 2. Upload local .env & gate prerequisites
if [ -f ".env" ]; then
    echo "[2/4] Uploading local .env to /content/.env..."
    colab upload .env /content/.env -s "$SESSION"
else
    echo "[!] Warning: No local .env found. Secrets must be configured in Colab Secrets."
fi

if [ -f "artifacts/task1/gates/kaggle_t4x2_report.json" ]; then
    echo "[*] Syncing Kaggle Dual-T4 report to /content/kaggle_t4x2_report.json..."
    colab upload artifacts/task1/gates/kaggle_t4x2_report.json /content/kaggle_t4x2_report.json -s "$SESSION" || true
fi

if [ -f "artifacts/task1/gates/colab_t4_report.json" ]; then
    echo "[*] Syncing Colab T4 report to /content/colab_t4_report.json..."
    colab upload artifacts/task1/gates/colab_t4_report.json /content/colab_t4_report.json -s "$SESSION" || true
fi

# 3. Execute notebook
echo "[3/4] Executing $NOTEBOOK on remote Colab VM..."
colab exec -s "$SESSION" -f "$NOTEBOOK" --timeout 14400

# 4. Download output artifacts before VM release
echo "[4/4] Retrieving remote artifacts..."
if [ "$GPU_MODE" = "T4" ]; then
    mkdir -p artifacts/task1/gates
    colab download /content/artifacts/task1/gates/colab_t4_report.json artifacts/task1/gates/colab_t4_report.json -s "$SESSION" || \
    colab download /content/colab_t4_report.json artifacts/task1/gates/colab_t4_report.json -s "$SESSION" || true
elif [ "$GPU_MODE" = "A100" ]; then
    mkdir -p artifacts/task1/production
    colab download /content/legalir_production_run/submission.zip artifacts/task1/production/submission.zip -s "$SESSION" || true
    colab download /content/legalir_production_run/run_manifest.json artifacts/task1/production/run_manifest.json -s "$SESSION" || true
    colab download /content/legalir_production_run/checksums.sha256 artifacts/task1/production/checksums.sha256 -s "$SESSION" || true
fi

echo ""
echo "================================================================="
echo "[+] Remote notebook execution completed successfully!"
echo "    Session will now be automatically stopped."
echo "================================================================="
