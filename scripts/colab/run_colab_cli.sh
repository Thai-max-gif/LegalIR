#!/usr/bin/env bash
# ==============================================================================
# LegalIR Google Colab CLI Automation (A100 production only)
#
# Kaggle T4x2 is the pre-A100 hardware gate. The retired single-T4 notebook
# is no longer supported; this script provisions an A100 VM and executes
# notebooks/colab_a100_train.ipynb.
#
# Test hook: PYTHON_BIN overrides the interpreter used for local preflight
# (default: .venv/bin/python).
# ==============================================================================

set -Euo pipefail

GPU_MODE="${1:-A100}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"

# Filter sensitive or irrelevant variables out of .env before upload
LOCAL_ENV="${REPO_ROOT}/.env"
FILTERED_ENV="${REPO_ROOT}/.env.filtered"
if [ -f "$LOCAL_ENV" ]; then
    grep -E '^(HF_TOKEN|HF_TOKEN_WRITE|HF_TOKEN_READ|KAGGLE_API_TOKEN|KAGGLE_KEY|KAGGLE_USERNAME|HF_REPO_ID|HF_ALLOW_PUBLIC_REPO|LEGALIR_COMMIT_SHA)=' "$LOCAL_ENV" > "$FILTERED_ENV" || true
else
    touch "$FILTERED_ENV"
fi

case "$GPU_MODE" in
  A100)
    GPU="A100"
    NOTEBOOK="notebooks/colab_a100_train.ipynb"
    SESSION="legalir-a100-production-$(head -c 4 /dev/urandom | xxd -p)"
    NEW_FLAGS=("--gpu" "A100")

    EXPECTED_SHA="${LEGALIR_COMMIT_SHA:-$(git rev-parse HEAD)}"
    echo "Running local provenance preflight for A100..."
    if ! "$PYTHON_BIN" scripts/colab/bootstrap.py --expected-sha "$EXPECTED_SHA"; then
        echo "[!] Preflight failed. Aborting before Colab allocation." >&2
        rm -f "$FILTERED_ENV"
        exit 1
    fi
    echo "{\"expected_sha\": \"$EXPECTED_SHA\"}" > legalir_launch.json
    ;;
  *)
    echo "[!] Error: Invalid GPU mode '$GPU_MODE'. Usage: $0 A100 (single-T4 mode retired; Kaggle T4x2 is the pre-A100 gate)" >&2
    rm -f "$FILTERED_ENV"
    exit 2
    ;;
esac

echo "================================================================="
echo "LegalIR Google Colab CLI Automation"
echo "  • Target Mode : $GPU_MODE"
echo "  • Session     : $SESSION"
echo "  • Notebook    : $NOTEBOOK"
echo "================================================================="

# Create a temporary copy of the notebook to run, preventing local dirtying
TMP_NOTEBOOK="${NOTEBOOK}.tmp.ipynb"
cp "$NOTEBOOK" "$TMP_NOTEBOOK"

EXEC_EXIT_CODE=0

cleanup() {
  echo ""
  echo "[*] Retrieving remote artifacts..."
  mkdir -p artifacts/task1/production
  colab download /content/legalir_production_run/recovery.tar.gz artifacts/task1/production/recovery.tar.gz -s "$SESSION" >/dev/null 2>&1 || true
  colab download /content/legalir_production_run/training.log artifacts/task1/production/training.log -s "$SESSION" >/dev/null 2>&1 || true
  colab download /content/legalir_production_run/run_manifest.json artifacts/task1/production/run_manifest.json -s "$SESSION" >/dev/null 2>&1 || true
  colab download /content/legalir_production_run/submission.zip artifacts/task1/production/submission.zip -s "$SESSION" >/dev/null 2>&1 || true
  colab download /content/legalir_production_run/submission.json artifacts/task1/production/submission.json -s "$SESSION" >/dev/null 2>&1 || true

  echo "[*] Cleaning up: Stopping Colab session '$SESSION' to release compute..."
  if colab stop -s "$SESSION" >/dev/null 2>&1; then
      echo "[+] Colab session '$SESSION' stopped."
  else
      echo "[-] Failed to stop Colab session '$SESSION'."
  fi
  rm -f "$FILTERED_ENV" "$TMP_NOTEBOOK" legalir_launch.json

  if [ $EXEC_EXIT_CODE -ne 0 ]; then
      echo "[!] Remote notebook execution failed (exit code $EXEC_EXIT_CODE)."
      exit $EXEC_EXIT_CODE
  fi
}

trap cleanup EXIT INT TERM

# 1. Allocate VM
echo "[1/4] Allocating Colab VM with ${NEW_FLAGS[*]}..."
colab new -s "$SESSION" "${NEW_FLAGS[@]}"

# 2. Upload local .env & gate prerequisites
if [ -s "$FILTERED_ENV" ]; then
    echo "[2/4] Uploading filtered local .env to /content/.env..."
    colab upload "$FILTERED_ENV" /content/.env -s "$SESSION"
else
    echo "[!] Warning: No relevant secrets found in local .env. Secrets must be configured in Colab Secrets."
fi

colab upload legalir_launch.json /content/legalir_launch.json -s "$SESSION" || true

# We assume standard artifacts are synced by Git now, but we'll upload if they are local-only
for f in artifacts/task1/gates/kaggle_t4x2_report.json artifacts/task1/freeze/production_freeze.json; do
  if [ -f "$f" ]; then
      colab upload "$f" "/content/$(basename "$f")" -s "$SESSION" >/dev/null 2>&1 || true
  fi
done

# NOTE: COLAB_TIMEOUT is in SECONDS (colab exec --timeout <float> seconds).
# Do NOT pass milliseconds. Default 40000s = 11.1h (exceeds typical Colab
# VM lifetime — expect preemption on long 5-fold OOF runs).
TIMEOUT="${COLAB_TIMEOUT:-40000}"
echo "[3/4] Executing $NOTEBOOK on remote Colab VM (timeout ${TIMEOUT}s)..."
set +e
colab exec -s "$SESSION" -f "$TMP_NOTEBOOK" --timeout "$TIMEOUT"
EXEC_EXIT_CODE=$?
set -e

echo "[+] Remote notebook execution completed!"
