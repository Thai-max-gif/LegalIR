#!/usr/bin/env bash
# ==============================================================================
# LegalIR Compulsory Local Verification & Test Suite
# Validates Python syntax, canonical dataset, notebooks, parameter budget,
# and all modular test suites before allowing push to GitHub.
# ==============================================================================

set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# Resolve python binary
if [ -x ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
else
    PYTHON_BIN="python"
fi

echo "================================================================="
echo "LegalIR Compulsory Local Verification Gate"
echo "  • Python Binary: $PYTHON_BIN"
echo "  • Repo Root    : $REPO_ROOT"
echo "================================================================="

# 1. Python syntax compilation
echo ""
echo "[1/6] Compiling Python syntax across src/ and scripts/..."
$PYTHON_BIN -m compileall -q src scripts
echo "[+] Python syntax compilation PASSED."

# 2. Canonical dataset validation
echo ""
echo "[2/6] Validating canonical dataset (schema, counts, splits, manifest)..."
$PYTHON_BIN -m pytest -v tests/dataset/
echo "[+] Canonical dataset validation PASSED."

# 3. Notebooks validation
echo ""
echo "[3/6] Validating notebooks (JSON schema, cell syntax, zero-drift parity)..."
$PYTHON_BIN -m pytest -v tests/notebook/
echo "[+] Notebooks validation PASSED."

# 4. Learned Parameter Budget (< 4B)
echo ""
echo "[4/6] Auditing learned parameter budget (< 4,000,000,000 cap)..."
$PYTHON_BIN scripts/audit_parameters.py
echo "[+] Parameter budget audit PASSED."

# 5. Full modular test suite
echo ""
echo "[5/6] Executing full modular test suites..."
$PYTHON_BIN -m pytest -q tests/
echo "[+] Full modular test suites PASSED."

# 6. Offline pipeline smoke
echo ""
echo "[6/6] Verifying offline pipeline smoke..."
$PYTHON_BIN scripts/smoke_kaggle_pipeline.py --tiny --run-mode smoke
echo "[+] Offline pipeline smoke PASSED."

echo ""
echo "================================================================="
echo "[+] ALL COMPULSORY LOCAL TESTS AND VALIDATION GATES PASSED."
echo "    Dataset, notebooks, and models are verified."
echo "    Safe to push to GitHub!"
echo "================================================================="
exit 0
