#!/bin/bash
set -e

# LegalIR Task 1 Main Test Execution Gate
# Executes syntax, dataset, notebooks, parameter budget, and full test suite

PYTHON_BIN="python3"
if [ -x ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
elif [ -x ".venv-ml/bin/python" ]; then
    PYTHON_BIN=".venv-ml/bin/python"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
fi

exec $PYTHON_BIN scripts/verify_prepush.py "$@"
