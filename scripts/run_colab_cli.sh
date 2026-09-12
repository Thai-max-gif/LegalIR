#!/usr/bin/env bash
# Backward-compatible wrapper for Colab CLI runner
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$REPO_ROOT/scripts/colab/run_colab_cli.sh" "$@"
