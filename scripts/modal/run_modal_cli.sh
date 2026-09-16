#!/usr/bin/env bash
# ==============================================================================
# LegalIR Modal A100 pre-dispatch CPU gate (recommended entrypoint).
#
# Fail before invoking Modal when CPU provenance is wrong, so known-invalid
# commits never reach image build/dispatch. Remote repeats trust checks;
# local checks alone cannot validate remote secret values or GPU.
#
# Order:
#   1. Validate args/tools/SHA before any cloud command.
#   2. Require a clean tracked/untracked tree (ignored artifacts excluded).
#   3. CPU provenance preflight (scripts/colab/bootstrap.py).
#   4. Dispatch via `modal run` with the same SHA in the environment.
#
# Remote order (in run_modal_a100.py): checkout+provenance, then HF access,
# then dataset download/fingerprint, then train. HF failure precedes expensive
# data acquisition. No auto-retry.
#
# Boolean flag spelling confirmed via installed SDK:
#   modal run scripts/modal/run_modal_a100.py --help
# shows `--hf-allow-public-repo / --no-hf-allow-public-repo`.
# Absent consent stays private-only; explicit --hf-allow-public-repo forwards
# once and is recorded in the manifest.
#
# Test hooks:
#   PYTHON_BIN (default .venv/bin/python) for preflight,
#   MODAL_BIN (default .venv/bin/modal) for dispatch.
# ==============================================================================

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
MODAL_BIN="${MODAL_BIN:-.venv/bin/modal}"

HF_PUBLIC_FLAG=""
SHOW_HELP=0

for arg in "$@"; do
  case "$arg" in
    --hf-allow-public-repo)
      if [ -n "$HF_PUBLIC_FLAG" ]; then
        echo "[!] Duplicate --hf-allow-public-repo flag." >&2
        exit 2
      fi
      HF_PUBLIC_FLAG="--hf-allow-public-repo"
      ;;
    --no-hf-allow-public-repo)
      if [ -n "$HF_PUBLIC_FLAG" ]; then
        echo "[!] Duplicate public-repo flag." >&2
        exit 2
      fi
      HF_PUBLIC_FLAG="--no-hf-allow-public-repo"
      ;;
    -h|--help)
      SHOW_HELP=1
      ;;
    *)
      echo "[!] Unknown argument: $arg (expected --hf-allow-public-repo)" >&2
      exit 2
      ;;
  esac
done

if [ "$SHOW_HELP" -eq 1 ]; then
  cat <<'EOF'
Usage: scripts/modal/run_modal_cli.sh [--hf-allow-public-repo]

Recommended Modal entrypoint. Validates CPU provenance before dispatch.
  --hf-allow-public-repo   Explicit opt-in to push to an existing PUBLIC HF
                           repo (recorded in manifest). Absent means
                           private-only (fail closed).
EOF
  exit 0
fi

if [ ! -x "$PYTHON_BIN" ] && [ ! -f "$PYTHON_BIN" ]; then
  echo "[!] PYTHON_BIN not found: $PYTHON_BIN" >&2
  exit 2
fi
if [ ! -x "$MODAL_BIN" ] && [ ! -f "$MODAL_BIN" ]; then
  echo "[!] MODAL_BIN not found: $MODAL_BIN (install modal CLI and authenticate)" >&2
  exit 2
fi

# Selected SHA from env or exact local HEAD.
if [ -n "${LEGALIR_COMMIT_SHA:-}" ]; then
  EXPECTED_SHA="$LEGALIR_COMMIT_SHA"
else
  if ! EXPECTED_SHA="$(git rev-parse HEAD 2>/dev/null)"; then
    echo "[!] Could not determine local HEAD; set LEGALIR_COMMIT_SHA." >&2
    exit 2
  fi
fi
if ! echo "$EXPECTED_SHA" | grep -Eq '^[0-9a-f]{40}$'; then
  echo "[!] LEGALIR_COMMIT_SHA must be an exact 40-char lowercase SHA, got '$EXPECTED_SHA'." >&2
  exit 2
fi

# Reject mismatch between selected SHA and local HEAD.
LOCAL_HEAD="$(git rev-parse HEAD 2>/dev/null || true)"
if [ -n "$LOCAL_HEAD" ] && [ "$LOCAL_HEAD" != "$EXPECTED_SHA" ]; then
  echo "[!] Selected SHA $EXPECTED_SHA does not match local HEAD $LOCAL_HEAD; refusing to dispatch." >&2
  exit 2
fi

# Require clean tracked/untracked tree (ignored files excluded by porcelain).
if [ -n "$(git status --porcelain=v1 2>/dev/null)" ]; then
  echo "[!] Working tree is dirty; commit or stash before Modal dispatch." >&2
  git status --porcelain=v1 >&2 || true
  exit 2
fi

# CPU provenance gate before any cloud command (image build/dispatch).
echo "[*] Local CPU provenance preflight for $EXPECTED_SHA..."
if ! "$PYTHON_BIN" scripts/colab/bootstrap.py --expected-sha "$EXPECTED_SHA"; then
  echo "[!] Local CPU provenance failed; aborting before Modal dispatch." >&2
  exit 1
fi

# Forward explicit consent once; absent stays private-only.
MODAL_ARGS=()
if [ -n "$HF_PUBLIC_FLAG" ]; then
  MODAL_ARGS+=("$HF_PUBLIC_FLAG")
else
  MODAL_ARGS+=("--no-hf-allow-public-repo")
fi

echo "[*] Dispatching to Modal for $EXPECTED_SHA ${MODAL_ARGS[*]}..."
echo "[*] NOTE: invoking Modal can build an image before remote preflight runs;"
echo "    local checks cannot validate remote secrets or GPU. Remote repeats"
echo "    checkout/provenance, HF access, dataset, then train. No auto-retry."
LEGALIR_COMMIT_SHA="$EXPECTED_SHA" "$MODAL_BIN" run scripts/modal/run_modal_a100.py "${MODAL_ARGS[@]}"
