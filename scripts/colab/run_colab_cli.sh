#!/usr/bin/env bash
# ==============================================================================
# LegalIR Google Colab CLI Automation (A100 production only)
#
# Kaggle T4x2 is the pre-A100 hardware gate. The retired single-T4 notebook
# is no longer supported; this script provisions an A100 VM and executes
# notebooks/colab_a100_train.ipynb.
#
# Supervised attempt with best-effort recovery. This does NOT establish
# survival of VM eviction or local-machine failure (SIGKILL, laptop shutdown,
# network loss, Colab eviction can defeat local cleanup). A VM-local tarball
# is not independent durable backup. Unattended Colab with durable checkpoint
# retention is a separate design requiring explicitly approved external
# storage; it is not part of this minimal repair.
#
# Supervising machine must stay awake and connected. Record the session ID
# before allocation. Confirm shutdown in the provider session/runtime view,
# not only a local success string. Local process termination is not evidence
# of remote shutdown: cleanup always attempts a confirmed stop, and a failed
# stop is an incident even when the local wrapper exits.
#
# Test hooks:
#   PYTHON_BIN overrides the interpreter for local preflight
#     (default: .venv/bin/python).
#   CLI_TIMEOUT_PYTHON is the real interpreter for the deadline helper
#     (default: .venv/bin/python). Tests must not stub deadline enforcement
#     away via PYTHON_BIN; set CLI_TIMEOUT_PYTHON to a real Python.
#   COLAB_NEW_TIMEOUT / COLAB_UPLOAD_TIMEOUT override the allocation and
#     per-upload deadlines (defaults below). Tests set these small; operators
#     need separate cost approval to raise them.
#
# Timeout policy (§4.3):
#   COLAB_TIMEOUT default 18000s (5h) is the whole-notebook wall-clock bound
#   enforced externally around the complete execution command. The same value
#   is also passed to `colab exec --timeout`, which the installed CLI applies
#   PER CELL (it loops over cells and supplies the timeout to each
#   runtime.execute_code call), so the per-cell flag alone cannot bound a
#   multi-cell notebook. Setup time and bounded cleanup time are separate and
#   additional. None of these is a platform-enforced billing cap. Larger
#   values need separate cost approval. A wait timeout does not kill the
#   remote process without a confirmed stop.
#   Allocation: COLAB_NEW_TIMEOUT default 180s. Required uploads:
#   COLAB_UPLOAD_TIMEOUT default 120s each. On timeout the primary failure
#   (124) is preserved and bounded shutdown of the recorded session is still
#   attempted, including ambiguous allocation outcomes.
#   Cleanup downloads: 15s manifest/log, 45s archive, 15s ZIP/JSON
#     (105s total max command time plus small termination overhead).
#   Stop: 30s, then actionable nonzero failure; no unbounded retry.
# ==============================================================================

set -Eeuo pipefail

GPU_MODE="${1:-A100}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
CLI_TIMEOUT_PYTHON="${CLI_TIMEOUT_PYTHON:-.venv/bin/python}"

# --- Early mode validation (no allocation on invalid mode) ---
case "$GPU_MODE" in
  A100)
    NOTEBOOK="notebooks/colab_a100_train.ipynb"
    ;;
  *)
    echo "[!] Error: Invalid GPU mode '$GPU_MODE'. Usage: $0 A100 (single-T4 mode retired; Kaggle T4x2 is the pre-A100 gate)" >&2
    exit 2
    ;;
esac

# --- Validate all deadlines before allocation ---
TIMEOUT="${COLAB_TIMEOUT:-18000}"
NEW_TIMEOUT="${COLAB_NEW_TIMEOUT:-180}"
UPLOAD_TIMEOUT="${COLAB_UPLOAD_TIMEOUT:-120}"
for _spec in "COLAB_TIMEOUT:$TIMEOUT" "COLAB_NEW_TIMEOUT:$NEW_TIMEOUT" "COLAB_UPLOAD_TIMEOUT:$UPLOAD_TIMEOUT"; do
  _name="${_spec%%:*}"
  _val="${_spec#*:}"
  if ! "$CLI_TIMEOUT_PYTHON" -c "import math,sys; v=float(sys.argv[1]); sys.exit(0 if (math.isfinite(v) and v>0) else 1)" "$_val" 2>/dev/null; then
    echo "[!] Invalid $_name: '$_val' (must be a positive finite number of seconds)" >&2
    exit 2
  fi
done
# NOTE: COLAB_TIMEOUT is in SECONDS. Do NOT pass milliseconds. Default 18000s
# = 5h whole-notebook wall clock, not a billing cap. Larger values need
# separate cost approval.

# --- Private temp files (umask 077) and traps BEFORE secret-bearing files ---
umask 077
TMPDIR_PRIVATE="$(mktemp -d -t legalir-colab-XXXXXX)"
FILTERED_ENV="$TMPDIR_PRIVATE/.env.filtered"
LAUNCH_JSON="$TMPDIR_PRIVATE/legalir_launch.json"
TMP_NOTEBOOK="$TMPDIR_PRIVATE/colab_a100_train.tmp.ipynb"

# State for truthful exit precedence.
PRIMARY_RC=0
EXEC_RC=0
EXEC_ATTEMPTED=0
ALLOC_ATTEMPTED=0
STOP_RC=0
CLEANUP_DONE=0
SESSION=""
RECOVERY_DIR=""
# PID of the currently supervised foreground child (0 when idle). On INT/TERM
# the handler terminates this child so a signal to the wrapper PID alone still
# reaches bounded cleanup promptly; bash would otherwise defer traps while
# waiting for a foreground command.
ACTIVE_CHILD=0

on_int() {
  if [ "$ACTIVE_CHILD" -ne 0 ]; then
    kill -TERM "$ACTIVE_CHILD" 2>/dev/null || true
  fi
  exit 130
}

on_term() {
  if [ "$ACTIVE_CHILD" -ne 0 ]; then
    kill -TERM "$ACTIVE_CHILD" 2>/dev/null || true
  fi
  exit 143
}

# Run a potentially-blocking command supervised: traps stay responsive to
# PID-only signals because the command runs in the background while the shell
# waits interruptibly. Returns the command's exit status.
run_fg() {
  "$@" &
  ACTIVE_CHILD=$!
  set +e
  wait "$ACTIVE_CHILD"
  local rc=$?
  set -e
  ACTIVE_CHILD=0
  return $rc
}

cleanup() {
  local trap_rc="${1:-0}"
  # Idempotent: run at most once. Remove traps and disable errexit so a
  # failed download cannot bypass stop.
  if [ "$CLEANUP_DONE" -eq 1 ]; then
    return 0
  fi
  CLEANUP_DONE=1
  trap - EXIT INT TERM
  set +e

  # Preserve signal/primary code if no earlier primary failure.
  if [ "$PRIMARY_RC" -eq 0 ] && [ "$trap_rc" -ne 0 ]; then
    PRIMARY_RC="$trap_rc"
  fi

  # Bounded artifact recovery (only if exec was attempted and session exists).
  if [ "$ALLOC_ATTEMPTED" -eq 1 ] && [ "$EXEC_ATTEMPTED" -eq 1 ] && [ -n "$SESSION" ]; then
    echo ""
    echo "[*] Retrieving remote artifacts to $RECOVERY_DIR (bounded)..."
    mkdir -p "$RECOVERY_DIR"
    local dl_rc=0
    "$CLI_TIMEOUT_PYTHON" scripts/colab/cli_timeout.py 15 colab download /content/legalir_production_run/run_manifest.json "$RECOVERY_DIR/run_manifest.json" -s "$SESSION" >/dev/null 2>&1
    dl_rc=$?
    if [ $dl_rc -ne 0 ]; then echo "[!] Warning: manifest download failed (rc=$dl_rc)." >&2; fi
    "$CLI_TIMEOUT_PYTHON" scripts/colab/cli_timeout.py 15 colab download /content/legalir_production_run/training.log "$RECOVERY_DIR/training.log" -s "$SESSION" >/dev/null 2>&1
    dl_rc=$?
    if [ $dl_rc -ne 0 ]; then echo "[!] Warning: training.log download failed (rc=$dl_rc)." >&2; fi
    "$CLI_TIMEOUT_PYTHON" scripts/colab/cli_timeout.py 45 colab download /content/legalir_production_run/recovery.tar.gz "$RECOVERY_DIR/recovery.tar.gz" -s "$SESSION" >/dev/null 2>&1
    dl_rc=$?
    if [ $dl_rc -ne 0 ]; then echo "[!] Warning: recovery archive download failed (rc=$dl_rc)." >&2; fi
    "$CLI_TIMEOUT_PYTHON" scripts/colab/cli_timeout.py 15 colab download /content/legalir_production_run/submission.zip "$RECOVERY_DIR/submission.zip" -s "$SESSION" >/dev/null 2>&1
    dl_rc=$?
    if [ $dl_rc -ne 0 ]; then echo "[!] Warning: submission.zip download failed (rc=$dl_rc)." >&2; fi
    "$CLI_TIMEOUT_PYTHON" scripts/colab/cli_timeout.py 15 colab download /content/legalir_production_run/submission.json "$RECOVERY_DIR/submission.json" -s "$SESSION" >/dev/null 2>&1
    dl_rc=$?
    if [ $dl_rc -ne 0 ]; then echo "[!] Warning: submission.json download failed (rc=$dl_rc)." >&2; fi
  fi

  # Bounded stop (attempted once even after ambiguous allocation failure,
  # in case allocation succeeded server-side).
  if [ "$ALLOC_ATTEMPTED" -eq 1 ] && [ -n "$SESSION" ]; then
    echo "[*] Cleaning up: Stopping Colab session '$SESSION' to release compute..."
    "$CLI_TIMEOUT_PYTHON" scripts/colab/cli_timeout.py 30 colab stop -s "$SESSION" >/dev/null 2>&1
    STOP_RC=$?
    if [ $STOP_RC -eq 0 ]; then
      echo "[+] Colab session '$SESSION' stopped."
    else
      echo "[-] Failed to stop Colab session '$SESSION' (rc=$STOP_RC). Remediation: colab stop -s $SESSION" >&2
    fi
  fi

  # Remove only this run's temp files; preserve all original .env files.
  rm -rf "$TMPDIR_PRIVATE"

  # Exit precedence: primary nonzero wins; otherwise failed stop=70;
  # otherwise incomplete required recovery=74; otherwise 0.
  if [ "$PRIMARY_RC" -ne 0 ]; then
    if [ "$STOP_RC" -ne 0 ]; then
      echo "[!] Secondary stop failure (rc=$STOP_RC) for session '$SESSION' preserved alongside primary rc=$PRIMARY_RC." >&2
    fi
    exit "$PRIMARY_RC"
  fi
  if [ "$STOP_RC" -ne 0 ]; then
    exit 70
  fi
  # Primary success requires local manifest/log plus either archive or both
  # submission files. This is a local handoff requirement, not proof the
  # archive contains valid model weights; downstream verification must
  # inspect it.
  if [ "$EXEC_ATTEMPTED" -eq 1 ]; then
    local has_manifest=0 has_log=0 has_archive=0 has_zip=0 has_json=0
    [ -s "$RECOVERY_DIR/run_manifest.json" ] && has_manifest=1
    [ -s "$RECOVERY_DIR/training.log" ] && has_log=1
    [ -s "$RECOVERY_DIR/recovery.tar.gz" ] && has_archive=1
    [ -s "$RECOVERY_DIR/submission.zip" ] && has_zip=1
    [ -s "$RECOVERY_DIR/submission.json" ] && has_json=1
    if [ "$has_manifest" -eq 0 ] || [ "$has_log" -eq 0 ]; then
      echo "[!] Incomplete recovery in $RECOVERY_DIR: manifest/log required." >&2
      exit 74
    fi
    if [ "$has_archive" -eq 0 ] && { [ "$has_zip" -eq 0 ] || [ "$has_json" -eq 0 ]; }; then
      echo "[!] Incomplete recovery in $RECOVERY_DIR: need archive or both submission files." >&2
      exit 74
    fi
  fi
  exit 0
}

trap 'on_int' INT
trap 'on_term' TERM
trap 'cleanup "$?"' EXIT

# --- Filtered env (allowlisted, original preserved) ---
LOCAL_ENV="${REPO_ROOT}/.env"
if [ -f "$LOCAL_ENV" ]; then
  grep -E '^(HF_TOKEN|HF_TOKEN_WRITE|HF_TOKEN_READ|KAGGLE_API_TOKEN|KAGGLE_KEY|KAGGLE_USERNAME|HF_REPO_ID|HF_ALLOW_PUBLIC_REPO|LEGALIR_COMMIT_SHA)=' "$LOCAL_ENV" > "$FILTERED_ENV" || true
else
  : > "$FILTERED_ENV"
fi

# --- Session and notebook prep ---
SESSION="legalir-a100-production-$(head -c 4 /dev/urandom | xxd -p)"
RECOVERY_DIR="artifacts/task1/production/$SESSION"
NEW_FLAGS=("--gpu" "A100")

EXPECTED_SHA="${LEGALIR_COMMIT_SHA:-$(git rev-parse HEAD)}"
echo "Running local provenance preflight for A100..."
set +e
run_fg "$PYTHON_BIN" scripts/colab/bootstrap.py --expected-sha "$EXPECTED_SHA"
preflight_rc=$?
set -e
if [ $preflight_rc -ne 0 ]; then
  echo "[!] Preflight failed (rc=$preflight_rc). Aborting before Colab allocation." >&2
  PRIMARY_RC=$preflight_rc
  # No allocation: cleanup will remove temp files and exit with PRIMARY_RC.
  exit "$PRIMARY_RC"
fi
echo "{\"expected_sha\": \"$EXPECTED_SHA\"}" > "$LAUNCH_JSON"

echo "================================================================="
echo "LegalIR Google Colab CLI Automation"
echo "  • Target Mode : $GPU_MODE"
echo "  • Session     : $SESSION"
echo "  • Notebook    : $NOTEBOOK"
echo "  • Exec wall clock : ${TIMEOUT}s total (plus setup/cleanup; not a billing cap)"
echo "================================================================="

# Private notebook copy to avoid dirtying the repo.
cp "$NOTEBOOK" "$TMP_NOTEBOOK"

# 1. Allocate VM with an explicit local deadline (track attempt before call for
# ambiguous-failure stop: a hang may mean server-side success).
echo "[1/4] Allocating Colab VM with ${NEW_FLAGS[*]} (deadline ${NEW_TIMEOUT}s)..."
ALLOC_ATTEMPTED=1
set +e
run_fg "$CLI_TIMEOUT_PYTHON" scripts/colab/cli_timeout.py "$NEW_TIMEOUT" colab new -s "$SESSION" "${NEW_FLAGS[@]}"
new_rc=$?
set -e
if [ $new_rc -ne 0 ]; then
  if [ $new_rc -eq 124 ]; then
    echo "[!] Colab allocation timed out after ${NEW_TIMEOUT}s for session '$SESSION'." >&2
  else
    echo "[!] Colab allocation failed (rc=$new_rc) for session '$SESSION'." >&2
  fi
  PRIMARY_RC=$new_rc
  exit "$PRIMARY_RC"
fi

# 2. Upload required inputs with explicit deadlines (failures block exec; no
# silent fallback).
if [ -s "$FILTERED_ENV" ]; then
  echo "[2/4] Uploading filtered local .env to /content/.env (deadline ${UPLOAD_TIMEOUT}s)..."
  set +e
  run_fg "$CLI_TIMEOUT_PYTHON" scripts/colab/cli_timeout.py "$UPLOAD_TIMEOUT" colab upload "$FILTERED_ENV" /content/.env -s "$SESSION"
  up_rc=$?
  set -e
  if [ $up_rc -ne 0 ]; then
    echo "[!] Required filtered .env upload failed (rc=$up_rc)." >&2
    PRIMARY_RC=$up_rc
    exit "$PRIMARY_RC"
  fi
else
  echo "[!] Warning: No relevant secrets found in local .env. Secrets must be configured in Colab Secrets."
fi

echo "[2/4] Uploading launch config (deadline ${UPLOAD_TIMEOUT}s)..."
set +e
run_fg "$CLI_TIMEOUT_PYTHON" scripts/colab/cli_timeout.py "$UPLOAD_TIMEOUT" colab upload "$LAUNCH_JSON" /content/legalir_launch.json -s "$SESSION"
up_rc=$?
set -e
if [ $up_rc -ne 0 ]; then
  echo "[!] Required launch JSON upload failed (rc=$up_rc)." >&2
  PRIMARY_RC=$up_rc
  exit "$PRIMARY_RC"
fi

# Gate/freeze files: if present locally, they override repo copies and their
# upload is required (failure must not silently fall back to other evidence).
for f in artifacts/task1/gates/kaggle_t4x2_report.json artifacts/task1/freeze/production_freeze.json; do
  if [ -f "$f" ]; then
    echo "[2/4] Uploading $f (deadline ${UPLOAD_TIMEOUT}s)..."
    set +e
    run_fg "$CLI_TIMEOUT_PYTHON" scripts/colab/cli_timeout.py "$UPLOAD_TIMEOUT" colab upload "$f" "/content/$(basename "$f")" -s "$SESSION"
    up_rc=$?
    set -e
    if [ $up_rc -ne 0 ]; then
      echo "[!] Required upload failed for $f (rc=$up_rc); refusing to fall back to different evidence." >&2
      PRIMARY_RC=$up_rc
      exit "$PRIMARY_RC"
    fi
  fi
done

# 3. Execute notebook. COLAB_TIMEOUT is BOTH the whole-notebook wall-clock bound
# (outer helper, kills an overrunning client) and the per-cell bound passed to
# `colab exec --timeout` (the CLI applies it separately to each cell). A slow
# multi-cell notebook therefore trips the outer deadline even when every
# individual cell is below it. Neither bound is a billing cap.
echo "[3/4] Executing $NOTEBOOK on remote Colab VM (wall clock ${TIMEOUT}s)..."
EXEC_ATTEMPTED=1
set +e
run_fg "$CLI_TIMEOUT_PYTHON" scripts/colab/cli_timeout.py "$TIMEOUT" colab exec -s "$SESSION" -f "$TMP_NOTEBOOK" --timeout "$TIMEOUT"
EXEC_RC=$?
set -e
if [ $EXEC_RC -ne 0 ]; then
  if [ $EXEC_RC -eq 124 ]; then
    echo "[!] Remote notebook execution exceeded the ${TIMEOUT}s whole-notebook wall clock." >&2
  else
    echo "[!] Remote notebook execution failed (exit code $EXEC_RC)." >&2
  fi
  PRIMARY_RC=$EXEC_RC
  exit "$PRIMARY_RC"
fi

echo "[+] Remote notebook execution completed successfully."
# Normal return triggers EXIT trap which performs bounded recovery + stop.
