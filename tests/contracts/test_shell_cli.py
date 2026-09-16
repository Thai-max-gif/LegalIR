"""Isolated shell CLI regressions: fixture repos only, never real .env or VMs."""
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

REAL_WRAPPER = Path("scripts/colab/run_colab_cli.sh").resolve()
REAL_HELPER = Path("scripts/colab/cli_timeout.py").resolve()
REAL_PYTHON = Path(".venv/bin/python").resolve()
if not REAL_PYTHON.is_file():
    REAL_PYTHON = Path(sys.executable)


def _make_fixture_repo(tmp_path: Path) -> Path:
    """Copy the real wrapper+helper into a temp repo-shaped dir."""
    repo = tmp_path / "repo"
    (repo / "scripts/colab").mkdir(parents=True)
    (repo / "notebooks").mkdir(parents=True)
    shutil.copy2(REAL_WRAPPER, repo / "scripts/colab/run_colab_cli.sh")
    shutil.copy2(REAL_HELPER, repo / "scripts/colab/cli_timeout.py")
    (repo / "notebooks/colab_a100_train.ipynb").write_text('{"cells":[]}', encoding="utf-8")
    return repo


def _write_stubs(bin_dir: Path, log_file: Path, capture_dir: Path, opts: dict):
    """Fake `colab` records calls, captures filtered env during upload, simulates RCs."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    capture_dir.mkdir(parents=True, exist_ok=True)
    # opts via env files: write a config shell snippet sourced by fake.
    cfg = bin_dir / "fake_cfg.sh"
    lines = []
    for k, v in opts.items():
        # Escape single quotes.
        vv = str(v).replace("'", "'\"'\"'")
        lines.append(f"{k}='{vv}'")
    cfg.write_text("\n".join(lines) + "\n", encoding="utf-8")
    fake = bin_dir / "colab"
    fake.write_text(
        f"""#!/bin/sh
CFG="{cfg}"
LOG="{log_file}"
CAP="{capture_dir}"
[ -f "$CFG" ] && . "$CFG"
echo "$@" >> "$LOG"
cmd="$1"
if [ "$cmd" = "new" ]; then
  exit ${{FAKE_NEW_RC:-0}}
fi
if [ "$cmd" = "upload" ]; then
  src="$2"
  # Capture filtered env content during upload, before cleanup removes it.
  # Only .env.filtered proves filtering; launch JSON and gate files must not
  # overwrite that evidence.
  case "$src" in
    *.env.filtered)
      if [ -f "$src" ]; then
        cp "$src" "$CAP/filtered_capture.txt" 2>/dev/null || true
      fi
      ;;
  esac
  echo "UPLOAD_SRC:$src" >> "$LOG"
  # Fail specific uploads if requested (substring match on src or dest).
  case "$*" in
    *kaggle_t4x2_report*|*production_freeze*|*legalir_launch*)
      if [ -n "${{FAKE_UPLOAD_REQ_RC:-}}" ]; then exit "$FAKE_UPLOAD_REQ_RC"; fi
      ;;
  esac
  if [ -n "${{FAKE_UPLOAD_RC:-}}" ]; then exit "$FAKE_UPLOAD_RC"; fi
  exit 0
fi
if [ "$cmd" = "exec" ]; then
  if [ -n "${{FAKE_EXEC_SLEEP:-}}" ]; then sleep "$FAKE_EXEC_SLEEP"; fi
  exit ${{FAKE_EXEC_RC:-0}}
fi
if [ "$cmd" = "download" ]; then
  remote="$2"
  local_path="$3"
  base="$(basename "$remote")"
  # Hang simulation for timeout tests.
  case "$FAKE_DOWNLOAD_HANG" in
    *"$base"*) sleep 30 ;;
  esac
  if [ -n "$FAKE_DOWNLOAD_FAIL" ]; then
    case "$FAKE_DOWNLOAD_FAIL" in
      *"$base"*) exit 1 ;;
    esac
  fi
  mkdir -p "$(dirname "$local_path")" 2>/dev/null || true
  # Write non-empty content unless explicitly empty.
  case "$FAKE_DOWNLOAD_EMPTY" in
    *"$base"*) : > "$local_path" ;;
    *) echo "dummy-$base" > "$local_path" ;;
  esac
  exit 0
fi
if [ "$cmd" = "stop" ]; then
  if [ -n "${{FAKE_STOP_HANG:-}}" ]; then sleep 30; fi
  exit ${{FAKE_STOP_RC:-0}}
fi
exit 0
""",
        encoding="utf-8",
    )
    fake.chmod(0o755)
    # PYTHON_BIN stub for preflight.
    py_stub = bin_dir / "py_stub"
    pre_rc = opts.get("FAKE_PREFLIGHT_RC", "0")
    py_stub.write_text(f"#!/bin/sh\nexit {pre_rc}\n", encoding="utf-8")
    py_stub.chmod(0o755)
    return fake, py_stub


def _run_wrapper(repo: Path, bin_dir: Path, extra_env: dict | None = None, args=("A100",)):
    env = os.environ.copy()
    env["PATH"] = str(bin_dir) + ":" + env.get("PATH", "")
    env["PYTHON_BIN"] = str(bin_dir / "py_stub")
    env["CLI_TIMEOUT_PYTHON"] = str(REAL_PYTHON)
    env["LEGALIR_COMMIT_SHA"] = "a" * 40
    # Isolate git: avoid reading real repo state.
    env["GIT_CEILING_DIRECTORIES"] = str(repo.parent)
    if extra_env:
        env.update(extra_env)
    script = repo / "scripts/colab/run_colab_cli.sh"
    res = subprocess.run(
        [str(script), *args],
        cwd=str(repo),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return res


def _log_text(log_file: Path) -> str:
    return log_file.read_text(encoding="utf-8") if log_file.is_file() else ""


def test_preflight_failure_blocks_allocation_and_cleans(tmp_path):
    repo = _make_fixture_repo(tmp_path)
    bin_dir = tmp_path / "bin"
    log = tmp_path / "colab.log"
    cap = tmp_path / "cap"
    _write_stubs(bin_dir, log, cap, {"FAKE_PREFLIGHT_RC": "9"})
    res = _run_wrapper(repo, bin_dir)
    assert res.returncode == 9
    text = _log_text(log)
    assert "new" not in text
    assert "upload" not in text
    assert "exec" not in text
    # No repo-root temp leftovers (old unsafe .env.filtered/legalir_launch.json).
    assert not (repo / ".env.filtered").exists()
    assert not (repo / "legalir_launch.json").exists()
    assert not (repo / "notebooks/colab_a100_train.ipynb.tmp.ipynb").exists()


def test_allocation_failure_preserves_code_and_attempts_stop(tmp_path):
    repo = _make_fixture_repo(tmp_path)
    bin_dir = tmp_path / "bin"
    log = tmp_path / "colab.log"
    cap = tmp_path / "cap"
    _write_stubs(bin_dir, log, cap, {"FAKE_NEW_RC": "23"})
    res = _run_wrapper(repo, bin_dir)
    assert res.returncode == 23
    text = _log_text(log)
    assert "new" in text
    assert "upload" not in text
    assert "exec" not in text
    assert "stop -s" in text


def test_required_upload_failure_blocks_exec(tmp_path):
    repo = _make_fixture_repo(tmp_path)
    bin_dir = tmp_path / "bin"
    log = tmp_path / "colab.log"
    cap = tmp_path / "cap"
    _write_stubs(bin_dir, log, cap, {"FAKE_UPLOAD_RC": "24"})
    res = _run_wrapper(repo, bin_dir)
    assert res.returncode == 24
    text = _log_text(log)
    assert "exec" not in text
    assert "stop -s" in text


def test_exec_failure_preserves_code_and_stops_once(tmp_path):
    repo = _make_fixture_repo(tmp_path)
    bin_dir = tmp_path / "bin"
    log = tmp_path / "colab.log"
    cap = tmp_path / "cap"
    _write_stubs(bin_dir, log, cap, {"FAKE_EXEC_RC": "42"})
    res = _run_wrapper(repo, bin_dir)
    assert res.returncode == 42
    text = _log_text(log)
    assert text.count("stop -s") == 1


def test_stop_failure_after_success_is_70_with_session(tmp_path):
    repo = _make_fixture_repo(tmp_path)
    bin_dir = tmp_path / "bin"
    log = tmp_path / "colab.log"
    cap = tmp_path / "cap"
    _write_stubs(bin_dir, log, cap, {"FAKE_EXEC_RC": "0", "FAKE_STOP_RC": "1"})
    res = _run_wrapper(repo, bin_dir)
    assert res.returncode == 70
    combined = (res.stdout or "") + (res.stderr or "")
    assert "colab stop -s" in combined
    assert "legalir-a100-production-" in combined


def test_exec_failure_plus_stop_failure_preserves_primary(tmp_path):
    repo = _make_fixture_repo(tmp_path)
    bin_dir = tmp_path / "bin"
    log = tmp_path / "colab.log"
    cap = tmp_path / "cap"
    _write_stubs(bin_dir, log, cap, {"FAKE_EXEC_RC": "42", "FAKE_STOP_RC": "1"})
    res = _run_wrapper(repo, bin_dir)
    assert res.returncode == 42
    combined = (res.stdout or "") + (res.stderr or "")
    assert "stop" in combined.lower()


def test_download_hang_does_not_block_stop(tmp_path):
    repo = _make_fixture_repo(tmp_path)
    bin_dir = tmp_path / "bin"
    log = tmp_path / "colab.log"
    cap = tmp_path / "cap"
    # Hang the archive download (45s budget in prod, but helper enforced).
    # Use 15s manifest timeout path? Archive hangs 30s, helper 45s would wait
    # 30s; to keep test fast, hang manifest (15s budget) instead.
    _write_stubs(bin_dir, log, cap, {"FAKE_EXEC_RC": "0", "FAKE_DOWNLOAD_HANG": "run_manifest.json"})
    start = time.monotonic()
    res = _run_wrapper(repo, bin_dir)
    elapsed = time.monotonic() - start
    # Manifest hang times out (15s) but remaining downloads+stop still run.
    # Total bounded: 15 (manifest timeout) + 15+45+15+15 + 30 stop ≈ 135s worst,
    # but with one hang it should still finish and attempt stop.
    assert "stop -s" in _log_text(log)
    # Should not hang indefinitely; allow generous upper bound for CI.
    assert elapsed < 120
    # Primary was success but manifest missing → 74 unless stop failed.
    assert res.returncode in (74, 70)


def test_stop_hang_is_bounded_to_70(tmp_path):
    repo = _make_fixture_repo(tmp_path)
    bin_dir = tmp_path / "bin"
    log = tmp_path / "colab.log"
    cap = tmp_path / "cap"
    _write_stubs(bin_dir, log, cap, {"FAKE_EXEC_RC": "0", "FAKE_STOP_HANG": "1"})
    start = time.monotonic()
    res = _run_wrapper(repo, bin_dir)
    elapsed = time.monotonic() - start
    assert res.returncode == 70
    assert elapsed < 60
    assert "colab stop -s" in ((res.stdout or "") + (res.stderr or ""))


def test_successful_recovery_uses_session_dir(tmp_path):
    repo = _make_fixture_repo(tmp_path)
    bin_dir = tmp_path / "bin"
    log = tmp_path / "colab.log"
    cap = tmp_path / "cap"
    _write_stubs(bin_dir, log, cap, {"FAKE_EXEC_RC": "0"})
    res = _run_wrapper(repo, bin_dir)
    assert res.returncode == 0
    prod = repo / "artifacts/task1/production"
    assert prod.is_dir()
    subdirs = [p for p in prod.iterdir() if p.is_dir()]
    assert len(subdirs) == 1
    sess_dir = subdirs[0]
    assert sess_dir.name.startswith("legalir-a100-production-")
    assert (sess_dir / "run_manifest.json").is_file()
    assert (sess_dir / "training.log").is_file()
    # No shared filenames overwritten at top level.
    assert not (prod / "recovery.tar.gz").is_file()


def test_success_but_missing_recovery_is_74(tmp_path):
    repo = _make_fixture_repo(tmp_path)
    bin_dir = tmp_path / "bin"
    log = tmp_path / "colab.log"
    cap = tmp_path / "cap"
    # Fail archive and both submission files → incomplete.
    _write_stubs(
        bin_dir,
        log,
        cap,
        {"FAKE_EXEC_RC": "0", "FAKE_DOWNLOAD_FAIL": "recovery.tar.gz submission.zip submission.json"},
    )
    res = _run_wrapper(repo, bin_dir)
    assert res.returncode == 74


def test_rejects_retired_t4_without_allocation(tmp_path):
    repo = _make_fixture_repo(tmp_path)
    bin_dir = tmp_path / "bin"
    log = tmp_path / "colab.log"
    cap = tmp_path / "cap"
    _write_stubs(bin_dir, log, cap, {})
    res = _run_wrapper(repo, bin_dir, args=("T4",))
    assert res.returncode == 2
    assert "new" not in _log_text(log)


def test_env_filtering_captured_during_upload(tmp_path):
    repo = _make_fixture_repo(tmp_path)
    (repo / ".env").write_text(
        "HF_TOKEN_WRITE=hf_test\nSECRET_KEY=bad\nHF_ALLOW_PUBLIC_REPO=1\n", encoding="utf-8"
    )
    bin_dir = tmp_path / "bin"
    log = tmp_path / "colab.log"
    cap = tmp_path / "cap"
    _write_stubs(bin_dir, log, cap, {"FAKE_EXEC_RC": "0"})
    res = _run_wrapper(repo, bin_dir)
    assert res.returncode == 0
    captured = (cap / "filtered_capture.txt").read_text(encoding="utf-8") if (cap / "filtered_capture.txt").is_file() else ""
    assert "HF_TOKEN_WRITE" in captured
    assert "HF_ALLOW_PUBLIC_REPO" in captured
    assert "SECRET_KEY" not in captured
    # Original preserved, no repo-root filtered leftovers.
    assert "SECRET_KEY" in (repo / ".env").read_text(encoding="utf-8")
    assert not (repo / ".env.filtered").exists()
    # Private temp removed (no leak under /tmp from this run is asserted via
    # absence of repo leftovers; TMPDIR itself is mktemp-managed).


def test_colab_timeout_validation(tmp_path):
    repo = _make_fixture_repo(tmp_path)
    bin_dir = tmp_path / "bin"
    log = tmp_path / "colab.log"
    cap = tmp_path / "cap"
    _write_stubs(bin_dir, log, cap, {})
    res = _run_wrapper(repo, bin_dir, extra_env={"COLAB_TIMEOUT": "not-a-number"})
    assert res.returncode == 2
    assert "new" not in _log_text(log)


def _run_wrapper_signal(tmp_path, sig):
    import os as _os

    repo = _make_fixture_repo(tmp_path)
    bin_dir = tmp_path / "bin"
    log = tmp_path / "colab.log"
    cap = tmp_path / "cap"
    _write_stubs(bin_dir, log, cap, {"FAKE_EXEC_SLEEP": "10", "FAKE_EXEC_RC": "0"})
    env = _os.environ.copy()
    env["PATH"] = str(bin_dir) + ":" + env.get("PATH", "")
    env["PYTHON_BIN"] = str(bin_dir / "py_stub")
    env["CLI_TIMEOUT_PYTHON"] = str(REAL_PYTHON)
    env["LEGALIR_COMMIT_SHA"] = "a" * 40
    env["GIT_CEILING_DIRECTORIES"] = str(repo.parent)
    script = repo / "scripts/colab/run_colab_cli.sh"
    proc = subprocess.Popen(
        [str(script), "A100"], cwd=str(repo), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        start_new_session=True,
    )
    try:
        time.sleep(1.5)
        # Send signal to the wrapper process group so child exec also sees it,
        # but helper timeouts still bound downloads/stop.
        try:
            _os.killpg(proc.pid, sig)
        except ProcessLookupError:
            pass
        try:
            out, err = proc.communicate(timeout=60)
        except subprocess.TimeoutExpired:
            try:
                _os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            out, err = proc.communicate()
        rc = proc.returncode
    finally:
        # Ensure no leaked fixture processes.
        try:
            _os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            pass
    return rc, _log_text(log), repo


def test_sigint_cleanup_once_130(tmp_path):
    rc, text, repo = _run_wrapper_signal(tmp_path, signal.SIGINT)
    assert rc == 130
    assert text.count("stop -s") == 1
    assert not (repo / ".env.filtered").exists()


def test_sigterm_cleanup_once_143(tmp_path):
    rc, text, repo = _run_wrapper_signal(tmp_path, signal.SIGTERM)
    assert rc == 143
    assert text.count("stop -s") == 1
    assert not (repo / ".env.filtered").exists()
