"""Contract tests for scripts/modal/run_modal_cli.sh (fixture git repos, fake modal)."""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REAL_WRAPPER = Path("scripts/modal/run_modal_cli.sh").resolve()


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "scripts/modal").mkdir(parents=True)
    (repo / "scripts/colab").mkdir(parents=True)
    shutil.copy2(REAL_WRAPPER, repo / "scripts/modal/run_modal_cli.sh")
    (repo / "scripts/modal/run_modal_cli.sh").chmod(0o755)
    return repo


def _git(repo: Path, *args: str, env_extra: dict | None = None):
    env = os.environ.copy()
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    if env_extra:
        env.update(env_extra)
    return subprocess.check_output(["git", *args], cwd=str(repo), env=env, text=True).strip()


def _init_git_repo(repo: Path):
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t.t")
    _git(repo, "config", "user.name", "t")
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "init")


def _write_fakes(bin_dir: Path, opts: dict):
    bin_dir.mkdir(parents=True, exist_ok=True)
    log = bin_dir / "modal.log"
    py_stub = bin_dir / "py_stub"
    modal_fake = bin_dir / "modal"
    pre_rc = opts.get("PRE_RC", "0")
    py_stub.write_text(f"#!/bin/sh\necho \"$@\" >> \"{bin_dir / 'py.log'}\"\nexit {pre_rc}\n", encoding="utf-8")
    py_stub.chmod(0o755)
    # Fake modal records invocation and env; never touches the cloud.
    modal_fake.write_text(
        f"""#!/bin/sh
echo "$@" >> "{log}"
echo "ENV_SHA:$LEGALIR_COMMIT_SHA" >> "{log}"
exit {opts.get('MODAL_RC', '0')}
""",
        encoding="utf-8",
    )
    modal_fake.chmod(0o755)
    return py_stub, modal_fake, log


def _run(repo: Path, bin_dir: Path, args=(), extra_env=None):
    env = os.environ.copy()
    py_stub, modal_fake, log = _write_fakes(bin_dir, extra_env or {})
    # Override to fixture fakes.
    env["PYTHON_BIN"] = str(py_stub)
    env["MODAL_BIN"] = str(modal_fake)
    env["GIT_CEILING_DIRECTORIES"] = str(repo.parent)
    if extra_env and "LEGALIR_COMMIT_SHA" in extra_env:
        env["LEGALIR_COMMIT_SHA"] = extra_env["LEGALIR_COMMIT_SHA"]
    else:
        env.pop("LEGALIR_COMMIT_SHA", None)
    # Ensure possiblereal LEGALIR_COMMIT_SHA from outer env doesn't leak.
    script = repo / "scripts/modal/run_modal_cli.sh"
    res = subprocess.run([str(script), *args], cwd=str(repo), env=env, capture_output=True, text=True, timeout=20)
    return res, log


def test_preflight_failure_never_invokes_modal(tmp_path):
    repo = _make_repo(tmp_path)
    _init_git_repo(repo)
    bin_dir = tmp_path / "bin"
    res, log = _run(repo, bin_dir, extra_env={"PRE_RC": "9"})
    assert res.returncode != 0
    assert not log.is_file() or "run" not in log.read_text(encoding="utf-8")


def test_dirty_tree_fails_before_dispatch(tmp_path):
    repo = _make_repo(tmp_path)
    _init_git_repo(repo)
    (repo / "dirty.txt").write_text("untracked\n", encoding="utf-8")
    bin_dir = tmp_path / "bin"
    res, log = _run(repo, bin_dir)
    assert res.returncode == 2
    assert not log.is_file() or "run" not in log.read_text(encoding="utf-8")


def test_invalid_sha_fails_before_dispatch(tmp_path):
    repo = _make_repo(tmp_path)
    _init_git_repo(repo)
    bin_dir = tmp_path / "bin"
    res, log = _run(repo, bin_dir, extra_env={"PRE_RC": "0", "LEGALIR_COMMIT_SHA": "not-a-sha"})
    # Wrapper validates SHA before cloud; env plumbing passes explicit bad SHA.
    # _run maps LEGALIR_COMMIT_SHA via extra_env; ensure wrapper rejects.
    assert res.returncode == 2
    assert not log.is_file() or "run" not in log.read_text(encoding="utf-8")


def test_sha_mismatch_rejected(tmp_path):
    repo = _make_repo(tmp_path)
    _init_git_repo(repo)
    head = _git(repo, "rev-parse", "HEAD")
    other = "0" * 40
    assert other != head
    bin_dir = tmp_path / "bin"
    res, log = _run(repo, bin_dir, extra_env={"LEGALIR_COMMIT_SHA": other})
    assert res.returncode == 2
    assert "does not match local HEAD" in (res.stderr or "")


def test_absent_consent_forwards_no_public_once(tmp_path):
    repo = _make_repo(tmp_path)
    _init_git_repo(repo)
    bin_dir = tmp_path / "bin"
    res, log = _run(repo, bin_dir)
    assert res.returncode == 0
    text = log.read_text(encoding="utf-8")
    assert "--no-hf-allow-public-repo" in text
    assert text.count("--hf-allow-public-repo") == 0 or "--no-hf-allow-public-repo" in text
    # Forwarded once.
    assert text.count("--no-hf-allow-public-repo") == 1
    assert "--hf-allow-public-repo" not in text.replace("--no-hf-allow-public-repo", "")


def test_explicit_consent_forwarded_once(tmp_path):
    repo = _make_repo(tmp_path)
    _init_git_repo(repo)
    bin_dir = tmp_path / "bin"
    res, log = _run(repo, bin_dir, args=("--hf-allow-public-repo",))
    assert res.returncode == 0
    text = log.read_text(encoding="utf-8")
    # Explicit flag appears once, not duplicated, and no --no- variant.
    assert "--no-hf-allow-public-repo" not in text
    assert text.count("--hf-allow-public-repo") == 1
