"""Detach supervision contract: attached is default, --detach is explicit opt-in."""
import os
import shutil
import subprocess
from pathlib import Path

REAL_WRAPPER = Path("scripts/modal/run_modal_cli.sh").resolve()


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "scripts/modal").mkdir(parents=True)
    (repo / "scripts/colab").mkdir(parents=True)
    shutil.copy2(REAL_WRAPPER, repo / "scripts/modal/run_modal_cli.sh")
    (repo / "scripts/modal/run_modal_cli.sh").chmod(0o755)
    return repo


def _git(repo: Path, *args: str):
    env = os.environ.copy()
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    return subprocess.check_output(["git", *args], cwd=str(repo), env=env, text=True).strip()


def _init_git_repo(repo: Path):
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t.t")
    _git(repo, "config", "user.name", "t")
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "init")


def _write_fakes(bin_dir: Path):
    bin_dir.mkdir(parents=True, exist_ok=True)
    log = bin_dir / "modal.log"
    py_stub = bin_dir / "py_stub"
    modal_fake = bin_dir / "modal"
    py_stub.write_text(f"#!/bin/sh\necho \"$@\" >> \"{bin_dir / 'py.log'}\"\nexit 0\n", encoding="utf-8")
    py_stub.chmod(0o755)
    modal_fake.write_text(
        f"""#!/bin/sh
echo "$@" >> "{log}"
echo "ENV_SHA:$LEGALIR_COMMIT_SHA" >> "{log}"
exit 0
""",
        encoding="utf-8",
    )
    modal_fake.chmod(0o755)
    return py_stub, modal_fake, log


def _run(repo: Path, bin_dir: Path, args=()):
    env = os.environ.copy()
    py_stub, modal_fake, log = _write_fakes(bin_dir)
    env["PYTHON_BIN"] = str(py_stub)
    env["MODAL_BIN"] = str(modal_fake)
    env.pop("LEGALIR_COMMIT_SHA", None)
    res = subprocess.run(
        [str(repo / "scripts/modal/run_modal_cli.sh"), *args],
        cwd=str(repo), env=env, capture_output=True, text=True, timeout=20,
    )
    return res, log


def test_default_is_attached_without_detach_flag(tmp_path):
    repo = _make_repo(tmp_path)
    _init_git_repo(repo)
    res, log = _run(repo, tmp_path / "bin")
    assert res.returncode == 0
    text = log.read_text(encoding="utf-8")
    assert "--detach" not in text
    combined = (res.stdout or "") + (res.stderr or "")
    assert "ATTACHED mode" in combined
    assert "terminates remote tasks" in combined


def test_detach_is_explicit_opt_in(tmp_path):
    repo = _make_repo(tmp_path)
    _init_git_repo(repo)
    res, log = _run(repo, tmp_path / "bin", args=("--detach",))
    assert res.returncode == 0
    text = log.read_text(encoding="utf-8")
    assert "--detach" in text
    combined = (res.stdout or "") + (res.stderr or "")
    assert "DETACHED mode" in combined
    assert "spending supervision" in combined


def test_duplicate_detach_rejected(tmp_path):
    repo = _make_repo(tmp_path)
    _init_git_repo(repo)
    res, _ = _run(repo, tmp_path / "bin", args=("--detach", "--detach"))
    assert res.returncode == 2
