"""Offline dispatch contracts for the Windows/Linux Python entrypoint."""
import subprocess

import pytest

from scripts.modal import launch


@pytest.fixture
def cli(monkeypatch):
    import scripts.colab.bootstrap as boot

    monkeypatch.delenv("LEGALIR_COMMIT_SHA", raising=False)
    monkeypatch.delenv("MODAL_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("LEGALIR_TIME_GATE_SECONDS", raising=False)
    monkeypatch.delenv("LEGALIR_BYPASS_T4_GATE", raising=False)
    monkeypatch.setattr(subprocess, "check_output", lambda cmd, **kw: "a" * 40 if "rev-parse" in cmd else "")
    monkeypatch.setattr(boot, "verify_launch", lambda *a, **kw: {})
    calls = []

    def run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", run)
    return calls


def test_check_only_never_dispatches(cli):
    assert launch.main(["--check-only"]) == 0
    assert cli == []


def test_private_dispatch_pins_profile_environment_and_timeout(cli, monkeypatch):
    monkeypatch.setenv("MODAL_TIMEOUT_SECONDS", "28800")
    assert launch.main(["--private", "--detach"]) == 0
    cmd, kwargs = cli[0]
    assert cmd == [launch.sys.executable, "-m", "modal", "run", "--profile", "zunuoivalutre",
                   "--env", "main", "--detach", "scripts/modal/run_modal_a100.py",
                   "--no-hf-allow-public-repo", "--private"]
    assert kwargs["env"]["LEGALIR_TIME_GATE_SECONDS"] == "28800"
    assert kwargs["env"]["LEGALIR_COMMIT_SHA"] == "a" * 40


def test_gate_failure_never_dispatches(cli, monkeypatch):
    import scripts.colab.bootstrap as boot

    def fail(*args, **kwargs):
        raise RuntimeError("stale evidence")

    monkeypatch.setattr(boot, "verify_launch", fail)
    assert launch.main([]) == 1
    assert cli == []


def test_dirty_tree_never_dispatches(cli, monkeypatch):
    monkeypatch.setattr(subprocess, "check_output", lambda cmd, **kw: "a" * 40 if "rev-parse" in cmd else " M runtime.py")
    assert launch.main([]) == 1
    assert cli == []


def test_timeout_mismatch_never_dispatches(cli, monkeypatch):
    monkeypatch.setenv("LEGALIR_TIME_GATE_SECONDS", "18000")
    assert launch.main([]) == 1
    assert cli == []


def test_bypass_flag_forwarded_explicitly(cli):
    assert launch.main(["--private", "--bypass-t4-gate", "--detach"]) == 0
    cmd, kwargs = cli[0]
    assert "--bypass-t4-gate" in cmd
    assert kwargs["env"]["LEGALIR_BYPASS_T4_GATE"] == "1"


def test_default_dispatch_has_no_bypass(cli):
    assert launch.main(["--private"]) == 0
    cmd, kwargs = cli[0]
    assert "--bypass-t4-gate" not in cmd
    assert "LEGALIR_BYPASS_T4_GATE" not in kwargs["env"]
