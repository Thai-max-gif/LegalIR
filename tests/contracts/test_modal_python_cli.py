"""Offline dispatch contracts for the Windows/Linux Python entrypoint."""
import subprocess
import os

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


@pytest.mark.parametrize("original", [None, "0"])
@pytest.mark.parametrize("fails", [False, True])
def test_bypass_applied_before_preflight_and_restored(cli, monkeypatch, original, fails):
    import scripts.colab.bootstrap as boot

    if original is not None:
        monkeypatch.setenv("LEGALIR_BYPASS_T4_GATE", original)

    def verify(*args, **kwargs):
        assert os.environ.get("LEGALIR_BYPASS_T4_GATE") == "1"
        if fails:
            raise RuntimeError("config mismatch")

    monkeypatch.setattr(boot, "verify_launch", verify)
    assert launch.main(["--check-only", "--bypass-t4-gate"]) == (1 if fails else 0)
    assert os.environ.get("LEGALIR_BYPASS_T4_GATE") == original
    assert cli == []


def test_real_validator_bypass_still_checks_config(monkeypatch, tmp_path):
    import json
    from src.release.fingerprints import fingerprint_structured_config

    # Real validator and JSON/hash checks; only Git subprocesses are stubbed.
    root = tmp_path
    config = root / "configs/algorithm/legalir_v2.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("training: {}\n", encoding="utf-8")
    freeze_path = root / "artifacts/task1/freeze/production_freeze.json"
    freeze_path.parent.mkdir(parents=True)
    report = root / "artifacts/task1/gates/kaggle_t4x2_report.json"
    report.parent.mkdir(parents=True)
    report.write_text("{}", encoding="utf-8")
    freeze = {"git_sha": "b" * 40,
              "algorithm_config_sha256": fingerprint_structured_config(config),
              "dataset": {"manifest_sha256": "fixture"}}
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
    monkeypatch.setattr(launch, "REPO_ROOT", root)
    for key in ("LEGALIR_BYPASS_T4_GATE", "LEGALIR_COMMIT_SHA",
                "MODAL_TIMEOUT_SECONDS", "LEGALIR_TIME_GATE_SECONDS"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(subprocess, "check_output",
                        lambda cmd, **kw: "a" * 40 if "rev-parse" in cmd else "")

    def no_dispatch(*args, **kwargs):
        pytest.fail("check-only must not dispatch")

    monkeypatch.setattr(subprocess, "run", no_dispatch)
    assert launch.main(["--check-only", "--bypass-t4-gate"]) == 0
    assert "LEGALIR_BYPASS_T4_GATE" not in os.environ
    freeze["algorithm_config_sha256"] = "wrong"
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
    assert launch.main(["--check-only", "--bypass-t4-gate"]) == 1
