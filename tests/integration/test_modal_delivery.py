"""Offline orchestration regressions for Modal durable delivery, not GPU evidence.

These tests verify path preservation only (dummy bytes), not reloadable
weights or survival across a real container kill. No resume is implied.
"""
from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path

import pytest


def _install_fake_modal():
    """Install an offline fake `modal` module before importing the launcher.

    No SDK cloud call, secret lookup, image build, or real Volume creation
    may occur in these tests.
    """
    if "modal" in sys.modules:
        real = sys.modules["modal"]
        # If the real SDK is already loaded, keep it but tests must not use it.
        # Prefer a fake only when explicitly requested via env? For offline
        # determinism, always replace with a fake for this test module.
        del sys.modules["modal"]

    fake = types.ModuleType("modal")

    class _FakeImage:
        def debian_slim(self, *a, **k):
            return self

        def apt_install(self, *a, **k):
            return self

        def pip_install(self, *a, **k):
            return self

    class _FakeVolume:
        @classmethod
        def from_name(cls, *a, **k):
            # Never touch the cloud; return a no-op placeholder. Tests replace
            # the launcher's `volume` attribute with a recording fake.
            class _V:
                def commit(self):
                    return None

            return _V()

    class _FakeSecret:
        @classmethod
        def from_name(cls, *a, **k):
            return object()

    class _FakeApp:
        def __init__(self, *a, **k):
            pass

        def function(self, *dargs, **dkwargs):
            def deco(fn):
                # Preserve the original function for direct offline calls and
                # expose a .remote alias used by main().
                fn.remote = fn
                fn._modal_fake_kwargs = dkwargs
                return fn

            return deco

        def local_entrypoint(self, *dargs, **dkwargs):
            def deco(fn):
                return fn

            return deco

    fake.App = _FakeApp
    fake.Image = _FakeImage()
    fake.Volume = _FakeVolume
    fake.Secret = _FakeSecret
    sys.modules["modal"] = fake
    return fake


def _load_launcher():
    _install_fake_modal()
    # Ensure a fresh import picks up the fake modal module.
    for mod in ("scripts.modal.run_modal_a100", "scripts.modal"):
        sys.modules.pop(mod, None)
    import scripts.modal.run_modal_a100 as m

    return importlib.reload(m)


@pytest.fixture
def launcher(monkeypatch, tmp_path):
    mod = _load_launcher()
    # Redirect Volume mount, repo checkout, and dataset to fixtures. Never
    # touch /root or the developer's real checkout.
    volume_root = tmp_path / "volume"
    volume_root.mkdir()
    repo_dir = tmp_path / "LegalIR"
    repo_dir.mkdir()
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    monkeypatch.setattr(mod, "VOLUME_MOUNT", str(volume_root))
    monkeypatch.setenv("LEGALIR_MODAL_REPO_DIR", str(repo_dir))
    monkeypatch.setenv("LEGALIR_MODAL_DATASET_DIR", str(dataset_dir))
    # Avoid changing the test process cwd.
    monkeypatch.setattr(mod.os, "chdir", lambda *a, **k: None)
    # Stub git clone/checkout.
    monkeypatch.setattr(
        mod.subprocess,
        "run",
        lambda *a, **k: types.SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    return mod


@pytest.fixture
def offline_stubs(launcher, monkeypatch):
    """Stub bootstrap, HF preflight, pipeline, and Volume commit."""
    calls = {"verify": [], "prepare": [], "preflight": [], "pipeline": [], "commits": []}
    received = {}

    import scripts.colab.bootstrap as boot
    import scripts.gates.run_a100 as gate
    import scripts.run_colab_train as wrapper

    def fake_verify(sha, kaggle_report, freeze_file, repo_root=None):
        calls["verify"].append((sha, str(kaggle_report), str(freeze_file)))
        return {"ok": True}

    def fake_prepare(dataset_dir, freeze_file=None):
        calls["prepare"].append((str(dataset_dir), str(freeze_file) if freeze_file else None))
        return Path(dataset_dir)

    def fake_preflight(repo_id, token=None, allow_public_repo=False):
        calls["preflight"].append((repo_id, allow_public_repo))
        return True, "offline preflight ok"

    def fake_pipeline(**kwargs):
        received.update(kwargs)
        out = Path(kwargs["output_dir"])
        # Simulate training writing directly into the durable attempt path
        # while training runs, including training.log.
        (out / "training.log").write_text("offline training log\n", encoding="utf-8")
        adapter = out / "checkpoints/reranker_final"
        adapter.mkdir(parents=True, exist_ok=True)
        (adapter / "adapter_model.safetensors").write_bytes(b"fixture-weights")
        (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
        (out / "run_manifest.json").write_text(json.dumps({"run_id": "offline"}), encoding="utf-8")
        (out / "recovery.tar.gz").write_bytes(b"fixture-archive")
        return {"status": "COMPLETED", "verdict": "PASS", "huggingface": {"uploaded": True}}

    monkeypatch.setattr(boot, "verify_launch", fake_verify)
    monkeypatch.setattr(boot, "prepare_dataset", fake_prepare)
    monkeypatch.setattr(gate, "preflight_huggingface_access", fake_preflight)
    monkeypatch.setattr(wrapper, "run_colab_production_training", fake_pipeline)

    class FakeVolume:
        def commit(self):
            calls["commits"].append(1)

    monkeypatch.setattr(launcher, "volume", FakeVolume())
    return calls, received


VALID_SHA = "a" * 40


def _read_state(attempt_dir: Path) -> dict:
    return json.loads((attempt_dir / "launcher_state.json").read_text(encoding="utf-8"))


def test_late_upload_failure_preserves_weights_and_archive(launcher, offline_stubs, monkeypatch):
    """Baseline P0 reproduction: graceful late failure must keep real files."""
    calls, received = offline_stubs
    import scripts.run_colab_train as wrapper

    def failing_pipeline(**kwargs):
        out = Path(kwargs["output_dir"])
        received.update(kwargs)
        (out / "training.log").write_text("log while training\n", encoding="utf-8")
        adapter = out / "checkpoints/reranker_final"
        adapter.mkdir(parents=True, exist_ok=True)
        (adapter / "adapter_model.safetensors").write_bytes(b"fixture-weights")
        (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
        (out / "run_manifest.json").write_text(json.dumps({"run_id": "x"}), encoding="utf-8")
        (out / "recovery.tar.gz").write_bytes(b"fixture-archive")
        raise RuntimeError("hf_test late upload failure")

    monkeypatch.setattr(wrapper, "run_colab_production_training", failing_pipeline)

    with pytest.raises(RuntimeError, match="late upload"):
        launcher.run_production_training(VALID_SHA, hf_allow_public_repo=False)

    output_dir = Path(received["output_dir"])
    volume_root = Path(launcher.VOLUME_MOUNT)
    assert output_dir.is_relative_to(volume_root)
    assert (output_dir / "checkpoints/reranker_final/adapter_model.safetensors").read_bytes() == b"fixture-weights"
    assert (output_dir / "recovery.tar.gz").is_file()
    assert (output_dir / "training.log").is_file()
    assert calls["commits"], "The graceful failure path must attempt a Volume commit"
    state = _read_state(output_dir)
    assert state["phase"] == "failed"
    assert state["outcome"] == "failed"


def test_two_distinct_attempts(launcher, offline_stubs):
    calls, received = offline_stubs
    launcher.run_production_training(VALID_SHA)
    first = Path(received["output_dir"])
    launcher.run_production_training(VALID_SHA)
    second = Path(received["output_dir"])
    assert first != second
    assert first.is_relative_to(Path(launcher.VOLUME_MOUNT))
    assert second.is_relative_to(Path(launcher.VOLUME_MOUNT))
    assert first.is_dir() and second.is_dir()


def test_invalid_sha_rejected_before_cloud_work(launcher, offline_stubs):
    calls, _ = offline_stubs
    with pytest.raises(ValueError, match="40-character"):
        launcher.run_production_training("not-a-sha")
    with pytest.raises(ValueError, match="40-character"):
        launcher.run_production_training("A" * 40)  # must be lowercase
    assert calls["verify"] == []
    assert calls["pipeline"] == []
    assert calls["commits"] == []


def test_preflight_failure_before_training(launcher, offline_stubs, monkeypatch):
    calls, received = offline_stubs
    import scripts.gates.run_a100 as gate

    def deny(*a, **k):
        return False, "denied"

    monkeypatch.setattr(gate, "preflight_huggingface_access", deny)
    with pytest.raises(RuntimeError, match="preflight"):
        launcher.run_production_training(VALID_SHA)
    # No expensive pipeline work after failed preflight.
    assert received == {}
    volume_root = Path(launcher.VOLUME_MOUNT)
    attempts = list((volume_root / VALID_SHA / "attempts").iterdir())
    assert len(attempts) == 1
    state = _read_state(attempts[0])
    assert state["phase"] == "failed"


def test_pipeline_success_commits_attempt(launcher, offline_stubs):
    calls, received = offline_stubs
    report = launcher.run_production_training(VALID_SHA)
    assert report["status"] == "COMPLETED"
    output_dir = Path(received["output_dir"])
    assert output_dir.is_relative_to(Path(launcher.VOLUME_MOUNT))
    assert (output_dir / "training.log").is_file()
    assert calls["commits"]
    state = _read_state(output_dir)
    assert state["phase"] == "completed"
    assert state["outcome"] == "completed"


def test_primary_failure_plus_commit_failure_preserves_primary(launcher, offline_stubs, monkeypatch, capsys):
    calls, received = offline_stubs
    import scripts.run_colab_train as wrapper

    def failing_pipeline(**kwargs):
        out = Path(kwargs["output_dir"])
        received.update(kwargs)
        (out / "training.log").write_text("log\n", encoding="utf-8")
        (out / "checkpoints/reranker_final/adapter_model.safetensors").parent.mkdir(parents=True, exist_ok=True)
        (out / "checkpoints/reranker_final/adapter_model.safetensors").write_bytes(b"fixture-weights")
        (out / "recovery.tar.gz").write_bytes(b"fixture-archive")
        raise RuntimeError("primary boom")

    monkeypatch.setattr(wrapper, "run_colab_production_training", failing_pipeline)

    class FailingVolume:
        def commit(self):
            calls["commits"].append(1)
            raise OSError("commit down")

    monkeypatch.setattr(launcher, "volume", FailingVolume())
    with pytest.raises(RuntimeError, match="primary boom"):
        launcher.run_production_training(VALID_SHA)
    out = capsys.readouterr().out
    assert "Volume commit failed" in out
    assert "OSError" in out


def test_success_plus_commit_failure_is_not_success(launcher, offline_stubs, monkeypatch):
    calls, received = offline_stubs

    class FailingVolume:
        def commit(self):
            # Allow early best-effort commits, fail only the final one.
            calls["commits"].append(1)
            if len(calls["commits"]) >= 4:
                raise OSError("final commit down")

    monkeypatch.setattr(launcher, "volume", FailingVolume())
    with pytest.raises(RuntimeError, match="Volume persistence failed"):
        launcher.run_production_training(VALID_SHA)
    output_dir = Path(received["output_dir"])
    assert (output_dir / "checkpoints/reranker_final/adapter_model.safetensors").is_file()


def test_launcher_state_redaction(launcher, offline_stubs, monkeypatch):
    calls, received = offline_stubs
    secret = "hf_test_secret_value_123"
    monkeypatch.setenv("HF_TOKEN", secret)
    import scripts.run_colab_train as wrapper

    def failing_pipeline(**kwargs):
        out = Path(kwargs["output_dir"])
        received.update(kwargs)
        (out / "training.log").write_text("log\n", encoding="utf-8")
        raise RuntimeError(f"boom containing {secret}")

    monkeypatch.setattr(wrapper, "run_colab_production_training", failing_pipeline)
    with pytest.raises(RuntimeError):
        launcher.run_production_training(VALID_SHA)
    output_dir = Path(received["output_dir"])
    raw = (output_dir / "launcher_state.json").read_text(encoding="utf-8")
    assert secret not in raw
    assert "boom containing" not in raw
    state = json.loads(raw)
    # Only exception class, never text, tokens, or env dicts.
    assert state["exception_class"] == "RuntimeError"
    assert set(state) == {
        "attempt_id",
        "expected_sha",
        "phase",
        "outcome",
        "exception_class",
        "started_utc",
        "updated_utc",
    }


def test_state_write_failure_cannot_mask_primary_or_skip_commit(launcher, offline_stubs, monkeypatch):
    """Failure metadata is best-effort: an OSError while writing state must
    neither replace the original pipeline error nor skip the final commit."""
    calls, received = offline_stubs
    import scripts.run_colab_train as wrapper

    def failing_pipeline(**kwargs):
        out = Path(kwargs["output_dir"])
        received.update(kwargs)
        received["commits_at_pipeline_entry"] = len(calls["commits"])
        (out / "training.log").write_text("log\n", encoding="utf-8")
        (out / "checkpoints/reranker_final/adapter_model.safetensors").parent.mkdir(parents=True, exist_ok=True)
        (out / "checkpoints/reranker_final/adapter_model.safetensors").write_bytes(b"fixture-weights")
        (out / "recovery.tar.gz").write_bytes(b"fixture-archive")
        raise RuntimeError("primary boom")

    monkeypatch.setattr(wrapper, "run_colab_production_training", failing_pipeline)

    def broken_state(*a, **k):
        raise OSError("state disk down")

    monkeypatch.setattr(launcher, "_write_launcher_state", broken_state)
    with pytest.raises(RuntimeError, match="primary boom"):
        launcher.run_production_training(VALID_SHA)
    # A commit was attempted AFTER the pipeline failure, not merely before it.
    assert len(calls["commits"]) > received["commits_at_pipeline_entry"]


def test_dataset_failure_gets_final_commit(launcher, offline_stubs, monkeypatch):
    """Lifecycle-wide finalization: dataset-stage failures also attempt a
    final commit and preserve the original error."""
    calls, received = offline_stubs
    import scripts.colab.bootstrap as boot

    def failing_prepare(dataset_dir, freeze_file=None):
        received["commits_at_dataset_entry"] = len(calls["commits"])
        raise RuntimeError("dataset boom")

    monkeypatch.setattr(boot, "prepare_dataset", failing_prepare)
    with pytest.raises(RuntimeError, match="dataset boom"):
        launcher.run_production_training(VALID_SHA)
    assert len(calls["commits"]) > received["commits_at_dataset_entry"]
    assert received == {"commits_at_dataset_entry": received["commits_at_dataset_entry"]}
    volume_root = Path(launcher.VOLUME_MOUNT)
    attempts = list((volume_root / VALID_SHA / "attempts").iterdir())
    assert len(attempts) == 1
    state = _read_state(attempts[0])
    assert state["phase"] == "failed"
    assert state["outcome"] == "failed"
    assert state["exception_class"] == "RuntimeError"


def test_create_attempt_dir_validates_sha(tmp_path):
    mod = _load_launcher()
    root = tmp_path / "v"
    root.mkdir()
    with pytest.raises(ValueError):
        mod.create_attempt_dir(root, "short")
    with pytest.raises(ValueError):
        mod.create_attempt_dir(root, "A" * 40)
    p1 = mod.create_attempt_dir(root, VALID_SHA)
    p2 = mod.create_attempt_dir(root, VALID_SHA)
    assert p1 != p2
    assert p1.parent.parent.name == VALID_SHA


def test_default_consent_is_false():
    import inspect

    mod = _load_launcher()
    sig = inspect.signature(mod.run_production_training)
    assert sig.parameters["hf_allow_public_repo"].default is False
    sig_main = inspect.signature(mod.main)
    assert sig_main.parameters["hf_allow_public_repo"].default is False


def test_modal_forwards_explicit_consent(launcher, monkeypatch):
    forwarded = {}

    def fake_remote(sha, hf_allow_public_repo=False):
        forwarded["sha"] = sha
        forwarded["consent"] = hf_allow_public_repo
        return {"ok": True}

    monkeypatch.setattr(launcher.run_production_training, "remote", fake_remote)
    # Avoid real CPU provenance validation for this forwarding unit test.
    import scripts.colab.bootstrap as boot

    monkeypatch.setattr(boot, "verify_launch", lambda *a, **k: {"ok": True})
    monkeypatch.setenv("LEGALIR_COMMIT_SHA", VALID_SHA)
    launcher.main(hf_allow_public_repo=True)
    assert forwarded["consent"] is True
    assert forwarded["sha"] == VALID_SHA
    launcher.main(hf_allow_public_repo=False)
    assert forwarded["consent"] is False
