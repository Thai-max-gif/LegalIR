"""Offline regressions for the real Colab launch/delivery boundaries, not GPU evidence."""
import json
from pathlib import Path
from types import SimpleNamespace

import huggingface_hub
import pytest

from scripts.gates import run_a100
from scripts import run_colab_train


@pytest.fixture
def hub(monkeypatch):
    calls = []

    class FakeApi:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))

        def whoami(self):
            return {"name": "test-owner"}

        def repo_info(self, **kwargs):
            return SimpleNamespace()

        def create_repo(self, **kwargs):
            calls.append(("create", kwargs))

        def auth_check(self, **kwargs):
            calls.append(("auth", kwargs))

        def upload_folder(self, **kwargs):
            calls.append(("upload", kwargs))
            return SimpleNamespace(oid="a" * 40)

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    for key in ("HF_TOKEN", "HF_TOKEN_WRITE", "HF_TOKEN_READ"):
        monkeypatch.delenv(key, raising=False)
    return calls, FakeApi


def test_release_requires_token(hub):
    ok, detail = run_a100.preflight_huggingface_access("test-owner/model")
    assert not ok
    assert "token" in detail.lower()


def test_preflight_checks_write_permission(hub):
    calls, _ = hub
    ok, _ = run_a100.preflight_huggingface_access("test-owner/model", "hf_test")
    assert ok
    assert ("auth", {"repo_id": "test-owner/model", "repo_type": "model", "write": True}) in calls


def test_preflight_network_failure_is_not_success(hub, monkeypatch):
    _, api = hub
    def unavailable(self):
        raise ConnectionError("offline")
    monkeypatch.setattr(api, "whoami", unavailable)
    assert run_a100.preflight_huggingface_access("test-owner/model", "hf_test")[0] is False


def test_preflight_read_only_token_fails_without_leaking_token(hub, monkeypatch):
    _, api = hub
    def denied(self, **kwargs):
        raise PermissionError("denied hf_test_secret")
    monkeypatch.setattr(api, "auth_check", denied)
    ok, detail = run_a100.preflight_huggingface_access("test-owner/model", "hf_test_secret")
    assert not ok
    assert "hf_test_secret" not in detail


def make_deliverables(root):
    for name in (
        "checkpoints/reranker_final/adapter_model.safetensors",
        "checkpoints/reranker_final/adapter_config.json",
        "checkpoints/reranker_final/tokenizer.json",
        "checkpoints/fusion_final/model.txt",
        "cv/cv_report.json", "training.log", "submission.zip", "submission.json",
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("test fixture")
    (root / "run_manifest.json").write_text(json.dumps({"run_id": "test-run"}))


def test_upload_limits_scope_and_keeps_runs_separate(tmp_path, hub):
    make_deliverables(tmp_path)
    for name in (".env", "kaggle.json", "documents.parquet", "unrelated.txt"):
        (tmp_path / name).write_text("must not upload")
    calls, _ = hub
    assert run_a100.upload_artifacts_to_huggingface(tmp_path, "test-owner/model", "hf_test") == "a" * 40
    upload = next(kwargs for name, kwargs in calls if name == "upload")
    assert upload["path_in_repo"] == "runs/test-run"
    selected = set(upload["allow_patterns"])
    assert "checkpoints/reranker_final/adapter_model.safetensors" in selected
    assert "checkpoints/reranker_final/tokenizer.json" in selected
    assert "training.log" in selected
    assert not selected.intersection({".env", "kaggle.json", "documents.parquet", "unrelated.txt"})


def test_upload_failure_raises_instead_of_succeeding_locally(tmp_path, hub, monkeypatch):
    make_deliverables(tmp_path)
    _, api = hub
    def unavailable(self, **kwargs):
        raise ConnectionError("hf_test_secret")
    monkeypatch.setattr(api, "upload_folder", unavailable)
    with pytest.raises(RuntimeError, match="upload failed") as error:
        run_a100.upload_artifacts_to_huggingface(tmp_path, "test-owner/model", "hf_test_secret")
    assert "hf_test_secret" not in str(error.value)


def test_missing_adapter_blocks_upload(tmp_path, hub):
    make_deliverables(tmp_path)
    (tmp_path / "checkpoints/reranker_final/adapter_model.safetensors").unlink()
    with pytest.raises(RuntimeError, match="adapter"):
        run_a100.upload_artifacts_to_huggingface(tmp_path, "test-owner/model", "hf_test")
    assert not any(name == "upload" for name, _ in hub[0])


def test_wrapper_forwards_explicit_hf_token(tmp_path, monkeypatch):
    captured = {}
    def gate(**kwargs):
        captured.update(kwargs)
        return {}
    monkeypatch.setattr(run_colab_train, "run_a100_production_gate", gate)
    run_colab_train.run_colab_production_training(tmp_path, tmp_path, hf_token="hf_test")
    assert captured["hf_token"] == "hf_test"


def test_wrapper_does_not_replace_explicit_missing_report(tmp_path, monkeypatch):
    captured = {}
    def gate(**kwargs):
        captured.update(kwargs)
        return {}
    monkeypatch.setattr(run_colab_train, "run_a100_production_gate", gate)
    missing = tmp_path / "missing.json"
    run_colab_train.run_colab_production_training(tmp_path, tmp_path, smoke_report_path=missing)
    assert captured["kaggle_report_path"] == missing
