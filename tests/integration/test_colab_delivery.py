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
            # Known private repository by default; positive tests represent a
            # verified private target, not an unknown/metadata-missing case.
            return SimpleNamespace(private=True)

        def create_repo(self, **kwargs):
            calls.append(("create", kwargs))

        def auth_check(self, **kwargs):
            calls.append(("auth", kwargs))

        def upload_folder(self, **kwargs):
            calls.append(("upload", kwargs))
            return SimpleNamespace(oid="a" * 40)

        def upload_file(self, **kwargs):
            calls.append(("upload_file", kwargs))
            return SimpleNamespace(oid="b" * 40)

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


@pytest.mark.parametrize(
    "private, consent, expected",
    [(True, False, True), (False, False, False),
     (False, True, True), (None, False, False), (None, True, False)],
)
def test_preflight_visibility_policy(hub, monkeypatch, private, consent, expected):
    _, api = hub
    monkeypatch.setattr(api, "repo_info", lambda self, **kw: SimpleNamespace(private=private))
    ok, _ = run_a100.preflight_huggingface_access(
        "test-owner/model", "hf_test", allow_public_repo=consent
    )
    assert ok is expected


@pytest.mark.parametrize(
    "private, consent, should_upload",
    [(True, False, True), (False, False, False),
     (False, True, True), (None, False, False), (None, True, False)],
)
def test_upload_visibility_policy(tmp_path, hub, monkeypatch, private, consent, should_upload):
    make_deliverables(tmp_path)
    calls, api = hub
    monkeypatch.setattr(api, "repo_info", lambda self, **kw: SimpleNamespace(private=private))
    if should_upload:
        commit = run_a100.upload_artifacts_to_huggingface(
            tmp_path, "test-owner/model", "hf_test", allow_public_repo=consent
        )
        assert commit == "a" * 40
    else:
        with pytest.raises(RuntimeError):
            run_a100.upload_artifacts_to_huggingface(
                tmp_path, "test-owner/model", "hf_test", allow_public_repo=consent
            )
        assert not any(name == "upload" for name, _ in calls)


def test_preflight_lookup_exception_does_not_leak_token(hub, monkeypatch):
    _, api = hub

    def boom(self, **kwargs):
        raise ConnectionError("lookup failed for hf_test_secret_embedded")

    monkeypatch.setattr(api, "repo_info", boom)
    ok, detail = run_a100.preflight_huggingface_access("test-owner/model", "hf_test_secret_embedded")
    assert ok is False
    assert "hf_test_secret_embedded" not in detail


def test_upload_lookup_exception_does_not_leak_token(tmp_path, hub, monkeypatch):
    make_deliverables(tmp_path)
    _, api = hub

    def boom(self, **kwargs):
        raise ConnectionError("lookup failed for hf_test_secret_embedded")

    monkeypatch.setattr(api, "repo_info", boom)
    with pytest.raises(RuntimeError, match="upload failed") as error:
        run_a100.upload_artifacts_to_huggingface(tmp_path, "test-owner/model", "hf_test_secret_embedded")
    assert "hf_test_secret_embedded" not in str(error.value)


def test_upload_unknown_visibility_blocks_without_upload(tmp_path, hub, monkeypatch):
    make_deliverables(tmp_path)
    calls, api = hub
    monkeypatch.setattr(api, "repo_info", lambda self, **kw: SimpleNamespace())
    with pytest.raises(RuntimeError, match="visibility|upload failed"):
        run_a100.upload_artifacts_to_huggingface(tmp_path, "test-owner/model", "hf_test")
    assert not any(name == "upload" for name, _ in calls)


def test_generated_consent_defaults_fail_closed():
    from scripts.generate_notebooks import build_colab_train_notebook

    cells = build_colab_train_notebook("a" * 40)["cells"]
    secrets_src = "".join(cells[1]["source"])
    assert "HF_ALLOW_PUBLIC_REPO" in secrets_src
    setup_src = "".join(cells[3]["source"])
    # Literal "1" is the only opt-in; absent defaults to "0" (private-only).
    assert 'os.environ.get("HF_ALLOW_PUBLIC_REPO", "0") == "1"' in setup_src
    assert 'os.environ.get("HF_ALLOW_PUBLIC_REPO", "1")' not in setup_src


def test_generated_consent_parsing_accepts_only_literal_one():
    # Simulate the generated expression for absent/0/1/arbitrary values.
    def parse(env_value, present=True):
        env = {"HF_ALLOW_PUBLIC_REPO": env_value} if present else {}
        return env.get("HF_ALLOW_PUBLIC_REPO", "0") == "1"

    assert parse(None, present=False) is False
    assert parse("0") is False
    assert parse("1") is True
    assert parse("true") is False
    assert parse("yes") is False
    assert parse("") is False


def _setup_receipt_gate_fixture(tmp_path, monkeypatch, receipt_oid="b" * 40, receipt_raises=False):
    """Stub the full A100 gate up to the HF receipt stage (offline, no GPU)."""
    import zipfile
    from types import SimpleNamespace as NS

    from scripts.gates import run_a100 as gate_mod
    import huggingface_hub

    sha = "a" * 40
    manifest_hash = "c" * 64
    algo_hash = "d" * 64

    dataset_dir = tmp_path / "data"
    dataset_dir.mkdir()
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    k_report = {
        "stage": "KAGGLE_T4X2",
        "verdict": "PASS",
        "git_sha": sha,
        "dataset_manifest_sha256": manifest_hash,
        "algorithm_config_sha256": algo_hash,
    }
    k_path = tmp_path / "kaggle.json"
    k_path.write_text(json.dumps(k_report), encoding="utf-8")
    freeze = {
        "git_sha": sha,
        "dataset": {"manifest_sha256": manifest_hash},
        "algorithm_config_sha256": algo_hash,
    }
    f_path = tmp_path / "freeze.json"
    f_path.write_text(json.dumps(freeze), encoding="utf-8")

    # Hardware / provenance stubs (patch gate module attributes for top imports).
    monkeypatch.setattr(
        gate_mod, "verify_device_contract", lambda *a, **k: NS(device_names=["NVIDIA A100"], device_count=1)
    )
    monkeypatch.setattr(gate_mod, "assert_exact_git_sha", lambda *a, **k: sha)
    monkeypatch.setattr(
        gate_mod,
        "verify_dataset_fingerprint",
        lambda *a, **k: NS(is_valid=True, manifest_sha256=manifest_hash, verified_files={}),
    )
    monkeypatch.setattr(gate_mod, "fingerprint_structured_config", lambda *a, **k: algo_hash)
    monkeypatch.setattr(gate_mod, "verify_prior_gate_reports", lambda **k: NS(is_valid=True, kaggle_report_sha256="e" * 64))
    monkeypatch.setattr(gate_mod, "validate_submission_zip", lambda *a, **k: {"is_valid": True, "errors": []})
    monkeypatch.setattr(gate_mod, "preflight_huggingface_access", lambda *a, **k: (True, "offline ok"))
    monkeypatch.setattr(gate_mod, "upload_artifacts_to_huggingface", lambda **k: "a" * 40)

    # Pipeline stub writes a minimal valid bundle (offline, no GPU training).
    def fake_pipeline(**kwargs):
        out = Path(kwargs.get("working_dir") or kwargs.get("output_dir") or out_dir)
        out.mkdir(parents=True, exist_ok=True)
        sub = {f"q_{i}": {"answer": ["101", "102"]} for i in range(5)}
        with zipfile.ZipFile(out / "submission.zip", "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("submission.json", json.dumps(sub))
        (out / "submission.json").write_text(json.dumps(sub), encoding="utf-8")
        ad = out / "checkpoints/reranker_final"
        ad.mkdir(parents=True, exist_ok=True)
        (ad / "adapter_model.safetensors").write_bytes(b"x")
        (ad / "adapter_config.json").write_text("{}", encoding="utf-8")
        return NS(status="COMPLETED")

    import src.pipeline.kaggle_train as pipe_mod

    monkeypatch.setattr(pipe_mod, "run_kaggle_pipeline", fake_pipeline)
    # Gate imports run_kaggle_pipeline inside the function from the same module
    # path; ensure the patched attribute is visible there too.
    import sys as _sys

    _sys.modules["src.pipeline.kaggle_train"].run_kaggle_pipeline = fake_pipeline

    # Receipt upload stub.
    calls = []

    class FakeReceiptApi:
        def __init__(self, *a, **k):
            pass

        def upload_file(self, **kwargs):
            calls.append(kwargs)
            if receipt_raises:
                raise ConnectionError("receipt boom")
            return NS(oid=receipt_oid)

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeReceiptApi)
    # resolve_hf_token is used for receipt upload; force a valid token.
    monkeypatch.setattr(gate_mod, "resolve_hf_token", lambda *a, **k: "hf_test")
    return gate_mod, sha, dataset_dir, out_dir, k_path, f_path, calls


def test_receipt_requires_valid_oid(tmp_path, monkeypatch):
    gate_mod, sha, data_dir, out_dir, k_path, f_path, _ = _setup_receipt_gate_fixture(
        tmp_path, monkeypatch, receipt_oid="not-a-sha"
    )
    with pytest.raises(RuntimeError, match="manifest upload failed"):
        gate_mod.run_a100_production_gate(
            dataset_dir=data_dir,
            output_dir=out_dir,
            expected_sha=sha,
            kaggle_report_path=k_path,
            freeze_file_path=f_path,
            mock=False,
            allow_non_a100=True,
            hf_repo="test-owner/model",
            hf_token="hf_test",
        )
    # Partial artifacts remain recoverable; no RELEASED claim.
    saved = json.loads((out_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert saved.get("status") != "RELEASED"
    assert "manifest_commit_sha" not in saved.get("huggingface", {})


def test_receipt_missing_oid_blocks_release(tmp_path, monkeypatch):
    gate_mod, sha, data_dir, out_dir, k_path, f_path, _ = _setup_receipt_gate_fixture(
        tmp_path, monkeypatch, receipt_oid=None
    )
    with pytest.raises(RuntimeError, match="manifest upload failed"):
        gate_mod.run_a100_production_gate(
            dataset_dir=data_dir,
            output_dir=out_dir,
            expected_sha=sha,
            kaggle_report_path=k_path,
            freeze_file_path=f_path,
            mock=False,
            allow_non_a100=True,
            hf_repo="test-owner/model",
            hf_token="hf_test",
        )
    saved = json.loads((out_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert saved.get("status") != "RELEASED"


def test_receipt_upload_failure_keeps_artifacts(tmp_path, monkeypatch):
    gate_mod, sha, data_dir, out_dir, k_path, f_path, _ = _setup_receipt_gate_fixture(
        tmp_path, monkeypatch, receipt_raises=True
    )
    with pytest.raises(RuntimeError, match="manifest upload failed"):
        gate_mod.run_a100_production_gate(
            dataset_dir=data_dir,
            output_dir=out_dir,
            expected_sha=sha,
            kaggle_report_path=k_path,
            freeze_file_path=f_path,
            mock=False,
            allow_non_a100=True,
            hf_repo="test-owner/model",
            hf_token="hf_test",
        )
    # Artifact commit exists locally; manifest is not RELEASED but files remain.
    assert (out_dir / "submission.zip").is_file()
    saved = json.loads((out_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert saved.get("status") != "RELEASED"


def test_successful_release_receipt_consistency(tmp_path, monkeypatch):
    gate_mod, sha, data_dir, out_dir, k_path, f_path, _ = _setup_receipt_gate_fixture(
        tmp_path, monkeypatch, receipt_oid="b" * 40
    )
    manifest = gate_mod.run_a100_production_gate(
        dataset_dir=data_dir,
        output_dir=out_dir,
        expected_sha=sha,
        kaggle_report_path=k_path,
        freeze_file_path=f_path,
        mock=False,
        allow_non_a100=True,
        hf_repo="test-owner/model",
        hf_token="hf_test",
        hf_allow_public_repo=False,
    )
    import re

    assert manifest["status"] == "RELEASED"
    hf = manifest["huggingface"]
    assert hf["repo_id"] == "test-owner/model"
    assert re.fullmatch(r"[0-9a-f]{40}", hf["commit_sha"])
    assert hf["path_in_repo"] == f"runs/{manifest['run_id']}"
    assert hf["public_repo_override"] is False
    assert re.fullmatch(r"[0-9a-f]{40}", hf["manifest_commit_sha"])
    # Local file matches returned manifest.
    saved = json.loads((out_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert saved == manifest
