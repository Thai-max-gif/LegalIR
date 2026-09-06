import json
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch

from src.production.final_train import train_final_adapter
from src.training.train_reranker import train_reranker
from src.ranking.reranker import CrossEncoderReranker


def test_nonmock_final_rejects_mock_base_model(tmp_path):
    pairs_file = tmp_path / "pairs.parquet"
    pd.DataFrame({
        "query_id": ["q1", "q1"],
        "query_text": ["q", "q"],
        "doc_id": ["dA", "dB"],
        "label": [1.0, 0.0],
        "evidence_text": ["eA", "eB"],
    }).to_parquet(pairs_file)

    adapter_out = tmp_path / "adapter"
    with pytest.raises(ValueError, match="mock.*not allowed|real base model"):
        train_final_adapter(
            pairs_path=pairs_file,
            output_adapter_dir=adapter_out,
            runtime_config={"base_model_name": "mock"},
            mock_run=False,
        )


def test_run_kaggle_final_passes_frozen_reranker_config(tmp_path, monkeypatch):
    import scripts.run_kaggle_final as rkf

    # Mock canonical verification and bundle verification
    monkeypatch.setattr(rkf, "verify_canonical_dataset", lambda p: (True, {"sha256": "abc"}, []))
    monkeypatch.setattr(rkf, "verify_production_bundle", lambda p: (True, []))

    recorded_cfg = {}

    def fake_train_final_adapter(pairs_path, output_adapter_dir, runtime_config=None, mock_run=False):
        recorded_cfg.update(runtime_config or {})
        return {"status": "PASS"}

    monkeypatch.setattr(rkf, "train_final_adapter", fake_train_final_adapter)

    # Set up dummy bundle with production_lock.json
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    lock_data = {
        "status": "LOCKED",
        "config": {
            "reranker": {
                "base_model_name": "BAAI/bge-reranker-v2-m3",
                "max_length": 512,
                "effective_batch_size": 16,
                "lora_r": 16,
            }
        }
    }
    (bundle_dir / "production_lock.json").write_text(json.dumps(lock_data), encoding="utf-8")
    (bundle_dir / "final_training_pairs.parquet").touch()
    (bundle_dir / "public_candidates.parquet").touch()
    (bundle_dir / "public_evidence.parquet").touch()

    dataset_dir = tmp_path / "data"
    dataset_dir.mkdir()
    (dataset_dir / "public-official.json").write_text('{"q1": {"question": "hoi"}}', encoding="utf-8")

    monkeypatch.setattr(rkf, "rerank_and_fuse_public_predictions", lambda **kw: {"q1": ["docA"]})
    monkeypatch.setattr(rkf, "validate_submission", lambda *a, **kw: (True, []))
    monkeypatch.setattr(rkf, "package_submission", lambda *a, **kw: (tmp_path / "sub.json", tmp_path / "sub.zip"))

    test_args = [
        "run_kaggle_final.py",
        "--dataset-dir", str(dataset_dir),
        "--bundle-dir", str(bundle_dir),
        "--output-dir", str(tmp_path / "out"),
    ]
    monkeypatch.setattr("sys.argv", test_args)

    with pytest.raises(SystemExit) as exc_info:
        rkf.main()
    assert exc_info.value.code == 0
    assert recorded_cfg.get("base_model_name") == "BAAI/bge-reranker-v2-m3"
    assert recorded_cfg.get("effective_batch_size") == 16


def test_nonmock_train_model_load_failure_is_fatal(tmp_path):
    pairs_file = tmp_path / "pairs.parquet"
    pd.DataFrame({
        "query_id": ["q1", "q1"],
        "query_text": ["q", "q"],
        "doc_id": ["dA", "dB"],
        "label": [1.0, 0.0],
        "evidence_text": ["eA", "eB"],
    }).to_parquet(pairs_file)

    out_dir = tmp_path / "out"

    with patch("transformers.AutoTokenizer.from_pretrained", side_effect=OSError("HF network unreachable")):
        with pytest.raises((RuntimeError, OSError)):
            train_reranker(
                pairs_file=pairs_file,
                output_dir=out_dir,
                base_model_name="BAAI/bge-reranker-v2-m3",
                enforce_full_coverage_steps=False,
            )


def test_nonmock_inference_model_load_failure_is_fatal():
    reranker = CrossEncoderReranker(
        model_name="BAAI/bge-reranker-v2-m3",
        local_files_only=True,
    )
    with patch("transformers.AutoTokenizer.from_pretrained", side_effect=OSError("Tokenizer load error")):
        with pytest.raises((RuntimeError, OSError)):
            reranker.ensure_loaded()


def test_real_adapter_manifest_base_model_matches_production_lock(tmp_path, monkeypatch):
    pairs_file = tmp_path / "pairs.parquet"
    pd.DataFrame({
        "query_id": ["q1", "q1"],
        "query_text": ["q", "q"],
        "doc_id": ["dA", "dB"],
        "label": [1.0, 0.0],
        "evidence_text": ["eA", "eB"],
    }).to_parquet(pairs_file)

    adapter_out = tmp_path / "adapter"

    # Simulate real training return
    fake_report = {
        "status": "PASS",
        "optimizer_steps": 10,
        "final_loss": 0.15,
        "param_diff": 0.02,
        "base_model": "BAAI/bge-reranker-v2-m3",
        "trainable_parameters": 45000000,
        "device": "cpu",
        "peft_config": {"r": 16, "lora_alpha": 32},
        "query_coverage": {"seen": 1, "total": 1},
    }

    monkeypatch.setattr("src.production.final_train.train_reranker", lambda **kw: fake_report)
    monkeypatch.setattr("src.production.final_train.CrossEncoderReranker", lambda **kw: type("DummyReranker", (), {
        "ensure_loaded": lambda self: None,
        "score_pairs": lambda self, pairs, batch_size=1: [0.9],
        "model": type("DummyModel", (), {"peft_config": {"default": type("C", (), {"base_model_name_or_path": "BAAI/bge-reranker-v2-m3"})()}})(),
    })())

    # Write dummy adapter files for directory hash
    adapter_out.mkdir(parents=True)
    (adapter_out / "adapter_config.json").write_text('{"base_model_name_or_path": "BAAI/bge-reranker-v2-m3"}', encoding="utf-8")
    (adapter_out / "adapter_model.bin").write_text("weights", encoding="utf-8")

    report = train_final_adapter(
        pairs_path=pairs_file,
        output_adapter_dir=adapter_out,
        runtime_config={"base_model_name": "BAAI/bge-reranker-v2-m3"},
        mock_run=False,
    )

    manifest_path = adapter_out / "final_run_manifest.json"
    assert manifest_path.is_file()
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["base_model"] == "BAAI/bge-reranker-v2-m3"
    assert "adapter_sha256" in manifest
    assert "optimizer_steps" in manifest
    assert manifest["optimizer_steps"] == 10
