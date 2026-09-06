import json
import pytest
import pandas as pd
from pathlib import Path

from src.production.final_train import train_final_adapter


def test_final_train_manifest_breaks_down_all_parameter_classes(tmp_path, monkeypatch):
    pairs_file = tmp_path / "pairs.parquet"
    pd.DataFrame({
        "query_id": ["q1", "q1"],
        "query_text": ["q", "q"],
        "doc_id": ["dA", "dB"],
        "label": [1.0, 0.0],
        "evidence_text": ["eA", "eB"],
    }).to_parquet(pairs_file)

    adapter_out = tmp_path / "adapter"

    fake_report = {
        "status": "PASS",
        "optimizer_steps": 10,
        "final_loss": 0.15,
        "param_diff": 0.02,
        "trainable_parameters": 8_128_513,
    }

    monkeypatch.setattr("src.production.final_train.train_reranker", lambda **kw: fake_report)
    monkeypatch.setattr("src.production.final_train.CrossEncoderReranker", lambda **kw: type("DummyReranker", (), {
        "ensure_loaded": lambda self: None,
        "score_pairs": lambda self, pairs, batch_size=1: [0.9],
        "model": type("DummyModel", (), {
            "peft_config": {"default": type("C", (), {"base_model_name_or_path": "BAAI/bge-reranker-v2-m3"})()},
            "named_parameters": lambda self: [("lora_a", type("T", (), {"numel": lambda: 8_128_513})())],
        })(),
    })())

    adapter_out.mkdir(parents=True, exist_ok=True)
    (adapter_out / "adapter_config.json").write_text('{"base_model_name_or_path": "BAAI/bge-reranker-v2-m3"}', encoding="utf-8")
    (adapter_out / "adapter_model.bin").write_text("weights", encoding="utf-8")

    report = train_final_adapter(
        pairs_path=pairs_file,
        output_adapter_dir=adapter_out,
        runtime_config={
            "base_model_name": "BAAI/bge-reranker-v2-m3",
            "expected_query_count": 1,
        },
        mock_run=False,
    )

    manifest_p = adapter_out / "final_run_manifest.json"
    assert manifest_p.is_file()
    with open(manifest_p, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Must distinguish parameter classes
    assert "base_model_parameters" in manifest
    assert "trainable_parameters" in manifest
    assert "dense_retriever_parameters" in manifest
    assert "total_system_learned_parameters" in manifest
    assert manifest["total_system_learned_parameters"] < 4_000_000_000
    assert manifest["base_model_parameters"] == 567_755_777
    assert manifest["dense_retriever_parameters"] == 134_998_272


def test_final_train_rejects_query_coverage_gap(tmp_path):
    pairs_file = tmp_path / "pairs.parquet"
    # Only 1 query when 7000 expected
    pd.DataFrame({
        "query_id": ["q1", "q1"],
        "query_text": ["q", "q"],
        "doc_id": ["dA", "dB"],
        "label": [1.0, 0.0],
        "evidence_text": ["eA", "eB"],
    }).to_parquet(pairs_file)

    adapter_out = tmp_path / "adapter"

    with pytest.raises(ValueError, match="query coverage gap|expected 7000"):
        train_final_adapter(
            pairs_path=pairs_file,
            output_adapter_dir=adapter_out,
            runtime_config={"base_model_name": "BAAI/bge-reranker-v2-m3", "expected_query_count": 7000},
            mock_run=False,
        )


def test_final_train_requires_positive_and_negative_for_all_queries(tmp_path, monkeypatch):
    pairs_file = tmp_path / "pairs.parquet"
    # q1 has positive and negative, but q2 has only positive!
    pd.DataFrame({
        "query_id": ["q1", "q1", "q2"],
        "query_text": ["q1", "q1", "q2"],
        "doc_id": ["dA", "dB", "dC"],
        "label": [1.0, 0.0, 1.0],
        "evidence_text": ["eA", "eB", "eC"],
    }).to_parquet(pairs_file)

    adapter_out = tmp_path / "adapter"

    with pytest.raises(ValueError, match="negative pairs|positive pairs|query coverage"):
        train_final_adapter(
            pairs_path=pairs_file,
            output_adapter_dir=adapter_out,
            runtime_config={"base_model_name": "BAAI/bge-reranker-v2-m3", "expected_query_count": 2},
            mock_run=False,
        )


def test_final_train_rejects_exceeding_4b_budget(tmp_path, monkeypatch):
    pairs_file = tmp_path / "pairs.parquet"
    pd.DataFrame({
        "query_id": ["q1", "q1"],
        "query_text": ["q", "q"],
        "doc_id": ["dA", "dB"],
        "label": [1.0, 0.0],
        "evidence_text": ["eA", "eB"],
    }).to_parquet(pairs_file)

    adapter_out = tmp_path / "adapter"

    fake_report = {
        "status": "PASS",
        "optimizer_steps": 10,
        "final_loss": 0.15,
        "param_diff": 0.02,
        "trainable_parameters": 3_500_000_000,
    }

    monkeypatch.setattr("src.production.final_train.train_reranker", lambda **kw: fake_report)
    monkeypatch.setattr("src.production.final_train.CrossEncoderReranker", lambda **kw: type("DummyReranker", (), {
        "ensure_loaded": lambda self: None,
        "score_pairs": lambda self, pairs, batch_size=1: [0.9],
        "model": type("DummyModel", (), {
            "peft_config": {"default": type("C", (), {"base_model_name_or_path": "BAAI/bge-reranker-v2-m3"})()},
        })(),
    })())

    adapter_out.mkdir(parents=True, exist_ok=True)
    (adapter_out / "adapter_config.json").write_text('{"base_model_name_or_path": "BAAI/bge-reranker-v2-m3"}', encoding="utf-8")
    (adapter_out / "adapter_model.bin").write_text("weights", encoding="utf-8")

    # Base (567M) + Dense (135M) + Trainable (3.5B) = 4.2B >= 4B
    with pytest.raises(ValueError, match="budget exceeded|4,000,000,000"):
        train_final_adapter(
            pairs_path=pairs_file,
            output_adapter_dir=adapter_out,
            runtime_config={"base_model_name": "BAAI/bge-reranker-v2-m3", "expected_query_count": 1},
            mock_run=False,
        )
