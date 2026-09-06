import json
import zipfile
import pytest
import pandas as pd
from pathlib import Path
from src.production.final_train import train_final_adapter
from src.production.public_rerank import rerank_and_fuse_public_predictions
from src.production.submission import validate_submission, package_submission
from src.retrieval.static_cache import StaticCacheWriter, StaticCandidateRecord


def test_validate_submission_valid():
    expected_qids = {f"pub_{i}" for i in range(10)}
    sub = {f"pub_{i}": [f"doc_{j}" for j in range(5)] for i in range(10)}
    is_valid, errors = validate_submission(sub, expected_qids=expected_qids, max_predictions=5)
    assert is_valid is True
    assert len(errors) == 0


def test_validate_submission_missing_key():
    expected_qids = {"q1", "q2"}
    sub = {"q1": ["d1", "d2"]}
    is_valid, errors = validate_submission(sub, expected_qids=expected_qids)
    assert is_valid is False
    assert any("Missing 1 query IDs" in e for e in errors)


def test_validate_submission_duplicate_predictions():
    expected_qids = {"q1"}
    sub = {"q1": ["d1", "d1", "d2"]}
    is_valid, errors = validate_submission(sub, expected_qids=expected_qids)
    assert is_valid is False
    assert any("Duplicate document IDs" in e for e in errors)


def test_validate_submission_too_many_predictions():
    expected_qids = {"q1"}
    sub = {"q1": ["d1", "d2", "d3", "d4", "d5", "d6"]}
    is_valid, errors = validate_submission(sub, expected_qids=expected_qids, max_predictions=5)
    assert is_valid is False
    assert any("exceeds max 5" in e for e in errors)


def test_package_submission(tmp_path):
    sub = {f"q_{i}": ["d1", "d2", "d3"] for i in range(5)}
    out_dir = tmp_path / "submission_out"
    json_p, zip_p = package_submission(sub, out_dir=out_dir)

    assert json_p.is_file()
    assert zip_p.is_file()
    assert zipfile.is_zipfile(zip_p)

    with zipfile.ZipFile(zip_p, "r") as zf:
        assert "submission.json" in zf.namelist()


def test_final_train_nonmock_calls_train_reranker(tmp_path):
    pairs_file = tmp_path / "final_pairs.parquet"
    df = pd.DataFrame({
        "query_id": ["q1", "q1"],
        "query_text": ["hoi luat 1", "hoi luat 1"],
        "doc_id": ["docA", "docB"],
        "label": [1.0, 0.0],
        "evidence_text": ["van ban A", "van ban B"],
        "fold": [0, 0],
    })
    df.to_parquet(pairs_file, index=False)

    adapter_out = tmp_path / "adapter_final"
    report = train_final_adapter(
        pairs_path=pairs_file,
        output_adapter_dir=adapter_out,
        runtime_config={
            "base_model_name": "mock",
            "allow_mock_base_model": True,
            "max_steps": 1,
            "batch_size": 2,
            "device": "cpu",
            "enforce_full_coverage_steps": False,
        },
        mock_run=False,
    )
    assert report["status"] == "PASS"
    assert report["optimizer_steps"] >= 1
    assert report["param_diff"] > 0
    assert (adapter_out / "adapter_config.json").is_file()
    assert (adapter_out / "final_run_manifest.json").is_file()


def test_final_train_rejects_zero_optimizer_steps(tmp_path, monkeypatch):
    from src.production import final_train

    pairs_file = tmp_path / "pairs.parquet"
    pd.DataFrame({
        "query_id": ["q1", "q1"],
        "label": [1.0, 0.0],
        "query_text": ["q", "q"],
        "doc_id": ["dA", "dB"],
        "evidence_text": ["eA", "eB"],
    }).to_parquet(pairs_file)

    def fake_train(**kwargs):
        return {"global_steps": 0, "final_loss": 0.5, "param_diff": 0.1}

    monkeypatch.setattr(final_train, "train_reranker", fake_train)
    with pytest.raises(ValueError, match=r"optimizer_steps \(0\) <= 0"):
        final_train.train_final_adapter(
            pairs_file,
            tmp_path / "ad",
            runtime_config={"base_model_name": "mock", "allow_mock_base_model": True},
            mock_run=False,
        )


def test_final_train_rejects_zero_param_diff(tmp_path, monkeypatch):
    from src.production import final_train

    pairs_file = tmp_path / "pairs.parquet"
    pd.DataFrame({
        "query_id": ["q1", "q1"],
        "label": [1.0, 0.0],
        "query_text": ["q", "q"],
        "doc_id": ["dA", "dB"],
        "evidence_text": ["eA", "eB"],
    }).to_parquet(pairs_file)

    def fake_train(**kwargs):
        return {"global_steps": 5, "final_loss": 0.5, "param_diff": 0.0}

    monkeypatch.setattr(final_train, "train_reranker", fake_train)
    with pytest.raises(ValueError, match=r"param_diff \(0\.0\) <= 0"):
        final_train.train_final_adapter(
            pairs_file,
            tmp_path / "ad",
            runtime_config={"base_model_name": "mock", "allow_mock_base_model": True},
            mock_run=False,
        )


def test_public_rerank_reads_production_lock_and_uses_frozen_fusion(tmp_path):
    cands_p = tmp_path / "public_candidates.parquet"
    writer = StaticCacheWriter(str(cands_p))
    writer.write_record(StaticCandidateRecord("pub1", "bm25", 1, "docA", 10.0))
    writer.write_record(StaticCandidateRecord("pub1", "bm25", 2, "docB", 8.0))
    writer.close()

    lock_p = tmp_path / "production_lock.json"
    lock_data = {
        "status": "LOCKED",
        "config": {
            "fusion": {
                "weights": {"bm25": 1.0, "reranker": 2.5},
                "top_k": 5,
                "rerank_k": 10,
            }
        },
    }
    lock_p.write_text(json.dumps(lock_data), encoding="utf-8")

    preds = rerank_and_fuse_public_predictions(
        public_candidates_path=cands_p,
        production_lock_path=lock_p,
        top_k=2,
    )
    assert "pub1" in preds
    assert preds["pub1"][0] == "docA"
    assert preds["pub1"][1] == "docB"


def test_public_rerank_uses_public_evidence(tmp_path):
    cands_p = tmp_path / "public_candidates.parquet"
    writer = StaticCacheWriter(str(cands_p))
    writer.write_record(StaticCandidateRecord("pub1", "bm25", 1, "docA", 10.0))
    writer.close()

    ev_p = tmp_path / "public_evidence.parquet"
    pd.DataFrame({
        "query_id": ["pub1"],
        "doc_id": ["docA"],
        "evidence_text": ["[DOCUMENT] Van ban A [EVIDENCE 1] Noi dung"],
    }).to_parquet(ev_p)

    lock_p = tmp_path / "production_lock.json"
    lock_data = {
        "status": "LOCKED",
        "config": {"fusion": {"weights": {"bm25": 1.0, "reranker": 2.5}}},
    }
    lock_p.write_text(json.dumps(lock_data), encoding="utf-8")

    preds = rerank_and_fuse_public_predictions(
        public_candidates_path=cands_p,
        production_lock_path=lock_p,
        public_evidence_path=ev_p,
        top_k=1,
    )
    assert preds["pub1"] == ["docA"]
