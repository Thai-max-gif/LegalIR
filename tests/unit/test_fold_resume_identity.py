"""Fold resume must validate identity; stale artifacts must force recompute."""
import json
from pathlib import Path

import pandas as pd

from src.pipeline.oof_runner import OOFRunner, _fold_expected_identity


def _minimal_runner(tmp_path: Path) -> OOFRunner:
    r = OOFRunner(
        data_dir=tmp_path / "data",
        index_dir=tmp_path / "idx",
        output_dir=tmp_path / "cv",
        config_path=None,
        num_folds=1,
        candidate_k=150,
        rerank_k=50,
        use_reranker=False,
        reranker_model="BAAI/bge-reranker-v2-m3",
        train_reranker_per_fold=False,
        smoke=False,
    )
    r.split_provenance = {"random_5fold": {"sha256": "abc123", "source": "input"}}
    return r


def test_expected_identity_hashes_are_stable():
    info = {"train_query_ids": ["q1", "q2"], "val_query_ids": ["q3"]}
    a = _fold_expected_identity(0, info, smoke=False, smoke_sample_size=20,
                                reranker_model="BAAI/bge-reranker-v2-m3",
                                candidate_k=150, rerank_k=50, precision="bf16",
                                split_provenance={"random_5fold": {"sha256": "abc"}})
    b = _fold_expected_identity(0, info, smoke=False, smoke_sample_size=20,
                                reranker_model="BAAI/bge-reranker-v2-m3",
                                candidate_k=150, rerank_k=50, precision="bf16",
                                split_provenance={"random_5fold": {"sha256": "abc"}})
    assert a == b
    c = _fold_expected_identity(0, {"train_query_ids": ["q1"], "val_query_ids": ["q3"]},
                                smoke=False, smoke_sample_size=20,
                                reranker_model="BAAI/bge-reranker-v2-m3",
                                candidate_k=150, rerank_k=50, precision="bf16",
                                split_provenance=None)
    assert c["train_ids_sha256"] != a["train_ids_sha256"]


def test_reuse_accepts_matching_identity(tmp_path):
    r = _minimal_runner(tmp_path)
    fold_info = {"train_query_ids": ["q1", "q2"], "val_query_ids": ["q3", "q4"]}
    fold_dir = tmp_path / "cv" / "fold_0"
    fold_dir.mkdir(parents=True, exist_ok=True)
    identity = r._expected_fold_identity(0, fold_info)
    identity.update({"status": "COMPLETED", "queries_count": 2, "recall@5": 0.5})
    (fold_dir / "complete.json").write_text(json.dumps(identity), encoding="utf-8")
    pd.DataFrame([
        {"query_id": "q3", "predicted_doc_ids": ["d1"]},
        {"query_id": "q4", "predicted_doc_ids": ["d2"]},
    ]).to_parquet(fold_dir / "predictions.parquet", index=False)
    pd.DataFrame([
        {"query_id": "q3", "candidate_doc_ids": ["d1"]},
        {"query_id": "q4", "candidate_doc_ids": ["d2"]},
    ]).to_parquet(fold_dir / "candidates.parquet", index=False)
    (fold_dir / "metrics.json").write_text(json.dumps({"fold": 0, "recall@5": 0.5}), encoding="utf-8")

    preds, cands, feats, metrics = r._try_reuse_completed_fold(0, fold_info, fold_dir)
    assert preds is not None and metrics is not None
    assert set(preds.keys()) == {"q3", "q4"}


def test_reuse_rejects_split_change(tmp_path):
    r = _minimal_runner(tmp_path)
    fold_info = {"train_query_ids": ["q1", "q2"], "val_query_ids": ["q3", "q4"]}
    fold_dir = tmp_path / "cv" / "fold_0"
    fold_dir.mkdir(parents=True, exist_ok=True)
    identity = r._expected_fold_identity(0, fold_info)
    identity.update({"status": "COMPLETED", "queries_count": 2})
    (fold_dir / "complete.json").write_text(json.dumps(identity), encoding="utf-8")
    pd.DataFrame([{"query_id": "q3", "predicted_doc_ids": ["d1"]},
                  {"query_id": "q4", "predicted_doc_ids": ["d2"]}]).to_parquet(
        fold_dir / "predictions.parquet", index=False)
    (fold_dir / "metrics.json").write_text(json.dumps({"fold": 0}), encoding="utf-8")

    # New splits with different val set must not reuse.
    new_info = {"train_query_ids": ["q1", "q2"], "val_query_ids": ["q3", "qX"]}
    preds, _, _, metrics = r._try_reuse_completed_fold(0, new_info, fold_dir)
    assert preds is None and metrics is None


def test_reuse_rejects_prediction_id_mismatch(tmp_path):
    r = _minimal_runner(tmp_path)
    fold_info = {"train_query_ids": ["q1"], "val_query_ids": ["q3", "q4"]}
    fold_dir = tmp_path / "cv" / "fold_0"
    fold_dir.mkdir(parents=True, exist_ok=True)
    identity = r._expected_fold_identity(0, fold_info)
    identity.update({"status": "COMPLETED", "queries_count": 1})
    (fold_dir / "complete.json").write_text(json.dumps(identity), encoding="utf-8")
    # Only one of two expected queries present.
    pd.DataFrame([{"query_id": "q3", "predicted_doc_ids": ["d1"]}]).to_parquet(
        fold_dir / "predictions.parquet", index=False)
    (fold_dir / "metrics.json").write_text(json.dumps({"fold": 0}), encoding="utf-8")

    preds, _, _, metrics = r._try_reuse_completed_fold(0, fold_info, fold_dir)
    assert preds is None and metrics is None


def test_reuse_rejects_config_change(tmp_path):
    r = _minimal_runner(tmp_path)
    fold_info = {"train_query_ids": ["q1"], "val_query_ids": ["q3"]}
    fold_dir = tmp_path / "cv" / "fold_0"
    fold_dir.mkdir(parents=True, exist_ok=True)
    identity = r._expected_fold_identity(0, fold_info)
    identity["reranker_model"] = "SOME/OTHER-MODEL"
    identity.update({"status": "COMPLETED", "queries_count": 1})
    (fold_dir / "complete.json").write_text(json.dumps(identity), encoding="utf-8")
    pd.DataFrame([{"query_id": "q3", "predicted_doc_ids": ["d1"]}]).to_parquet(
        fold_dir / "predictions.parquet", index=False)
    (fold_dir / "metrics.json").write_text(json.dumps({"fold": 0}), encoding="utf-8")

    preds, _, _, metrics = r._try_reuse_completed_fold(0, fold_info, fold_dir)
    assert preds is None and metrics is None
