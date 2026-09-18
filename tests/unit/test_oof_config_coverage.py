"""Effective FULL config identity + exact OOF coverage."""
import json
from pathlib import Path

import pandas as pd

from src.pipeline.oof_runner import OOFRunner


def _tiny_repo(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"doc_id": "d1"}, {"doc_id": "d2"}]).to_parquet(data_dir / "documents.parquet", index=False)
    pd.DataFrame([
        {"query_id": "q1", "question_norm": "a"},
        {"query_id": "q2", "question_norm": "b"},
        {"query_id": "q3", "question_norm": "c"},
    ]).to_parquet(data_dir / "queries_train.parquet", index=False)
    pd.DataFrame([
        {"query_id": "q1", "doc_id": "d1"},
        {"query_id": "q2", "doc_id": "d1"},
        {"query_id": "q3", "doc_id": "d2"},
    ]).to_parquet(data_dir / "qrels_train.parquet", index=False)
    splits = [{"train_query_ids": ["q1", "q2"], "val_query_ids": ["q3"]},
              {"train_query_ids": ["q3"], "val_query_ids": ["q1", "q2"]}]
    sp = data_dir / "splits" / "random_5fold.json"
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps(splits), encoding="utf-8")
    return data_dir, sp


def test_resolved_config_binds_full_depths_and_hash(tmp_path):
    data_dir, sp = _tiny_repo(tmp_path)
    r = OOFRunner(data_dir=data_dir, index_dir=tmp_path / "idx", output_dir=tmp_path / "cv",
                  splits_path=sp, config_path=None, num_folds=2,
                  candidate_k=150, rerank_k=50, smoke=False)
    cfg = r.resolved_run_config()
    # FULL orchestration depths, not algorithm-YAML 100/30.
    assert cfg["candidate_k"] == 150 and cfg["rerank_k"] == 50
    assert len(cfg["config_sha256"]) == 64
    r2 = OOFRunner(data_dir=data_dir, index_dir=tmp_path / "idx", output_dir=tmp_path / "cv",
                   splits_path=sp, config_path=None, num_folds=2,
                   candidate_k=100, rerank_k=30, smoke=False)
    assert r2.resolved_run_config()["config_sha256"] != cfg["config_sha256"]


def test_oof_coverage_failure_when_fold_missing(monkeypatch, tmp_path):
    data_dir, sp = _tiny_repo(tmp_path)
    r = OOFRunner(data_dir=data_dir, index_dir=tmp_path / "idx", output_dir=tmp_path / "cv",
                  splits_path=sp, config_path=None, num_folds=2,
                  candidate_k=10, rerank_k=5, use_reranker=False,
                  reranker_model="mock", smoke=True, doc_disjoint=False)
    import src.pipeline.oof_runner as oof_mod
    monkeypatch.setattr(oof_mod, "audit_system_parameters",
                        lambda **k: {"total_learned_parameters": 1,
                                     "total_parameters_billions": 0.0,
                                     "budget_utilization_pct": 0.0})

    _orig_run_fold = OOFRunner.run_fold

    def one_fold_only(self, fold_idx=0, fold_info=None, reranker=None):
        if fold_idx == 1:
            return {}, {}, [], {"recall@5": 0.0, "recall@1": 0.0, "recall@3": 0.0,
                                "precision@5": 0.0, "candidate_recall@20": 0.0,
                                "candidate_recall@50": 0.0, "candidate_recall@100": 0.0,
                                "candidate_recall@150": 0.0, "elapsed_seconds": 0.0}, {}
        return _orig_run_fold(self, fold_idx=fold_idx, fold_info=fold_info, reranker=reranker)

    # run_fold for fold 0 returns real results; fold 1 returns nothing:
    # global coverage assertion must fail instead of silently scoring subset.
    import pytest
    monkeypatch.setattr(OOFRunner, "run_fold", one_fold_only)
    with pytest.raises(AssertionError, match="coverage failed"):
        r.run()
