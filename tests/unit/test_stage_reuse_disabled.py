"""F4: completed-stage reuse disabled by default; opt-in only."""
import json
from pathlib import Path

import pandas as pd

from src.pipeline.oof_runner import OOFRunner


def _tiny_data(data_dir: Path):
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"doc_id": "d1", "title": "T1"},
        {"doc_id": "d2", "title": "T2"},
    ]).to_parquet(data_dir / "documents.parquet", index=False)
    pd.DataFrame([
        {"query_id": "q1", "question_norm": "alpha"},
        {"query_id": "q2", "question_norm": "beta"},
        {"query_id": "q3", "question_norm": "gamma"},
    ]).to_parquet(data_dir / "queries_train.parquet", index=False)
    pd.DataFrame([
        {"query_id": "q1", "doc_id": "d1"},
        {"query_id": "q2", "doc_id": "d1"},
        {"query_id": "q3", "doc_id": "d2"},
    ]).to_parquet(data_dir / "qrels_train.parquet", index=False)


def _write_valid_fold_artifacts(runner: OOFRunner, fold_info: dict, out_dir: Path):
    fold_dir = out_dir / "fold_0"
    fold_dir.mkdir(parents=True, exist_ok=True)
    identity = runner._expected_fold_identity(0, fold_info)
    raw_val = [str(x) for x in fold_info["val_query_ids"]]
    identity.update({"status": "COMPLETED", "queries_count": len(raw_val), "recall@5": 0.5})
    (fold_dir / "complete.json").write_text(json.dumps(identity), encoding="utf-8")
    pd.DataFrame([{"query_id": q, "predicted_doc_ids": ["d1"]} for q in raw_val]).to_parquet(
        fold_dir / "predictions.parquet", index=False)
    pd.DataFrame([{"query_id": q, "candidate_doc_ids": ["d1"]} for q in raw_val]).to_parquet(
        fold_dir / "candidates.parquet", index=False)
    (fold_dir / "metrics.json").write_text(json.dumps({"fold": 0, "recall@5": 0.5,
                                                        "recall@1": 0.5, "recall@3": 0.5,
                                                        "precision@5": 0.2}), encoding="utf-8")


def _make_runner(tmp_path: Path, **kw):
    data_dir = tmp_path / "data"
    _tiny_data(data_dir)
    splits = [{"train_query_ids": ["q1", "q2"], "val_query_ids": ["q3"]},
              {"train_query_ids": ["q3"], "val_query_ids": ["q1", "q2"]}]
    splits_path = data_dir / "splits" / "random_5fold.json"
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    splits_path.write_text(json.dumps(splits), encoding="utf-8")
    return OOFRunner(
        data_dir=data_dir,
        index_dir=tmp_path / "idx",
        output_dir=tmp_path / "cv",
        splits_path=splits_path,
        config_path=None,
        num_folds=1,
        candidate_k=10,
        rerank_k=5,
        use_reranker=False,
        reranker_model="mock",
        train_reranker_per_fold=False,
        smoke=True,
        smoke_sample_size=20,
        doc_disjoint=False,
        **kw,
    ), splits[0]


def test_default_disables_reuse(monkeypatch, tmp_path):
    runner, fold_info = _make_runner(tmp_path)
    assert runner.allow_stage_reuse is False
    _write_valid_fold_artifacts(runner, fold_info, tmp_path / "cv")
    calls = []
    orig_run_fold = OOFRunner.run_fold

    def counting_run_fold(self, fold_idx=0, fold_info=None, reranker=None):
        calls.append(fold_idx)
        return orig_run_fold(self, fold_idx=fold_idx, fold_info=fold_info, reranker=reranker)

    monkeypatch.setattr(OOFRunner, "run_fold", counting_run_fold)
    # Stub audit to avoid config dependency.
    import src.pipeline.oof_runner as oof_mod
    monkeypatch.setattr(oof_mod, "audit_system_parameters",
                        lambda **k: {"total_learned_parameters": 1,
                                     "total_parameters_billions": 0.0,
                                     "budget_utilization_pct": 0.0})
    runner.run()
    assert calls == [0], "disabled reuse must recompute the fold"


def test_opt_in_reuses_valid_artifacts(monkeypatch, tmp_path):
    runner, fold_info = _make_runner(tmp_path, allow_stage_reuse=True)
    assert runner.allow_stage_reuse is True
    _write_valid_fold_artifacts(runner, fold_info, tmp_path / "cv")
    calls = []
    orig_run_fold = OOFRunner.run_fold

    def counting_run_fold(self, fold_idx=0, fold_info=None, reranker=None):
        calls.append(fold_idx)
        return orig_run_fold(self, fold_idx=fold_idx, fold_info=fold_info, reranker=reranker)

    monkeypatch.setattr(OOFRunner, "run_fold", counting_run_fold)
    import src.pipeline.oof_runner as oof_mod
    monkeypatch.setattr(oof_mod, "audit_system_parameters",
                        lambda **k: {"total_learned_parameters": 1,
                                     "total_parameters_billions": 0.0,
                                     "budget_utilization_pct": 0.0})
    report = runner.run()
    assert calls == [], "opt-in valid reuse must skip run_fold"
    assert report["total_evaluated_queries"] == 1
