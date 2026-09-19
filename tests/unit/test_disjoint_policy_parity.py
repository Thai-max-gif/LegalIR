"""Disjoint trained evaluation must use the fixed predeclared RRF policy.

Regression: the disjoint pass previously fed reranked candidates directly
into the selector, so its score did not evaluate the submission scoring
policy (fixed RRF + selector) used by public inference.
"""
import json
from pathlib import Path

import pandas as pd

from src.pipeline.oof_runner import OOFRunner
from src.ranking.evidence_pack import EvidencePackBuilder
from src.ranking.fusion import ReciprocalRankFusion
from src.ranking.reranker import CrossEncoderReranker
from src.retrieval.hybrid_search import HybridSearchEngine


def _evidence():
    return EvidencePackBuilder(macro_chunks=[
        {"chunk_id": "docA-c", "doc_id": "docA", "text_norm": "docA evidence"},
        {"chunk_id": "docB-c", "doc_id": "docB", "text_norm": "docB evidence"},
    ])


def _tiny_repo(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"doc_id": "docA", "title": "A"}, {"doc_id": "docB", "title": "B"}]).to_parquet(
        data_dir / "documents.parquet", index=False)
    pd.DataFrame([
        {"query_id": "q1", "question_norm": "train question"},
        {"query_id": "q2", "question_norm": "heldout question"},
    ]).to_parquet(data_dir / "queries_train.parquet", index=False)
    pd.DataFrame([
        {"query_id": "q1", "doc_id": "docB"},
        {"query_id": "q2", "doc_id": "docA"},
    ]).to_parquet(data_dir / "qrels_train.parquet", index=False)
    splits = [{"train_query_ids": ["q1", "q2"], "val_query_ids": ["q1"]},
              {"train_query_ids": ["q1"], "val_query_ids": ["q2"]}]
    sp = data_dir / "splits" / "random_5fold.json"
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps(splits), encoding="utf-8")
    dj = {"train_query_ids": ["q1"], "val_query_ids": ["q2"]}
    djp = data_dir / "splits" / "doc_disjoint_split.json"
    djp.write_text(json.dumps(dj), encoding="utf-8")
    return data_dir, sp, djp


def _branch_cands():
    # docB looks best to the reranker alone; docA dominates every branch.
    return [
        {"doc_id": "docB", "bm25_rank": 999.0, "bm25_pyvi_rank": 999.0,
         "dense_rank": 999.0, "memory_rank": 999.0, "exact_score": 0.0},
        {"doc_id": "docA", "bm25_rank": 1.0, "bm25_pyvi_rank": 1.0,
         "dense_rank": 1.0, "memory_rank": 1.0, "exact_score": 1.0},
    ]


def test_disjoint_trained_pass_applies_fixed_rrf(monkeypatch, tmp_path):
    data_dir, sp, djp = _tiny_repo(tmp_path)
    runner = OOFRunner(
        data_dir=data_dir, index_dir=tmp_path / "idx", output_dir=tmp_path / "cv",
        splits_path=sp, doc_disjoint_splits_path=djp, config_path=None,
        num_folds=2, candidate_k=10, rerank_k=5, use_reranker=True,
        reranker_model="mock", train_reranker_per_fold=False, smoke=True,
    )
    runner.split_provenance = {}
    runner.load_data()
    runner.evidence_builder = _evidence()

    monkeypatch.setattr(
        HybridSearchEngine, "search_candidates",
        lambda self, query="", top_k=10, exclude_qid=None, q_emb=None, **k: [dict(c) for c in _branch_cands()],
    )

    def stub_score(pairs, batch_size=16, max_length=384):
        return [10.0 if "docB" in passage else -5.0 for _, passage in pairs]

    reranker = CrossEncoderReranker(model_name="mock", score_fn=stub_score)

    rrf_calls: list = []
    orig_predict = ReciprocalRankFusion.predict

    def spy_predict(self, records, **kwargs):
        rrf_calls.append([str(c.get("doc_id")) for c in records])
        return orig_predict(self, records, **kwargs)

    monkeypatch.setattr(ReciprocalRankFusion, "predict", spy_predict)

    report = runner.run_document_disjoint_evaluation(reranker=reranker)

    # Fixed RRF fused the reranked window before selection.
    assert rrf_calls, "fixed RRF was not applied in the disjoint trained pass"
    assert rrf_calls[0] == ["docB", "docA"], "RRF must receive reranker-ordered candidates"
    # Branch mass outweighs the lone reranker signal: docA first, unlike
    # selector-only order which would keep reranker-top docB first (MRR 0.5).
    assert report["trained_reranker_system"]["val_queries"] == 1
    assert report["fusion_policy"] == "predeclared_rrf"
    assert report["trained_reranker_system"]["mrr"] == 1.0


def test_oof_fold_applies_same_fixed_rrf_policy(monkeypatch, tmp_path):
    """OOF folds must validate the submission scoring policy, not raw
    reranker order — the confirmatory OOF score certifies the RRF+selector
    pipeline that public inference runs."""
    data_dir, sp, _djp = _tiny_repo(tmp_path)
    runner = OOFRunner(
        data_dir=data_dir, index_dir=tmp_path / "idx", output_dir=tmp_path / "cv",
        splits_path=sp, config_path=None,
        num_folds=2, candidate_k=10, rerank_k=5, use_reranker=True,
        reranker_model="mock", train_reranker_per_fold=False, smoke=True,
    )
    runner.load_data()
    runner.evidence_builder = _evidence()

    monkeypatch.setattr(
        HybridSearchEngine, "search_candidates",
        lambda self, query="", top_k=10, exclude_qid=None, q_emb=None, **k: [dict(c) for c in _branch_cands()],
    )

    def stub_score(pairs, batch_size=16, max_length=384):
        return [10.0 if "docB" in passage else -5.0 for _, passage in pairs]

    reranker = CrossEncoderReranker(model_name="mock", score_fn=stub_score)

    rrf_calls: list = []
    orig_predict = ReciprocalRankFusion.predict

    def spy_predict(self, records, **kwargs):
        rrf_calls.append([str(c.get("doc_id")) for c in records])
        return orig_predict(self, records, **kwargs)

    monkeypatch.setattr(ReciprocalRankFusion, "predict", spy_predict)

    fold_info = {"train_query_ids": ["q1"], "val_query_ids": ["q2"]}
    f_preds, _cands, _feats, f_metrics, _rt = runner.run_fold(
        fold_idx=0, fold_info=fold_info, reranker=reranker)

    assert rrf_calls, "fixed RRF was not applied in the OOF fold pass"
    assert rrf_calls[0] == ["docB", "docA"], "RRF must receive reranker-ordered candidates"
    assert f_preds["q2"][0] == "docA"
    assert f_metrics["mrr"] == 1.0
