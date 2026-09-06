import json
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock

from src.production.public_rerank import (
    rerank_and_fuse_public_predictions,
    normalize_public_queries,
)
from src.ranking.fusion import ReciprocalRankFusion, LightGBMRanker
from src.retrieval.static_cache import StaticCacheWriter, StaticCandidateRecord


def test_public_json_is_normalized_to_question_text():
    raw_public = {
        "q1": {"question": "Quy định về hợp đồng dân sự?", "answer": None},
        "q2": "Thời hiệu khởi kiện vụ án dân sự?",
        "q3": {"question": "Quyền của người sử dụng đất"},
    }
    normalized = normalize_public_queries(raw_public)
    assert normalized["q1"] == "Quy định về hợp đồng dân sự?"
    assert normalized["q2"] == "Thời hiệu khởi kiện vụ án dân sự?"
    assert normalized["q3"] == "Quyền của người sử dụng đất"


def test_public_reranker_never_receives_dict_as_query_text(tmp_path, monkeypatch):
    cands_p = tmp_path / "public_candidates.parquet"
    writer = StaticCacheWriter(str(cands_p))
    writer.write_record(StaticCandidateRecord("q1", "bm25_legal", 1, "doc1", 10.0))
    writer.write_record(StaticCandidateRecord("q1", "dense", 1, "doc1", 0.9))
    writer.close()

    lock_p = tmp_path / "production_lock.json"
    lock_data = {
        "status": "LOCKED",
        "config": {
            "fusion": {
                "method": "reciprocal_rank_fusion",
                "k": 60,
                "weights": {"bm25": 1.0, "dense": 1.0, "rerank": 2.0},
            }
        }
    }
    lock_p.write_text(json.dumps(lock_data), encoding="utf-8")

    evidence_p = tmp_path / "public_evidence.parquet"
    pd.DataFrame({
        "query_id": ["q1"],
        "doc_id": ["doc1"],
        "evidence_text": ["noi dung dieu luat"],
    }).to_parquet(evidence_p)

    raw_public = {
        "q1": {"question": "Hợp đồng lao động", "answer": None}
    }

    mock_reranker = MagicMock()
    mock_reranker.score_pairs.return_value = [0.95]

    monkeypatch.setattr("src.production.public_rerank.CrossEncoderReranker", lambda **kw: mock_reranker)

    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()

    preds = rerank_and_fuse_public_predictions(
        public_candidates_path=cands_p,
        production_lock_path=lock_p,
        adapter_dir=adapter_dir,
        public_evidence_path=evidence_p,
        public_queries_dict=raw_public,
    )

    assert mock_reranker.score_pairs.called
    pairs_passed = mock_reranker.score_pairs.call_args[0][0]
    assert len(pairs_passed) == 1
    query_text, evidence_text = pairs_passed[0]
    assert isinstance(query_text, str)
    assert query_text == "Hợp đồng lao động"
    assert not query_text.startswith("{")


def test_public_rrf_matches_ranking_fusion_on_same_candidate_records(tmp_path):
    cands_p = tmp_path / "public_candidates.parquet"
    writer = StaticCacheWriter(str(cands_p))
    # Query 1 with 2 docs across branches
    writer.write_record(StaticCandidateRecord("q1", "bm25_legal", 1, "docA", 10.0))
    writer.write_record(StaticCandidateRecord("q1", "bm25_pyvi", 2, "docA", 9.0))
    writer.write_record(StaticCandidateRecord("q1", "dense", 1, "docB", 0.95))
    writer.write_record(StaticCandidateRecord("q1", "bm25_legal", 2, "docB", 8.0))
    writer.close()

    weights = {"bm25": 1.0, "bm25_pyvi": 1.2, "dense": 1.5, "rerank": 2.0}
    lock_p = tmp_path / "production_lock.json"
    lock_data = {
        "status": "LOCKED",
        "config": {
            "fusion": {
                "method": "reciprocal_rank_fusion",
                "k": 60,
                "weights": weights,
            }
        }
    }
    lock_p.write_text(json.dumps(lock_data), encoding="utf-8")

    adapter_scores = {"q1": {"docA": 2.0, "docB": -1.0}}

    preds = rerank_and_fuse_public_predictions(
        public_candidates_path=cands_p,
        production_lock_path=lock_p,
        adapter_scores=adapter_scores,
        top_k=2,
    )

    # Reference computation using ReciprocalRankFusion class directly
    rf = ReciprocalRankFusion(k=60, weights=weights)
    cand_records = [
        {
            "doc_id": "docA",
            "raw_bm25_rank": 1,
            "pyvi_bm25_rank": 2,
            "dense_rank": None,
            "exact_score": 0.0,
            "reranker_score": 2.0,
        },
        {
            "doc_id": "docB",
            "raw_bm25_rank": 2,
            "pyvi_bm25_rank": None,
            "dense_rank": 1,
            "exact_score": 0.0,
            "reranker_score": -1.0,
        }
    ]
    ref_ranked = rf.rank_candidates(cand_records)
    expected_top = [r["doc_id"] for r in ref_ranked]

    assert preds["q1"] == expected_top


def test_public_learned_fusion_matches_oof_feature_schema(tmp_path):
    cands_p = tmp_path / "public_candidates.parquet"
    writer = StaticCacheWriter(str(cands_p))
    writer.write_record(StaticCandidateRecord("q1", "bm25_legal", 1, "docA", 10.0))
    writer.write_record(StaticCandidateRecord("q1", "dense", 1, "docB", 0.95))
    writer.close()

    # Train a minimal linear fallback ranker on core features
    from src.ranking.fusion import LinearRanker
    lr = LinearRanker(feature_cols=["raw_bm25_rank", "dense_rank", "reranker_score"])
    X_dummy = pd.DataFrame({
        "raw_bm25_rank": [1.0, 999.0],
        "dense_rank": [999.0, 1.0],
        "reranker_score": [1.0, 0.5],
    })
    y_dummy = [1.0, 0.0]
    lr.fit(X_dummy, y_dummy)

    fusion_model_path = tmp_path / "fusion_model.json"
    lr.save(fusion_model_path)

    lock_p = tmp_path / "production_lock.json"
    lock_data = {
        "status": "LOCKED",
        "config": {
            "fusion": {
                "method": "learned_ranker",
                "model_file": str(fusion_model_path),
            }
        }
    }
    lock_p.write_text(json.dumps(lock_data), encoding="utf-8")

    adapter_scores = {"q1": {"docA": 1.0, "docB": 0.5}}

    preds = rerank_and_fuse_public_predictions(
        public_candidates_path=cands_p,
        production_lock_path=lock_p,
        fusion_model_path=fusion_model_path,
        adapter_scores=adapter_scores,
        top_k=2,
    )
    assert "q1" in preds
    assert len(preds["q1"]) == 2
    assert preds["q1"][0] == "docA"


def test_public_learned_winner_requires_real_fusion_payload(tmp_path):
    cands_p = tmp_path / "public_candidates.parquet"
    writer = StaticCacheWriter(str(cands_p))
    writer.write_record(StaticCandidateRecord("q1", "bm25_legal", 1, "docA", 10.0))
    writer.close()

    lock_p = tmp_path / "production_lock.json"
    lock_data = {
        "status": "LOCKED",
        "config": {
            "fusion": {
                "method": "learned_ranker",
                "model_file": "missing_model.txt",
            }
        }
    }
    lock_p.write_text(json.dumps(lock_data), encoding="utf-8")

    with pytest.raises((RuntimeError, FileNotFoundError)):
        rerank_and_fuse_public_predictions(
            public_candidates_path=cands_p,
            production_lock_path=lock_p,
            fusion_model_path=tmp_path / "non_existent_fusion_descriptor.json",
        )


def test_public_top5_is_unique_complete_and_official(tmp_path):
    cands_p = tmp_path / "public_candidates.parquet"
    writer = StaticCacheWriter(str(cands_p))
    for i in range(10):
        writer.write_record(StaticCandidateRecord("q1", "bm25_legal", i + 1, f"doc_{i}", 10.0 - i))
    writer.close()

    lock_p = tmp_path / "production_lock.json"
    lock_data = {
        "status": "LOCKED",
        "config": {
            "fusion": {
                "method": "reciprocal_rank_fusion",
                "k": 60,
                "weights": {"bm25": 1.0},
                "top_k": 5,
            }
        }
    }
    lock_p.write_text(json.dumps(lock_data), encoding="utf-8")

    preds = rerank_and_fuse_public_predictions(
        public_candidates_path=cands_p,
        production_lock_path=lock_p,
        top_k=5,
    )
    assert len(preds["q1"]) == 5
    assert len(set(preds["q1"])) == 5
    assert preds["q1"] == [f"doc_{i}" for i in range(5)]
