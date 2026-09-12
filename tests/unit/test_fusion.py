import pytest
from src.ranking.fusion import ReciprocalRankFusion, LightGBMRanker
from src.ranking.selector import TopKSelector

def test_rrf_fusion():
    candidates = [
        {"doc_id": "doc1", "bm25_rank": 1, "exact_match_score": 1.0, "memory_score": 0.0, "reranker_score": 2.5},
        {"doc_id": "doc2", "bm25_rank": 2, "exact_match_score": 0.0, "memory_score": 0.9, "reranker_score": 1.0},
        {"doc_id": "doc3", "bm25_rank": 20, "exact_match_score": 0.0, "memory_score": 0.0, "reranker_score": -1.0}
    ]
    fuser = ReciprocalRankFusion()
    ranked = fuser.rank_candidates(candidates)
    assert len(ranked) == 3
    assert ranked[0]["doc_id"] == "doc1"

def test_topk_selector():
    candidates = [
        {"doc_id": "doc1", "final_score": 0.9},
        {"doc_id": "doc2", "final_score": 0.8},
        {"doc_id": "doc1", "final_score": 0.75}, # duplicate
        {"doc_id": "doc3", "final_score": 0.7},
        {"doc_id": "doc4", "final_score": 0.6},
        {"doc_id": "doc5", "final_score": 0.5},
        {"doc_id": "doc6", "final_score": 0.4}
    ]
    selector = TopKSelector(max_k=5)
    selected = selector.select(candidates)
    assert len(selected) == 5
    assert len(set(selected)) == 5
    assert selected == ["doc1", "doc2", "doc3", "doc4", "doc5"]
