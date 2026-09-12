import pytest

from src.retrieval.hybrid_search import HybridSearchEngine
from src.retrieval.types import CandidateRecord


class StubRetriever:
    def __init__(self, return_items):
        self.return_items = return_items

    def retrieve(self, query, top_k=50):
        return self.return_items[:top_k]


class StubExactMatcher:
    def __init__(self, return_dict):
        self.return_dict = return_dict
        self.calls = []

    def match(self, query):
        self.calls.append(query)
        return self.return_dict


class StubQuestionMemory:
    def __init__(self, return_items):
        self.return_items = return_items
        self.calls = []

    def search(self, query, top_k=5):
        self.calls.append((query, top_k))
        return self.return_items[:top_k]


def test_hybrid_search_unions_all_branches_and_sorts_stably():
    bm25_hits = [{"doc_id": "doc1", "score": 10.0, "bm25_score": 10.0, "bm25_best_score": 10.0}]
    exact_hits = {"doc2": {"score": 1.0, "exact_legal_number": True}}
    dense_hits = [{"doc_id": "doc3", "score": 0.95, "dense_score": 0.95, "dense_best_score": 0.95}]

    engine = HybridSearchEngine(
        bm25_retriever=StubRetriever(bm25_hits),
        exact_matcher=StubExactMatcher(exact_hits),
        dense_retriever=StubRetriever(dense_hits),
    )

    cands = engine.search_candidates("query", top_k=150)
    cand_doc_ids = {c["doc_id"] for c in cands}
    assert {"doc1", "doc2", "doc3"} <= cand_doc_ids

    # Check that CandidateRecord contains expected feature fields
    for c in cands:
        assert "rrf_score" in c
        assert "source_count" in c
        assert "exact_legal_number" in c
        assert "bm25_best_score" in c
        assert "dense_best_score" in c


def test_hybrid_search_4_branches_uses_unique_ids_and_weighted_rrf():
    bm25 = StubRetriever([
        {"doc_id": "shared", "score": 10.0, "bm25_score": 10.0},
        {"doc_id": "bm25-only", "score": 9.0, "bm25_score": 9.0},
    ])
    dense = StubRetriever([
        {"doc_id": "dense-only", "score": 0.9, "dense_score": 0.9},
        {"doc_id": "shared", "score": 0.8, "dense_score": 0.8},
    ])
    memory = StubQuestionMemory([
        {"doc_id": "memory-only", "score": 0.95, "memory_score": 0.95},
        {"doc_id": "shared", "score": 0.90, "memory_score": 0.90},
    ])
    exact = StubExactMatcher({
        "exact-only": {"score": 1.0, "exact_legal_number": True},
        "shared": {"score": 0.8, "exact_title": True},
    })

    engine = HybridSearchEngine(
        bm25_retriever=bm25,
        dense_retriever=dense,
        question_memory=memory,
        exact_matcher=exact,
    )

    candidates = engine.search("query", top_k_candidates=10, rrf_k=60)

    assert {candidate["doc_id"] for candidate in candidates} == {
        "shared", "bm25-only", "dense-only", "memory-only", "exact-only",
    }
    shared = next(candidate for candidate in candidates if candidate["doc_id"] == "shared")
    assert shared["source_count"] == 4
    assert set(shared["branch_contributions"]) == {"bm25", "dense", "memory", "exact"}
    assert shared["rrf_score"] == sum(shared["branch_contributions"].values())
    assert shared["rrf_score"] == pytest.approx(
        1.0 / 61 + 1.2 / 62 + 2.0 / 62 + 2.5 / 62
    )
    assert shared["branch_ranks"] == {"bm25": 1, "dense": 2, "memory": 2, "exact": 2}
    assert len({candidate["doc_id"] for candidate in candidates}) == len(candidates)
    assert bm25.return_items and dense.return_items
    assert memory.calls == [("query", 10)]
    assert exact.calls == ["query"]
