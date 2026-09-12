from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.retrieval.bm25_micro import BM25MicroRetriever, tokenize_legal
from src.retrieval.bm25_pyvi import BM25PyViRetriever, tokenize_pyvi
from src.retrieval.candidate_union import (
    DEFAULT_CANDIDATE_CUTOFFS,
    build_candidate_features,
    evaluate_candidate_recall,
)
from src.retrieval.dense_macro import DenseMacroRetriever
from src.retrieval.exact_matcher import ExactMatcher
from src.retrieval.hybrid_search import HybridSearchEngine
from src.retrieval.question_memory import TrainQuestionMemory


# --------------------------------------------------------------------------
# 1. Raw / Legal BM25 Corpus & Query Tokenizer Consistency
# --------------------------------------------------------------------------

def test_raw_legal_bm25_tokenizer_consistency():
    legal_text = "Căn cứ Nghị định số 123/2020/NĐ-CP và Điều 15 Khoản 2 Điểm a Thông tư 15/2021/TT-BTC năm 2023"
    tokens = tokenize_legal(legal_text)

    # Must preserve statutory identifiers
    assert "123/2020/nđ-cp" in tokens
    assert "15/2021/tt-btc" in tokens
    assert "điều" in tokens
    assert "15" in tokens
    assert "khoản" in tokens
    assert "2" in tokens
    assert "điểm" in tokens
    assert "a" in tokens
    assert "2023" in tokens

    # Consistency test: same input tokenized twice yields exact same list
    assert tokenize_legal(legal_text) == tokens


def test_bm25_micro_save_load_roundtrip(tmp_path: Path):
    chunks = [
        {
            "chunk_id": "c1",
            "doc_id": "doc1",
            "legal_number": "123/2020/NĐ-CP",
            "title": "Nghị định 123",
            "article": "Điều 1",
            "text_norm": "Quy định về hóa đơn chứng từ",
        },
        {
            "chunk_id": "c2",
            "doc_id": "doc2",
            "legal_number": "15/2021/TT-BTC",
            "title": "Thông tư 15",
            "article": "Điều 2",
            "text_norm": "Hướng dẫn thực hiện hóa đơn",
        },
    ]

    retriever = BM25MicroRetriever()
    retriever.fit(chunks)

    save_file = tmp_path / "bm25_index.pkl"
    retriever.save(save_file)
    loaded = BM25MicroRetriever.load(save_file)

    res_orig = retriever.retrieve("123/2020/NĐ-CP", top_k=2)
    res_loaded = loaded.retrieve("123/2020/NĐ-CP", top_k=2)

    assert len(res_orig) == len(res_loaded)
    assert res_orig[0]["doc_id"] == res_loaded[0]["doc_id"]
    assert pytest.approx(res_orig[0]["score"]) == res_loaded[0]["score"]


# --------------------------------------------------------------------------
# 2. PyVi BM25 Corpus & Query Tokenizer Consistency
# --------------------------------------------------------------------------

def test_pyvi_bm25_tokenizer_consistency():
    text = "Thủ tục đăng ký quyền sử dụng đất đai và bất động sản"
    tokens = tokenize_pyvi(text)

    # PyVi segments multi-word phrases with underscores
    assert any("đăng_ký" in tok or "đất_đai" in tok or "bất_động_sản" in tok for tok in tokens)

    # Same text tokenized as query yields exact same tokens
    query_tokens = tokenize_pyvi("đăng ký quyền sử dụng đất đai")
    for tok in query_tokens:
        if "_" in tok:
            assert tok in tokens


def test_bm25_pyvi_save_load_roundtrip(tmp_path: Path):
    chunks = [
        {
            "chunk_id": "c1",
            "doc_id": "doc1",
            "title": "Luật Đất đai",
            "article": "Điều 1",
            "text_norm": "Quy định về quản lý và sử dụng đất đai",
        },
        {
            "chunk_id": "c2",
            "doc_id": "doc2",
            "title": "Luật Nhà ở",
            "article": "Điều 2",
            "text_norm": "Quy định về sở hữu và phát triển nhà ở",
        },
    ]

    retriever = BM25PyViRetriever()
    retriever.fit(chunks)

    save_file = tmp_path / "bm25_pyvi.pkl"
    retriever.save(save_file)
    loaded = BM25PyViRetriever.load(save_file)

    res_orig = retriever.retrieve("sử dụng đất đai", top_k=2)
    res_loaded = loaded.retrieve("sử dụng đất đai", top_k=2)

    assert len(res_orig) == len(res_loaded)
    assert res_orig[0]["doc_id"] == res_loaded[0]["doc_id"]
    assert pytest.approx(res_orig[0]["score"]) == res_loaded[0]["score"]


# --------------------------------------------------------------------------
# 3. Legal Boost Actually Improves Rank for Statutory Queries
# --------------------------------------------------------------------------

def test_legal_boost_improves_rank_for_statutory_references():
    chunks = [
        {
            "chunk_id": "c_general",
            "doc_id": "doc_general",
            "legal_number": "01/2015/NĐ-CP",
            "title": "Nghị định quy định chung",
            "article": "Điều 1",
            "clause": "Khoản 1",
            "point": "",
            "year": "2015",
            "doc_type": "Nghị định",
            "text_norm": "quy định xử phạt vi phạm giao thông đường bộ tốc độ xe máy phạt tiền",
        },
        {
            "chunk_id": "c_target",
            "doc_id": "doc_target",
            "legal_number": "100/2019/NĐ-CP",
            "title": "Nghị định 100 quy định xử phạt",
            "article": "Điều 5",
            "clause": "Khoản 3",
            "point": "Điểm a",
            "year": "2019",
            "doc_type": "Nghị định",
            "text_norm": "xử phạt người điều khiển phương tiện vi phạm quy định",
        },
    ]

    retriever = BM25MicroRetriever(
        legal_boost_weights={"legal_number": 10.0, "article": 5.0, "clause": 2.0, "point": 1.0}
    ).fit(chunks)

    # Query with specific statutory reference to doc_target
    query = "Mức phạt theo Điều 5 Khoản 3 Điểm a Nghị định 100/2019/NĐ-CP"
    results = retriever.retrieve(query, top_k=2)

    assert len(results) == 2
    # doc_target MUST be rank 1 due to statutory boosting on legal_number, article, clause, point
    assert results[0]["doc_id"] == "doc_target"
    assert results[0]["bm25_legal_boost"] > 0
    assert results[0]["score"] > results[1]["score"]


# --------------------------------------------------------------------------
# 4. Exact Matcher Handling Null / NaN Metadata
# --------------------------------------------------------------------------

def test_exact_matcher_handles_null_and_nan_metadata():
    docs_with_nans = [
        {
            "doc_id": "valid_doc",
            "legal_number": "61/2020/QH14",
            "title": "Luật Đầu tư 2020",
            "year": 2020,
            "doc_type": "Luật",
            "article": "Điều 15",
            "clause": "Khoản 2",
            "point": "Điểm a",
        },
        {
            "doc_id": "nan_doc",
            "legal_number": float("nan"),
            "title": None,
            "year": np.nan,
            "doc_type": float("nan"),
            "article": None,
            "clause": float("nan"),
            "point": None,
        },
        {
            "doc_id": None,  # invalid doc id
            "legal_number": "99/9999/TT-BXD",
            "title": "Invalid Document",
        },
    ]

    matcher = ExactMatcher(docs_with_nans)
    matches = matcher.match("Theo Điều 15 Khoản 2 Điểm a Luật Đầu tư 2020 số 61/2020/QH14")

    assert "valid_doc" in matches
    assert "nan_doc" not in matches
    assert None not in matches

    features = matches["valid_doc"]
    assert features["exact_legal_number"] is True
    assert features["exact_title"] is True
    assert features["exact_year"] is True
    assert features["exact_doc_type"] is True
    assert features["exact_article"] is True
    assert features["exact_clause"] is True
    assert features["exact_point"] is True
    assert features["exact_title_overlap"] > 0.0
    assert features["exact_score"] > 0.0


# --------------------------------------------------------------------------
# 5. Candidate Union Determinism & Recall @200 Evaluation
# --------------------------------------------------------------------------

class MockRetriever:
    def __init__(self, hits):
        self.hits = hits

    def retrieve(self, query, top_k=200):
        return self.hits[:top_k]


def test_candidate_union_determinism_and_features():
    bm25_hits = [{"doc_id": f"doc_{i}", "score": 20.0 - i, "bm25_score": 20.0 - i} for i in range(10)]
    pyvi_hits = [{"doc_id": f"doc_{i+5}", "score": 15.0 - i, "bm25_pyvi_score": 15.0 - i} for i in range(10)]
    dense_hits = [{"doc_id": f"doc_{i+8}", "score": 0.95 - i*0.01, "dense_score": 0.95 - i*0.01} for i in range(10)]

    engine = HybridSearchEngine(
        bm25_retriever=MockRetriever(bm25_hits),
        bm25_pyvi_retriever=MockRetriever(pyvi_hits),
        dense_retriever=MockRetriever(dense_hits),
    )

    cands1 = engine.search("query", top_k_candidates=200)
    cands2 = engine.search("query", top_k_candidates=200)

    # Determinism test
    assert [c["doc_id"] for c in cands1] == [c["doc_id"] for c in cands2]
    assert [c["rrf_score"] for c in cands1] == [c["rrf_score"] for c in cands2]

    # Verify all expected documents are retrieved in union
    retrieved_ids = {c["doc_id"] for c in cands1}
    assert {f"doc_{i}" for i in range(18)} <= retrieved_ids

    # Test candidate feature builder
    df_features = build_candidate_features({"q1": cands1}, qrels={"q1": ["doc_0", "doc_5"]})
    assert len(df_features) == len(cands1)
    assert "query_id" in df_features.columns
    assert "doc_id" in df_features.columns
    assert "rrf_score" in df_features.columns
    assert "bm25_score" in df_features.columns
    assert "bm25_pyvi_score" in df_features.columns
    assert "dense_score" in df_features.columns
    assert "label" in df_features.columns
    assert df_features[df_features["doc_id"] == "doc_0"]["label"].values[0] == 1


def test_candidate_recall_at_200_evaluation():
    # 200 candidates for q1
    cands = {"q1": [f"doc_{i}" for i in range(250)]}
    ground_truth = {"q1": ["doc_15", "doc_45", "doc_95", "doc_140", "doc_195", "doc_220"]}

    metrics = evaluate_candidate_recall(
        cands,
        ground_truth,
        cutoffs=[20, 50, 100, 150, 200],
    )

    assert set(metrics.keys()) == {
        "candidate_recall@20",
        "candidate_recall@50",
        "candidate_recall@100",
        "candidate_recall@150",
        "candidate_recall@200",
    }
    assert metrics["candidate_recall@20"] == pytest.approx(1 / 6)
    assert metrics["candidate_recall@50"] == pytest.approx(2 / 6)
    assert metrics["candidate_recall@100"] == pytest.approx(3 / 6)
    assert metrics["candidate_recall@150"] == pytest.approx(4 / 6)
    assert metrics["candidate_recall@200"] == pytest.approx(5 / 6)


# --------------------------------------------------------------------------
# 6. Dense Query Caching
# --------------------------------------------------------------------------

def test_dense_macro_query_caching(tmp_path: Path):
    cache_file = tmp_path / "query_cache.npy"

    def mock_encoder(texts):
        return np.array([[float(len(t)), 1.0] for t in texts], dtype=np.float32)

    retriever = DenseMacroRetriever.from_arrays(query_encoder=mock_encoder)
    queries = ["câu hỏi 1", "câu hỏi dài hơn số 2"]

    embs1 = retriever.encode_and_cache_queries(queries, cache_file)
    assert cache_file.exists()
    assert embs1.shape == (2, 2)

    # Second call loads from cache without calling encoder
    retriever_no_encoder = DenseMacroRetriever.from_arrays(query_encoder=None)
    embs2 = retriever_no_encoder.encode_and_cache_queries(queries, cache_file)
    assert np.allclose(embs1, embs2)


# --------------------------------------------------------------------------
# 7. Question Memory Fold Isolation and Self-Query Exclusion
# --------------------------------------------------------------------------

def test_question_memory_fold_isolation_and_exclude_qid():
    train_queries = {
        "train_1": "quy định về đất đai",
        "train_2": "thủ tục đăng ký kinh doanh",
    }
    train_qrels = {
        "train_1": ["doc_land"],
        "train_2": ["doc_biz"],
    }

    memory = TrainQuestionMemory(min_similarity=0.7, use_dense=False)
    memory.fit(train_queries, train_qrels)

    # When querying with train_1 without exclude_qid, doc_land is returned
    hits = memory.search("quy định về đất đai", top_k=5)
    assert any(h["doc_id"] == "doc_land" for h in hits)

    # When querying with train_1 WITH exclude_qid="train_1", train_1 is excluded from voting
    hits_excluded = memory.search("quy định về đất đai", top_k=5, exclude_qid="train_1")
    assert not any(h["doc_id"] == "doc_land" for h in hits_excluded)
