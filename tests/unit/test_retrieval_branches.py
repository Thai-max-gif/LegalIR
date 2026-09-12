import pytest
import pandas as pd
from src.retrieval.exact_matcher import ExactMatcher
from src.retrieval.question_memory import QuestionMemory
from src.retrieval.bm25_micro import BM25MicroRetriever
from src.retrieval.hybrid_search import HybridSearchEngine


def test_exact_matcher():
    docs = [
        {"doc_id": "740", "title": "Quyết định 5868/QĐ-BYT 2018", "legal_number": "5868/QĐ-BYT", "year": "2018", "doc_type": "Quyết định"},
        {"doc_id": "2113", "title": "Luật Đầu tư 2020", "legal_number": "61/2020/QH14", "year": "2020", "doc_type": "Luật"}
    ]
    matcher = ExactMatcher(docs)

    # Query with exact legal number
    matches = matcher.match("Căn cứ Quyết định số 5868/QĐ-BYT quy định như thế nào?")
    assert "740" in matches
    assert matches["740"]["score"] > 0.8

    # Query with law title and year
    matches2 = matcher.match("Theo Luật Đầu tư năm 2020 dự án là gì?")
    assert "2113" in matches2
    assert matches2["2113"]["score"] > 0.8


def test_question_memory():
    train_queries = [
        {"query_id": "1", "question_norm": "dự án đầu tư là gì?", "doc_ids": ["2113"]},
        {"query_id": "2", "question_norm": "thời hạn cấp giấy phép lái xe", "doc_ids": ["999"]}
    ]
    memory = QuestionMemory(train_queries)

    # Exact match query
    res = memory.retrieve("Dự án đầu tư là gì?", exclude_qid=None)
    assert "2113" in res
    assert res["2113"]["score"] > 0.8

    # Self-exclusion test
    res_ex = memory.retrieve("dự án đầu tư là gì?", exclude_qid="1")
    assert "2113" not in res_ex


def test_bm25_micro_and_hybrid():
    chunks = [
        {"chunk_id": "c1", "doc_id": "doc1", "granularity": "micro", "text_norm": "quy định về thủ tục cấp giấy phép xây dựng"},
        {"chunk_id": "c2", "doc_id": "doc2", "granularity": "micro", "text_norm": "thời hạn giải quyết hồ sơ đăng ký kinh doanh"}
    ]
    bm25 = BM25MicroRetriever()
    bm25.fit(chunks)

    res = bm25.retrieve("thủ tục cấp giấy phép xây dựng", top_k=5)
    assert len(res) > 0
    assert res[0]["doc_id"] == "doc1"
