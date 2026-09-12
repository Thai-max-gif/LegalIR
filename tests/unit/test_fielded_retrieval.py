from src.retrieval.bm25_micro import BM25MicroRetriever
from src.retrieval.exact_matcher import ExactMatcher


def test_legal_number_field_outranks_body_only_match():
    chunks = [
        {
            "chunk_id": "c1",
            "doc_id": "1",
            "legal_number": "61/2020/QH14",
            "title": "Luật Đầu tư",
            "article": "Điều 1",
            "text_norm": "văn bản này quy định chung không nhắc số",
            "link": ""
        },
        {
            "chunk_id": "c2",
            "doc_id": "2",
            "legal_number": "",
            "title": "Thông tư khác",
            "article": "Điều 2",
            "text_norm": "căn cứ luật số 61/2020/qh14 để thực hiện",
            "link": ""
        },
    ]
    retriever = BM25MicroRetriever(field_weights={"legal_number": 5.0, "body": 1.0}).fit(chunks)
    res = retriever.retrieve("61/2020/QH14", top_k=2)
    assert len(res) >= 1
    # Document 1 has legal_number matching, gets 5x weight on legal_number field
    assert res[0]["doc_id"] == "1"
    assert "bm25_best_score" in res[0]
    assert "bm25_best_chunk_id" in res[0]


def test_exact_match_returns_separate_flags():
    docs = [
        {
            "doc_id": "1",
            "legal_number": "61/2020/QH14",
            "title": "Luật Đầu tư 2020",
            "year": "2020",
            "doc_type": "Luật",
            "name_raw": "Luat-Dau-tu-61-2020-QH14",
            "link": "https://example.com/luat-dau-tu-2020"
        }
    ]
    matcher = ExactMatcher(docs)
    res = matcher.match("Theo Điều 2 Luật Đầu tư 2020 số 61/2020/QH14")
    assert "1" in res
    m = res["1"]
    assert m["exact_legal_number"] is True
    assert m["exact_title"] is True
    assert m["exact_year"] is True
    assert m["exact_doc_type"] is True
    assert m["score"] > 0
