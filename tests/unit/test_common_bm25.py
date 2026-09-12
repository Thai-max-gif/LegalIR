from src.common.bm25 import BM25Retriever

def test_bm25_retriever_fit_search():
    corpus = [
        {"chunk_id": "c1", "doc_id": "100", "text_raw": "Nghị định 44/2023/NĐ-CP về giảm thuế giá trị gia tăng.", "text_norm": "nghị định 44/2023/nđ-cp về giảm thuế giá trị gia tăng"},
        {"chunk_id": "c2", "doc_id": "200", "text_raw": "Thông tư 58/2020/TT-BCA về quy trình đăng ký xe.", "text_norm": "thông tư 58/2020/tt-bca về quy trình đăng ký xe"}
    ]
    retriever = BM25Retriever()
    retriever.fit(corpus)

    results = retriever.search("44/2023/NĐ-CP", top_k=5)
    assert len(results) > 0
    assert results[0]["doc_id"] == "100"
