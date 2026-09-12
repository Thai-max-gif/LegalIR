from src.common.reranker import BGEReranker

def test_bge_reranker_mock():
    reranker = BGEReranker(model_name="mock")
    candidates = [{"doc_id": "100"}, {"doc_id": "200"}]
    evidence = ["Giảm thuế GTGT quy định cụ thể", "Đăng ký phương tiện giao thông"]
    res = reranker.rerank_candidates("thuế GTGT", candidates, evidence_texts=evidence, top_k=2)
    assert len(res) == 2
    assert res[0]["doc_id"] == "100"
