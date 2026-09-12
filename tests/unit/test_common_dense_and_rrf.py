import numpy as np
from src.common.dense_dek21 import DEk21Retriever
from src.common.rrf import reciprocal_rank_fusion

def test_dense_retriever_mock():
    corpus = [
        {"chunk_id": "c1", "doc_id": "100", "article": "Điều 1", "text_raw": "Quy định về thuế GTGT."},
        {"chunk_id": "c2", "doc_id": "200", "article": "Điều 2", "text_raw": "Quy định về đăng ký xe."}
    ]
    retriever = DEk21Retriever(model_name="mock")
    retriever.fit(corpus)
    res = retriever.search("thuế GTGT", top_k=5)
    assert len(res) > 0

def test_reciprocal_rank_fusion():
    run1 = [{"doc_id": "100"}, {"doc_id": "200"}]
    run2 = [{"doc_id": "200"}, {"doc_id": "300"}]
    fused = reciprocal_rank_fusion([run1, run2], k=60)
    assert len(fused) == 3
    # doc 200 appears in both runs so should have high score
    assert fused[0]["doc_id"] == "200"
