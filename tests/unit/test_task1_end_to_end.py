from src.common.bm25 import BM25Retriever
from src.common.dense_dek21 import DEk21Retriever
from src.common.reranker import BGEReranker
from src.common.evidence import EvidencePackBuilder
from src.task1.memory import QuestionMemory
from src.task1.retrieve import CandidateRetriever, LegalMatcher
from src.task1.rerank import DocumentReranker
from src.task1.selector import TopKSelector
from src.task1.predict import LegalIRPipeline

def test_legalir_pipeline_end_to_end():
    corpus = [
        {"chunk_id": "c1", "doc_id": "100", "article": "Điều 10", "text_raw": "Nghị định 44/2023/NĐ-CP giảm thuế 2%."},
        {"chunk_id": "c2", "doc_id": "200", "article": "Điều 5", "text_raw": "Thông tư 58/2020/TT-BCA về đăng ký xe máy."}
    ]
    bm25 = BM25Retriever()
    bm25.fit(corpus)

    dense = DEk21Retriever(model_name="mock")
    dense.fit(corpus)

    memory = QuestionMemory()
    memory.fit({"q1": "thuế GTGT 44/2023"}, {"q1": ["100"]})

    exact = LegalMatcher(doc_index={"100": {"legal_number": "44/2023/NĐ-CP", "name_raw": "Nghi-dinh-44-2023-ND-CP"}})

    retriever = CandidateRetriever(bm25=bm25, dense=dense, memory=memory, exact=exact)
    reranker = DocumentReranker(
        reranker=BGEReranker(model_name="mock"),
        evidence_builder=EvidencePackBuilder(),
        doc_map={"100": {"doc_id": "100", "title": "Nghị định 44/2023/NĐ-CP", "legal_number": "44/2023/NĐ-CP"}},
        chunk_map={"100": [corpus[0]]}
    )
    selector = TopKSelector(max_k=5, min_k=1)

    pipeline = LegalIRPipeline(
        retriever=retriever,
        reranker=reranker,
        selector=selector,
        valid_doc_ids={"100", "200"}
    )

    preds = pipeline.predict_batch([{"id": "q_test", "question": "Nghị định 44/2023/NĐ-CP giảm thuế GTGT bao nhiêu?"}])
    assert "q_test" in preds
    assert 1 <= len(preds["q_test"]["answer"]) <= 5
    assert preds["q_test"]["answer"][0] == "100"
