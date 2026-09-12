import json
import zipfile
from pathlib import Path
from src.pipeline.predict import LegalIRPipeline
from src.ranking.evidence_pack import EvidencePackBuilder
from src.ranking.fusion import ReciprocalRankFusion
from src.ranking.selector import TopKSelector
from src.retrieval.bm25_micro import BM25MicroRetriever
from src.retrieval.exact_matcher import ExactMatcher
from src.retrieval.hybrid_search import HybridSearchEngine
from src.retrieval.question_memory import QuestionMemory


def test_pipeline_predict_end_to_end(tmp_path: Path):
    docs = [
        {"doc_id": "1", "title": "Luật Đầu tư 2020", "legal_number": "61/2020/QH14", "year": "2020", "doc_type": "Luật"},
        {"doc_id": "2", "title": "Luật Doanh nghiệp 2020", "legal_number": "59/2020/QH14", "year": "2020", "doc_type": "Luật"},
    ]
    chunks = [
        {"chunk_id": "c1", "doc_id": "1", "granularity": "micro", "text_norm": "quy định về dự án đầu tư"},
        {"chunk_id": "c2", "doc_id": "2", "granularity": "micro", "text_norm": "thành lập doanh nghiệp cổ phần"},
    ]

    bm25 = BM25MicroRetriever().fit(chunks)
    exact = ExactMatcher(docs)
    memory = QuestionMemory([])
    engine = HybridSearchEngine(bm25_retriever=bm25, exact_matcher=exact, question_memory=memory)

    evidence_builder = EvidencePackBuilder(macro_chunks=chunks, doc_metadata={"1": docs[0], "2": docs[1]})
    fuser = ReciprocalRankFusion()
    selector = TopKSelector(max_k=5)

    pipeline = LegalIRPipeline(
        hybrid_engine=engine,
        evidence_builder=evidence_builder,
        reranker=None,
        ranker=fuser,
        selector=selector,
    )

    preds = pipeline.predict_batch({"q1": "dự án đầu tư là gì", "q2": "thành lập doanh nghiệp"})
    assert "q1" in preds
    assert "q2" in preds
    assert 1 <= len(preds["q1"]["answer"]) <= 5
    assert preds["q1"]["answer"][0] == "1"
    assert preds["q2"]["answer"][0] == "2"


def test_pipeline_fallback_uses_valid_documents_when_defaults_are_unavailable():
    class EmptyEngine:
        def search_candidates(self, query, exclude_qid=None, top_k=150):
            return []

    pipeline = LegalIRPipeline(
        hybrid_engine=EmptyEngine(),
        evidence_builder=EvidencePackBuilder(macro_chunks=[]),
        selector=TopKSelector(max_k=5, min_k=1),
        fallback_doc_ids=["2113", "58389", "84570"],
        valid_doc_ids={"doc-b", "doc-a"},
    )

    answer = pipeline.predict_one("q1", "a question with no retrieval hits")

    assert answer == ["doc-a", "doc-b"]
    assert 1 <= len(answer) <= 5
