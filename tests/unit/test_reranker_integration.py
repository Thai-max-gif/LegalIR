from src.ranking.evidence_pack import EvidencePackBuilder
from src.ranking.reranker import CrossEncoderReranker
from src.retrieval.types import CandidateRecord


def test_evidence_contains_required_sections():
    macro_chunks = [
        {
            "doc_id": "1",
            "chunk_id": "c1",
            "article": "Điều 2. Phạm vi điều chỉnh",
            "text_norm": "Luật này quy định về hoạt động đầu tư kinh doanh.",
            "is_empty": False,
        }
    ]
    doc_metadata = {
        "1": {
            "doc_id": "1",
            "title": "Luật Đầu tư 2020",
            "legal_number": "61/2020/QH14",
            "year": "2020",
        }
    }
    builder = EvidencePackBuilder(macro_chunks=macro_chunks, doc_metadata=doc_metadata)
    packs = builder.build(query="phạm vi điều chỉnh của luật đầu tư", doc_id="1", max_chunks=2)
    assert len(packs) >= 1
    evidence_text = packs[0]["text"]
    assert "[VĂN BẢN]" in evidence_text
    assert "[ĐIỀU KHOẢN]" in evidence_text
    assert "[NỘI DUNG]" in evidence_text
    assert "c1" == packs[0]["chunk_id"]


def test_reranker_aggregates_document_evidence_scores():
    class MockScoreModel:
        def score_pairs(self, pairs, batch_size=16, max_length=512):
            # Return predictable scores based on text
            return [3.5 if "c1" in p[1] else 1.5 for p in pairs]

    reranker = CrossEncoderReranker(model_name="mock")
    reranker.score_fn = MockScoreModel().score_pairs

    macro_chunks = [
        {"doc_id": "1", "chunk_id": "c1", "article": "Điều 1", "text_norm": "Nội dung c1", "is_empty": False},
        {"doc_id": "1", "chunk_id": "c2", "article": "Điều 2", "text_norm": "Nội dung c2", "is_empty": False},
    ]
    doc_meta = {"1": {"doc_id": "1", "title": "Luật 1", "legal_number": "01/2020"}}
    builder = EvidencePackBuilder(macro_chunks=macro_chunks, doc_metadata=doc_meta)

    cands: list[CandidateRecord] = [
        {"doc_id": "1", "rrf_score": 0.05, "bm25_score": 10.0}
    ]

    reranked = reranker.rerank(query="câu hỏi", candidates=cands, evidence_builder=builder, top_k=10)
    assert len(reranked) == 1
    doc = reranked[0]
    assert doc["reranker_best_score"] == 3.5
    assert doc["reranker_second_score"] == 1.5
    assert doc["reranker_margin"] == 2.0
    assert doc["reranker_best_chunk_id"] == "c1"
    assert doc["evidence_chunk_count"] == 2
