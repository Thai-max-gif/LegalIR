from src.ranking.evidence_pack import EvidencePackBuilder


def test_structured_evidence_pack_includes_query_document_and_two_evidence_sections():
    chunks = [
        {"chunk_id": "c1", "doc_id": "doc-1", "article": "Điều 1", "text_norm": "First evidence."},
        {"chunk_id": "c2", "doc_id": "doc-1", "article": "Điều 2", "text_norm": "Second evidence."},
    ]
    builder = EvidencePackBuilder(
        macro_chunks=chunks,
        doc_metadata={"doc-1": {"title": "Law One", "legal_number": "01/2020"}},
        max_chunks=2,
    )

    pack = builder.build_pack("What applies?", "doc-1")

    assert pack == (
        "[DOCUMENT] Law One 01/2020 "
        "[EVIDENCE 1] First evidence. [EVIDENCE 2] Second evidence."
    )
    assert "[QUESTION]" not in pack


def test_structured_evidence_pack_omits_nan_metadata_and_text():
    builder = EvidencePackBuilder(
        macro_chunks=[
            {"chunk_id": "c1", "doc_id": "doc-1", "text_norm": float("nan"), "text_raw": ""}
        ],
        doc_metadata={"doc-1": {"title": float("nan"), "legal_number": "nan"}},
    )

    pack = builder.build_pack("Question", "doc-1")

    assert "[DOCUMENT] Văn bản doc-1" in pack
    assert "nan" not in pack.lower()
