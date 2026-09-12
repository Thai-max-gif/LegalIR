from src.common.evidence import EvidencePackBuilder

def test_evidence_pack_builder():
    builder = EvidencePackBuilder(max_chunks_per_doc=2)
    doc_info = {"title": "Nghị định 44/2023/NĐ-CP", "legal_number": "44/2023/NĐ-CP"}
    chunks = [
        {"article": "Điều 1", "text_raw": "Giảm thuế GTGT 2%."},
        {"article": "Điều 2", "text_raw": "Hiệu lực thi hành từ ngày 01/7/2023."}
    ]
    ev_text = builder.build_evidence_text("Giảm thuế GTGT bao nhiêu %?", doc_info, chunks)
    assert "[QUESTION]" not in ev_text
    assert "[DOCUMENT]" in ev_text
    assert "[EVIDENCE 1]" in ev_text
    assert "[EVIDENCE 2]" in ev_text

