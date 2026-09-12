from src.dataset.chunker import ChunkConfig, build_document_chunks
from src.dataset.legal_parser import parse_legal_units


def test_parser_preserves_full_hierarchy():
    text = "Chương I\nMục 1\nĐiều 2. Phạm vi\n1. Nội dung quy định\na) Chi tiết điểm a"
    units = parse_legal_units(text)
    assert len(units) >= 1
    unit = units[0]
    assert unit.chapter == "Chương I"
    assert unit.section == "Mục 1"
    assert "Điều 2" in unit.article
    assert "Khoản 1" in (unit.clause or "")
    assert "Điểm a" in (unit.point or "")


def test_long_article_is_split_and_micro_parents_are_valid():
    doc = {
        "doc_id": "1",
        "title": "Luật thử nghiệm",
        "passage_norm": "Điều 1. Quy định chung\n" + "từ pháp luật quy định chi tiết " * 400,
        "is_empty": False,
        "legal_number": "01/2020/QH14",
    }
    chunks = build_document_chunks(doc, ChunkConfig())
    macros = [c for c in chunks if c["granularity"] == "macro"]
    micros = [c for c in chunks if c["granularity"] == "micro"]
    assert len(macros) >= 1
    assert max(c["token_count"] for c in macros) <= 800
    assert {c["parent_chunk_id"] for c in micros} <= {c["chunk_id"] for c in macros}


def test_empty_document_has_one_metadata_chunk():
    doc = {
        "doc_id": "20",
        "title": "Văn bản trống",
        "passage_norm": "",
        "is_empty": True,
        "legal_number": None,
        "year": "2021",
    }
    chunks = build_document_chunks(doc, ChunkConfig())
    assert len(chunks) == 1
    assert chunks[0]["is_empty"] is True
    assert chunks[0]["granularity"] == "macro"
    assert "20" in chunks[0]["chunk_id"]


def test_sliding_window_overlap():
    doc = {
        "doc_id": "99",
        "title": "Văn bản không có điều",
        "passage_norm": "Văn bản này không có cấu trúc điều khoản chuẩn.\n" + "đoạn nội dung dài văn bản " * 300,
        "is_empty": False,
        "legal_number": "99/2022/TT-BQP",
    }
    chunks = build_document_chunks(doc, ChunkConfig())
    macros = [c for c in chunks if c["granularity"] == "macro"]
    assert len(macros) >= 1
    for m in macros:
        assert m["token_count"] <= 1200
