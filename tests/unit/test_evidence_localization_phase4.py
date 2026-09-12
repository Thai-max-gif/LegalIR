import time
import pytest

from src.ranking.evidence_pack import EvidencePackBuilder
from src.ranking.reranker import CrossEncoderReranker, DocumentReranker
from src.training.positive_localizer import PositiveLocalizer


def test_late_article_localization_dieu_61():
    """Verify that a query mentioning Điều 61 selects the Điều 61 chunk from a long document."""
    chunks = []
    # Create 70 chunks for doc 100 with Điều 1 to Điều 70
    for i in range(1, 71):
        chunks.append({
            "doc_id": "doc_law_100",
            "chunk_id": f"chunk_macro_{i:03d}",
            "article": f"Điều {i}. Nội dung điều khoản số {i}",
            "text_norm": f"Quy định chi tiết thi hành đối với Điều {i} về thủ tục hành chính số {i}.",
            "text_raw": f"Quy định chi tiết thi hành đối với Điều {i} về thủ tục hành chính số {i}.",
        })

    # Add specific high-signal text to Điều 61
    chunks[60] = {
        "doc_id": "doc_law_100",
        "chunk_id": "chunk_macro_061",
        "article": "Điều 61. Thẩm quyền quyết định chủ trương đầu tư",
        "text_norm": "Thủ tướng Chính phủ chấp thuận chủ trương đầu tư đối với các dự án đặc biệt quan trọng.",
        "text_raw": "Thủ tướng Chính phủ chấp thuận chủ trương đầu tư đối với các dự án đặc biệt quan trọng.",
    }

    doc_meta = {
        "doc_law_100": {
            "doc_id": "doc_law_100",
            "title": "Luật Đầu tư 2020",
            "legal_number": "61/2020/QH14",
        }
    }

    builder = EvidencePackBuilder(macro_chunks=chunks, doc_metadata=doc_meta, max_chunks=2)
    query = "Theo quy định tại Điều 61, dự án nào do Thủ tướng Chính phủ chấp thuận chủ trương đầu tư?"

    # Build evidence pack
    pack = builder.build_pack(query, "doc_law_100", max_chunks=2)

    assert "chunk_macro_061" in [c["chunk_id"] for c in builder._select_chunks(query, "doc_law_100", None, max_chunks=2)]
    assert "Thủ tướng Chính phủ chấp thuận chủ trương đầu tư" in pack
    assert "[DOCUMENT] Luật Đầu tư 2020 61/2020/QH14" in pack
    assert "[EVIDENCE 1]" in pack

    # Also verify PositiveLocalizer localizes Điều 61 correctly
    localizer = PositiveLocalizer(macro_chunks=chunks)
    pos_chunk = localizer.localize(query, "doc_law_100")
    assert pos_chunk is not None
    assert pos_chunk["chunk_id"] == "chunk_macro_061"


def test_clause_and_point_localization():
    """Verify that a query mentioning Khoản 3 Điểm a selects the matching clause/point chunk."""
    chunks = [
        {
            "doc_id": "doc_decree_10",
            "chunk_id": "c_k1",
            "article": "Điều 15. Hồ sơ cấp phép",
            "clause": "Khoản 1",
            "point": "",
            "text_norm": "Khoản 1. Đơn đề nghị cấp giấy phép kinh doanh theo mẫu số 01.",
        },
        {
            "doc_id": "doc_decree_10",
            "chunk_id": "c_k2",
            "article": "Điều 15. Hồ sơ cấp phép",
            "clause": "Khoản 2",
            "point": "",
            "text_norm": "Khoản 2. Bản sao Giấy chứng nhận đăng ký doanh nghiệp.",
        },
        {
            "doc_id": "doc_decree_10",
            "chunk_id": "c_k3_pa",
            "article": "Điều 15. Hồ sơ cấp phép",
            "clause": "Khoản 3",
            "point": "Điểm a",
            "text_norm": "Khoản 3. Điều kiện về vốn: Điểm a) Vốn điều lệ tối thiểu 50 tỷ đồng Việt Nam đối với dịch vụ tài chính.",
        },
    ]

    doc_meta = {
        "doc_decree_10": {
            "title": "Nghị định về điều kiện cấp phép",
            "legal_number": "15/2021/NĐ-CP",
        }
    }

    builder = EvidencePackBuilder(macro_chunks=chunks, doc_metadata=doc_meta, max_chunks=1)
    query = "Hồ sơ vốn điều lệ tối thiểu 50 tỷ đồng tại Khoản 3 Điểm a Điều 15 gồm những gì?"

    pack = builder.build_pack(query, "doc_decree_10", max_chunks=1)
    assert "Vốn điều lệ tối thiểu 50 tỷ đồng" in pack
    assert "[EVIDENCE 1]" in pack

    localizer = PositiveLocalizer(macro_chunks=chunks)
    best = localizer.localize(query, "doc_decree_10")
    assert best["chunk_id"] == "c_k3_pa"


def test_deduplication_of_redundant_chunks():
    """Verify that near-identical chunks are deduplicated in favor of complementary chunks."""
    chunks = [
        {
            "doc_id": "doc_dup",
            "chunk_id": "c_dup_1",
            "article": "Điều 5",
            "text_norm": "Quy định về bảo vệ môi trường trong hoạt động sản xuất kinh doanh tại khu công nghiệp.",
        },
        {
            "doc_id": "doc_dup",
            "chunk_id": "c_dup_2",
            "article": "Điều 5",
            "text_norm": "Quy định về bảo vệ môi trường trong hoạt động sản xuất kinh doanh tại khu công nghiệp.",
        },
        {
            "doc_id": "doc_dup",
            "chunk_id": "c_comp_3",
            "article": "Điều 6",
            "text_norm": "Trách nhiệm của ban quản lý khu công nghiệp trong công tác quan trắc chất thải.",
        },
    ]

    builder = EvidencePackBuilder(macro_chunks=chunks, max_chunks=2)
    selected = builder._select_chunks("bảo vệ môi trường khu công nghiệp và trách nhiệm quan trắc", "doc_dup", None, max_chunks=2)

    assert len(selected) == 2
    selected_ids = {c["chunk_id"] for c in selected}
    assert "c_comp_3" in selected_ids
    # Only one of the duplicate chunks should be picked
    assert not ("c_dup_1" in selected_ids and "c_dup_2" in selected_ids)


def test_token_budget_awareness_and_truncation():
    """Verify that token / char budget limits truncate evidence cleanly while preserving headers."""
    long_text_1 = " ".join(["Quy định chi tiết về xử phạt vi phạm hành chính."] * 30)
    long_text_2 = " ".join(["Trình tự thủ tục áp dụng biện pháp khắc phục hậu quả."] * 30)

    chunks = [
        {"doc_id": "doc_long", "chunk_id": "c1", "article": "Điều 1", "text_norm": long_text_1},
        {"doc_id": "doc_long", "chunk_id": "c2", "article": "Điều 2", "text_norm": long_text_2},
    ]
    doc_meta = {
        "doc_long": {
            "title": "Nghị định xử phạt vi phạm",
            "legal_number": "100/2019/NĐ-CP",
        }
    }

    builder = EvidencePackBuilder(
        macro_chunks=chunks,
        doc_metadata=doc_meta,
        max_chunks=2,
        max_chars=350,
        max_tokens=80,
    )

    pack = builder.build_pack("xử phạt vi phạm hành chính", "doc_long", max_chars=350, max_tokens=80)

    assert "[DOCUMENT] Nghị định xử phạt vi phạm 100/2019/NĐ-CP" in pack
    assert "[EVIDENCE 1]" in pack
    assert len(pack) <= 450
    # Make sure text does not end with trailing space or broken word
    assert not pack.endswith("  ")


def test_deterministic_output_formatting():
    """Verify structured pack formatting matches exact expected schema."""
    chunks = [
        {"doc_id": "d1", "chunk_id": "c1", "article": "Điều 10", "text_norm": "Quy định một."},
        {"doc_id": "d1", "chunk_id": "c2", "article": "Điều 11", "text_norm": "Quy định hai."},
    ]
    doc_meta = {"d1": {"title": "Luật Mẫu", "legal_number": "99/2023"}}

    builder = EvidencePackBuilder(macro_chunks=chunks, doc_metadata=doc_meta, max_chunks=2)

    # Standard format with question
    pack_with_q = builder.build_pack("câu hỏi?", "d1", include_question=True)
    assert pack_with_q.startswith("[QUESTION] câu hỏi? [DOCUMENT] Luật Mẫu 99/2023 [EVIDENCE 1]")
    assert "[EVIDENCE 2]" in pack_with_q

    # Standard format without question
    pack_no_q = builder.build_pack("câu hỏi?", "d1", include_question=False)
    assert pack_no_q.startswith("[DOCUMENT] Luật Mẫu 99/2023 [EVIDENCE 1]")
    assert "[QUESTION]" not in pack_no_q


def test_evidence_pack_construction_speed_150_candidates():
    """Verify that building evidence packs for 150 candidates executes in under 50ms."""
    chunks = []
    doc_meta = {}
    for d_idx in range(1, 151):
        did = f"doc_{d_idx}"
        doc_meta[did] = {
            "doc_id": did,
            "title": f"Văn bản quy phạm pháp luật số {d_idx}",
            "legal_number": f"{d_idx}/2022/NĐ-CP",
        }
        for c_idx in range(1, 11):
            chunks.append({
                "doc_id": did,
                "chunk_id": f"{did}_c{c_idx}",
                "article": f"Điều {c_idx}. Quy định chi tiết {c_idx}",
                "clause": f"Khoản {c_idx % 3 + 1}",
                "text_norm": f"Nội dung quy định chi tiết của Điều {c_idx} đối với hồ sơ thủ tục cấp phép văn bản {d_idx}.",
            })

    builder = EvidencePackBuilder(macro_chunks=chunks, doc_metadata=doc_meta, max_chunks=2)
    query = "Theo quy định tại Điều 5, thủ tục cấp phép và hồ sơ gồm những gì?"
    candidate_doc_ids = list(doc_meta.keys())

    # Measure construction time for all 150 candidates
    t0 = time.perf_counter()
    packs = [builder.build_pack(query, did, max_chunks=2) for did in candidate_doc_ids]
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert len(packs) == 150
    assert elapsed_ms < 50.0, f"Evidence pack construction took {elapsed_ms:.2f} ms (must be < 50 ms)"


def test_reranker_integration_with_localized_evidence():
    """Verify that CrossEncoderReranker and DocumentReranker use query-aware evidence properly."""
    chunks = [
        {
            "doc_id": "doc_target",
            "chunk_id": "c_early",
            "article": "Điều 1. Phạm vi",
            "text_norm": "Luật này áp dụng chung cho mọi tổ chức cá nhân.",
        },
        {
            "doc_id": "doc_target",
            "chunk_id": "c_late_target",
            "article": "Điều 61. Thẩm quyền quyết định",
            "text_norm": "Thủ tướng Chính phủ có thẩm quyền phê duyệt dự án nhóm A.",
        },
        {
            "doc_id": "doc_irrelevant",
            "chunk_id": "c_irr",
            "article": "Điều 1. Phạm vi",
            "text_norm": "Quy chuẩn kỹ thuật về môi trường nước thải y tế.",
        },
    ]

    doc_meta = {
        "doc_target": {"title": "Luật Đầu tư", "legal_number": "61/2020"},
        "doc_irrelevant": {"title": "Quy chuẩn Y tế", "legal_number": "01/2020"},
    }

    builder = EvidencePackBuilder(macro_chunks=chunks, doc_metadata=doc_meta, max_chunks=2)

    def mock_score_fn(pairs, **kwargs):
        scores = []
        for q, passage in pairs:
            if "phê duyệt dự án nhóm A" in passage or "Luật Đầu tư" in passage:
                scores.append(10.0)
            else:
                scores.append(1.0)
        return scores

    reranker = CrossEncoderReranker(model_name="mock", score_fn=mock_score_fn)
    cands = [
        {"doc_id": "doc_irrelevant", "bm25_score": 5.0},
        {"doc_id": "doc_target", "bm25_score": 4.0},
    ]

    query = "Dự án nhóm A theo Điều 61 do ai phê duyệt?"
    reranked = reranker.rerank(query, cands, evidence_builder=builder, top_k=2)

    assert len(reranked) == 2
    assert reranked[0]["doc_id"] == "doc_target"
    assert reranked[0]["reranker_best_chunk_id"] == "c_late_target"


def test_prior_retrieval_chunk_id_boost():
    """Verify that prior best chunk from dense/BM25 retrieval boosts chunk selection."""
    chunks = [
        {"doc_id": "d1", "chunk_id": "c_generic_1", "article": "Điều 1", "text_norm": "Nội dung chung về tài chính."},
        {"doc_id": "d1", "chunk_id": "c_prior_dense", "article": "Điều 2", "text_norm": "Nội dung chung về tài chính ngân hàng."},
    ]
    builder = EvidencePackBuilder(macro_chunks=chunks, max_chunks=1)

    cand_with_prior = {"doc_id": "d1", "dense_best_chunk_id": "c_prior_dense"}
    selected = builder._select_chunks("tài chính", "d1", cand_with_prior, max_chunks=1)

    assert len(selected) == 1
    assert selected[0]["chunk_id"] == "c_prior_dense"


def test_configurable_token_budget_256_384_512():
    """Verify pack sizing under various token budget limits (256, 384, 512)."""
    long_article = " ".join(["Nội dung quy định chi tiết về quản lý tài sản công và đấu thầu."] * 40)
    chunks = [
        {"doc_id": "d_tok", "chunk_id": "c1", "article": "Điều 1", "text_norm": long_article},
        {"doc_id": "d_tok", "chunk_id": "c2", "article": "Điều 2", "text_norm": long_article},
        {"doc_id": "d_tok", "chunk_id": "c3", "article": "Điều 3", "text_norm": long_article},
    ]
    doc_meta = {"d_tok": {"title": "Luật Quản lý tài sản công", "legal_number": "15/2017/QH14"}}

    builder = EvidencePackBuilder(macro_chunks=chunks, doc_metadata=doc_meta, max_chunks=3)

    for token_limit in (256, 384, 512):
        pack = builder.build_pack(
            "quy định tài sản công",
            "d_tok",
            max_tokens=token_limit,
            max_chunks=3,
        )
        assert "[DOCUMENT] Luật Quản lý tài sản công 15/2017/QH14" in pack
        assert "[EVIDENCE 1]" in pack
        # Token count should be strictly bounded
        from src.ranking.evidence_pack import tokenize
        tok_count = len(tokenize(pack))
        assert tok_count <= token_limit + 10, f"Token count {tok_count} exceeded budget {token_limit}"


def test_empty_and_single_chunk_documents():
    """Verify that empty and single-chunk documents produce deterministic, valid evidence packs."""
    builder = EvidencePackBuilder(macro_chunks=[], doc_metadata={"empty_doc": {"title": "Văn bản rỗng"}})

    pack_empty = builder.build_pack("câu hỏi?", "empty_doc")
    assert "[DOCUMENT] Văn bản rỗng" in pack_empty
    assert "[EVIDENCE 1]" in pack_empty

    records = builder.build("câu hỏi?", "empty_doc")
    assert len(records) >= 1
    assert records[0]["doc_id"] == "empty_doc"


def test_document_reranker_wrapper_integration():
    """Verify DocumentReranker correctly integrates with EvidencePackBuilder."""
    chunks = [
        {"doc_id": "d1", "chunk_id": "c1", "article": "Điều 1", "text_norm": "Thủ tục xin cấp phép xây dựng nhà ở."},
        {"doc_id": "d2", "chunk_id": "c2", "article": "Điều 1", "text_norm": "Quy định về bảo vệ rừng tự nhiên."},
    ]
    doc_meta = {
        "d1": {"title": "Luật Xây dựng", "legal_number": "50/2014"},
        "d2": {"title": "Luật Lâm nghiệp", "legal_number": "16/2017"},
    }
    builder = EvidencePackBuilder(macro_chunks=chunks, doc_metadata=doc_meta)

    def mock_score(pairs, **kwargs):
        return [10.0 if "luật xây dựng" in p[1].lower() else 1.0 for p in pairs]

    cross_encoder = CrossEncoderReranker(model_name="mock", score_fn=mock_score)
    doc_reranker = DocumentReranker(
        reranker=cross_encoder,
        evidence_builder=builder,
        doc_map=doc_meta,
    )

    cands = [{"doc_id": "d2"}, {"doc_id": "d1"}]
    reranked = doc_reranker.rerank_documents("xin cấp phép xây dựng nhà ở", cands, top_k=2)

    assert len(reranked) == 2
    assert reranked[0]["doc_id"] == "d1"
    assert reranked[0]["final_rank"] == 1

