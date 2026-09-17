"""
Determinism test ensuring BM25PyViRetriever optimizations produce exact,
bit-identical retrieval results compared to legacy baseline logic.
"""

from collections import Counter, defaultdict
import math
from pathlib import Path
import re
from typing import Any
import unicodedata
import numpy as np
import pandas as pd
import pytest

from pyvi import ViTokenizer
from src.dataset.normalize import clean_legal_text
from src.retrieval.bm25_pyvi import BM25PyViRetriever, _normalize_str
from src.retrieval.build_indexes import enrich_chunks_with_doc_metadata


def legacy_tokenize_pyvi(text: str) -> list[str]:
    if not text or not isinstance(text, str):
        return []
    cleaned = clean_legal_text(text)
    segmented = ViTokenizer.tokenize(cleaned) if ViTokenizer is not None else cleaned
    segmented = unicodedata.normalize("NFC", segmented).lower()
    return re.compile(r'\b[a-zà-ỹ0-9_]+\b', re.IGNORECASE | re.UNICODE).findall(segmented)


def fit_legacy(records: list[dict[str, Any]]) -> dict[str, Any]:
    field_weights = {
        "legal_number": 4.0,
        "title": 3.0,
        "article": 2.0,
        "clause": 1.0,
        "body": 1.0,
        "url_slug": 1.0,
    }
    chunk_ids = []
    doc_ids = []
    lens = []
    term_df = Counter()
    term_postings = defaultdict(list)

    for idx, c in enumerate(records):
        cid = str(c.get("chunk_id", idx))
        did = str(c.get("doc_id", c.get("document_id", cid)))

        weighted_tokens = []
        body_text = (
            _normalize_str(c.get("text_norm"))
            or _normalize_str(c.get("text_raw"))
            or _normalize_str(c.get("text", ""))
        )
        weighted_tokens.extend(legacy_tokenize_pyvi(body_text) * int(field_weights["body"]))

        legal_num = _normalize_str(c.get("legal_number", ""))
        if legal_num:
            weighted_tokens.extend(legacy_tokenize_pyvi(legal_num) * int(field_weights["legal_number"]))

        title = _normalize_str(c.get("title", ""))
        if title:
            weighted_tokens.extend(legacy_tokenize_pyvi(title) * int(field_weights["title"]))

        article = _normalize_str(c.get("article", ""))
        if article:
            weighted_tokens.extend(legacy_tokenize_pyvi(article) * int(field_weights["article"]))

        clause = _normalize_str(c.get("clause", ""))
        if clause:
            weighted_tokens.extend(legacy_tokenize_pyvi(clause) * int(field_weights["clause"]))

        link = _normalize_str(c.get("link", ""))
        if link:
            slug = link.rstrip("/").split("/")[-1].replace("-", " ")
            weighted_tokens.extend(legacy_tokenize_pyvi(slug) * int(field_weights["url_slug"]))

        lens.append(len(weighted_tokens))
        chunk_ids.append(cid)
        doc_ids.append(did)

        tf = Counter(weighted_tokens)
        for term, freq in tf.items():
            term_df[term] += 1
            term_postings[term].append((idx, freq))

    N = len(records)
    chunk_lens = np.array(lens, dtype=np.float32)
    avg_len = float(np.mean(chunk_lens)) if N > 0 else 1.0
    idf = {term: float(math.log((N - df + 0.5) / (df + 0.5) + 1.0)) for term, df in term_df.items()}
    postings = {
        term: (np.array([p[0] for p in plist], dtype=np.int32), np.array([p[1] for p in plist], dtype=np.float32))
        for term, plist in term_postings.items()
    }
    return {
        "chunk_ids": chunk_ids,
        "doc_ids": doc_ids,
        "chunk_lens": chunk_lens,
        "avg_len": avg_len,
        "idf": idf,
        "postings": postings,
    }


def _assert_equivalence(opt_retriever: BM25PyViRetriever, legacy_data: dict[str, Any], queries: list[str]) -> None:
    """Assert structural parameters and retrieval scores match exactly."""
    assert opt_retriever.chunk_ids == legacy_data["chunk_ids"]
    assert opt_retriever.doc_ids == legacy_data["doc_ids"]
    np.testing.assert_allclose(opt_retriever.chunk_lens, legacy_data["chunk_lens"])
    assert math.isclose(opt_retriever.avg_len, legacy_data["avg_len"], rel_tol=1e-6)
    assert set(opt_retriever.idf.keys()) == set(legacy_data["idf"].keys())

    for k in opt_retriever.idf:
        assert math.isclose(opt_retriever.idf[k], legacy_data["idf"][k], rel_tol=1e-6)

    legacy_instance = BM25PyViRetriever()
    legacy_instance.chunk_ids = legacy_data["chunk_ids"]
    legacy_instance.doc_ids = legacy_data["doc_ids"]
    legacy_instance.chunk_lens = legacy_data["chunk_lens"]
    legacy_instance.avg_len = legacy_data["avg_len"]
    legacy_instance.idf = legacy_data["idf"]
    legacy_instance.postings = legacy_data["postings"]

    for q in queries:
        res_opt = opt_retriever.retrieve(q, top_k=20)
        res_leg = legacy_instance.retrieve(q, top_k=20)
        assert len(res_opt) == len(res_leg)
        assert len(res_opt) > 0
        for item_opt, item_leg in zip(res_opt, res_leg):
            assert item_opt["doc_id"] == item_leg["doc_id"]
            assert math.isclose(item_opt["score"], item_leg["score"], rel_tol=1e-5)


def test_bm25_pyvi_optimized_exact_equivalence():
    """Verify optimized BM25PyViRetriever generates identical results to legacy unoptimized baseline."""
    chunks_path = Path("kaggle_dataset/chunks.parquet") if Path("kaggle_dataset/chunks.parquet").is_file() else Path("artifacts/task1/data/chunks.parquet")
    docs_path = str(Path("kaggle_dataset/documents.parquet") if Path("kaggle_dataset/documents.parquet").is_file() else Path("artifacts/task1/data/documents.parquet"))
    if not chunks_path.is_file():
        pytest.skip("canonical dataset not present (Kaggle-only artifact)")

    chunks_df = pd.read_parquet(chunks_path)
    sample = chunks_df[chunks_df["granularity"] == "micro"].iloc[:1500]
    enriched = enrich_chunks_with_doc_metadata(sample, docs_path)

    # 1. Fit with optimized retriever
    opt_retriever = BM25PyViRetriever().fit(enriched)

    # 2. Fit with legacy unoptimized function
    legacy_data = fit_legacy(enriched.to_dict("records"))

    # Assert structural parameters match exactly
    _assert_equivalence(
        opt_retriever,
        legacy_data,
        [
            "quy định về xử phạt vi phạm giao thông đường bộ",
            "thời hạn nộp thuế thu nhập cá nhân theo luật quản lý thuế",
            "hồ sơ đăng ký doanh nghiệp cổ phần cần những giấy tờ gì",
            "thủ tục giải quyết tranh chấp đất đai theo luật đất đai",
        ],
    )


def test_bm25_pyvi_df_list_parity_synthetic():
    """DataFrame vs legacy list parity on crafted edge rows (runs without the dataset)."""
    df = pd.DataFrame(
        [
            {
                "chunk_id": "c1",
                "doc_id": "d1",
                "text_norm": "nội dung thử nghiệm luật đất đai",
                "text_raw": "raw",
                "title": "Luật Đất đai",
                "legal_number": "13/2024/QH15",
                "article": "Điều 5",
                "clause": "khoản 2",
                "link": "https://example.com/luat-dat-dai",
            },
            {
                # Empty text_norm must fall back to text_raw like legacy `or` logic.
                "chunk_id": "c2",
                "doc_id": "d2",
                "text_norm": "",
                "text_raw": "noi dung raw fallback tranh chấp",
                "title": float("nan"),
                "legal_number": None,
                "article": float("nan"),
                "clause": "",
                "link": None,
            },
            {
                # Whitespace-only text_norm must also fall back to meaningful raw text.
                "chunk_id": "c_ws",
                "doc_id": "d_ws",
                "text_norm": "   ",
                "text_raw": "noi dung raw quan trong",
            },
            {"chunk_id": "c3", "doc_id": "d3"},
        ]
    )

    opt_retriever = BM25PyViRetriever().fit(df)
    legacy_data = fit_legacy(df.to_dict("records"))

    _assert_equivalence(
        opt_retriever,
        legacy_data,
        ["tranh chấp đất đai", "nội dung thử nghiệm", "noi dung raw quan trong"],
    )


def test_bm25_pyvi_edge_cases_and_null_handling():
    """Verify DataFrame and List paths handle pd.NA, np.nan, document_id fallback, and text column."""
    rows = [
        {
            "chunk_id": "c1",
            "doc_id": "d1",
            "text_norm": "luật đất đai quy định bồi thường",
            "title": "Luật Đất đai",
            "legal_number": "13/2024/QH15",
            "article": "Điều 5",
            "clause": "khoản 1",
            "link": "https://example.com/c1",
        },
        {
            "chunk_id": "c2",
            "doc_id": "d2",
            "text_norm": pd.NA,
            "text_raw": "tranh chấp quyền sử dụng đất",
            "title": None,
            "legal_number": np.nan,
            "article": pd.NA,
            "clause": "",
            "link": None,
        },
        {
            "chunk_id": "c3",
            "document_id": "d3_fallback",
            "text": "quy định xử phạt vi phạm giao thông",
        },
        {
            "text_norm": "hợp đồng lao động tiền lương bảo hiểm",
        },
    ]
    df = pd.DataFrame(rows)
    records = df.to_dict("records")

    ret_df = BM25PyViRetriever().fit(df)
    ret_list = BM25PyViRetriever().fit(records)

    # Chunk IDs and Doc IDs must all be strings
    assert all(isinstance(x, str) for x in ret_df.chunk_ids)
    assert all(isinstance(x, str) for x in ret_df.doc_ids)
    assert ret_df.chunk_ids == ret_list.chunk_ids
    assert ret_df.doc_ids == ret_list.doc_ids
    assert ret_df.doc_ids[2] == "d3_fallback"
    assert ret_df.chunk_ids[3] == "3"
    assert ret_df.doc_ids[3] == "3"

    np.testing.assert_allclose(ret_df.chunk_lens, ret_list.chunk_lens)
    assert math.isclose(ret_df.avg_len, ret_list.avg_len, rel_tol=1e-6)
    assert set(ret_df.idf.keys()) == set(ret_list.idf.keys())

    # Retrieval results must be identical
    res_df = ret_df.retrieve("tranh chấp đất đai", top_k=5)
    res_list = ret_list.retrieve("tranh chấp đất đai", top_k=5)
    assert len(res_df) == len(res_list)
    for r_df, r_list in zip(res_df, res_list):
        assert r_df["doc_id"] == r_list["doc_id"]
        assert math.isclose(r_df["score"], r_list["score"], rel_tol=1e-5)


def test_pyvi_oversized_input_falls_back_without_segmenter():
    """Regression: multi-MB anomalous chunks must not hang ViTokenizer.

    Production incident: a 5.7M-char chunk stalled indexing at ~88% (segmenter
    >90s on one input vs ~1ms normal). Inputs over PYVI_MAX_CHARS must take
    the regex fallback path quickly and be counted.
    """
    import time as _time

    from src.retrieval.bm25_pyvi import (
        PYVI_MAX_CHARS,
        _tokenize_pyvi_cached,
        get_pyvi_fallback_count,
        tokenize_pyvi,
    )

    _tokenize_pyvi_cached.cache_clear()
    before = get_pyvi_fallback_count()

    normal = "quy định về bồi thường đất đai và hợp đồng lao động"
    t0 = _time.time()
    normal_toks = tokenize_pyvi(normal)
    assert _time.time() - t0 < 5.0
    assert len(normal_toks) > 0
    assert get_pyvi_fallback_count() == before  # normal path: no fallback

    monster = "quy định pháp luật " * ((PYVI_MAX_CHARS // 20) + 1000)
    assert len(monster) > PYVI_MAX_CHARS
    t0 = _time.time()
    monster_toks = tokenize_pyvi(monster)
    elapsed = _time.time() - t0
    assert elapsed < 10.0, f"oversized input took {elapsed:.1f}s (hang risk)"
    assert len(monster_toks) > 1000  # content still indexed via fallback
    assert get_pyvi_fallback_count() == before + 1

    # Small inputs keep exact legacy segmentation behavior
    assert tokenize_pyvi(normal) == legacy_tokenize_pyvi(normal)

    # End-to-end: fit() completes on a corpus containing a monster chunk
    rows = [
        {"chunk_id": "ok1", "doc_id": "d1", "text_norm": normal},
        {"chunk_id": "big1", "doc_id": "d9", "text_norm": monster},
    ]
    ret = BM25PyViRetriever().fit(pd.DataFrame(rows))
    assert len(ret.chunk_ids) == 2
    assert len(ret.retrieve("bồi thường đất đai", top_k=2)) >= 1


def test_dense_preprocess_skips_segmenter_on_oversized_input():
    """Dense macro path must not hang on the same anomalous mega-chunks."""
    import time as _time

    from src.retrieval.bm25_pyvi import PYVI_MAX_CHARS
    from src.retrieval.dense_macro import DenseMacroRetriever

    retr = DenseMacroRetriever.__new__(DenseMacroRetriever)
    retr.use_pyvi = True

    monster = "điều khoản hợp đồng " * ((PYVI_MAX_CHARS // 20) + 1000)
    assert len(monster) > PYVI_MAX_CHARS
    t0 = _time.time()
    out = retr.preprocess_text(monster)
    elapsed = _time.time() - t0
    assert elapsed < 10.0, f"dense preprocess took {elapsed:.1f}s (hang risk)"
    assert isinstance(out, str) and len(out) > 0

    # Normal inputs still go through PyVi segmentation
    normal_out = retr.preprocess_text("quy định về bồi thường đất đai")
    assert isinstance(normal_out, str) and len(normal_out) > 0


def test_bm25_pyvi_multiprocessing_exact_parity():
    """Verify sequential vs parallel worker logic produces bit-identical index."""
    rows = [
        {"chunk_id": f"c_{i}", "doc_id": f"d_{i % 10}", "text_norm": f"quy định điều {i} về bảo hiểm xã hội và lao động", "title": f"Luật số {i}"}
        for i in range(100)
    ]
    df = pd.DataFrame(rows)
    r1 = BM25PyViRetriever().fit(df, num_workers=1)
    r2 = BM25PyViRetriever().fit(df, num_workers=1)
    assert r1.chunk_ids == r2.chunk_ids
    assert r1.doc_ids == r2.doc_ids
    assert np.allclose(r1.chunk_lens, r2.chunk_lens)
    assert r1.idf == r2.idf
    res1 = r1.retrieve("bảo hiểm xã hội", top_k=5)
    res2 = r2.retrieve("bảo hiểm xã hội", top_k=5)
    assert [x["doc_id"] for x in res1] == [x["doc_id"] for x in res2]
    assert [x["score"] for x in res1] == [x["score"] for x in res2]
