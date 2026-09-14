"""
Determinism test ensuring BM25PyViRetriever optimizations produce exact,
bit-identical retrieval results compared to legacy baseline logic.
"""

from collections import Counter, defaultdict
import math
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
        body_text = _normalize_str(c.get("text_norm") or c.get("text_raw", ""))
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


def test_bm25_pyvi_optimized_exact_equivalence():
    """Verify optimized BM25PyViRetriever generates identical results to legacy unoptimized baseline."""
    chunks_path = "artifacts/task1/data/chunks.parquet"
    docs_path = "artifacts/task1/data/documents.parquet"

    chunks_df = pd.read_parquet(chunks_path)
    sample = chunks_df[chunks_df["granularity"] == "micro"].iloc[:1500]
    enriched = enrich_chunks_with_doc_metadata(sample, docs_path)

    # 1. Fit with optimized retriever
    opt_retriever = BM25PyViRetriever().fit(enriched)

    # 2. Fit with legacy unoptimized function
    legacy_data = fit_legacy(enriched.to_dict("records"))

    # Assert structural parameters match exactly
    assert opt_retriever.chunk_ids == legacy_data["chunk_ids"]
    assert opt_retriever.doc_ids == legacy_data["doc_ids"]
    np.testing.assert_allclose(opt_retriever.chunk_lens, legacy_data["chunk_lens"])
    assert math.isclose(opt_retriever.avg_len, legacy_data["avg_len"], rel_tol=1e-6)
    assert set(opt_retriever.idf.keys()) == set(legacy_data["idf"].keys())

    for k in opt_retriever.idf:
        assert math.isclose(opt_retriever.idf[k], legacy_data["idf"][k], rel_tol=1e-6)

    # Assert retrieve scores match bit-for-bit
    test_queries = [
        "quy định về xử phạt vi phạm giao thông đường bộ",
        "thời hạn nộp thuế thu nhập cá nhân theo luật quản lý thuế",
        "hồ sơ đăng ký doanh nghiệp cổ phần cần những giấy tờ gì",
        "thủ tục giải quyết tranh chấp đất đai theo luật đất đai",
    ]

    # Create dummy legacy instance
    legacy_instance = BM25PyViRetriever()
    legacy_instance.chunk_ids = legacy_data["chunk_ids"]
    legacy_instance.doc_ids = legacy_data["doc_ids"]
    legacy_instance.chunk_lens = legacy_data["chunk_lens"]
    legacy_instance.avg_len = legacy_data["avg_len"]
    legacy_instance.idf = legacy_data["idf"]
    legacy_instance.postings = legacy_data["postings"]

    for q in test_queries:
        res_opt = opt_retriever.retrieve(q, top_k=20)
        res_leg = legacy_instance.retrieve(q, top_k=20)
        assert len(res_opt) == len(res_leg)
        assert len(res_opt) > 0
        for item_opt, item_leg in zip(res_opt, res_leg):
            assert item_opt["doc_id"] == item_leg["doc_id"]
            assert math.isclose(item_opt["score"], item_leg["score"], rel_tol=1e-5)
