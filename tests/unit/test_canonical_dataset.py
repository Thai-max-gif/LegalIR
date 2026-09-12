import os
import json
import pytest
import pandas as pd
from src.dataset.validator import validate_canonical_dataset


def test_canonical_dataset_invariants(tmp_path):
    # Create mock canonical files
    docs_df = pd.DataFrame([
        {
            "doc_id": "740",
            "name_raw": "Quyet-dinh-5868-QD-BYT",
            "title": "Quyết định 5868/QĐ-BYT 2018",
            "link": "https://example.com/740",
            "passage_raw": "Điều 1. Phạm vi\nĐiều 2. Đối tượng",
            "passage_norm": "điều 1. phạm vi\nđiều 2. đối tượng",
            "legal_number": "5868/QĐ-BYT",
            "year": "2018",
            "doc_type": "Quyết định",
            "is_empty": False
        }
    ])

    chunks_df = pd.DataFrame([
        {
            "chunk_id": "740_macro_001",
            "doc_id": "740",
            "granularity": "macro",
            "chapter": None,
            "section": None,
            "article": "Điều 1",
            "clause": None,
            "point": None,
            "text_raw": "Điều 1. Phạm vi",
            "text_norm": "điều 1. phạm vi",
            "parent_chunk_id": None,
            "token_count": 10,
            "is_empty": False
        },
        {
            "chunk_id": "740_micro_001",
            "doc_id": "740",
            "granularity": "micro",
            "chapter": None,
            "section": None,
            "article": "Điều 1",
            "clause": "Khoản 1",
            "point": None,
            "text_raw": "Điều 1. Phạm vi",
            "text_norm": "điều 1. phạm vi",
            "parent_chunk_id": "740_macro_001",
            "token_count": 5,
            "is_empty": False
        }
    ])

    queries_df = pd.DataFrame([
        {
            "query_id": "101",
            "question_raw": "Quy định phạm vi là gì?",
            "question_norm": "quy định phạm vi là gì?",
            "gold_count": 1
        }
    ])

    qrels_df = pd.DataFrame([
        {
            "query_id": "101",
            "doc_id": "740",
            "relevance": 1
        }
    ])

    data_dir = tmp_path / "v1"
    data_dir.mkdir(parents=True)
    docs_df.to_parquet(data_dir / "documents.parquet")
    chunks_df.to_parquet(data_dir / "chunks.parquet")
    queries_df.to_parquet(data_dir / "queries_train.parquet")
    qrels_df.to_parquet(data_dir / "qrels_train.parquet")

    report = validate_canonical_dataset(str(data_dir))
    assert report["is_valid"] is True
    assert report["total_documents"] == 1
    assert report["total_chunks"] == 2
    assert report["total_micro_chunks"] == 1
    assert report["total_macro_chunks"] == 1
    assert report["total_queries"] == 1
    assert report["total_qrels"] == 1
