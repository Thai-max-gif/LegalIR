import json
from pathlib import Path
import pandas as pd
import pytest

from src.dataset.validator import validate_canonical_dataset


def test_validator_reports_duplicate_ids_and_cross_doc_parent(tmp_path: Path):
    pd.DataFrame([
        {"doc_id": "1", "name_raw": "doc1", "title": "Doc 1", "link": "", "passage_raw": "p1", "passage_norm": "p1", "legal_number": None, "year": None, "doc_type": "Luật", "is_empty": False},
        {"doc_id": "1", "name_raw": "doc1 dup", "title": "Doc 1 dup", "link": "", "passage_raw": "p1", "passage_norm": "p1", "legal_number": None, "year": None, "doc_type": "Luật", "is_empty": False},
    ]).to_parquet(tmp_path / "documents.parquet")

    pd.DataFrame([
        {"chunk_id": "m1", "doc_id": "1", "granularity": "macro", "chapter": None, "section": None, "article": "Điều 1", "clause": None, "point": None, "text_raw": "raw", "text_norm": "norm", "parent_chunk_id": None, "token_count": 10, "is_empty": False},
        {"chunk_id": "u1", "doc_id": "2", "granularity": "micro", "chapter": None, "section": None, "article": "Điều 1", "clause": None, "point": None, "text_raw": "raw", "text_norm": "norm", "parent_chunk_id": "m1", "token_count": 10, "is_empty": False},
    ]).to_parquet(tmp_path / "chunks.parquet")

    pd.DataFrame([{"query_id": "q1", "question_raw": "raw", "question_norm": "norm", "gold_count": 1}]).to_parquet(tmp_path / "queries_train.parquet")
    pd.DataFrame([{"query_id": "missing_qid", "doc_id": "1", "relevance": 1}]).to_parquet(tmp_path / "qrels_train.parquet")

    report = validate_canonical_dataset(str(tmp_path), expected_document_count=2)
    err_text = " ".join(report["errors"])
    assert "duplicate document IDs" in err_text
    assert "unknown query IDs" in err_text
    assert "cross-document parent" in err_text
