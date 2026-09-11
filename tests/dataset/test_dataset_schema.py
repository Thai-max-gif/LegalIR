from pathlib import Path
import pyarrow.parquet as pq
import pytest
from src.data.canonical import discover_canonical_dataset_dir


@pytest.fixture
def dataset_dir() -> Path:
    return discover_canonical_dataset_dir()


def test_documents_schema(dataset_dir: Path):
    doc_path = dataset_dir / "documents.parquet"
    assert doc_path.is_file(), f"Missing documents.parquet at {dataset_dir}"
    table = pq.read_table(doc_path)
    cols = set(table.column_names)
    assert "doc_id" in cols
    assert any(c in cols for c in ("passage_norm", "text_norm", "text_raw", "text"))


def test_chunks_schema(dataset_dir: Path):
    chunk_path = dataset_dir / "chunks.parquet"
    assert chunk_path.is_file(), f"Missing chunks.parquet at {dataset_dir}"
    schema = pq.read_schema(str(chunk_path))
    cols = set(schema.names)
    assert "chunk_id" in cols
    assert "doc_id" in cols
    assert any(c in cols for c in ("chunk_text", "text_norm", "text_raw", "text"))


def test_queries_schema(dataset_dir: Path):
    query_path = dataset_dir / "queries_train.parquet"
    assert query_path.is_file(), f"Missing queries_train.parquet at {dataset_dir}"
    table = pq.read_table(query_path)
    cols = set(table.column_names)
    assert "query_id" in cols
    assert any(c in cols for c in ("question_norm", "question_raw", "query_text", "text"))


def test_qrels_schema(dataset_dir: Path):
    qrel_path = dataset_dir / "qrels_train.parquet"
    assert qrel_path.is_file(), f"Missing qrels_train.parquet at {dataset_dir}"
    table = pq.read_table(qrel_path)
    cols = set(table.column_names)
    assert "query_id" in cols
    assert "doc_id" in cols
