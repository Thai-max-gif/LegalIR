from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.data.dataset_validator import validate_dataset


def test_validator_reports_dangling_chunk_fk(tmp_path: Path):
    (tmp_path / "canonical").mkdir()
    pq.write_table(pa.table({"document_id": ["d1"], "document_text": ["text"], "source_ref": ["src"], "content_sha256": ["a"]}), tmp_path / "canonical/documents.parquet")
    pq.write_table(pa.table({"chunk_id": ["c1"], "document_id": ["missing"], "chunk_text": ["text"], "chunk_level": ["micro"], "order_index": [0], "content_sha256": ["b"]}), tmp_path / "canonical/chunks.parquet")
    report = validate_dataset(tmp_path)
    assert report["status"] == "FAIL"
    assert any("dangling document_id" in x["reason"] for x in report["issues"])


def test_validator_accepts_observed_flat_alias_layout(tmp_path: Path):
    pq.write_table(pa.table({"doc_id": ["d1"], "passage_norm": ["text"], "link": ["src"]}), tmp_path / "documents.parquet")
    pq.write_table(pa.table({"chunk_id": ["c1"], "doc_id": ["d1"], "text_norm": ["text"], "granularity": ["article"]}), tmp_path / "chunks.parquet")
    pq.write_table(pa.table({"query_id": ["q1"], "question_norm": ["what"]}), tmp_path / "queries_train.parquet")
    pq.write_table(pa.table({"query_id": ["q1"], "doc_id": ["d1"], "relevance": [1]}), tmp_path / "qrels_train.parquet")
    report = validate_dataset(tmp_path)
    assert report["status"] == "PASS"
    assert report["layout"] == "flat"
    assert report["column_mappings"]["documents"]["document_id"] == "doc_id"
    assert report["column_mappings"]["queries"]["query_text"] == "question_norm"
