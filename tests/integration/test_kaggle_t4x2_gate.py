"""
Integration tests for Kaggle Dual-T4 CUDA Gate (scripts/gates/run_kaggle_t4x2.py).
Ensures fail-closed behavior on hardware mismatch, dataset tampering, or fallback attempts.
"""

from __future__ import annotations

import json
import pytest
from pathlib import Path
from unittest.mock import patch

from src.release.fingerprints import generate_dataset_manifest
from scripts.gates.run_kaggle_t4x2 import run_kaggle_t4x2_gate


@pytest.fixture
def mock_canonical_data(tmp_path: Path) -> Path:
    """Create a minimal real-schema parquet dataset for gate testing."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    data_dir = tmp_path / "canonical_data"
    data_dir.mkdir(parents=True)

    # Documents
    docs_table = pa.Table.from_pydict({
        "doc_id": ["1", "2", "3", "4"],
        "text": ["Official legal decree text about commerce", "Civil code article on obligations", "Tax law provisions", "Penal code article on fraud"],
    })
    pq.write_table(docs_table, data_dir / "documents.parquet")

    # Chunks
    chunks_table = pa.Table.from_pydict({
        "chunk_id": ["c1", "c2", "c3", "c4"],
        "doc_id": ["1", "2", "3", "4"],
        "text": ["Official legal decree text about commerce", "Civil code article on obligations", "Tax law provisions", "Penal code article on fraud"],
    })
    pq.write_table(chunks_table, data_dir / "chunks.parquet")

    # Queries train
    queries_table = pa.Table.from_pydict({
        "query_id": ["q1", "q2"],
        "question_norm": ["What are commercial rules?", "What are fraud penalties?"],
    })
    pq.write_table(queries_table, data_dir / "queries_train.parquet")

    # Qrels train
    qrels_table = pa.Table.from_pydict({
        "query_id": ["q1", "q2"],
        "doc_id": ["1", "4"],
    })
    pq.write_table(qrels_table, data_dir / "qrels_train.parquet")

    # Public official json
    (data_dir / "public-official.json").write_text(json.dumps([{"query_id": "pub1", "question": "test"}]), encoding="utf-8")

    # Generate manifest
    manifest = generate_dataset_manifest(data_dir)
    (data_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return data_dir


def test_kaggle_gate_rejects_single_gpu_in_production(mock_canonical_data: Path, tmp_path: Path):
    """Kaggle dual-T4 gate must fail if fewer than 2 CUDA devices are present."""
    out_dir = tmp_path / "gate_out"
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.device_count", return_value=1), \
         patch("torch.cuda.get_device_name", return_value="Tesla T4"):
        with pytest.raises(RuntimeError, match="requires >= 2 CUDA devices"):
            run_kaggle_t4x2_gate(
                dataset_dir=mock_canonical_data,
                output_dir=out_dir,
                mock=False,
            )


def test_kaggle_gate_fails_on_tampered_dataset(mock_canonical_data: Path, tmp_path: Path):
    """Kaggle gate must fail closed if any dataset file is mutated."""
    out_dir = tmp_path / "gate_out"
    # Tamper with documents.parquet
    (mock_canonical_data / "documents.parquet").write_bytes(b"corrupted_bytes")

    valid_sha = "718efb7ba4565fa5b863f05927122484f8e58c2f"
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.device_count", return_value=2), \
         patch("torch.cuda.get_device_name", return_value="Tesla T4"), \
         patch("src.release.fingerprints.get_git_head_sha", return_value=valid_sha):
        with pytest.raises(RuntimeError, match="Dataset file 'documents.parquet' failed checksum"):
            run_kaggle_t4x2_gate(
                dataset_dir=mock_canonical_data,
                output_dir=out_dir,
                expected_sha=valid_sha,
                mock=False,
            )


def test_kaggle_gate_mock_mode_produces_compliant_report(mock_canonical_data: Path, tmp_path: Path):
    """Mock mode executes without GPU and produces a valid gate report."""
    out_dir = tmp_path / "gate_out"
    report = run_kaggle_t4x2_gate(
        dataset_dir=mock_canonical_data,
        output_dir=out_dir,
        mock=True,
    )
    assert report["stage"] == "KAGGLE_T4X2"
    assert report["verdict"] in ("PASS", "DEBUG_ONLY")
    assert report["real_models_only"] is True
    assert report["cpu_fallback_used"] is False
    assert report["model_fallback_used"] is False
    assert (out_dir / "kaggle_t4x2_report.json").is_file()
