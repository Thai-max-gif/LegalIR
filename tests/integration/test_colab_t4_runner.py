"""
Integration tests for Colab Single-T4 Gate Runner (scripts/gates/run_colab_t4.py).
Ensures single-device topology (cuda:0 / cuda:0) and strict upstream Kaggle report verification.
"""

from __future__ import annotations

import json
import pytest
from pathlib import Path
from unittest.mock import patch

from scripts.gates.run_colab_t4 import run_colab_t4_gate
from src.release.fingerprints import generate_dataset_manifest, compute_canonical_json_hash


@pytest.fixture
def mock_dataset_and_kaggle_report(tmp_path: Path) -> tuple[Path, Path]:
    """Create minimal canonical data and a valid Kaggle report fixture."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    data_dir = tmp_path / "canonical_data"
    data_dir.mkdir(parents=True)

    docs_table = pa.Table.from_pydict({
        "doc_id": ["1", "2"],
        "text": ["Civil law contract rules", "Labor code working hours"],
    })
    pq.write_table(docs_table, data_dir / "documents.parquet")

    chunks_table = pa.Table.from_pydict({
        "chunk_id": ["c1", "c2"],
        "doc_id": ["1", "2"],
        "text": ["Civil law contract rules", "Labor code working hours"],
    })
    pq.write_table(chunks_table, data_dir / "chunks.parquet")

    queries_table = pa.Table.from_pydict({
        "query_id": ["q1"],
        "question_norm": ["What are contract rules?"],
    })
    pq.write_table(queries_table, data_dir / "queries_train.parquet")

    qrels_table = pa.Table.from_pydict({
        "query_id": ["q1"],
        "doc_id": ["1"],
    })
    pq.write_table(qrels_table, data_dir / "qrels_train.parquet")

    (data_dir / "public-official.json").write_text(json.dumps([{"query_id": "pub1", "question": "test"}]), encoding="utf-8")

    manifest = generate_dataset_manifest(data_dir)
    (data_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Kaggle report
    from src.release.fingerprints import fingerprint_structured_config
    algo_cfg_path = Path(__file__).resolve().parent.parent.parent / "configs" / "algorithm" / "legalir_v2.yaml"
    algo_hash = fingerprint_structured_config(algo_cfg_path)

    valid_sha = "718efb7ba4565fa5b863f05927122484f8e58c2f"
    kaggle_report = {
        "stage": "KAGGLE_T4X2",
        "verdict": "PASS",
        "git_sha": valid_sha,
        "dataset_manifest_sha256": manifest["manifest_sha256"],
        "algorithm_config_sha256": algo_hash,
        "gpu_count": 2,
    }
    kaggle_report_path = tmp_path / "kaggle_t4x2_report.json"
    kaggle_report_path.write_text(json.dumps(kaggle_report, indent=2), encoding="utf-8")

    return data_dir, kaggle_report_path


def test_colab_t4_gate_requires_kaggle_report(mock_dataset_and_kaggle_report, tmp_path: Path):
    """Colab T4 gate fails if Kaggle report does not exist."""
    data_dir, _ = mock_dataset_and_kaggle_report
    out_dir = tmp_path / "colab_out"
    missing_kaggle = tmp_path / "nonexistent_kaggle_report.json"

    with pytest.raises(FileNotFoundError, match="Kaggle report not found"):
        run_colab_t4_gate(
            dataset_dir=data_dir,
            output_dir=out_dir,
            kaggle_report_path=missing_kaggle,
            mock=True,
        )


def test_colab_t4_gate_mock_produces_compliant_report(mock_dataset_and_kaggle_report, tmp_path: Path):
    """Colab T4 gate in mock mode successfully emits colab_t4_report.json."""
    data_dir, kaggle_report_path = mock_dataset_and_kaggle_report
    out_dir = tmp_path / "colab_out"

    report = run_colab_t4_gate(
        dataset_dir=data_dir,
        output_dir=out_dir,
        kaggle_report_path=kaggle_report_path,
        mock=True,
    )
    assert report["stage"] == "COLAB_SINGLE_T4"
    assert report["dense_device"] == "cuda:0"
    assert report["reranker_device"] == "cuda:0"
    assert report["single_gpu_production_path"] is True
    assert (out_dir / "colab_t4_report.json").is_file()


def test_colab_t4_gate_rejects_mismatched_sha(mock_dataset_and_kaggle_report, tmp_path: Path):
    """Colab T4 gate rejects Kaggle report if its git_sha does not match."""
    data_dir, kaggle_report_path = mock_dataset_and_kaggle_report
    out_dir = tmp_path / "colab_out"

    # Mismatch SHA in kaggle report
    kaggle_data = json.loads(kaggle_report_path.read_text(encoding="utf-8"))
    kaggle_data["git_sha"] = "0" * 40
    kaggle_report_path.write_text(json.dumps(kaggle_data, indent=2), encoding="utf-8")

    with pytest.raises(RuntimeError, match="Git SHA mismatch"):
        run_colab_t4_gate(
            dataset_dir=data_dir,
            output_dir=out_dir,
            kaggle_report_path=kaggle_report_path,
            mock=True,
        )
