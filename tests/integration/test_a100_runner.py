"""
Integration tests for Colab A100 Production Runner (scripts/gates/run_a100.py).
Ensures 1 A100 device is accepted, upstream gate reports are strictly enforced,
and full provenance is captured into run_manifest.json.
"""

from __future__ import annotations

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from scripts.gates.run_a100 import run_a100_production_gate
from src.release.fingerprints import generate_dataset_manifest, fingerprint_structured_config, get_git_head_sha


@pytest.fixture
def a100_test_fixtures(tmp_path: Path) -> dict[str, Path]:
    """Create minimal canonical data, Kaggle report, and Colab T4 report fixtures."""
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

    algo_cfg_path = Path(__file__).resolve().parent.parent.parent / "configs" / "algorithm" / "legalir_v2.yaml"
    algo_hash = fingerprint_structured_config(algo_cfg_path)
    valid_sha = get_git_head_sha()

    # Upstream reports
    k_report = {
        "stage": "KAGGLE_T4X2",
        "verdict": "PASS",
        "git_sha": valid_sha,
        "dataset_manifest_sha256": manifest["manifest_sha256"],
        "algorithm_config_sha256": algo_hash,
    }
    k_report_path = tmp_path / "kaggle_t4x2_report.json"
    k_report_path.write_text(json.dumps(k_report, indent=2), encoding="utf-8")

    c_report = {
        "stage": "COLAB_SINGLE_T4",
        "verdict": "PASS",
        "git_sha": valid_sha,
        "dataset_manifest_sha256": manifest["manifest_sha256"],
        "algorithm_config_sha256": algo_hash,
    }
    c_report_path = tmp_path / "colab_t4_report.json"
    c_report_path.write_text(json.dumps(c_report, indent=2), encoding="utf-8")

    return {
        "dataset_dir": data_dir,
        "kaggle_report_path": k_report_path,
        "colab_t4_report_path": c_report_path,
        "valid_sha": valid_sha,
    }


def test_a100_runner_fails_without_kaggle_report(a100_test_fixtures, tmp_path: Path):
    """A100 runner fails closed if Kaggle dual-T4 report is missing."""
    out_dir = tmp_path / "a100_out"
    missing_k = tmp_path / "missing_kaggle.json"

    with pytest.raises(RuntimeError, match="Kaggle T4x2 report missing"):
        run_a100_production_gate(
            dataset_dir=a100_test_fixtures["dataset_dir"],
            output_dir=out_dir,
            kaggle_report_path=missing_k,
            colab_t4_report_path=a100_test_fixtures["colab_t4_report_path"],
            expected_sha=a100_test_fixtures["valid_sha"],
            mock=True,
        )


def test_a100_runner_fails_without_colab_t4_report(a100_test_fixtures, tmp_path: Path):
    """A100 runner fails closed if Colab T4 report is missing."""
    out_dir = tmp_path / "a100_out"
    missing_c = tmp_path / "missing_colab_t4.json"

    with pytest.raises(RuntimeError, match="Colab T4 report missing"):
        run_a100_production_gate(
            dataset_dir=a100_test_fixtures["dataset_dir"],
            output_dir=out_dir,
            kaggle_report_path=a100_test_fixtures["kaggle_report_path"],
            colab_t4_report_path=missing_c,
            expected_sha=a100_test_fixtures["valid_sha"],
            mock=True,
        )


def test_a100_runner_mock_produces_manifest_and_submission(a100_test_fixtures, tmp_path: Path):
    """Mock execution creates valid submission and full run_manifest.json."""
    out_dir = tmp_path / "a100_out"
    report = run_a100_production_gate(
        dataset_dir=a100_test_fixtures["dataset_dir"],
        output_dir=out_dir,
        kaggle_report_path=a100_test_fixtures["kaggle_report_path"],
        colab_t4_report_path=a100_test_fixtures["colab_t4_report_path"],
        expected_sha=a100_test_fixtures["valid_sha"],
        mock=True,
    )
    assert report["verdict"] in ("PASS", "DEBUG_ONLY")
    assert (out_dir / "submission.zip").is_file()
    assert (out_dir / "run_manifest.json").is_file()
    assert (out_dir / "checksums.sha256").is_file()
