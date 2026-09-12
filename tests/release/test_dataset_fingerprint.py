"""
Tests for canonical dataset manifest and critical file fingerprint validation.
Validates that one-byte mutation in any critical file causes fail-closed abort.
"""

from __future__ import annotations

import json
import pytest
from pathlib import Path

from src.release.fingerprints import (
    compute_file_sha256,
    generate_dataset_manifest,
    verify_dataset_fingerprint,
    DatasetFingerprintMismatchError,
)


@pytest.fixture
def temp_dataset_dir(tmp_path: Path) -> Path:
    """Create a minimal valid canonical dataset structure with a valid manifest."""
    data_dir = tmp_path / "dataset"
    data_dir.mkdir(parents=True)

    files = {
        "documents.parquet": b"documents_content_12345",
        "chunks.parquet": b"chunks_content_12345",
        "queries_train.parquet": b"queries_train_content_12345",
        "qrels_train.parquet": b"qrels_train_content_12345",
        "public-official.json": b'{"queries": ["q1"]}',
    }
    for filename, content in files.items():
        (data_dir / filename).write_bytes(content)

    manifest = generate_dataset_manifest(data_dir)
    (data_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return data_dir


def test_valid_dataset_passes(temp_dataset_dir: Path):
    """Untampered dataset passes verification."""
    manifest_path = temp_dataset_dir / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    res = verify_dataset_fingerprint(temp_dataset_dir, expected_manifest_hash=manifest["manifest_sha256"])
    assert res.is_valid
    assert len(res.verified_files) == 5


def test_mutate_one_byte_causes_fail_closed(temp_dataset_dir: Path):
    """Mutating even a single byte in any critical file raises DatasetFingerprintMismatchError."""
    target_file = temp_dataset_dir / "documents.parquet"
    original_bytes = target_file.read_bytes()
    # Mutate the last byte
    mutated_bytes = original_bytes[:-1] + b"X"
    target_file.write_bytes(mutated_bytes)

    manifest_path = temp_dataset_dir / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    with pytest.raises(DatasetFingerprintMismatchError, match="documents.parquet"):
        verify_dataset_fingerprint(temp_dataset_dir, expected_manifest_hash=manifest["manifest_sha256"])


def test_manifest_tampering_causes_fail_closed(temp_dataset_dir: Path):
    """Altering the manifest itself fails verification against the expected hash."""
    manifest_path = temp_dataset_dir / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hash = manifest["manifest_sha256"]

    # Alter the stored manifest_sha256 inside the manifest
    manifest["manifest_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    with pytest.raises(DatasetFingerprintMismatchError, match="Manifest checksum mismatch"):
        verify_dataset_fingerprint(temp_dataset_dir, expected_manifest_hash=expected_hash)
