import json
import pytest
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

from src.core.hashing import sha256_file, sha256_bytes, sha256_directory
from src.core.manifests import Manifest, PreflightManifest, JobManifest, BundleManifest
from src.data.canonical import (
    CanonicalDatasetIdentity,
    verify_canonical_dataset,
    read_parquet_metadata_fast,
)


def test_hashing_basic(tmp_path):
    f = tmp_path / "hello.txt"
    f.write_text("legalir", encoding="utf-8")
    expected = "75b8e385e64fe831ac99dc94e9fcc66f6f07b79f94e9dfadef5e67d3a9f264e2"
    assert sha256_file(f) == expected
    assert sha256_bytes(b"legalir") == expected


def test_hashing_directory(tmp_path):
    d = tmp_path / "sub"
    d.mkdir()
    (d / "b.txt").write_text("file b", encoding="utf-8")
    (d / "a.txt").write_text("file a", encoding="utf-8")
    h1 = sha256_directory(d)
    assert len(h1) == 64
    # Determinism check
    h2 = sha256_directory(d)
    assert h1 == h2


def test_manifest_roundtrip(tmp_path):
    m = PreflightManifest(
        dataset_name="task1_canonical",
        dataset_version="v2",
        runtime_commit="a0efb25",
        status="PASS",
        details={"num_docs": 8532, "num_chunks": 1153876},
    )
    p = tmp_path / "preflight.json"
    m.save(p)

    loaded = PreflightManifest.load(p)
    assert loaded.dataset_name == "task1_canonical"
    assert loaded.status == "PASS"
    assert loaded.details["num_docs"] == 8532


def test_canonical_identity_dataclass():
    ident = CanonicalDatasetIdentity(
        dataset_name="task1_canonical",
        version="v2",
        schema_version="hierarchical_micro_macro_v2",
        num_docs=8532,
        num_chunks=1153876,
        num_micro=934416,
        num_macro=219460,
        num_train_queries=7000,
        num_qrels=7637,
        num_public_queries=1000,
        num_duplicate_groups=4,
    )
    assert ident.num_docs == 8532
    assert ident.num_train_queries == 7000
    assert ident.is_canonical_match() is True


def test_read_parquet_metadata_fast(tmp_path):
    table = pa.Table.from_pydict({"doc_id": ["d1", "d2"], "text": ["t1", "t2"]})
    pq_path = tmp_path / "test.parquet"
    pq.write_table(table, str(pq_path))

    meta = read_parquet_metadata_fast(pq_path)
    assert meta["num_rows"] == 2
    assert meta["num_columns"] == 2
    assert "doc_id" in meta["columns"]


def test_duplicate_groups_repo_fallback(tmp_path):
    from src.data.canonical import resolve_duplicate_groups_path
    # Path without duplicate_groups.json should fallback to repo artifacts/task1/data
    empty_dir = tmp_path / "empty_ds"
    empty_dir.mkdir()
    fallback = resolve_duplicate_groups_path(empty_dir)
    assert fallback is not None
    assert fallback.is_file()
    assert "duplicate_groups.json" in fallback.name


def test_real_canonical_fixture_uses_granularity_and_text_norm():
    p = Path("kaggle_dataset/chunks.parquet") if Path("kaggle_dataset/chunks.parquet").is_file() else Path("artifacts/task1/data/chunks.parquet")
    if p.is_file():
        schema_names = pq.read_schema(str(p)).names
        assert "granularity" in schema_names
        assert "text_norm" in schema_names
        assert "text_raw" in schema_names
