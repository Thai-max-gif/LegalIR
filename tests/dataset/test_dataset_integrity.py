import json
from pathlib import Path
import pyarrow.parquet as pq
import pytest
from src.data.canonical import (
    EXPECTED_DOCS,
    EXPECTED_DUPLICATE_GROUPS,
    EXPECTED_PUBLIC_QUERIES,
    EXPECTED_QRELS,
    EXPECTED_TRAIN_QUERIES,
    discover_canonical_dataset_dir,
    resolve_duplicate_groups_path,
)


@pytest.fixture
def dataset_dir() -> Path:
    d = discover_canonical_dataset_dir()
    if not (d / "documents.parquet").is_file():
        pytest.skip("Canonical parquet dataset not present in git checkout (published directly on Kaggle).")
    return d


def test_record_counts(dataset_dir: Path):
    docs = pq.read_table(dataset_dir / "documents.parquet")
    queries = pq.read_table(dataset_dir / "queries_train.parquet")
    qrels = pq.read_table(dataset_dir / "qrels_train.parquet")

    assert len(docs) == EXPECTED_DOCS
    assert len(queries) == EXPECTED_TRAIN_QUERIES
    assert len(qrels) == EXPECTED_QRELS


def test_public_queries_count(dataset_dir: Path):
    pub_path = dataset_dir / "public-official.json"
    if not pub_path.is_file():
        # Fallback to root or artifacts
        pub_path = Path("artifacts/task1/data/public-official.json")
    assert pub_path.is_file(), f"Missing public-official.json"
    with open(pub_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data) == EXPECTED_PUBLIC_QUERIES


def test_duplicate_groups(dataset_dir: Path):
    dup_path = resolve_duplicate_groups_path(dataset_dir)
    assert dup_path is not None and dup_path.is_file(), "Missing duplicate_groups.json"
    with open(dup_path, "r", encoding="utf-8") as f:
        groups = json.load(f)
    assert len(groups) == EXPECTED_DUPLICATE_GROUPS
