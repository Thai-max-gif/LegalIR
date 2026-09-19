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
        # Fallback to kaggle_dataset or root
        pub_path = Path("kaggle_dataset/public-official.json")
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


def test_private_queries_count_and_isolation(dataset_dir: Path):
    priv_path = dataset_dir / "private-official.json"
    if not priv_path.is_file():
        priv_path = Path("kaggle_dataset/private-official.json")
    if not priv_path.is_file():
        pytest.skip("private-official.json not found in dataset directory.")

    with open(priv_path, "r", encoding="utf-8") as f:
        priv_data = json.load(f)

    assert len(priv_data) == 2080, f"Expected 2080 private queries, got {len(priv_data)}"

    # Schema integrity
    for qid, obj in priv_data.items():
        assert isinstance(qid, str) and qid.isdigit()
        assert isinstance(obj, dict) and "question" in obj and "answer" in obj
        assert isinstance(obj["question"], str) and len(obj["question"]) > 0
        assert obj["answer"] is None

    # Complete query ID isolation against train and public
    queries = pq.read_table(dataset_dir / "queries_train.parquet")
    train_qids = set(queries["query_id"].to_pylist())
    priv_qids = set(priv_data.keys())
    assert len(priv_qids & train_qids) == 0, "Leakage: private query IDs overlap with train queries!"

    pub_path = dataset_dir / "public-official.json"
    if not pub_path.is_file():
        pub_path = Path("kaggle_dataset/public-official.json")
    if pub_path.is_file():
        with open(pub_path, "r", encoding="utf-8") as f:
            pub_data = json.load(f)
        pub_qids = set(pub_data.keys())
        assert len(priv_qids & pub_qids) == 0, "Overlap: private query IDs overlap with public queries!"
