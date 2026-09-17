"""Canonical Task 1 v2 dataset verification and metadata inspection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
import pyarrow.parquet as pq

from src.core.hashing import sha256_file

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

EXPECTED_DOCS = 8532
EXPECTED_CHUNKS = 1153876
EXPECTED_MICRO = 934416
EXPECTED_MACRO = 219460
EXPECTED_TRAIN_QUERIES = 7000
EXPECTED_QRELS = 7637
EXPECTED_PUBLIC_QUERIES = 1000
EXPECTED_DUPLICATE_GROUPS = 4


def resolve_duplicate_groups_path(dataset_dir: Union[str, Path]) -> Optional[Path]:
    """
    Resolve path to duplicate_groups.json using the fallback chain:
    dataset root -> dataset root/splits -> repo kaggle_dataset -> repo artifacts/task1/data -> None
    """
    d = Path(dataset_dir)
    candidates = [
        d / "duplicate_groups.json",
        d / "splits" / "duplicate_groups.json",
        REPO_ROOT / "kaggle_dataset" / "duplicate_groups.json",
        REPO_ROOT / "kaggle_dataset" / "splits" / "duplicate_groups.json",
        REPO_ROOT / "artifacts" / "task1" / "data" / "duplicate_groups.json",
        REPO_ROOT / "data" / "task1_canonical_v2" / "duplicate_groups.json",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def resolve_split_path(dataset_dir: Union[str, Path], filename: str) -> Optional[Path]:
    """
    Resolve path to a split artifact file using the fallback chain:
    dataset root -> dataset root/splits -> repo kaggle_dataset -> repo artifacts/task1/data -> None
    """
    d = Path(dataset_dir)
    candidates = [
        d / filename,
        d / "splits" / filename,
        REPO_ROOT / "kaggle_dataset" / filename,
        REPO_ROOT / "kaggle_dataset" / "splits" / filename,
        REPO_ROOT / "artifacts" / "task1" / "data" / filename,
        REPO_ROOT / "data" / "task1_canonical_v2" / filename,
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


@dataclass
class CanonicalDatasetIdentity:
    """Canonical dataset metadata and verification parameters."""

    dataset_name: str = "task1_canonical"
    version: str = "v2"
    schema_version: str = "hierarchical_micro_macro_v2"
    num_docs: int = EXPECTED_DOCS
    num_chunks: int = EXPECTED_CHUNKS
    num_micro: int = EXPECTED_MICRO
    num_macro: int = EXPECTED_MACRO
    num_train_queries: int = EXPECTED_TRAIN_QUERIES
    num_qrels: int = EXPECTED_QRELS
    num_public_queries: int = EXPECTED_PUBLIC_QUERIES
    num_duplicate_groups: int = EXPECTED_DUPLICATE_GROUPS

    def is_canonical_match(self) -> bool:
        return (
            self.num_docs == EXPECTED_DOCS
            and self.num_chunks == EXPECTED_CHUNKS
            and self.num_micro == EXPECTED_MICRO
            and self.num_macro == EXPECTED_MACRO
            and self.num_train_queries == EXPECTED_TRAIN_QUERIES
            and self.num_qrels == EXPECTED_QRELS
            and self.num_public_queries == EXPECTED_PUBLIC_QUERIES
            and self.num_duplicate_groups == EXPECTED_DUPLICATE_GROUPS
        )


def read_parquet_metadata_fast(path: Union[str, Path]) -> Dict[str, Any]:
    """Read Parquet metadata without loading the dataset into memory."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    parquet_file = pq.ParquetFile(str(path))
    metadata = parquet_file.metadata
    schema = parquet_file.schema_arrow

    return {
        "num_rows": metadata.num_rows,
        "num_columns": metadata.num_columns,
        "num_row_groups": metadata.num_row_groups,
        "columns": [col.name for col in schema],
        "size_bytes": path.stat().st_size,
    }


def verify_canonical_dataset(
    dataset_dir: Union[str, Path],
) -> Tuple[bool, CanonicalDatasetIdentity, List[str]]:
    """
    Verify canonical Task 1 dataset against authoritative specification.
    Returns (is_valid, identity, list_of_errors).
    """
    d = Path(dataset_dir)
    errors: List[str] = []

    if not d.is_dir():
        if (REPO_ROOT / "artifacts" / "task1" / "data").is_dir():
            d = REPO_ROOT / "artifacts" / "task1" / "data"
        elif (REPO_ROOT / dataset_dir).is_dir():
            d = REPO_ROOT / dataset_dir
        else:
            return False, CanonicalDatasetIdentity(), [f"Dataset directory not found: {d}"]

    # Required files
    docs_p = d / "documents.parquet"
    chunks_p = d / "chunks.parquet"
    queries_train_p = d / "queries_train.parquet"
    qrels_train_p = d / "qrels_train.parquet"
    public_p = d / "public-official.json"
    dup_p = d / "duplicate_groups.json"

    num_docs = 0
    num_chunks = 0
    num_micro = 0
    num_macro = 0
    num_train_queries = 0
    num_qrels = 0
    num_public_queries = 0
    num_duplicate_groups = 0

    if docs_p.is_file():
        meta = read_parquet_metadata_fast(docs_p)
        num_docs = meta["num_rows"]
        if num_docs != EXPECTED_DOCS:
            errors.append(f"Expected {EXPECTED_DOCS} docs, found {num_docs}")
    else:
        errors.append(f"Missing {docs_p.name}")

    if chunks_p.is_file():
        meta = read_parquet_metadata_fast(chunks_p)
        num_chunks = meta["num_rows"]
        if num_chunks != EXPECTED_CHUNKS:
            errors.append(f"Expected {EXPECTED_CHUNKS} chunks, found {num_chunks}")
        # To count micro/macro, inspect table granularity or chunk_type column efficiently
        try:
            schema_names = pq.read_schema(str(chunks_p)).names
            type_col = "granularity" if "granularity" in schema_names else "chunk_type"
            tbl = pq.read_table(str(chunks_p), columns=[type_col])
            chunk_types = tbl[type_col].to_pylist()
            num_micro = sum(1 for t in chunk_types if t == "micro")
            num_macro = sum(1 for t in chunk_types if t == "macro")
            if num_micro != EXPECTED_MICRO:
                errors.append(f"Expected {EXPECTED_MICRO} micro chunks, found {num_micro}")
            if num_macro != EXPECTED_MACRO:
                errors.append(f"Expected {EXPECTED_MACRO} macro chunks, found {num_macro}")
        except Exception as e:
            errors.append(f"Error inspecting chunk granularity: {e}")
    else:
        errors.append(f"Missing {chunks_p.name}")

    if queries_train_p.is_file():
        meta = read_parquet_metadata_fast(queries_train_p)
        num_train_queries = meta["num_rows"]
        if num_train_queries != EXPECTED_TRAIN_QUERIES:
            errors.append(f"Expected {EXPECTED_TRAIN_QUERIES} train queries, found {num_train_queries}")
    else:
        errors.append(f"Missing {queries_train_p.name}")

    if qrels_train_p.is_file():
        meta = read_parquet_metadata_fast(qrels_train_p)
        num_qrels = meta["num_rows"]
        if num_qrels != EXPECTED_QRELS:
            errors.append(f"Expected {EXPECTED_QRELS} qrels, found {num_qrels}")
    else:
        errors.append(f"Missing {qrels_train_p.name}")

    if public_p.is_file():
        try:
            with open(public_p, "r", encoding="utf-8") as f:
                pub_data = json.load(f)
            num_public_queries = len(pub_data)
            if num_public_queries != EXPECTED_PUBLIC_QUERIES:
                errors.append(f"Expected {EXPECTED_PUBLIC_QUERIES} public queries, found {num_public_queries}")
        except Exception as e:
            errors.append(f"Failed parsing public-official.json: {e}")
    else:
        errors.append(f"Missing {public_p.name}")

    resolved_dup_p = resolve_duplicate_groups_path(d)
    if resolved_dup_p and resolved_dup_p.is_file():
        try:
            with open(resolved_dup_p, "r", encoding="utf-8") as f:
                dup_data = json.load(f)
            num_duplicate_groups = len(dup_data)
            if num_duplicate_groups != EXPECTED_DUPLICATE_GROUPS:
                errors.append(f"Expected {EXPECTED_DUPLICATE_GROUPS} duplicate groups, found {num_duplicate_groups}")
        except Exception as e:
            errors.append(f"Failed parsing duplicate_groups.json: {e}")
    else:
        errors.append(f"Missing duplicate_groups.json in {d} or fallback locations")

    identity = CanonicalDatasetIdentity(
        num_docs=num_docs,
        num_chunks=num_chunks,
        num_micro=num_micro,
        num_macro=num_macro,
        num_train_queries=num_train_queries,
        num_qrels=num_qrels,
        num_public_queries=num_public_queries,
        num_duplicate_groups=num_duplicate_groups,
    )

    is_valid = len(errors) == 0 and identity.is_canonical_match()
    return is_valid, identity, errors


def discover_canonical_dataset_dir(preferred_path: Union[str, Path, None] = None) -> Path:
    """
    Discover the root directory containing the canonical Task 1 dataset across
    Kaggle, Google Colab, and local environments.
    """
    import os
    if preferred_path:
        p = Path(preferred_path)
        if (p / "documents.parquet").is_file() or (p / "canonical" / "documents.parquet").is_file():
            return p

    env_path = os.environ.get("LEGALIR_DATA_ROOT") or os.environ.get("LEGALIR_DATASET_DIR")
    if env_path:
        p = Path(env_path)
        if (p / "documents.parquet").is_file() or (p / "canonical" / "documents.parquet").is_file():
            return p

    # Dynamic search under /kaggle/input if running on Kaggle
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        for doc_file in kaggle_input.rglob("documents.parquet"):
            candidate = doc_file.parent
            if (candidate / "queries_train.parquet").is_file():
                return candidate
            if (candidate.parent / "queries_train.parquet").is_file():
                return candidate.parent

    candidates = [
        REPO_ROOT / "kaggle_dataset",
        Path("/content/kaggle_dataset"),
        Path("/content/LegalIR/kaggle_dataset"),
        Path("/kaggle/input/datasets/phucdangg/legalir-task1-clean-data"),
        Path("/kaggle/input/legalir-task1-clean-data"),
        Path("/kaggle/input/task1-canonical-v2"),
        Path("/content/drive/MyDrive/legalir-task1-clean-data"),
        Path("/content/legalir-task1-clean-data"),
        REPO_ROOT / "artifacts" / "task1" / "data",
    ]
    for c in candidates:
        if (c / "documents.parquet").is_file():
            return c
        if (c / "canonical" / "documents.parquet").is_file():
            return c / "canonical"

    # Default fallback to artifacts/task1/data
    return REPO_ROOT / "artifacts" / "task1" / "data"

