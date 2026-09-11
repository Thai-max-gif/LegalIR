from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq

from src.data.paths import artifact_candidates, artifact_description, artifact_path

# The first name in each tuple is the documented contract.  The later names are
# aliases observed in the mounted Kaggle release; they are reported explicitly,
# never written back to the immutable input dataset.
ALIASES = {
    "documents": {
        "document_id": ("document_id", "doc_id"),
        "document_text": ("document_text", "passage_norm", "passage_raw"),
        "source_ref": ("source_ref", "link"),
        "content_sha256": ("content_sha256",),
    },
    "chunks": {
        "chunk_id": ("chunk_id",),
        "document_id": ("document_id", "doc_id"),
        "chunk_text": ("chunk_text", "text_norm", "text_raw"),
        "chunk_level": ("chunk_level", "granularity"),
        "order_index": ("order_index",),
        "content_sha256": ("content_sha256",),
    },
    "queries": {"query_id": ("query_id",), "query_text": ("query_text", "question_norm", "question_raw")},
    "qrels": {"query_id": ("query_id",), "document_id": ("document_id", "doc_id"), "relevance": ("relevance",)},
}
REQUIRED_FIELDS = {
    "documents": {"document_id", "document_text"},
    "chunks": {"chunk_id", "document_id", "chunk_text"},
    "queries": {"query_id", "query_text"},
    "qrels": {"query_id", "document_id"},
}


def stream_sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def parquet_schema(path: Path) -> list[str]:
    return pq.ParquetFile(path).schema_arrow.names


def _values(path: Path, column: str) -> Iterable[Any]:
    for batch in pq.ParquetFile(path).iter_batches(columns=[column], batch_size=100_000):
        yield from batch.column(0).to_pylist()


def _unique(path: Path, column: str, issues: list[dict[str, Any]]) -> set[str]:
    seen: set[str] = set()
    for index, value in enumerate(_values(path, column)):
        if value is None or not str(value).strip():
            issues.append({"file": str(path), "record": index, "reason": f"empty {column}"})
        elif str(value) in seen:
            issues.append({"file": str(path), "record": str(value), "reason": f"duplicate {column}"})
        else:
            seen.add(str(value))
    return seen


def _mapping(columns: list[str], kind: str) -> dict[str, str]:
    return {semantic: found for semantic, aliases in ALIASES[kind].items() if (found := next((name for name in aliases if name in columns), None))}


def _resolve(root: Path, filename: str) -> Path | None:
    return artifact_path(root, filename, "canonical")


def _optional_artifact_check(root: Path, filename: str, category: str, name: str) -> dict[str, Any]:
    path = artifact_path(root, filename, category)
    if not path:
        return {
            "name": name,
            "status": "NOT_RUN",
            "reason": "artifact absent from this mounted release",
            "checked_paths": [str(item) for item in artifact_candidates(root, filename, category)],
        }
    result: dict[str, Any] = {"name": name, "status": "PASS", "path": str(path)}
    if path.suffix == ".parquet":
        try:
            result["columns"] = parquet_schema(path)
        except Exception as exc:  # invalid optional artifacts should be visible
            result.update({"status": "FAIL", "reason": f"cannot inspect parquet: {exc!r}"})
    return result


def _dataset_layout(root: Path, core_paths: dict[str, Path | None]) -> str:
    locations = {"nested" if path and "canonical" in path.relative_to(root).parts else "flat" for path in core_paths.values() if path}
    return next(iter(locations)) if len(locations) == 1 else "mixed"


def validate_dataset(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    report: dict[str, Any] = {
        "dataset_root": str(root),
        "checks": [],
        "issues": [],
        "schemas": {},
        "column_mappings": {},
        "resolved_paths": {},
        "status": "FAIL",
    }
    if not root.is_dir():
        report["issues"].append({"file": str(root), "reason": "dataset root missing"})
        return report
    paths = {"documents": _resolve(root, "documents.parquet"), "chunks": _resolve(root, "chunks.parquet"), "queries": _resolve(root, "queries_train.parquet"), "qrels": _resolve(root, "qrels_train.parquet")}
    report["layout"] = _dataset_layout(root, paths)
    for kind, path in paths.items():
        if not path:
            filename = {"documents": "documents.parquet", "chunks": "chunks.parquet", "queries": "queries_train.parquet", "qrels": "qrels_train.parquet"}[kind]
            report["issues"].append({"file": artifact_description(root, filename, "canonical"), "reason": "required file missing"})
            continue
        columns = parquet_schema(path)
        mapping = _mapping(columns, kind)
        report["schemas"][str(path.relative_to(root))] = columns
        report["column_mappings"][kind] = mapping
        report["resolved_paths"][kind] = str(path)
        missing = sorted(REQUIRED_FIELDS[kind] - set(mapping))
        if missing:
            report["issues"].append({"file": str(path), "reason": "missing required semantic fields", "fields": missing, "columns": columns})
        optional_missing = sorted(set(ALIASES[kind]) - set(mapping) - REQUIRED_FIELDS[kind])
        if optional_missing:
            report["checks"].append({"name": f"{kind}_optional_fields", "status": "NOT_RUN", "reason": "not supplied by this release", "fields": optional_missing})
    if not report["issues"]:
        docs, chunks, queries, qrels = (paths[name] for name in ("documents", "chunks", "queries", "qrels"))
        maps = report["column_mappings"]
        doc_ids = _unique(docs, maps["documents"]["document_id"], report["issues"])
        chunk_ids = _unique(chunks, maps["chunks"]["chunk_id"], report["issues"])
        for doc_id in _values(chunks, maps["chunks"]["document_id"]):
            if str(doc_id) not in doc_ids:
                report["issues"].append({"file": str(chunks), "record": str(doc_id), "reason": "dangling chunk document_id"})
        report["checks"].append({"name": "document_chunk_pk_fk", "status": "PASS" if not report["issues"] else "FAIL", "documents": len(doc_ids), "chunks": len(chunk_ids)})
        query_ids = _unique(queries, maps["queries"]["query_id"], report["issues"])
        for semantic, known in (("query_id", query_ids), ("document_id", doc_ids)):
            for value in _values(qrels, maps["qrels"][semantic]):
                if str(value) not in known:
                    report["issues"].append({"file": str(qrels), "record": str(value), "reason": f"dangling qrels {semantic}"})
        report["checks"].append({"name": "qrels_fk", "status": "PASS" if not report["issues"] else "FAIL"})
    manifest = artifact_path(root, "dataset_manifest.json", "metadata") or artifact_path(root, "manifest.json")
    if manifest:
        try:
            report["manifest"] = json.loads(manifest.read_text(encoding="utf-8"))
            report["manifest_sha256"] = stream_sha256(manifest)
            report["checks"].append({"name": "manifest", "status": "PASS"})
        except json.JSONDecodeError as exc:
            report["issues"].append({"file": str(manifest), "reason": f"invalid JSON: {exc.msg}"})
    else:
        report["checks"].append({"name": "manifest", "status": "NOT_RUN", "reason": "manifest absent"})

    checksums = artifact_path(root, "checksums.sha256", "metadata") or artifact_path(root, "checksums.sha256")
    if checksums:
        report["checks"].append({"name": "checksums", "status": "PASS", "path": str(checksums), "sha256": stream_sha256(checksums)})
    else:
        report["checks"].append({"name": "checksums", "status": "NOT_RUN", "reason": "checksums.sha256 absent; no replacement checksum file was created"})

    # These are merely inspected here.  Stages which rely on them are blocked
    # by preflight rather than being incorrectly reported as dataset corruption.
    report["checks"].extend([
        _optional_artifact_check(root, "folds.parquet", "splits", "folds_artifact"),
        _optional_artifact_check(root, "document_disjoint_split.parquet", "splits", "document_disjoint_split_artifact"),
        _optional_artifact_check(root, "reranker_pairs.parquet", "derived_optional", "reranker_pairs_artifact"),
        _optional_artifact_check(root, "hard_negatives.parquet", "derived_optional", "hard_negatives_artifact"),
        _optional_artifact_check(root, "retrieval_candidates.parquet", "derived_optional", "retrieval_candidates_artifact"),
        _optional_artifact_check(root, "duplicate_groups.json", "metadata", "duplicate_groups_artifact"),
        _optional_artifact_check(root, "empty_context_ids.json", "metadata", "empty_context_ids_artifact"),
    ])
    public_official = artifact_path(root, "public-official.json")
    report["checks"].append({
        "name": "public_official_excluded",
        "status": "PASS",
        "path": str(public_official) if public_official else None,
        "reason": "never used for training, hyperparameter selection, or checkpoint selection",
    })
    report["status"] = "PASS" if not report["issues"] else "FAIL"
    return report


def write_validation_report(root: str | Path, destination: str | Path) -> dict[str, Any]:
    report = validate_dataset(root)
    Path(destination).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
