"""Strict submission formatting, validation, and packaging for LegalIR Task 1."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any
import zipfile
import pandas as pd


def compute_sha256(file_or_bytes: str | Path | bytes) -> str:
    """Compute SHA-256 hex digest of a file or bytes."""
    if isinstance(file_or_bytes, bytes):
        return hashlib.sha256(file_or_bytes).hexdigest()
    p = Path(file_or_bytes)
    if not p.exists():
        return ""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_submission(
    predictions_or_file: dict[str, Any] | str | Path,
    expected_qids: set[str] | None = None,
    corpus_doc_ids: set[str] | None = None,
    public_json: str | Path | None = None,
    data_dir: str | Path | None = None,
    raise_on_error: bool | None = None,
) -> dict[str, Any]:
    """Validate submission structure against competition rules.

    Accepts either a loaded dictionary of predictions or a filepath to submission.json.
    """
    errors: list[str] = []

    if isinstance(predictions_or_file, (str, Path)):
        sub_path = Path(predictions_or_file)
        if not sub_path.exists():
            return {"is_valid": False, "errors": [f"File not found: {sub_path}"], "total_queries": 0}
        with open(sub_path, "r", encoding="utf-8") as f:
            predictions = json.load(f)
    else:
        predictions = predictions_or_file

    should_raise = (not isinstance(predictions_or_file, (str, Path))) if raise_on_error is None else bool(raise_on_error)

    if not isinstance(predictions, dict) or not predictions:
        msg = "Predictions must be a non-empty dictionary"
        if should_raise:
            raise ValueError(msg)
        return {"is_valid": False, "errors": [msg], "total_queries": 0}

    # If public_json is provided, load expected query IDs
    if expected_qids is None and public_json is not None and Path(public_json).exists():
        with open(public_json, "r", encoding="utf-8") as f:
            pub_data = json.load(f)
        expected_qids = set(str(k) for k in pub_data.keys())

    # If data_dir is provided, load valid corpus document IDs
    if corpus_doc_ids is None and data_dir is not None:
        docs_p = Path(data_dir) / "documents.parquet"
        if docs_p.exists():
            df_docs = pd.read_parquet(docs_p, columns=["doc_id"])
            corpus_doc_ids = set(df_docs["doc_id"].astype(str))

    pred_qids = set(str(k) for k in predictions.keys())

    if expected_qids is not None:
        expected_set = set(str(k) for k in expected_qids)
        if pred_qids != expected_set:
            missing = expected_set - pred_qids
            extra = pred_qids - expected_set
            msg = f"Submission query keys mismatch: {len(missing)} missing, {len(extra)} extra"
            errors.append(msg)
            if should_raise:
                raise ValueError(msg)

    for qid, qobj in predictions.items():
        if not isinstance(qobj, dict) or "answer" not in qobj:
            msg = f"Query {qid} prediction must be a dict with 'answer' key"
            errors.append(msg)
            if should_raise:
                raise ValueError(msg)
            continue

        answer = qobj["answer"]
        if not isinstance(answer, list):
            msg = f"Query {qid} answer must be a list of document IDs"
            errors.append(msg)
            if should_raise:
                raise ValueError(msg)
            continue

        if not (1 <= len(answer) <= 5):
            msg = f"Query {qid} answer length must be between 1 to 5, got {len(answer)}"
            errors.append(msg)
            if should_raise:
                raise ValueError(msg)

        if not all(isinstance(x, str) for x in answer):
            msg = f"Query {qid} answer IDs must all be strings"
            errors.append(msg)
            if should_raise:
                raise ValueError(msg)

        if len(answer) != len(set(answer)):
            msg = f"Query {qid} contains duplicate document IDs in answer: {answer}"
            errors.append(msg)
            if should_raise:
                raise ValueError(msg)

        if corpus_doc_ids is not None:
            unknown_docs = set(answer) - corpus_doc_ids
            if unknown_docs:
                msg = f"Query {qid} contains unknown document IDs not in corpus: {unknown_docs}"
                errors.append(msg)
                if should_raise:
                    raise ValueError(msg)

    is_valid = len(errors) == 0
    return {
        "is_valid": is_valid,
        "errors": errors,
        "total_queries": len(predictions),
    }


def validate_submission_zip(zip_path: str | Path) -> dict[str, Any]:
    """Validate that submission.zip contains strictly only submission.json at archive root,
    and that the payload satisfies the scorer schema ({qid: {"answer": [1-5 str]}})."""
    zip_path = Path(zip_path)
    if not zip_path.exists():
        return {"is_valid": False, "errors": [f"ZIP not found: {zip_path}"]}

    errors = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = zf.namelist()
        if namelist != ["submission.json"]:
            errors.append(f"ZIP must contain strictly ['submission.json'] at root, found: {namelist}")

        if "submission.json" in namelist:
            try:
                with zf.open("submission.json") as jf:
                    loaded = json.load(jf)
                if not isinstance(loaded, dict) or not loaded:
                    errors.append("submission.json inside zip is empty or invalid JSON")
                else:
                    # Schema check (no expected_qids/corpus here; structural only).
                    schema = validate_submission(loaded, raise_on_error=False)
                    if not schema.get("is_valid"):
                        errors.extend(schema.get("errors", []))
            except Exception as e:
                errors.append(f"Failed to read submission.json from zip: {e}")

    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
    }


def package_submission(
    predictions_or_file: dict[str, Any] | str | Path,
    json_path_or_zip: str | Path,
    zip_path: str | Path | None = None,
) -> Path:
    """Write submission.json and package into submission.zip.

    Supports:
      - package_submission(predictions, json_path, zip_path)
      - package_submission(json_path, zip_path)
    """
    if zip_path is None:
        actual_json_path = Path(predictions_or_file)
        actual_zip_path = Path(json_path_or_zip)
        json_bytes = actual_json_path.read_bytes()
    else:
        actual_json_path = Path(json_path_or_zip)
        actual_zip_path = Path(zip_path)
        if isinstance(predictions_or_file, (str, Path)):
            json_bytes = Path(predictions_or_file).read_bytes()
            if actual_json_path != Path(predictions_or_file):
                actual_json_path.parent.mkdir(parents=True, exist_ok=True)
                actual_json_path.write_bytes(json_bytes)
        else:
            json_bytes = (json.dumps(predictions_or_file, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
            actual_json_path.parent.mkdir(parents=True, exist_ok=True)
            actual_json_path.write_bytes(json_bytes)

    actual_zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(actual_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("submission.json", json_bytes)

    return actual_zip_path


def create_submission_manifest(
    submission_json_path: str | Path,
    submission_zip_path: str | Path,
    output_path: str | Path | None = None,
    git_commit: str = "unknown",
    config_path: str | Path | None = None,
    dataset_manifest_path: str | Path | None = None,
    parameter_total: int | None = None,
    model_names_and_revisions: list[dict[str, Any]] | None = None,
    all_answers_valid: bool = True,
    all_ids_valid: bool = True,
    extra_metadata: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create submission_manifest.json with integrity hashes and metadata."""
    json_path = Path(submission_json_path)
    zip_path = Path(submission_zip_path)

    query_count = 0
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            query_count = len(data)
        except Exception:
            pass

    manifest: dict[str, Any] = {
        "git_commit": str(git_commit),
        "config_sha256": compute_sha256(config_path) if config_path else "",
        "dataset_manifest_sha256": compute_sha256(dataset_manifest_path) if dataset_manifest_path else "",
        "query_count": query_count,
        "prediction_count": query_count,
        "all_answers_valid": bool(all_answers_valid),
        "all_ids_valid": bool(all_ids_valid),
        "parameter_total": parameter_total,
        "model_names_and_revisions": model_names_and_revisions or [],
        "submission_json_sha256": compute_sha256(json_path),
        "submission_zip_sha256": compute_sha256(zip_path),
        "submission_zip_size_bytes": zip_path.stat().st_size if zip_path.exists() else 0,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }

    if metadata:
        manifest["metadata"] = dict(metadata)
        manifest.update(metadata)
    if extra_metadata:
        manifest.update(extra_metadata)

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    return manifest


