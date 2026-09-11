from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REQUIRED_DENSE_IDENTITY = {"model_id", "model_revision", "pooling", "preprocessing", "embedding_dimension"}


def validate_index_identity(path: str | Path | None, configured_model_id: str | None, configured_revision: str | None) -> dict[str, Any]:
    if not path:
        return {"status": "BLOCKED", "reason": "retrieval.index_manifest_path is not configured"}
    source = Path(path)
    if not source.is_file():
        return {"status": "BLOCKED", "reason": f"index manifest missing: {source}"}
    try:
        manifest = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"status": "FAIL", "reason": f"invalid index manifest: {exc.msg}"}
    dense = manifest.get("dense", manifest)
    missing = sorted(REQUIRED_DENSE_IDENTITY - set(dense))
    if missing:
        return {"status": "BLOCKED", "reason": "dense index identity incomplete", "missing": missing, "manifest": manifest}
    mismatches = {}
    if configured_model_id and dense["model_id"] != configured_model_id:
        mismatches["model_id"] = [configured_model_id, dense["model_id"]]
    if configured_revision and dense["model_revision"] != configured_revision:
        mismatches["model_revision"] = [configured_revision, dense["model_revision"]]
    if mismatches:
        return {"status": "FAIL", "reason": "configured dense encoder incompatible with index", "mismatches": mismatches, "manifest": manifest}
    return {"status": "PASS", "manifest": manifest}
