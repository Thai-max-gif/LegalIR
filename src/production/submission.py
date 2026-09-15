"""Submission formatting, strict validation against competition rules, and zip packaging.

DEPRECATED for releases: the authoritative scorer-compatible packager is
``src.evaluation.submission`` (``{qid: {"answer": [...]}}``). This legacy
module accepts the bare-list format (``{qid: [...]}``) only for backward
compatibility with local tooling and converts to canonical form when packaging.
Do NOT use this module for A100→HF releases.
"""

from __future__ import annotations

import json
import warnings
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union


def validate_submission(
    submission: Dict[str, List[str]],
    expected_qids: Set[str],
    max_predictions: int = 5,
) -> Tuple[bool, List[str]]:
    """
    Validate submission dictionary against official competition constraints:
    - keysts match expected query IDs exactly
    - 1 to max_predictions predictions per query
    - unique document IDs per query
    """
    errors: List[str] = []
    actual_qids = set(submission.keys())

    missing = expected_qids - actual_qids
    if missing:
        errors.append(f"Missing {len(missing)} query IDs from submission.")

    extra = actual_qids - expected_qids
    if extra:
        errors.append(f"Submission contains {len(extra)} unexpected query IDs.")

    for qid, doc_ids in submission.items():
        if not isinstance(doc_ids, list):
            errors.append(f"Query {qid} predictions must be a list.")
            continue

        if len(doc_ids) < 1:
            errors.append(f"Query {qid} has 0 predictions (at least 1 required).")
        elif len(doc_ids) > max_predictions:
            errors.append(f"Query {qid} has {len(doc_ids)} predictions (exceeds max {max_predictions}).")

        if len(doc_ids) != len(set(doc_ids)):
            errors.append(f"Duplicate document IDs detected in query {qid}: {doc_ids}")

    return len(errors) == 0, errors


def package_submission(
    submission: Dict[str, List[str]],
    out_dir: Union[str, Path],
    filename_prefix: str = "submission",
) -> Tuple[Path, Path]:
    """Save submission.json and compress it into submission.zip.

    Accepts legacy bare-list format and converts to canonical
    ``{qid: {"answer": [...]}}`` via ``src.evaluation.submission`` so local
    tooling cannot emit scorer-incompatible zips.
    """
    warnings.warn(
        "src.production.submission is legacy; releases must use src.evaluation.submission",
        DeprecationWarning,
        stacklevel=2,
    )
    from src.evaluation.submission import package_submission as _canonical_package

    canonical: dict[str, dict[str, list[str]]] = {}
    for qid, val in dict(submission).items():
        if isinstance(val, dict) and "answer" in val:
            canonical[str(qid)] = {"answer": [str(x) for x in val["answer"]]}
        elif isinstance(val, (list, tuple)):
            canonical[str(qid)] = {"answer": [str(x) for x in val]}
        else:
            raise ValueError(f"Query {qid} predictions must be a list or {{'answer': [...]}}.")

    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    json_path = out_p / f"{filename_prefix}.json"
    zip_path = out_p / f"{filename_prefix}.zip"
    json_path.write_text(json.dumps(canonical, indent=2, ensure_ascii=False), encoding="utf-8")
    _canonical_package(json_path, zip_path)
    return json_path, zip_path
