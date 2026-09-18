#!/usr/bin/env python3
"""Independent CPU-only acceptance verifier (fix.md section 12).

Reconstructs the official pooled OOF metric from persisted per-query
predictions plus fixed expected splits. Declared paths only (no broad scans,
no credential reads). Exit 0 only for a complete matching PASS; all other
states exit nonzero with reasons on stdout as JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.release.acceptance import verify_acceptance


def _resolve(base: Path | None, value: str | None) -> Path | None:
    if not value:
        return None
    p = Path(value)
    if p.is_absolute():
        return p
    return (base / p) if base is not None else p


def _load_json(path: Path | None) -> object | None:
    if path is None or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Verify one FULL attempt receipt (CPU-only).")
    ap.add_argument("--receipt", required=True, help="Machine-readable acceptance receipt JSON.")
    ap.add_argument("--artifacts-dir", default=None, help="Base dir for relative artifact paths.")
    ap.add_argument("--oof-predictions", required=True, help="Persisted OOF predictions JSON.")
    ap.add_argument("--qrels", required=True, help="Qrels JSON {query_id: [doc_id, ...]}.")
    ap.add_argument("--splits", required=True, help="Fixed random_5fold splits JSON.")
    ap.add_argument("--corpus-ids", required=True, help="Corpus document IDs JSON list.")
    ap.add_argument("--disjoint-report", default=None)
    ap.add_argument("--submission", default=None)
    args = ap.parse_args(argv)

    base = Path(args.artifacts_dir) if args.artifacts_dir else None
    receipt = _load_json(_resolve(base, args.receipt))
    if not isinstance(receipt, dict):
        print(json.dumps({"verdict": "INCOMPLETE", "reasons": ["receipt unreadable"]}, indent=2))
        return 2
    oof = _load_json(_resolve(base, args.oof_predictions))
    qrels = _load_json(_resolve(base, args.qrels))
    splits = _load_json(_resolve(base, args.splits))
    corpus = _load_json(_resolve(base, args.corpus_ids))
    disjoint = _load_json(_resolve(base, args.disjoint_report))
    submission = _load_json(_resolve(base, args.submission))

    result = verify_acceptance(
        receipt,
        oof_predictions=oof if isinstance(oof, dict) else {},
        qrels=qrels if isinstance(qrels, dict) else {},
        splits=splits,
        corpus_doc_ids=set(corpus) if isinstance(corpus, list) else set(),
        disjoint_report=disjoint if isinstance(disjoint, dict) else None,
        submission=submission if isinstance(submission, dict) else None,
        artifacts_dir=base,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
