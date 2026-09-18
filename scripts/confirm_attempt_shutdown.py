#!/usr/bin/env python3
"""Record operator-confirmed provider shutdown on an attempt receipt.

The pipeline cannot observe its own provider shutdown, so receipts are born
with ``shutdown_confirmed: false``. After independently confirming termination
(e.g. ``modal app stop <id>`` output, Colab session view), the operator runs:

    python scripts/confirm_attempt_shutdown.py --receipt <attempt>/acceptance_receipt.json \
        --confirmed-by <name> --evidence "modal app stop <id> confirmed stopped"

The script flips only the shutdown flag, records who/what confirmed it, and
re-runs the CPU-only verifier against the same attempt directory. It never
touches scores, timings, or delivery flags.
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--receipt", required=True, help="Path to acceptance_receipt.json.")
    ap.add_argument("--confirmed-by", required=True, help="Operator identity.")
    ap.add_argument("--evidence", required=True, help="Provider stop confirmation evidence.")
    ap.add_argument("--artifacts-dir", default=None, help="Attempt dir (defaults to receipt parent).")
    args = ap.parse_args(argv)

    if not args.confirmed_by.strip() or not args.evidence.strip():
        print("[!] --confirmed-by and --evidence must both be non-empty.", file=sys.stderr)
        return 2
    receipt_path = Path(args.receipt)
    if not receipt_path.is_file():
        print(f"[!] Receipt not found: {receipt_path}", file=sys.stderr)
        return 2
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt = payload.get("receipt")
    if not isinstance(receipt, dict):
        print("[!] Receipt file has no 'receipt' object.", file=sys.stderr)
        return 2
    base = Path(args.artifacts_dir) if args.artifacts_dir else receipt_path.parent

    def _load(name: str, default=None):
        p = base / name
        if not p.is_file():
            return default
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return default

    receipt["shutdown_confirmed"] = True
    receipt["shutdown_confirmation"] = {
        "confirmed_by": args.confirmed_by.strip(),
        "evidence": args.evidence.strip(),
    }
    inputs = receipt.get("verify_inputs") or {}
    result = verify_acceptance(
        receipt,
        oof_predictions=_load(inputs.get("oof_predictions.json", "cv/oof_predictions.json"), {}),
        qrels=_load(inputs.get("qrels.json", "acceptance_inputs/qrels.json"), {}),
        splits=_load(inputs.get("splits.json", "acceptance_inputs/splits.json")),
        corpus_doc_ids=set(_load(inputs.get("corpus_ids.json", "acceptance_inputs/corpus_ids.json"), [])),
        disjoint_report=_load(inputs.get("doc_disjoint_report.json", "acceptance_inputs/doc_disjoint_report.json")),
        submission=_load(inputs.get("submission.json", "acceptance_inputs/submission.json")),
        artifacts_dir=base,
    )
    receipt["verdict"] = result["verdict"]
    receipt["reasons"] = result["reasons"]
    payload["receipt"] = receipt
    payload["verification"] = result
    receipt_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
