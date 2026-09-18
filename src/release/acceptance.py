"""Joint acceptance contract (fix.md sections 8 and 12).

CPU-only. Reconstructs the official pooled OOF metric from persisted
per-query predictions plus fixed expected splits — never trusts a claimed
summary score. Strict thresholds: elapsed must be *under* 18000 seconds
(exactly 18000 fails); pooled official OOF Recall@5 must be *above* 0.96
(exactly 0.96 fails); no rounding across thresholds.

Receipts use NOT_RUN defaults (null numerics) before measurement; no plausible
scores or durations are seeded.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

SCHEMA_VERSION = 1
TIME_GATE_SECONDS = 18000
QUALITY_GATE_RECALL5 = 0.96
SCORE_TOLERANCE = 1e-9

_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_MUTABLE_REVS = {"", "main", "master", "latest", "default"}


def not_run_receipt(attempt_id: str = "unassigned") -> dict[str, Any]:
    """Machine-readable receipt before measurement: null numerics, NOT_RUN."""
    return {
        "schema_version": SCHEMA_VERSION,
        "attempt_id": attempt_id,
        "runtime_sha": None,
        "release_sha": None,
        "dataset_digest": None,
        "split_random_5fold_sha256": None,
        "split_doc_disjoint_sha256": None,
        "config_sha256": None,
        "model_revisions": {"reranker": None, "dense": None},
        "backend": None,
        "protocol": "confirmation",
        "predeclared_policy_hash": None,
        "selection_protocol": "predeclared_rrf",
        "split_exposure_disclosure": None,
        "supervisor_start_utc": None,
        "supervisor_end_utc": None,
        "elapsed_seconds": None,
        "expected_oof_queries": None,
        "per_fold": [],
        "pooled_oof_recall@5": None,
        "pooled_oof_precision@5": None,
        "doc_disjoint": {"recall@5": None, "count": None, "complete": False},
        "training_jobs": [],
        "reload_ok": False,
        "submission_path": None,
        "artifacts": [],
        "delivery_confirmed": False,
        "shutdown_confirmed": False,
        "verdict": "NOT_RUN",
        "reasons": ["not measured"],
    }


def _is_immutable_rev(rev: Any) -> bool:
    if rev is None:
        return False
    s = str(rev).strip()
    if not s or s.lower() in _MUTABLE_REVS or len(s) != 40:
        return False
    return all(c in "0123456789abcdef" for c in s.lower())


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_adapter_reload_fresh(
    adapter_dir: str | Path,
    model_name: str,
    revision: str | None,
    timeout_s: int = 600,
) -> tuple[bool, str]:
    """Reload the saved adapter in a FRESH OS process and score one pair.

    Proves the persisted artifact (not in-memory state) loads under the pinned
    base revision. Returns ``(ok, detail)``; never raises. A missing adapter
    dir or manifest fails without spawning a process.
    """
    import subprocess
    import sys

    adapter = Path(adapter_dir)
    if not (adapter / "adapter_config.json").is_file():
        return False, "adapter_config.json missing"
    weights = adapter / "adapter_model.safetensors"
    if not weights.is_file():
        weights = adapter / "adapter_model.bin"
    if not weights.is_file():
        return False, "adapter weights missing"
    snippet = (
        "import json;"
        "from src.ranking.reranker import CrossEncoderReranker;"
        f"r=CrossEncoderReranker(model_name={model_name!r},adapter_path={str(adapter)!r},"
        f"device='cpu',revision={revision!r});"
        "r.ensure_loaded();"
        "s=r.score_pairs([('fresh reload probe','fresh reload probe')],batch_size=1);"
        "print(json.dumps({'ok':True,'score':float(s[0])}))"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", snippet],
            capture_output=True, text=True, timeout=int(timeout_s),
        )
    except subprocess.TimeoutExpired:
        return False, f"fresh reload timed out after {timeout_s}s"
    except Exception as exc:
        return False, f"fresh reload spawn failed: {type(exc).__name__}"
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-500:]
        return False, f"fresh reload exited {proc.returncode}: {tail}"
    try:
        payload = json.loads((proc.stdout or "").strip().splitlines()[-1])
        if payload.get("ok") is True:
            return True, f"fresh reload score={payload.get('score')}"
    except Exception:
        pass
    return False, f"fresh reload unparseable output: {(proc.stdout or '')[-200:]}"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _official_recall_precision(
    y_pred: Mapping[str, list[str]],
    y_true: Mapping[str, list[str]],
) -> tuple[float, float]:
    """Official per-query macro recall/precision (scoring.py semantics)."""
    if not y_true:
        raise ValueError("empty gold set")
    recalls: list[float] = []
    precisions: list[float] = []
    for qid, gold in y_true.items():
        pred = list(y_pred.get(str(qid), []))
        gold_list = [str(x) for x in gold]
        if not gold_list:
            recalls.append(0.0)
            precisions.append(0.0)
            continue
        inter = len(set(gold_list) & set(pred))
        recalls.append(inter / len(gold_list))
        precisions.append(inter / len(pred) if pred else 0.0)
    return float(sum(recalls) / len(recalls)), float(sum(precisions) / len(precisions))


def _expected_ids_from_splits(splits: Any, num_folds: int = 5) -> tuple[dict[int, list[str]], set[str]]:
    """Parse canonical random_5fold splits; require exactly five valid folds."""
    if not isinstance(splits, list) or len(splits) != num_folds:
        raise ValueError(f"expected exactly {num_folds} folds, got {len(splits) if isinstance(splits, list) else type(splits)}")
    per_fold: dict[int, list[str]] = {}
    for idx, fold in enumerate(splits):
        if not isinstance(fold, dict):
            raise ValueError(f"fold {idx} is not an object")
        val = fold.get("val_query_ids", fold.get("val", []))
        if not isinstance(val, list) or not val:
            raise ValueError(f"fold {idx} has no validation IDs")
        per_fold[idx] = [str(x) for x in val]
    return per_fold, {qid for ids in per_fold.values() for qid in ids}


def verify_acceptance(
    receipt: Mapping[str, Any],
    *,
    oof_predictions: Mapping[str, Any],
    qrels: Mapping[str, list[str]],
    splits: Any,
    corpus_doc_ids: set[str] | list[str],
    disjoint_report: Mapping[str, Any] | None,
    submission: Mapping[str, Any] | None,
    artifacts_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Verify one FULL attempt. Returns recomputed metrics + verdict + reasons.

    All inputs are in-memory structures loaded by the caller from declared
    paths (the CLI resolves those paths without broad scans). Checks, in order:
    schema/provenance identity, exact ID/fold coverage, per-query shape (five
    unique valid docs), recomputed pooled official score vs claimed (tolerance),
    strict time/quality gates, disjoint completeness, reload/delivery/shutdown,
    predeclared policy, artifact checksums.
    """
    reasons: list[str] = []
    details: dict[str, Any] = {}

    def fail(reason: str) -> None:
        reasons.append(reason)

    # --- Schema + provenance identity ---
    if int(receipt.get("schema_version", -1)) != SCHEMA_VERSION:
        fail(f"schema_version must be {SCHEMA_VERSION}")
    for key in ("runtime_sha", "release_sha"):
        val = receipt.get(key)
        if not isinstance(val, str) or not _SHA40.match(val):
            fail(f"{key} must be an exact 40-char lowercase SHA")
    for key in ("dataset_digest", "split_random_5fold_sha256", "config_sha256"):
        if not receipt.get(key):
            fail(f"{key} is missing")
    model_revs = receipt.get("model_revisions") or {}
    for mkey in ("reranker", "dense"):
        if not _is_immutable_rev(model_revs.get(mkey)):
            fail(f"model_revisions.{mkey} must be an immutable 40-char pin (got {model_revs.get(mkey)!r})")
    if receipt.get("selection_protocol") != "predeclared_rrf":
        fail(f"selection_protocol must be predeclared_rrf for confirmation (got {receipt.get('selection_protocol')!r})")
    if not receipt.get("predeclared_policy_hash"):
        fail("predeclared_policy_hash is missing")

    # --- Expected population from fixed splits ---
    try:
        per_fold, expected_ids = _expected_ids_from_splits(splits)
    except ValueError as exc:
        fail(str(exc))
        per_fold, expected_ids = {}, set()
    details["expected_oof_queries"] = len(expected_ids)
    if receipt.get("expected_oof_queries") not in (None, len(expected_ids)):
        fail(f"expected_oof_queries mismatch (receipt={receipt.get('expected_oof_queries')} splits={len(expected_ids)})")

    # --- Predictions: exact coverage + per-query shape ---
    corpus = {str(x) for x in corpus_doc_ids}
    norm_preds: dict[str, list[str]] = {}
    if not isinstance(oof_predictions, Mapping) or not oof_predictions:
        fail("oof predictions are missing or empty")
    else:
        for qid, val in oof_predictions.items():
            ans = val.get("answer") if isinstance(val, Mapping) else val
            if not isinstance(ans, list):
                fail(f"query {qid}: answer is not a list")
                continue
            norm_preds[str(qid)] = [str(x) for x in ans]
        if set(norm_preds.keys()) != expected_ids and expected_ids:
            missing = sorted(expected_ids - set(norm_preds.keys()))[:5]
            extra = sorted(set(norm_preds.keys()) - expected_ids)[:5]
            fail(f"OOF coverage mismatch (missing={missing} extra={extra})")
        for qid, docs in norm_preds.items():
            if len(docs) != 5:
                fail(f"query {qid}: expected exactly 5 documents, got {len(docs)}")
            if len(set(docs)) != len(docs):
                fail(f"query {qid}: duplicate document IDs")
            invalid = [d for d in docs if d not in corpus]
            if invalid:
                fail(f"query {qid}: invalid document IDs {invalid[:3]}")

    # --- Recomputed official score vs claimed (no trust in summary) ---
    recomputed_r5: float | None = None
    recomputed_p5: float | None = None
    if norm_preds and expected_ids and not any("coverage" in r or "exactly 5" in r or "duplicate" in r or "invalid" in r for r in reasons):
        gold = {qid: [str(x) for x in qrels.get(qid, [])] for qid in expected_ids}
        if any(not gold[qid] for qid in expected_ids):
            fail("qrels missing gold for expected queries")
        else:
            try:
                recomputed_r5, recomputed_p5 = _official_recall_precision(norm_preds, gold)
                details["recomputed_recall@5"] = recomputed_r5
                details["recomputed_precision@5"] = recomputed_p5
            except ValueError as exc:
                fail(f"scoring failed: {exc}")
    claimed_r5 = receipt.get("pooled_oof_recall@5")
    if recomputed_r5 is not None and isinstance(claimed_r5, (int, float)):
        if abs(float(claimed_r5) - recomputed_r5) > SCORE_TOLERANCE:
            fail(f"claimed pooled score {claimed_r5} inconsistent with recomputed {recomputed_r5:.6f}")
    elif recomputed_r5 is not None and claimed_r5 is None:
        fail("receipt omits claimed pooled_oof_recall@5")

    # Per-fold counts/scores present for all five folds.
    per_fold_receipt = receipt.get("per_fold") or []
    if len(per_fold_receipt) != 5:
        fail(f"per_fold must cover five folds (got {len(per_fold_receipt)})")

    # --- Timing endpoints must exist; elapsed must be a finite non-negative
    # duration strictly under the gate (negative/absent can never pass) ---
    for endpoint in ("supervisor_start_utc", "supervisor_end_utc"):
        if not isinstance(receipt.get(endpoint), str) or not receipt[endpoint].strip():
            fail(f"timing endpoint {endpoint} is absent")
    elapsed = receipt.get("elapsed_seconds")
    if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)):
        fail(f"cold elapsed {elapsed!r} is not a measured duration")
        pass_t = False
    elif not (0 <= float(elapsed) < TIME_GATE_SECONDS):
        fail(f"cold elapsed {elapsed} is not in [0, {TIME_GATE_SECONDS})s")
        pass_t = False
    else:
        pass_t = True
    pass_q = recomputed_r5 is not None and recomputed_r5 > QUALITY_GATE_RECALL5
    if recomputed_r5 is None:
        fail("quality gate unevaluable (no recomputed score)")
    elif not pass_q:
        fail(f"pooled official OOF Recall@5 {recomputed_r5:.6f} is not strictly above {QUALITY_GATE_RECALL5}")

    # --- Document-disjoint completeness (separate, never blended) ---
    # An empty report mapping is absent for acceptance purposes: it carries no
    # recall, counts, or identity to check.
    dj = receipt.get("doc_disjoint") or {}
    if not isinstance(disjoint_report, Mapping) or not disjoint_report:
        fail("document-disjoint report is absent")
    elif not dj.get("complete"):
        fail("document-disjoint is not marked complete")
    else:
        try:
            dj_sys = disjoint_report.get("trained_reranker_system") or {}
            dj_r5 = float(dj_sys.get("recall@5", disjoint_report.get("recall@5")))
            details["disjoint_recall@5"] = dj_r5
        except Exception:
            fail("document-disjoint report recall unreadable")

    # --- Reload / delivery / shutdown / artifacts ---
    if receipt.get("reload_ok") is not True:
        fail("final fresh-process reload is not confirmed")
    if receipt.get("delivery_confirmed") is not True:
        fail("durable delivery is not confirmed")
    if receipt.get("shutdown_confirmed") is not True:
        fail("provider shutdown is not confirmed")
    # Submission is a required deliverable: an absent submission can never pass,
    # and every delivered answer must hold exactly 5 unique valid documents.
    if not isinstance(submission, Mapping) or not submission:
        fail("submission is absent")
    else:
        for qid, val in submission.items():
            ans = val.get("answer") if isinstance(val, Mapping) else val
            if not isinstance(ans, list) or len(ans) != 5 or len({str(x) for x in ans}) != 5:
                fail(f"submission query {qid}: must hold exactly 5 unique documents")
                break
            if any(str(x) not in corpus for x in ans):
                fail(f"submission query {qid}: invalid document ID")
                break
    # Adapter weights must be inventoried WITH a checksum, and checksums are
    # verified against bytes. Without artifacts_dir there is nothing to verify
    # against, so the check fails closed instead of passing blind.
    artifacts = receipt.get("artifacts") or []
    if not artifacts:
        fail("artifact inventory is empty")
    elif artifacts_dir is None:
        fail("artifact checksums unverified without artifacts_dir")
    else:
        base = Path(artifacts_dir)
        has_weights = False
        for entry in artifacts:
            if not isinstance(entry, Mapping):
                fail(f"artifact entry is not an object: {entry!r}")
                continue
            rel = str(entry.get("path", ""))
            if not rel or rel.startswith("/") or ".." in rel.split("/"):
                fail(f"artifact path escapes inventory: {rel!r}")
                continue
            if "adapter_model" in rel:
                if not entry.get("sha256"):
                    fail(f"artifact checksum absent: {rel}")
                    continue
                has_weights = True
            if entry.get("sha256"):
                target = base / rel
                if not target.is_file():
                    fail(f"artifact missing: {rel}")
                elif _sha_file(target) != str(entry["sha256"]):
                    fail(f"artifact checksum mismatch: {rel}")
        if not has_weights:
            fail("artifact inventory omits final adapter weights with checksum")

    verdict = "PASS" if not reasons else ("FAIL" if recomputed_r5 is not None or elapsed is not None else "INCOMPLETE")
    return {
        "verdict": verdict,
        "pass_t": bool(pass_t),
        "pass_q": bool(pass_q),
        "reasons": reasons,
        "details": details,
        "recomputed_recall@5": recomputed_r5,
        "recomputed_precision@5": recomputed_p5,
    }
