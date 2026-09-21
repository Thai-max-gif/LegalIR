"""Section 12 acceptance verifier: failure fixtures + one positive control."""
import hashlib
import json
from pathlib import Path

import pytest

from src.release.acceptance import (
    TIME_GATE_SECONDS,
    not_run_receipt,
    verify_acceptance,
)

PIN_A = "a" * 40
PIN_B = "b" * 40
SHA40 = "c" * 40


def _five_fold_splits(nfolds=5, per_fold=2):
    splits = []
    q = 0
    for _ in range(nfolds):
        val = [f"q{q + i}" for i in range(per_fold)]
        q += per_fold
        splits.append({"train_query_ids": [], "val_query_ids": val})
    return splits


def _corpus(n=30):
    return [f"d{i}" for i in range(n)] + [f"q{i}-gold" for i in range(30)] + [f"q{i}-neg" for i in range(30)]


def _perfect_predictions(splits, corpus):
    preds = {}
    for fold in splits:
        for q in fold["val_query_ids"]:
            preds[q] = {"answer": [f"{q}-gold", f"{q}-neg", "d1", "d2", "d3"]}
    return preds


def _qrels_for(splits):
    return {q: [f"{q}-gold"] for fold in splits for q in fold["val_query_ids"]}


def _positive_receipt(tmp_path: Path, splits, preds, elapsed=17999.9, score=1.0):
    art = tmp_path / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    weights = art / "adapter_model.safetensors"
    weights.write_bytes(b"weights")
    manifest = art / "training_manifest.json"
    manifest.write_text(json.dumps({"ok": True}), encoding="utf-8")
    per_fold = []
    for i, fold in enumerate(splits):
        per_fold.append({"fold": i, "count": len(fold["val_query_ids"]), "recall@5": score})
    return {
        "schema_version": 1,
        "attempt_id": "att-1",
        "runtime_sha": SHA40,
        "release_sha": SHA40,
        "dataset_digest": "d" * 16,
        "split_random_5fold_sha256": "e" * 16,
        "split_doc_disjoint_sha256": "f" * 16,
        "config_sha256": "g" * 16,
        "model_revisions": {"reranker": PIN_A, "dense": PIN_B},
        "backend": "modal",
        "protocol": "confirmation",
        "predeclared_policy_hash": "h" * 16,
        "selection_protocol": "predeclared_rrf",
        "split_exposure_disclosure": "frozen splits; no tuning on held-out labels",
        "supervisor_start_utc": "2026-09-18T00:00:00Z",
        "supervisor_end_utc": "2026-09-18T04:59:59Z",
        "elapsed_seconds": elapsed,
        "expected_oof_queries": sum(len(f["val_query_ids"]) for f in splits),
        "per_fold": per_fold,
        "pooled_oof_recall@5": score,
        "pooled_oof_precision@5": 0.2,
        "doc_disjoint": {"recall@5": 0.8, "count": 2, "complete": True},
        "training_jobs": [{"job": "fold_0", "updates": 700}],
        "reload_ok": True,
        "submission_path": "submission.json",
        "artifacts": [
            {"path": "adapter_model.safetensors", "sha256": hashlib.sha256(b"weights").hexdigest()},
            {"path": "training_manifest.json", "sha256": hashlib.sha256(b'{"ok": true}').hexdigest()},
        ],
        "delivery_confirmed": True,
        "shutdown_confirmed": True,
        "verdict": "PASS",
        "reasons": [],
    }


def _disjoint_report():
    return {"recall@5": 0.8, "trained_reranker_system": {"recall@5": 0.8}}


def _submission_for(splits, corpus):
    return {q: {"answer": [f"{q}-gold", f"{q}-neg", "d1", "d2", "d3"]}
            for fold in splits for q in fold["val_query_ids"]}


def test_not_run_receipt_has_nulls_and_no_scores():
    r = not_run_receipt()
    assert r["verdict"] == "NOT_RUN"
    assert r["elapsed_seconds"] is None
    assert r["pooled_oof_recall@5"] is None
    assert r["reload_ok"] is False


def test_positive_control_passes(tmp_path):
    splits = _five_fold_splits()
    corpus = _corpus()
    preds = _perfect_predictions(splits, corpus)
    receipt = _positive_receipt(tmp_path, splits, preds)
    out = verify_acceptance(
        receipt, oof_predictions=preds, qrels=_qrels_for(splits), splits=splits,
        corpus_doc_ids=set(corpus), disjoint_report=_disjoint_report(),
        submission=_submission_for(splits, corpus), artifacts_dir=tmp_path / "artifacts",
    )
    assert out["verdict"] == "PASS", out["reasons"]
    assert out["recomputed_recall@5"] == pytest.approx(1.0)


def test_elapsed_boundary(tmp_path):
    splits = _five_fold_splits()
    corpus = _corpus()
    preds = _perfect_predictions(splits, corpus)
    ok = _positive_receipt(tmp_path, splits, preds, elapsed=float(TIME_GATE_SECONDS) - 0.1)
    out = verify_acceptance(ok, oof_predictions=preds, qrels=_qrels_for(splits), splits=splits,
                            corpus_doc_ids=set(corpus), disjoint_report=_disjoint_report(),
                            submission=_submission_for(splits, corpus), artifacts_dir=tmp_path / "artifacts")
    assert out["verdict"] == "PASS"
    bad = _positive_receipt(tmp_path, splits, preds, elapsed=float(TIME_GATE_SECONDS))
    out2 = verify_acceptance(bad, oof_predictions=preds, qrels=_qrels_for(splits), splits=splits,
                             corpus_doc_ids=set(corpus), disjoint_report=_disjoint_report(),
                             submission=_submission_for(splits, corpus), artifacts_dir=tmp_path / "artifacts")
    assert out2["verdict"] != "PASS" and any(str(TIME_GATE_SECONDS) in r for r in out2["reasons"])


def test_exact_threshold_fails(tmp_path):
    # 25 single-gold queries, 24 perfect + 1 miss => exactly 0.96 => FAIL.
    splits = _five_fold_splits(nfolds=5, per_fold=5)
    corpus = _corpus()
    preds = {}
    for fold in splits:
        for q in fold["val_query_ids"]:
            preds[q] = {"answer": [f"{q}-gold", f"{q}-neg", "d1", "d2", "d3"]}
    preds["q24"] = {"answer": ["d1", "d2", "d3", "d4", "d5"]}  # miss
    receipt = _positive_receipt(tmp_path, splits, preds, score=0.96)
    out = verify_acceptance(receipt, oof_predictions=preds, qrels=_qrels_for(splits), splits=splits,
                            corpus_doc_ids=set(corpus), disjoint_report=_disjoint_report(),
                            submission=_submission_for(splits, corpus), artifacts_dir=tmp_path / "artifacts")
    assert out["recomputed_recall@5"] == pytest.approx(0.96)
    assert out["verdict"] != "PASS" and any("strictly above" in r for r in out["reasons"])


def test_empty_training_jobs_and_placeholder_folds_fail(tmp_path):
    # fix.md F8: training_jobs=[] with per_fold=[{}]*5 must never PASS.
    splits = _five_fold_splits()
    corpus = _corpus()
    preds = _perfect_predictions(splits, corpus)
    receipt = _positive_receipt(tmp_path, splits, preds)
    receipt["training_jobs"] = []
    receipt["per_fold"] = [{}] * 5
    out = verify_acceptance(receipt, oof_predictions=preds, qrels=_qrels_for(splits), splits=splits,
                            corpus_doc_ids=set(corpus), disjoint_report=_disjoint_report(),
                            submission=_submission_for(splits, corpus), artifacts_dir=tmp_path / "artifacts")
    assert out["verdict"] != "PASS"
    assert any("training_jobs" in r for r in out["reasons"])
    assert any("per_fold" in r for r in out["reasons"])


def test_duplicate_fold_copies_fail(tmp_path):
    # fix.md F8: five copies of one fold must not PASS as five-fold evidence.
    fold = {"train_query_ids": [], "val_query_ids": ["q0", "q1"]}
    splits = [dict(fold) for _ in range(5)]
    corpus = _corpus()
    preds = {q: {"answer": [f"{q}-gold", f"{q}-neg", "d1", "d2", "d3"]} for q in ("q0", "q1")}
    receipt = _positive_receipt(tmp_path, splits, preds)
    out = verify_acceptance(receipt, oof_predictions=preds, qrels=_qrels_for(splits), splits=splits,
                            corpus_doc_ids=set(corpus), disjoint_report=_disjoint_report(),
                            submission=_submission_for(splits, corpus), artifacts_dir=tmp_path / "artifacts")
    assert out["verdict"] != "PASS"
    assert any("distinct" in r or "population" in r for r in out["reasons"])


def test_missing_query_fails(tmp_path):
    splits = _five_fold_splits()
    corpus = _corpus()
    preds = _perfect_predictions(splits, corpus)
    preds.pop("q0")
    receipt = _positive_receipt(tmp_path, splits, preds)
    out = verify_acceptance(receipt, oof_predictions=preds, qrels=_qrels_for(splits), splits=splits,
                            corpus_doc_ids=set(corpus), disjoint_report=_disjoint_report(),
                            submission=_submission_for(splits, corpus), artifacts_dir=tmp_path / "artifacts")
    assert out["verdict"] != "PASS" and any("coverage" in r for r in out["reasons"])


def test_four_folds_fail(tmp_path):
    splits = _five_fold_splits(nfolds=4)
    corpus = _corpus()
    preds = _perfect_predictions(splits, corpus)
    receipt = _positive_receipt(tmp_path, splits, preds)
    out = verify_acceptance(receipt, oof_predictions=preds, qrels=_qrels_for(splits), splits=splits,
                            corpus_doc_ids=set(corpus), disjoint_report=_disjoint_report(),
                            submission=_submission_for(splits, corpus), artifacts_dir=tmp_path / "artifacts")
    assert out["verdict"] != "PASS" and any("five folds" in r for r in out["reasons"])


def test_duplicate_ids_fail(tmp_path):
    splits = _five_fold_splits()
    corpus = _corpus()
    preds = _perfect_predictions(splits, corpus)
    preds["q0"] = {"answer": ["d1", "d1", "d2", "d3", "d4"]}
    receipt = _positive_receipt(tmp_path, splits, preds)
    out = verify_acceptance(receipt, oof_predictions=preds, qrels=_qrels_for(splits), splits=splits,
                            corpus_doc_ids=set(corpus), disjoint_report=_disjoint_report(),
                            submission=_submission_for(splits, corpus), artifacts_dir=tmp_path / "artifacts")
    assert out["verdict"] != "PASS" and any("duplicate" in r for r in out["reasons"])


def test_absent_disjoint_fails(tmp_path):
    splits = _five_fold_splits()
    corpus = _corpus()
    preds = _perfect_predictions(splits, corpus)
    receipt = _positive_receipt(tmp_path, splits, preds)
    out = verify_acceptance(receipt, oof_predictions=preds, qrels=_qrels_for(splits), splits=splits,
                            corpus_doc_ids=set(corpus), disjoint_report=None,
                            submission=_submission_for(splits, corpus), artifacts_dir=tmp_path / "artifacts")
    assert out["verdict"] != "PASS" and any("disjoint" in r for r in out["reasons"])


def test_mixed_runs_fail(tmp_path):
    splits = _five_fold_splits()
    corpus = _corpus()
    preds_b = {f"x{i}": {"answer": ["d1", "d2", "d3", "d4", "d5"]} for i in range(10)}
    receipt = _positive_receipt(tmp_path, splits, preds_b)  # timing from A, preds from B
    out = verify_acceptance(receipt, oof_predictions=preds_b, qrels=_qrels_for(splits), splits=splits,
                            corpus_doc_ids=set(corpus), disjoint_report=_disjoint_report(),
                            submission=_submission_for(splits, corpus), artifacts_dir=tmp_path / "artifacts")
    assert out["verdict"] != "PASS"


def test_stale_adapter_fails(tmp_path):
    splits = _five_fold_splits()
    corpus = _corpus()
    preds = _perfect_predictions(splits, corpus)
    receipt = _positive_receipt(tmp_path, splits, preds)
    receipt["model_revisions"] = {"reranker": "main", "dense": PIN_B}
    out = verify_acceptance(receipt, oof_predictions=preds, qrels=_qrels_for(splits), splits=splits,
                            corpus_doc_ids=set(corpus), disjoint_report=_disjoint_report(),
                            submission=_submission_for(splits, corpus), artifacts_dir=tmp_path / "artifacts")
    assert out["verdict"] != "PASS" and any("immutable" in r for r in out["reasons"])


def test_fake_summary_fails(tmp_path):
    splits = _five_fold_splits()
    corpus = _corpus()
    preds = _perfect_predictions(splits, corpus)
    receipt = _positive_receipt(tmp_path, splits, preds, score=0.5)  # recomputed is 1.0
    out = verify_acceptance(receipt, oof_predictions=preds, qrels=_qrels_for(splits), splits=splits,
                            corpus_doc_ids=set(corpus), disjoint_report=_disjoint_report(),
                            submission=_submission_for(splits, corpus), artifacts_dir=tmp_path / "artifacts")
    assert out["verdict"] != "PASS" and any("inconsistent" in r for r in out["reasons"])


def test_missing_weights_receipt_fails(tmp_path):
    splits = _five_fold_splits()
    corpus = _corpus()
    preds = _perfect_predictions(splits, corpus)
    receipt = _positive_receipt(tmp_path, splits, preds)
    receipt["artifacts"] = [{"path": "training_manifest.json",
                             "sha256": hashlib.sha256(b'{"ok": true}').hexdigest()}]
    out = verify_acceptance(receipt, oof_predictions=preds, qrels=_qrels_for(splits), splits=splits,
                            corpus_doc_ids=set(corpus), disjoint_report=_disjoint_report(),
                            submission=_submission_for(splits, corpus), artifacts_dir=tmp_path / "artifacts")
    assert out["verdict"] != "PASS" and any("weights" in r for r in out["reasons"])


def test_missing_submission_fails(tmp_path):
    splits = _five_fold_splits()
    corpus = _corpus()
    preds = _perfect_predictions(splits, corpus)
    receipt = _positive_receipt(tmp_path, splits, preds)
    out = verify_acceptance(receipt, oof_predictions=preds, qrels=_qrels_for(splits), splits=splits,
                            corpus_doc_ids=set(corpus), disjoint_report=_disjoint_report(),
                            submission=None, artifacts_dir=tmp_path / "artifacts")
    assert out["verdict"] != "PASS" and any("submission is absent" in r for r in out["reasons"])


def test_negative_elapsed_and_absent_endpoints_fail(tmp_path):
    splits = _five_fold_splits()
    corpus = _corpus()
    preds = _perfect_predictions(splits, corpus)
    receipt = _positive_receipt(tmp_path, splits, preds, elapsed=-5.0)
    receipt["supervisor_end_utc"] = ""
    out = verify_acceptance(receipt, oof_predictions=preds, qrels=_qrels_for(splits), splits=splits,
                            corpus_doc_ids=set(corpus), disjoint_report=_disjoint_report(),
                            submission=_submission_for(splits, corpus), artifacts_dir=tmp_path / "artifacts")
    assert out["verdict"] != "PASS"
    assert any("timing endpoint" in r for r in out["reasons"])
    assert any(f"[0, {TIME_GATE_SECONDS})" in r for r in out["reasons"])


def test_empty_disjoint_report_fails(tmp_path):
    splits = _five_fold_splits()
    corpus = _corpus()
    preds = _perfect_predictions(splits, corpus)
    receipt = _positive_receipt(tmp_path, splits, preds)
    out = verify_acceptance(receipt, oof_predictions=preds, qrels=_qrels_for(splits), splits=splits,
                            corpus_doc_ids=set(corpus), disjoint_report={},
                            submission=_submission_for(splits, corpus), artifacts_dir=tmp_path / "artifacts")
    assert out["verdict"] != "PASS" and any("disjoint" in r for r in out["reasons"])


def test_missing_weights_checksum_fails(tmp_path):
    import hashlib as _hl
    splits = _five_fold_splits()
    corpus = _corpus()
    preds = _perfect_predictions(splits, corpus)
    receipt = _positive_receipt(tmp_path, splits, preds)
    receipt["artifacts"] = [
        {"path": "adapter_model.safetensors"},  # checksum absent
        {"path": "training_manifest.json", "sha256": _hl.sha256(b'{"ok": true}').hexdigest()},
    ]
    out = verify_acceptance(receipt, oof_predictions=preds, qrels=_qrels_for(splits), splits=splits,
                            corpus_doc_ids=set(corpus), disjoint_report=_disjoint_report(),
                            submission=_submission_for(splits, corpus), artifacts_dir=tmp_path / "artifacts")
    assert out["verdict"] != "PASS" and any("checksum absent" in r for r in out["reasons"])


def test_failed_shutdown_fails(tmp_path):
    splits = _five_fold_splits()
    corpus = _corpus()
    preds = _perfect_predictions(splits, corpus)
    receipt = _positive_receipt(tmp_path, splits, preds)
    receipt["shutdown_confirmed"] = False
    out = verify_acceptance(receipt, oof_predictions=preds, qrels=_qrels_for(splits), splits=splits,
                            corpus_doc_ids=set(corpus), disjoint_report=_disjoint_report(),
                            submission=_submission_for(splits, corpus), artifacts_dir=tmp_path / "artifacts")
    assert out["verdict"] != "PASS" and any("shutdown" in r for r in out["reasons"])


def test_acceptance_verifier_2080_private_queries_submission(tmp_path):
    """Verify acceptance verifier works with 2,080 private round queries."""
    splits = _five_fold_splits()
    corpus = _corpus(n=50)
    preds = _perfect_predictions(splits, corpus)
    receipt = _positive_receipt(tmp_path, splits, preds)

    # 1. Valid 2080-query submission: every answer exactly 5 unique valid corpus documents
    sub_2080 = {
        f"priv_{i}": {"answer": [f"d{j % 50}" for j in range(i, i + 5)]}
        for i in range(2080)
    }
    expected_qids = set(sub_2080.keys())

    out_pass = verify_acceptance(
        receipt,
        oof_predictions=preds,
        qrels=_qrels_for(splits),
        splits=splits,
        corpus_doc_ids=set(corpus),
        disjoint_report=_disjoint_report(),
        submission=sub_2080,
        artifacts_dir=tmp_path / "artifacts",
        expected_submission_qids=expected_qids,
    )
    assert out_pass["verdict"] == "PASS", f"Valid 2080 submission failed: {out_pass['reasons']}"

    # 2. Incomplete 2080-query submission (missing query) fails
    sub_incomplete = dict(sub_2080)
    sub_incomplete.pop("priv_0")
    out_incomp = verify_acceptance(
        receipt,
        oof_predictions=preds,
        qrels=_qrels_for(splits),
        splits=splits,
        corpus_doc_ids=set(corpus),
        disjoint_report=_disjoint_report(),
        submission=sub_incomplete,
        artifacts_dir=tmp_path / "artifacts",
        expected_submission_qids=expected_qids,
    )
    assert out_incomp["verdict"] != "PASS"
    assert any("mismatch" in r for r in out_incomp["reasons"])

    # 3. Submission with non-5 length answer fails
    sub_len = dict(sub_2080)
    sub_len["priv_1"] = {"answer": ["d1", "d2", "d3"]}
    out_len = verify_acceptance(
        receipt,
        oof_predictions=preds,
        qrels=_qrels_for(splits),
        splits=splits,
        corpus_doc_ids=set(corpus),
        disjoint_report=_disjoint_report(),
        submission=sub_len,
        artifacts_dir=tmp_path / "artifacts",
        expected_submission_qids=expected_qids,
    )
    assert out_len["verdict"] != "PASS"
    assert any("exactly 5 unique documents" in r for r in out_len["reasons"])

    # 4. Submission with duplicate doc IDs fails
    sub_dup = dict(sub_2080)
    sub_dup["priv_2"] = {"answer": ["d1", "d1", "d2", "d3", "d4"]}
    out_dup = verify_acceptance(
        receipt,
        oof_predictions=preds,
        qrels=_qrels_for(splits),
        splits=splits,
        corpus_doc_ids=set(corpus),
        disjoint_report=_disjoint_report(),
        submission=sub_dup,
        artifacts_dir=tmp_path / "artifacts",
        expected_submission_qids=expected_qids,
    )
    assert out_dup["verdict"] != "PASS"
    assert any("exactly 5 unique documents" in r for r in out_dup["reasons"])

    # 5. Submission with invalid (unknown) doc ID fails
    sub_inv = dict(sub_2080)
    sub_inv["priv_3"] = {"answer": ["d1", "d2", "d3", "d4", "unknown_doc_999"]}
    out_inv = verify_acceptance(
        receipt,
        oof_predictions=preds,
        qrels=_qrels_for(splits),
        splits=splits,
        corpus_doc_ids=set(corpus),
        disjoint_report=_disjoint_report(),
        submission=sub_inv,
        artifacts_dir=tmp_path / "artifacts",
        expected_submission_qids=expected_qids,
    )
    assert out_inv["verdict"] != "PASS"
    assert any("invalid document ID" in r for r in out_inv["reasons"])
