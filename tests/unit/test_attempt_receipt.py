"""Attempt-bound acceptance: fresh reload, receipt builder, shutdown confirm."""
import json
from pathlib import Path

import pytest

from src.release.acceptance import verify_adapter_reload_fresh

PIN = "a" * 40


def test_fresh_reload_missing_config_no_subprocess(tmp_path, monkeypatch):
    d = tmp_path / "adapter"
    d.mkdir()
    called = []
    monkeypatch.setattr("subprocess.run", lambda *a, **k: called.append(1))
    ok, detail = verify_adapter_reload_fresh(d, "BAAI/bge-reranker-v2-m3", PIN)
    assert ok is False and "adapter_config.json missing" in detail
    assert called == []


def test_fresh_reload_missing_weights(tmp_path):
    d = tmp_path / "adapter"
    d.mkdir()
    (d / "adapter_config.json").write_text("{}", encoding="utf-8")
    ok, detail = verify_adapter_reload_fresh(d, "BAAI/bge-reranker-v2-m3", PIN)
    assert ok is False and "weights missing" in detail


def test_fresh_reload_success_and_failure_via_stub(tmp_path, monkeypatch):
    import subprocess as _sp

    d = tmp_path / "adapter"
    d.mkdir()
    (d / "adapter_config.json").write_text("{}", encoding="utf-8")
    (d / "adapter_model.safetensors").write_bytes(b"w")

    class _Proc:
        def __init__(self, rc, out, err=""):
            self.returncode = rc
            self.stdout = out
            self.stderr = err

    monkeypatch.setattr(_sp, "run", lambda *a, **k: _Proc(0, '{"ok": true, "score": 1.5}\n'))
    ok, detail = verify_adapter_reload_fresh(d, "m", None)
    assert ok is True and "1.5" in detail

    monkeypatch.setattr(_sp, "run", lambda *a, **k: _Proc(1, "", "boom"))
    ok, _ = verify_adapter_reload_fresh(d, "m", None)
    assert ok is False

    def _timeout(*a, **k):
        raise _sp.TimeoutExpired(cmd="x", timeout=1)

    monkeypatch.setattr(_sp, "run", _timeout)
    ok, detail = verify_adapter_reload_fresh(d, "m", None, timeout_s=1)
    assert ok is False and "timed out" in detail


def _five_fold_fixture():
    splits = [{"train_query_ids": [], "val_query_ids": [f"q{2 * i}", f"q{2 * i + 1}"]} for i in range(5)]
    corpus = [f"d{i}" for i in range(6)]
    qrels = {}
    oof = {}
    for fold in splits:
        for q in fold["val_query_ids"]:
            qrels[q] = [f"{q}-gold"]
            corpus.append(f"{q}-gold")
            oof[q] = {"answer": [f"{q}-gold", "d0", "d1", "d2", "d3"]}
    return splits, corpus, qrels, oof


def _working_dir(tmp_path: Path, splits, oof):
    work = tmp_path / "attempt-1"
    (work / "cv").mkdir(parents=True)
    (work / "cv" / "oof_predictions.json").write_text(json.dumps(oof), encoding="utf-8")
    ad = work / "checkpoints" / "reranker_final"
    ad.mkdir(parents=True)
    (ad / "adapter_model.safetensors").write_bytes(b"weights")
    (ad / "adapter_config.json").write_text("{}", encoding="utf-8")
    (ad / "training_manifest.json").write_text(json.dumps({"base_model_revision": PIN}), encoding="utf-8")
    sub = work / "submissions"
    sub.mkdir(parents=True)
    (sub / "submission.json").write_text(json.dumps({"pq": {"answer": ["d0", "d1", "d2", "d3", "d4"]}}), encoding="utf-8")
    (sub / "submission.zip").write_bytes(b"zip")
    (sub / "submission_manifest.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
    return work


def _stub_reload(monkeypatch):
    import src.release.acceptance as acc

    monkeypatch.setattr(acc, "verify_adapter_reload_fresh", lambda *a, **k: (True, "test reload"))


def test_builder_end_to_end_fails_closed_on_shutdown(tmp_path, monkeypatch):
    from src.pipeline.kaggle_train import build_attempt_acceptance_receipt

    splits, corpus, qrels, oof = _five_fold_fixture()
    work = _working_dir(tmp_path, splits, oof)
    _stub_reload(monkeypatch)
    cv_report = {
        "total_evaluated_queries": 10,
        "overall_aggregate_metrics": {"recall@5": 1.0, "precision@5": 0.2},
        "folds": [{"fold": i, "val_queries": 2, "recall@5": 1.0, "reranker_optimizer_steps": 7} for i in range(5)],
        "split_provenance": {"random_5fold": {"sha256": "s" * 16}},
        "resolved_config_sha256": "c" * 64,
    }
    fusion_report = {
        "winning_method": "reciprocal_rank_fusion",
        "manifest": {"feature_columns": ["rrf_score"]},
        "comparison": {"selection_protocol": "predeclared_rrf"},
    }
    disjoint = {"recall@5": 0.9, "trained_reranker_system": {"recall@5": 0.9, "val_queries": 2}}
    submission = {"pq": {"answer": ["d0", "d1", "d2", "d3", "d4"]}}
    receipt, verification = build_attempt_acceptance_receipt(
        working_path=work, cv_report=cv_report, fusion_report=fusion_report,
        doc_disjoint_report=disjoint, predictions=submission, qrels_dict=qrels,
        splits=splits, corpus_doc_ids=set(corpus),
        final_reranker_dir=work / "checkpoints" / "reranker_final",
        final_reranker_report={"optimizer_steps": 9, "base_model_revision": PIN},
        reranker_model="BAAI/bge-reranker-v2-m3", reranker_revision=PIN,
        git_sha="f" * 40, backend="modal",
        supervisor_start_utc="2026-09-18T00:00:00Z", supervisor_end_utc="2026-09-18T00:01:40Z",
        elapsed_seconds=100.0, submission_valid=True,
    )
    # Everything genuine except shutdown: exactly one honest failure reason.
    assert verification["verdict"] == "FAIL", verification["reasons"]
    assert verification["reasons"] == ["provider shutdown is not confirmed"]
    assert verification["recomputed_recall@5"] == pytest.approx(1.0)
    assert (work / "acceptance_inputs" / "splits.json").is_file()
    assert receipt["reload_ok"] is True
    assert receipt["verify_inputs"]["splits.json"] == "acceptance_inputs/splits.json"


def _build_test_receipt(work, splits, corpus, qrels, oof, monkeypatch):
    from src.pipeline.kaggle_train import build_attempt_acceptance_receipt

    _stub_reload(monkeypatch)
    cv_report = {
        "total_evaluated_queries": 10,
        "overall_aggregate_metrics": {"recall@5": 1.0, "precision@5": 0.2},
        "folds": [{"fold": i, "val_queries": 2, "recall@5": 1.0, "reranker_optimizer_steps": 7} for i in range(5)],
        "split_provenance": {"random_5fold": {"sha256": "s" * 16}},
        "resolved_config_sha256": "c" * 64,
    }
    fusion_report = {
        "winning_method": "reciprocal_rank_fusion",
        "manifest": {"feature_columns": ["rrf_score"]},
        "comparison": {"selection_protocol": "predeclared_rrf"},
    }
    disjoint = {"recall@5": 0.9, "trained_reranker_system": {"recall@5": 0.9, "val_queries": 2}}
    submission = {"pq": {"answer": ["d0", "d1", "d2", "d3", "d4"]}}
    return build_attempt_acceptance_receipt(
        working_path=work, cv_report=cv_report, fusion_report=fusion_report,
        doc_disjoint_report=disjoint, predictions=submission, qrels_dict=qrels,
        splits=splits, corpus_doc_ids=set(corpus),
        final_reranker_dir=work / "checkpoints" / "reranker_final",
        final_reranker_report={"optimizer_steps": 9, "base_model_revision": PIN},
        reranker_model="BAAI/bge-reranker-v2-m3", reranker_revision=PIN,
        git_sha="f" * 40, backend="modal",
        supervisor_start_utc="2026-09-18T00:00:00Z", supervisor_end_utc="2026-09-18T00:01:40Z",
        elapsed_seconds=100.0, submission_valid=True,
    )


def test_confirm_shutdown_flips_to_pass(tmp_path, monkeypatch):
    splits, corpus, qrels, oof = _five_fold_fixture()
    work = _working_dir(tmp_path, splits, oof)
    receipt, _ = _build_test_receipt(work, splits, corpus, qrels, oof, monkeypatch)
    receipt_path = work / "acceptance_receipt.json"
    receipt_path.write_text(json.dumps({"receipt": receipt, "verification": {}}), encoding="utf-8")

    import scripts.confirm_attempt_shutdown as confirm

    rc = confirm.main(["--receipt", str(receipt_path), "--confirmed-by", "op",
                       "--evidence", "modal app stop att-1 confirmed stopped"])
    assert rc == 0
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert payload["receipt"]["shutdown_confirmed"] is True
    assert payload["verification"]["verdict"] == "PASS"


def test_confirm_shutdown_rejects_empty_evidence(tmp_path):
    p = tmp_path / "acceptance_receipt.json"
    p.write_text(json.dumps({"receipt": {}, "verification": {}}), encoding="utf-8")
    import scripts.confirm_attempt_shutdown as confirm

    assert confirm.main(["--receipt", str(p), "--confirmed-by", "op", "--evidence", "  "]) == 2
