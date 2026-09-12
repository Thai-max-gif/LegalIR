import json
from pathlib import Path


STANDARD_CANDIDATE_CUTOFFS = (20, 50, 100)
FINAL_METRIC_KEYS = ("Recall@1", "Recall@3", "Recall@5", "Precision@5")


def _assert_final_metric_report(report_section):
    for metric_key in FINAL_METRIC_KEYS:
        machine_key = metric_key.lower().replace("@", "_at_")
        assert (
            metric_key in report_section
            or machine_key in report_section
            or f"mean_{machine_key}" in report_section
        )


def _resolve_artifact(paths: list[str]) -> Path:
    for p_str in paths:
        p = Path(p_str)
        if p.exists():
            return p
    return Path(paths[0])


def test_accepted_benchmark_reports_expose_dual_validation_metrics():
    model_path = _resolve_artifact([
        "artifacts/task1/benchmarks/accepted/final_model.json",
        "artifacts/task1/benchmarks/final_model.json",
        "artifacts/shared/benchmarks/accepted/final_model.json",
    ])
    assert model_path.exists(), f"final_model.json must exist, checked {model_path}"
    report = json.loads(model_path.read_text(encoding="utf-8"))

    assert set(STANDARD_CANDIDATE_CUTOFFS) <= set(report["candidate_cutoffs"])
    for split_name in ("random_5fold", "document_disjoint"):
        _assert_final_metric_report(report[split_name])

    random_candidates = report["random_5fold"]["candidate_recalls"]
    for cutoff in STANDARD_CANDIDATE_CUTOFFS:
        assert f"candidate_recall@{cutoff}" in random_candidates


def test_final_model_passes_all_acceptance_gates():
    model_path = _resolve_artifact([
        "artifacts/task1/benchmarks/accepted/final_model.json",
        "artifacts/task1/benchmarks/final_model.json",
        "artifacts/shared/benchmarks/accepted/final_model.json",
    ])
    assert model_path.exists(), "final_model.json must exist in benchmarks directory"
    report = json.loads(model_path.read_text(encoding="utf-8"))

    assert len(report["random_5fold"]["folds"]) == 5
    assert report["official_scorer_equivalent"] is True
    assert report["leakage_checks_passed"] is True

    # Baseline comparison
    baseline_path = _resolve_artifact([
        "artifacts/task1/benchmarks/accepted/strict_baseline.json",
        "artifacts/task1/benchmarks/strict_baseline.json",
        "artifacts/shared/benchmarks/accepted/strict_baseline.json",
    ])
    assert baseline_path.exists()
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

    assert report["random_5fold"]["mean_recall_at_5"] >= baseline["random_5fold"]["mean_recall_at_5"]
    assert report["document_disjoint"]["recall_at_5"] >= baseline["document_disjoint"]["recall_at_5"] - 0.01


def test_final_submission_manifest_integrity():
    sub_manifest_path = _resolve_artifact([
        "artifacts/task1/submissions/accepted/submission_manifest.json",
        "artifacts/task1/submissions/submission_manifest.json",
        "artifacts/shared/submissions/accepted/submission_manifest.json",
    ])
    assert sub_manifest_path.exists(), "submission_manifest.json must exist in submissions directory"
    manifest = json.loads(sub_manifest_path.read_text(encoding="utf-8"))

    assert manifest["total_queries"] == 1000
    assert "submission_json_sha256" in manifest
    assert "submission_zip_sha256" in manifest
    assert manifest["compliance_verified"] is True
