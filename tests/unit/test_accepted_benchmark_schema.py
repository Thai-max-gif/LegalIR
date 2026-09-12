import json
from pathlib import Path


def test_accepted_baseline_contains_all_five_folds():
    p = Path("artifacts/task1/benchmarks/accepted/strict_baseline.json")
    if not p.exists():
        p = Path("artifacts/task1/benchmarks/strict_baseline.json")
    if not p.exists():
        p = Path("artifacts/shared/benchmarks/accepted/strict_baseline.json")
    assert p.exists(), "strict_baseline.json must exist in artifacts/task1/benchmarks/ or artifacts/shared/benchmarks/accepted/"
    report = json.loads(p.read_text(encoding="utf-8"))
    assert len(report["random_5fold"]["folds"]) == 5
    assert report["leakage_checks_passed"] is True
    assert report["official_scorer_equivalent"] is True
    assert set(report["candidate_cutoffs"]) == {20, 50, 100, 150}
    assert report["label"] == "strict_baseline"
    assert "mean_recall_at_5" in report["random_5fold"]
    assert "recall_at_5" in report["document_disjoint"]
