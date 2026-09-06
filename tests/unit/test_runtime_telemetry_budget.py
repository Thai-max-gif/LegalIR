import json
import pytest
from pathlib import Path

from src.core.runtime_estimator import (
    estimate_kaggle_final_runtime,
    estimate_factory_runtime,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
COLAB_REPORT_PATH = REPO_ROOT / "artifacts" / "task1" / "colab_smoke_report.json"


def test_runtime_projection_uses_measured_telemetry():
    assert COLAB_REPORT_PATH.is_file(), f"Missing {COLAB_REPORT_PATH}"
    report = json.loads(COLAB_REPORT_PATH.read_text(encoding="utf-8"))

    projection = estimate_kaggle_final_runtime(report, safety_factor=1.2)
    assert "sec_per_step" in projection
    assert projection["sec_per_step"] > 0
    # ~2.9 - 20.2 sec/step on T4 depending on FP16 optimizations
    assert 2.0 <= projection["sec_per_step"] <= 35.0
    assert projection["total_projected_hours"] < 9.0
    assert projection["is_feasible_on_kaggle"] is True


def test_monolithic_full_is_not_final_kaggle_entrypoint():
    assert COLAB_REPORT_PATH.is_file()
    report = json.loads(COLAB_REPORT_PATH.read_text(encoding="utf-8"))

    factory_proj = estimate_factory_runtime(report)
    assert factory_proj["total_projected_hours"] > 9.0
    assert factory_proj["is_single_kaggle_session_feasible"] is False
    assert factory_proj["requires_resumable_sessions"] is True


def test_kaggle_final_projection_excludes_oof_and_doc_disjoint():
    report = json.loads(COLAB_REPORT_PATH.read_text(encoding="utf-8"))
    final_proj = estimate_kaggle_final_runtime(report)

    # Kaggle final must only include 1 final adapter (875 steps), not 5 folds (3500) + doc disjoint (700)
    assert final_proj["optimizer_steps"] == 875
    assert "oof_folds_sec" not in final_proj["stages"]
    assert "doc_disjoint_sec" not in final_proj["stages"]


def test_mock_smoke_cannot_authorize_final():
    mock_report = {
        "gpu_name": "CPU Mock",
        "result": "PASS",
        "optimizer_steps": 10,
        "stage_timings": {"reranker_training_sec": 0.5},
        "is_mock": True,
    }
    with pytest.raises(ValueError, match="Mock smoke|cannot authorize"):
        estimate_kaggle_final_runtime(mock_report)
