import json
from pathlib import Path
import subprocess
import sys
import pytest
from src.data.canonical import discover_canonical_dataset_dir

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_kaggle_smoke_runner_mock(tmp_path: Path):
    ds_dir = discover_canonical_dataset_dir()
    if not (ds_dir / "queries_train.parquet").is_file():
        from scripts.smoke_kaggle_pipeline import create_toy_canonical_dataset
        ds_dir = create_toy_canonical_dataset(tmp_path / "toy_data")

    out_dir = tmp_path / "smoke_out"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/run_kaggle_smoke.py"),
        "--dataset-dir", str(ds_dir),
        "--output-dir", str(out_dir),
        "--target-sha", "0000000000000000000000000000000000000000",
        "--mock",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    assert res.returncode == 0, f"Smoke runner failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"

    report_p = out_dir / "kaggle_smoke_report.json"
    assert report_p.is_file(), f"Missing report at {report_p}"

    data = json.loads(report_p.read_text(encoding="utf-8"))
    assert data.get("verdict") in ("PASS", "DEBUG_ONLY")
    assert data.get("stage") == "B1.1_KAGGLE_T4_SMOKE"
    metrics = data.get("smoke_metrics", {})
    assert metrics.get("weight_delta", 0) > 0
    assert metrics.get("sample_queries") > 0
