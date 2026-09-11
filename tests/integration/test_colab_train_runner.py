import json
from pathlib import Path
import subprocess
import sys
import pytest
from src.data.canonical import discover_canonical_dataset_dir

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_colab_train_runner_mock(tmp_path: Path):
    ds_dir = discover_canonical_dataset_dir()
    out_dir = tmp_path / "train_out"
    smoke_report = tmp_path / "kaggle_smoke_report.json"
    smoke_report.write_text(json.dumps({"verdict": "PASS", "stage": "B1.1_KAGGLE_T4_SMOKE"}), encoding="utf-8")

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/run_colab_train.py"),
        "--dataset-dir", str(ds_dir),
        "--output-dir", str(out_dir),
        "--smoke-report", str(smoke_report),
        "--mock",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    assert res.returncode == 0, f"Colab train failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"

    manifest_p = out_dir / "run_manifest.json"
    assert manifest_p.is_file(), f"Missing manifest at {manifest_p}"

    data = json.loads(manifest_p.read_text(encoding="utf-8"))
    assert data.get("stage") == "B1.2_COLAB_A100_PRODUCTION_RUN"
    assert data.get("status") == "COMPLETED"

    sub_zip = out_dir / "submission.zip"
    assert sub_zip.is_file(), f"Missing submission.zip at {sub_zip}"
