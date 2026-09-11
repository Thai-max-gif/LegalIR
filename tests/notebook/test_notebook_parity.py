from pathlib import Path
import subprocess
import sys
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_notebook_generator_zero_drift():
    gen_script = REPO_ROOT / "scripts" / "generate_notebooks.py"
    res = subprocess.run(
        [sys.executable, str(gen_script), "--check-drift"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert res.returncode == 0, f"Notebook drift detected:\n{res.stdout}\n{res.stderr}"
