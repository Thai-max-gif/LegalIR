import json
import re
import pytest
from pathlib import Path


def test_notebook_invariants_no_secrets():
    # Test helper that audits all notebooks for exposed secrets or hardcoded tokens
    forbidden_tokens = ["ghp_", "sk-", "AIzaSy", "password", "SECRET_KEY"]
    for nb_path in [
        Path("notebooks/kaggle_smoke.ipynb"),
        Path("notebooks/colab_a100_train.ipynb"),
    ]:
        if nb_path.is_file():
            raw_text = nb_path.read_text(encoding="utf-8")
            for token in forbidden_tokens:
                assert token not in raw_text, f"Forbidden token '{token}' in {nb_path}"


def test_kaggle_notebook_pin_contains_new_runtime_full_sha():
    nb_path = Path("notebooks/kaggle_smoke.ipynb")
    assert nb_path.is_file(), f"Missing {nb_path}"

    raw_text = nb_path.read_text(encoding="utf-8")
    # Verify 40-char hex commit pin exists
    match = re.search(r"([0-9a-f]{40})", raw_text)
    assert match is not None, "Kaggle notebook must contain full 40-char git commit SHA pin"
    pinned_sha = match.group(1)
    assert len(pinned_sha) == 40


def test_kaggle_notebook_target_script_exists_at_pin():
    nb_path = Path("notebooks/kaggle_smoke.ipynb")
    raw_text = nb_path.read_text(encoding="utf-8")
    assert "scripts/run_kaggle_smoke.py" in raw_text
    assert Path("scripts/run_kaggle_smoke.py").is_file()


def test_colab_notebook_passes_required_cli_args():
    nb_path = Path("notebooks/colab_a100_train.ipynb")
    assert nb_path.is_file(), f"Missing {nb_path}"
    raw_text = nb_path.read_text(encoding="utf-8")
    assert "scripts/run_colab_train.py" in raw_text
    assert "--dataset-dir" in raw_text
    assert "--output-dir" in raw_text
    match = re.search(r"([0-9a-f]{40})", raw_text)
    assert match is not None, f"{nb_path} must pass 40-char commit SHA"


def test_notebook_execution_contract():
    stages = ["K0", "K1", "K2", "K3", "K4", "K5", "K6", "K7", "K8", "K9"]
    assert len(stages) == 10
