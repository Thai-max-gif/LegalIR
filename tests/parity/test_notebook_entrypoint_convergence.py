import json
import pytest
from pathlib import Path

from scripts.check_notebook_parity import compute_sha256

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

DISTRIBUTED_FINAL_NOTEBOOKS = [
    REPO_ROOT / "notebooks" / "kaggle_t4x2_smoke.ipynb",
    REPO_ROOT / "notebooks" / "kaggle_smoke.ipynb",
    REPO_ROOT / "notebooks" / "colab_t4_smoke.ipynb",
    REPO_ROOT / "notebooks" / "colab_a100_train.ipynb",
]


def test_all_canonical_notebooks_use_dedicated_runners():
    assert (REPO_ROOT / "notebooks" / "kaggle_smoke.ipynb").is_file()
    assert (REPO_ROOT / "notebooks" / "colab_a100_train.ipynb").is_file()
    kaggle_txt = (REPO_ROOT / "notebooks" / "kaggle_smoke.ipynb").read_text(encoding="utf-8")
    colab_txt = (REPO_ROOT / "notebooks" / "colab_a100_train.ipynb").read_text(encoding="utf-8")
    assert "run_kaggle_t4x2" in kaggle_txt or "run_kaggle_smoke" in kaggle_txt
    assert "run_a100" in colab_txt or "run_colab_train" in colab_txt


def test_no_distributed_final_notebook_calls_monolithic_inline():
    for nb_path in DISTRIBUTED_FINAL_NOTEBOOKS:
        assert nb_path.is_file(), f"Notebook {nb_path} missing"
        content = nb_path.read_text(encoding="utf-8")
        assert "run_kaggle_pipeline(" not in content, (
            f"Competition notebook {nb_path} must not call monolithic run_kaggle_pipeline() directly in cell"
        )


def test_all_distributed_notebooks_pin_same_full_runtime_sha():
    pins = set()
    for nb_path in DISTRIBUTED_FINAL_NOTEBOOKS:
        content = nb_path.read_text(encoding="utf-8")
        # Find 40-char SHA in notebook text
        import re
        shas = re.findall(r"[0-9a-fA-F]{40}", content)
        assert len(shas) > 0, f"Notebook {nb_path} has no 40-char SHA pin"
        # The primary checkout or expected commit SHA
        pins.add(shas[0].lower())

    assert len(pins) == 1, f"Distributed notebooks pin different runtime SHAs: {pins}"


def test_all_distributed_notebooks_avoid_torch_reinstall():
    disallowed_patterns = [
        "pip install torch",
        "pip install --upgrade torch",
        "pip install torchvision",
        "conda install torch",
    ]
    for nb_path in DISTRIBUTED_FINAL_NOTEBOOKS:
        content = nb_path.read_text(encoding="utf-8")
        for pat in disallowed_patterns:
            assert pat not in content, f"Notebook {nb_path} contains forbidden torch reinstall pattern: {pat}"


def test_notebook_parity_covers_every_distributed_surface():
    from scripts.check_notebook_parity import check_all_notebook_parity
    is_valid, report = check_all_notebook_parity(REPO_ROOT)
    assert is_valid is True, f"Parity check failed: {report}"
