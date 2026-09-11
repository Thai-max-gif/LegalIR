import json
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

NOTEBOOK_PATHS = [
    REPO_ROOT / "notebooks/kaggle_smoke.ipynb",
    REPO_ROOT / "notebooks/colab_a100_train.ipynb",
]


@pytest.mark.parametrize("nb_path", NOTEBOOK_PATHS)
def test_notebook_json_and_metadata(nb_path: Path):
    assert nb_path.is_file(), f"Notebook missing: {nb_path}"
    data = json.loads(nb_path.read_text(encoding="utf-8"))
    assert data.get("nbformat") == 4
    assert "cells" in data
    assert len(data["cells"]) >= 4

    metadata = data.get("metadata", {})
    kernelspec = metadata.get("kernelspec", {})
    assert kernelspec.get("name") == "python3"
    assert kernelspec.get("language") == "python"
