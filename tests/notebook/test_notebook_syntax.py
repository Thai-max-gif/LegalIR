import ast
import json
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

NOTEBOOK_PATHS = [
    REPO_ROOT / "notebooks/kaggle_smoke.ipynb",
    REPO_ROOT / "notebooks/colab_a100_train.ipynb",
]


@pytest.mark.parametrize("nb_path", NOTEBOOK_PATHS)
def test_code_cells_valid_python_syntax(nb_path: Path):
    assert nb_path.is_file(), f"Notebook missing: {nb_path}"
    data = json.loads(nb_path.read_text(encoding="utf-8"))
    for idx, cell in enumerate(data.get("cells", [])):
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))
            clean_lines = [
                line for line in source.splitlines()
                if not line.strip().startswith(("!", "%"))
            ]
            clean_source = "\n".join(clean_lines)
            try:
                ast.parse(clean_source)
            except SyntaxError as e:
                pytest.fail(f"Syntax error in {nb_path.name} cell {idx}: {e}")
