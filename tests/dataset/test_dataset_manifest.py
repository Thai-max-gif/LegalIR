import hashlib
import json
from pathlib import Path
import pytest
from src.data.canonical import discover_canonical_dataset_dir


@pytest.fixture
def dataset_dir() -> Path:
    return discover_canonical_dataset_dir()


def test_manifest_metadata(dataset_dir: Path):
    manifest_p = dataset_dir / "manifest.json"
    if not manifest_p.is_file():
        manifest_p = Path("kaggle_dataset/manifest.json")
    assert manifest_p.is_file(), "Missing manifest.json"

    with open(manifest_p, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest.get("dataset") == "task1_canonical"
    assert manifest.get("version") == "v2"
    assert manifest.get("total_documents") == 8532
    assert manifest.get("total_chunks") == 1153876
    assert manifest.get("total_queries") == 7000
    assert manifest.get("total_qrels") == 7637
    assert manifest.get("total_duplicate_groups") == 4
    assert manifest.get("schema") == "hierarchical_micro_macro_v2"
