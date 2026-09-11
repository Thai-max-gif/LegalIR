from pathlib import Path

import pytest

from src.utils.source_discovery import inspect_source_dir, select_source


def make_source(root: Path) -> None:
    for relative in [
        "scripts/run_preflight.py",
        "scripts/run_smoke_test.py",
        "src/config.py",
        "configs/legalir_kaggle_t4_smoke.yaml",
    ]:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# fixture\n")


def test_explicit_source_must_have_signature(tmp_path: Path):
    invalid = tmp_path / "not_source"
    invalid.mkdir()
    with pytest.raises(ValueError):
        select_source(tmp_path, tmp_path / "working", source_root=invalid)


def test_multiple_sources_require_explicit_choice(tmp_path: Path):
    make_source(tmp_path / "one")
    make_source(tmp_path / "two")
    with pytest.raises(RuntimeError, match="Multiple verified"):
        select_source(tmp_path, tmp_path / "working")


def test_explicit_source_is_accepted(tmp_path: Path):
    root = tmp_path / "source"
    make_source(root)
    selected, diagnostic = select_source(tmp_path, tmp_path / "working", source_root=root)
    assert selected == root
    assert diagnostic["explicit_source_root"]["accepted"]
