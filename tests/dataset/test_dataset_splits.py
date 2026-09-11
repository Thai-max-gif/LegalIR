from pathlib import Path
import pytest
from src.data.canonical import discover_canonical_dataset_dir
from src.data.splits import load_5fold_splits, load_doc_disjoint_split


@pytest.fixture
def dataset_dir() -> Path:
    return discover_canonical_dataset_dir()


def test_5fold_splits_coverage(dataset_dir: Path):
    splits = load_5fold_splits(dataset_dir)
    assert len(splits) == 5
    all_val_qids = set()
    for fold in splits:
        assert len(fold.train_qids) == 5600
        assert len(fold.val_qids) == 1400
        assert len(set(fold.train_qids) & set(fold.val_qids)) == 0
        all_val_qids.update(fold.val_qids)
    assert len(all_val_qids) == 7000


def test_doc_disjoint_split_isolation(dataset_dir: Path):
    split = load_doc_disjoint_split(dataset_dir)
    assert len(split.train_doc_ids & split.val_doc_ids) == 0
    assert len(split.train_qids) > 0
    assert len(split.val_qids) > 0
