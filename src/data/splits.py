"""Dataset split loaders and split validation for 5-fold CV and document-disjoint."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from src.data.canonical import resolve_split_path


@dataclass
class FoldSplit:
    """A single cross-validation fold with train and validation query ID sets."""

    fold_id: int
    train_qids: Set[str]
    val_qids: Set[str]

    def validate(self) -> None:
        """Enforce disjointness."""
        overlap = self.train_qids.intersection(self.val_qids)
        if overlap:
            raise ValueError(f"Fold {self.fold_id} has {len(overlap)} overlapping query IDs between train and val")


@dataclass
class DocDisjointSplit:
    """Document-disjoint split definitions with query and document sets."""

    train_qids: Set[str]
    val_qids: Set[str]
    train_doc_ids: Set[str]
    val_doc_ids: Set[str]

    def validate(self) -> None:
        """Enforce strict disjointness on queries and documents."""
        q_overlap = self.train_qids.intersection(self.val_qids)
        if q_overlap:
            raise ValueError(f"Doc-disjoint split has {len(q_overlap)} overlapping query IDs")
        if not self.train_doc_ids or not self.val_doc_ids:
            raise ValueError(
                "Doc-disjoint split is missing train_doc_ids/val_doc_ids; regenerate "
                "doc_disjoint_split.json with src/evaluation/splits.py (fail-closed, no silent skip)."
            )
        d_overlap = self.train_doc_ids.intersection(self.val_doc_ids)
        if d_overlap:
            raise ValueError(f"Doc-disjoint split has {len(d_overlap)} overlapping document IDs")


def load_5fold_splits(dataset_dir: Union[str, Path]) -> List[FoldSplit]:
    """Load and validate the 5-fold cross-validation splits."""
    split_p = resolve_split_path(dataset_dir, "random_5fold.json")
    if not split_p or not split_p.is_file():
        # Try subfolder splits/random_5fold.json
        split_p = resolve_split_path(dataset_dir, "splits/random_5fold.json")
    if not split_p or not split_p.is_file():
        raise FileNotFoundError(f"Could not find random_5fold.json in {dataset_dir} or fallback locations")

    with open(split_p, "r", encoding="utf-8") as f:
        data = json.load(f)

    splits: List[FoldSplit] = []
    if isinstance(data, list):
        for idx, fold_dict in enumerate(data):
            train_qids = set(map(str, fold_dict.get("train_query_ids", fold_dict.get("train", []))))
            val_qids = set(map(str, fold_dict.get("val_query_ids", fold_dict.get("val", []))))
            fs = FoldSplit(fold_id=idx, train_qids=train_qids, val_qids=val_qids)
            fs.validate()
            splits.append(fs)
    elif isinstance(data, dict):
        for k, fold_dict in sorted(data.items(), key=lambda x: int(x[0])):
            train_qids = set(map(str, fold_dict.get("train_query_ids", fold_dict.get("train", []))))
            val_qids = set(map(str, fold_dict.get("val_query_ids", fold_dict.get("val", []))))
            fs = FoldSplit(fold_id=int(k), train_qids=train_qids, val_qids=val_qids)
            fs.validate()
            splits.append(fs)

    if len(splits) != 5:
        raise ValueError(f"Expected 5 folds, found {len(splits)}")

    return splits


def load_doc_disjoint_split(dataset_dir: Union[str, Path]) -> DocDisjointSplit:
    """Load and validate the document-disjoint split."""
    split_p = resolve_split_path(dataset_dir, "doc_disjoint_split.json")
    if not split_p or not split_p.is_file():
        split_p = resolve_split_path(dataset_dir, "splits/doc_disjoint_split.json")
    if not split_p or not split_p.is_file():
        raise FileNotFoundError(f"Could not find doc_disjoint_split.json in {dataset_dir} or fallback locations")

    with open(split_p, "r", encoding="utf-8") as f:
        data = json.load(f)

    train_qids = set(map(str, data.get("train_query_ids", data.get("train", []))))
    val_qids = set(map(str, data.get("val_query_ids", data.get("val", []))))
    train_doc_ids = set(map(str, data.get("train_doc_ids", [])))
    val_doc_ids = set(map(str, data.get("val_doc_ids", [])))

    if (not train_doc_ids or not val_doc_ids) and (train_qids and val_qids):
        # Migrate legacy splits (pre-doc_ids): recompute gold-doc sets from
        # qrels and verify disjointness instead of silently skipping the check.
        recomputed = _recompute_doc_ids(dataset_dir, train_qids, val_qids)
        if recomputed is not None:
            train_doc_ids, val_doc_ids = recomputed

    split = DocDisjointSplit(
        train_qids=train_qids,
        val_qids=val_qids,
        train_doc_ids=train_doc_ids,
        val_doc_ids=val_doc_ids,
    )
    split.validate()
    return split


def _recompute_doc_ids(
    dataset_dir: Union[str, Path], train_qids: Set[str], val_qids: Set[str]
) -> Optional[tuple[Set[str], Set[str]]]:
    """Recompute gold-doc sets from qrels_train.parquet for legacy split files."""
    base = Path(dataset_dir)
    candidates = [
        base / "qrels_train.parquet",
        base / "splits" / "qrels_train.parquet",
        base.parent / "qrels_train.parquet",
    ]
    qrels_p = next((p for p in candidates if p.is_file()), None)
    if qrels_p is None:
        try:
            from src.data.canonical import discover_canonical_dataset_dir

            canon = discover_canonical_dataset_dir()
            cand = canon / "qrels_train.parquet"
            if cand.is_file():
                qrels_p = cand
        except Exception:
            qrels_p = None
    if qrels_p is None:
        return None
    try:
        import pandas as pd

        df = pd.read_parquet(qrels_p, columns=["query_id", "doc_id"])
        q2d: dict[str, set[str]] = {}
        for qid, did in zip(df["query_id"].astype(str), df["doc_id"].astype(str)):
            q2d.setdefault(qid, set()).add(did)
        train_docs: set[str] = set()
        for q in train_qids:
            train_docs.update(q2d.get(q, set()))
        val_docs: set[str] = set()
        for q in val_qids:
            val_docs.update(q2d.get(q, set()))
        if not train_docs or not val_docs:
            return None
        return train_docs, val_docs
    except Exception:
        return None
