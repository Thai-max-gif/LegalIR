from collections import defaultdict
from pathlib import Path
from typing import Any
import argparse
import json
import random
import pandas as pd


def generate_random_5fold_split(queries: list, seed: int = 42, num_folds: int = 5) -> list[dict[str, Any]]:
    """Generate k-fold cross-validation query splits with deterministic ordering."""
    qids = sorted(list(set(str(q["query_id"]) if isinstance(q, dict) else str(q) for q in queries)))
    rng = random.Random(seed)
    shuffled_qids = list(qids)
    rng.shuffle(shuffled_qids)

    n = len(shuffled_qids)
    k = max(2, min(num_folds, n))
    fold_size = max(1, n // k)
    folds = []

    for i in range(k):
        val_start = i * fold_size
        val_end = (i + 1) * fold_size if i < k - 1 else n
        val_qids = sorted(shuffled_qids[val_start:val_end])
        val_set = set(val_qids)
        train_qids = sorted([qid for qid in qids if qid not in val_set])

        folds.append({
            "fold": i,
            "train": train_qids,
            "val": val_qids,
            "train_query_ids": train_qids,
            "val_query_ids": val_qids,
        })

    return folds


def generate_document_disjoint_split(
    queries: list,
    qrels: list,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> dict[str, Any]:
    """Generate a document-disjoint split where validation gold documents never appear in the train set."""
    rng = random.Random(seed)

    # Map query -> gold docs and doc -> queries
    q2d = defaultdict(set)
    d2q = defaultdict(set)
    for r in qrels:
        qid = str(r["query_id"])
        did = str(r["doc_id"])
        q2d[qid].add(did)
        d2q[did].add(qid)

    # Build connected components of queries that share gold documents
    visited_q = set()
    clusters = []  # each cluster is a set of query_ids

    all_qids = sorted(list(q2d.keys()))
    for q in all_qids:
        if q in visited_q:
            continue
        cluster = set()
        queue = [q]
        visited_q.add(q)

        while queue:
            curr_q = queue.pop()
            cluster.add(curr_q)
            for doc in q2d[curr_q]:
                for neighbor_q in d2q[doc]:
                    if neighbor_q not in visited_q:
                        visited_q.add(neighbor_q)
                        queue.append(neighbor_q)
        clusters.append(cluster)

    # Shuffle clusters and greedily assign to val up to val_ratio
    rng.shuffle(clusters)
    total_queries = len(all_qids)
    target_val_count = int(total_queries * val_ratio)

    val_qids = []
    train_qids = []

    for c in clusters:
        if len(val_qids) + len(c) <= target_val_count or len(val_qids) == 0:
            val_qids.extend(list(c))
        else:
            train_qids.extend(list(c))

    # Safety assertion
    val_docs = set()
    for q in val_qids:
        val_docs.update(q2d[q])

    train_docs = set()
    for q in train_qids:
        train_docs.update(q2d[q])

    overlap = val_docs & train_docs
    if len(overlap) > 0:
        raise AssertionError(f"Document-disjoint split violation! Overlap: {overlap}")

    return {
        "train": sorted(train_qids),
        "val": sorted(val_qids),
        "train_query_ids": sorted(train_qids),
        "val_query_ids": sorted(val_qids),
        "train_doc_count": len(train_docs),
        "val_doc_count": len(val_docs),
        "train_doc_ids": sorted(train_docs),
        "val_doc_ids": sorted(val_docs),
    }


def verify_fold_isolation(
    folds: list[dict[str, Any]],
    qrels: list[dict[str, Any]] | dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Strictly verify fold isolation properties:
    1. At least 2 folds.
    2. Each fold has non-empty disjoint train and val sets (train & val == empty).
    3. Val sets across all folds are pairwise disjoint.
    4. Union of all val sets covers all queries.
    5. If qrels provided, ensure validation queries' qrels are completely isolated.
    """
    if not folds or len(folds) < 2:
        raise AssertionError(f"Expected at least 2 folds, got {len(folds) if folds else 0}")

    all_val_ids: list[str] = []
    all_train_ids_fold0: set[str] = set()

    for idx, f in enumerate(folds):
        train_ids = set(str(x) for x in f.get("train_query_ids", f.get("train", [])))
        val_ids = set(str(x) for x in f.get("val_query_ids", f.get("val", [])))

        if not train_ids:
            raise AssertionError(f"Fold {idx} has empty train query set")
        if not val_ids:
            raise AssertionError(f"Fold {idx} has empty val query set")

        intersection = train_ids & val_ids
        if intersection:
            raise AssertionError(
                f"Fold {idx} has {len(intersection)} queries in both train and val: {sorted(list(intersection))[:5]}"
            )

        if idx == 0:
            all_train_ids_fold0 = train_ids | val_ids

        all_val_ids.extend(list(val_ids))

    # Check pairwise disjointness of validation sets
    if len(all_val_ids) != len(set(all_val_ids)):
        seen = set()
        duplicates = set()
        for qid in all_val_ids:
            if qid in seen:
                duplicates.add(qid)
            seen.add(qid)
        raise AssertionError(
            f"Validation sets across folds are not pairwise disjoint! Duplicates: {sorted(list(duplicates))[:5]}"
        )

    # Check that union of all val sets equals total query set
    if all_train_ids_fold0 and set(all_val_ids) != all_train_ids_fold0:
        missing = all_train_ids_fold0 - set(all_val_ids)
        extra = set(all_val_ids) - all_train_ids_fold0
        raise AssertionError(
            f"Fold validation sets do not partition full query set! Missing: {len(missing)}, Extra: {len(extra)}"
        )

    # If qrels provided, verify query-level qrel isolation. NOTE: random
    # 5-fold splits are query-disjoint by design but intentionally doc-leaky
    # (same gold doc may appear in train+val). Doc-level robustness is
    # covered by the separate document-disjoint split, not here.
    if qrels is not None:
        q2d = defaultdict(set)
        if isinstance(qrels, dict):
            for qid, docs in qrels.items():
                for d in docs:
                    q2d[str(qid)].add(str(d))
        else:
            for r in qrels:
                q2d[str(r["query_id"])].add(str(r["doc_id"]))

        for idx, f in enumerate(folds):
            val_ids = set(str(x) for x in f.get("val_query_ids", f.get("val", [])))
            train_ids = set(str(x) for x in f.get("train_query_ids", f.get("train", [])))

            # Query-level isolation (duplicate of per-fold check above, kept
            # explicit when qrels are supplied).
            leaked_qrels = val_ids & train_ids
            if leaked_qrels:
                raise AssertionError(f"Fold {idx} leaked validation queries into train: {leaked_qrels}")

            # Coverage: every val query should have at least one gold doc.
            missing_gold = sorted(q for q in val_ids if not q2d.get(q))
            if missing_gold:
                raise AssertionError(
                    f"Fold {idx} has {len(missing_gold)} val queries without gold qrels: {missing_gold[:5]}"
                )

    return {
        "is_isolated": True,
        "num_folds": len(folds),
        "total_queries": len(all_val_ids),
        "val_queries_per_fold": [len(f.get("val_query_ids", f.get("val", []))) for f in folds],
    }


def verify_document_disjoint_isolation(
    split: dict[str, Any],
    qrels: list[dict[str, Any]] | dict[str, list[str]],
) -> dict[str, Any]:
    """Strictly verify document-disjoint split isolation:
    1. train_qids and val_qids are disjoint.
    2. Gold documents for train queries and gold documents for val queries have ZERO overlap.
    """
    train_qids = set(str(x) for x in split.get("train_query_ids", split.get("train", [])))
    val_qids = set(str(x) for x in split.get("val_query_ids", split.get("val", [])))

    if not train_qids or not val_qids:
        raise AssertionError("Train or val query set in document-disjoint split is empty")

    query_overlap = train_qids & val_qids
    if query_overlap:
        raise AssertionError(f"Document-disjoint split has query overlap: {query_overlap}")

    q2d = defaultdict(set)
    if isinstance(qrels, dict):
        for qid, docs in qrels.items():
            for d in docs:
                q2d[str(qid)].add(str(d))
    else:
        for r in qrels:
            q2d[str(r["query_id"])].add(str(r["doc_id"]))

    train_docs = set()
    for q in train_qids:
        train_docs.update(q2d[q])

    val_docs = set()
    for q in val_qids:
        val_docs.update(q2d[q])

    doc_overlap = train_docs & val_docs
    if doc_overlap:
        raise AssertionError(
            f"Document-disjoint split violation! Overlap of {len(doc_overlap)} docs between train and val: {sorted(list(doc_overlap))[:5]}"
        )

    return {
        "is_disjoint": True,
        "train_queries": len(train_qids),
        "val_queries": len(val_qids),
        "train_docs": len(train_docs),
        "val_docs": len(val_docs),
    }


def create_all_splits(canonical_dir: str | Path):
    canonical_dir = Path(canonical_dir)
    splits_dir = canonical_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    queries_df = pd.read_parquet(canonical_dir / "queries_train.parquet")
    qrels_df = pd.read_parquet(canonical_dir / "qrels_train.parquet")

    queries = queries_df.to_dict(orient="records")
    qrels = qrels_df.to_dict(orient="records")

    print(f"Generating 5-fold random CV splits for {len(queries)} queries...")
    random_5fold = generate_random_5fold_split(queries, seed=42)
    verify_fold_isolation(random_5fold, qrels)
    (splits_dir / "random_5fold.json").write_text(json.dumps(random_5fold, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("Generating document-disjoint split...")
    doc_disjoint = generate_document_disjoint_split(queries, qrels, val_ratio=0.2, seed=42)
    verify_document_disjoint_isolation(doc_disjoint, qrels)
    (splits_dir / "doc_disjoint_split.json").write_text(json.dumps(doc_disjoint, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Splits successfully created and verified in {splits_dir}!")
    print(f"  Random 5-fold: 5 folds x {len(random_5fold[0]['val_query_ids'])} val queries")
    print(f"  Doc-disjoint: {len(doc_disjoint['train_query_ids'])} train queries ({doc_disjoint['train_doc_count']} docs) | {len(doc_disjoint['val_query_ids'])} val queries ({doc_disjoint['val_doc_count']} docs)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical_dir", type=str, default="artifacts/shared/canonical/v2")
    args = parser.parse_args()
    create_all_splits(args.canonical_dir)
