import json
from pathlib import Path
from src.evaluation.splits import generate_random_5fold_split, generate_document_disjoint_split


def test_splits_are_deterministic_and_partition_queries():
    queries = [{"query_id": str(i), "question_norm": f"câu {i}"} for i in range(100)]
    qrels = [{"query_id": str(i), "doc_id": str(i % 20), "relevance": 1} for i in range(100)]

    split1 = generate_random_5fold_split(queries, seed=42)
    split2 = generate_random_5fold_split(queries, seed=42)
    assert split1 == split2
    assert len(split1) == 5

    # Check that all queries are in each fold union and val sets are disjoint
    all_val_qids = set()
    for fold in split1:
        assert set(fold["train"]).isdisjoint(set(fold["val"]))
        all_val_qids.update(fold["val"])
    assert len(all_val_qids) == 100

    # Document disjoint
    doc_split1 = generate_document_disjoint_split(queries, qrels, val_ratio=0.2, seed=42)
    doc_split2 = generate_document_disjoint_split(queries, qrels, val_ratio=0.2, seed=42)
    assert doc_split1 == doc_split2

    train_docs = {q["doc_id"] for q in qrels if q["query_id"] in set(doc_split1["train"])}
    val_docs = {q["doc_id"] for q in qrels if q["query_id"] in set(doc_split1["val"])}
    assert train_docs.isdisjoint(val_docs)
