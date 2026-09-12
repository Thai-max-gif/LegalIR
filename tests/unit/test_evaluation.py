import pytest
import numpy as np
from src.evaluation.evaluator import evaluate_predictions, compute_candidate_recall
from src.evaluation.splits import generate_random_5fold_split, generate_document_disjoint_split

def test_codabench_scoring_rules():
    # Case 1: Exact match with 1 gold
    y_true = {"q1": ["docA"]}
    y_pred = {"q1": {"answer": ["docA", "docB", "docC"]}}
    res = evaluate_predictions(y_pred, y_true)
    assert res["recall"] == 1.0
    assert pytest.approx(res["precision"], 0.01) == 1.0 / 3.0

    # Case 2: > 5 predictions gives 0.0
    y_pred_invalid = {"q1": {"answer": ["docA", "docB", "docC", "docD", "docE", "docF"]}}
    res_inv = evaluate_predictions(y_pred_invalid, y_true)
    assert res_inv["recall"] == 0.0
    assert res_inv["precision"] == 0.0

    # Case 3: Empty prediction gives 0.0
    y_pred_empty = {"q1": {"answer": []}}
    res_empty = evaluate_predictions(y_pred_empty, y_true)
    assert res_empty["recall"] == 0.0
    assert res_empty["precision"] == 0.0

    # Case 4: Multi-positive gold matching partial
    y_true_multi = {"q1": ["docA", "docB"]}
    y_pred_partial = {"q1": {"answer": ["docA", "docC"]}}
    res_part = evaluate_predictions(y_pred_partial, y_true_multi)
    assert res_part["recall"] == 0.5
    assert res_part["precision"] == 0.5

def test_candidate_recall():
    y_true = {
        "q1": ["docA", "docB"],
        "q2": ["docC"]
    }
    candidates = {
        "q1": ["docX", "docA", "docY", "docB"],
        "q2": ["docZ", "docW"]
    }
    recall_at_2 = compute_candidate_recall(candidates, y_true, k=2)
    # q1 has docA in top 2 (1/2 = 0.5), q2 has 0 in top 2 (0/1 = 0.0) -> mean = 0.25
    assert recall_at_2 == 0.25

    recall_at_4 = compute_candidate_recall(candidates, y_true, k=4)
    # q1 has docA and docB in top 4 (2/2 = 1.0), q2 has 0 in top 4 (0.0) -> mean = 0.5
    assert recall_at_4 == 0.5

def test_split_generators():
    queries = [{"query_id": str(i)} for i in range(100)]
    qrels = [
        {"query_id": str(i), "doc_id": f"doc_{i % 10}"}
        for i in range(100)
    ]
    # Test random 5-fold
    folds = generate_random_5fold_split(queries, seed=42)
    assert len(folds) == 5
    all_val = []
    for f in folds:
        assert len(f["train_query_ids"]) + len(f["val_query_ids"]) == 100
        all_val.extend(f["val_query_ids"])
    assert len(set(all_val)) == 100

    # Test doc-disjoint split
    disjoint = generate_document_disjoint_split(queries, qrels, val_ratio=0.2, seed=42)
    train_q = set(disjoint["train_query_ids"])
    val_q = set(disjoint["val_query_ids"])
    assert len(train_q & val_q) == 0

    train_docs = set(r["doc_id"] for r in qrels if r["query_id"] in train_q)
    val_docs = set(r["doc_id"] for r in qrels if r["query_id"] in val_q)
    assert len(train_docs & val_docs) == 0
