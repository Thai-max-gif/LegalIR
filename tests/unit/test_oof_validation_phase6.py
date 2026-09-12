"""Comprehensive test suite for Phase 6: OOF Cross-Validation, Scorer Parity, and Metrics."""

import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

# Ensure scoring program directory is on sys.path
repo_root = Path(__file__).resolve().parents[1]
scoring_prog_dir = repo_root / "Scoring-Program-Task-LegalIR"
if str(scoring_prog_dir) not in sys.path:
    sys.path.insert(0, str(scoring_prog_dir))

from scoring import eval_retrieval as official_eval_retrieval
from src.evaluation.codabench_compat import assert_official_equivalence
from src.evaluation.evaluator import (
    DEFAULT_CANDIDATE_CUTOFFS,
    compute_candidate_cutoffs,
    compute_candidate_recall,
    evaluate_predictions,
    normalize_candidate_cutoffs,
)
from src.evaluation.splits import (
    generate_document_disjoint_split,
    generate_random_5fold_split,
    verify_document_disjoint_isolation,
    verify_fold_isolation,
)
from src.pipeline.oof_runner import OOFRunner
from src.retrieval.bm25_micro import BM25MicroRetriever
from src.retrieval.question_memory import TrainQuestionMemory


# -----------------------------------------------------------------------------
# 1. Official Scorer Parity Across All Edge Cases
# -----------------------------------------------------------------------------

def test_official_scorer_parity_perfect_single_and_multi_gold():
    """Verify exact parity between internal evaluator and official Codabench scorer on perfect matches."""
    # 1 gold, top 1 correct, 5 returned
    y_true = {"q1": ["docA"], "q2": ["docB"]}
    y_pred = {
        "q1": {"answer": ["docA", "docX", "docY", "docZ", "docW"]},
        "q2": {"answer": ["docB", "doc1", "doc2", "doc3", "doc4"]},
    }
    metrics = assert_official_equivalence(y_pred, y_true)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(0.2)  # 1/5 for each query

    # Multi-gold perfect match (3 golds, 5 preds)
    y_true_multi = {"q1": ["docA", "docB", "docC"]}
    y_pred_multi = {"q1": {"answer": ["docA", "docB", "docC", "docD", "docE"]}}
    metrics_multi = assert_official_equivalence(y_pred_multi, y_true_multi)
    assert metrics_multi["recall"] == pytest.approx(1.0)
    assert metrics_multi["precision"] == pytest.approx(3.0 / 5.0)


def test_official_scorer_parity_partial_and_zero_matches():
    """Verify parity on partial matches and zero-recall queries."""
    y_true = {
        "q1": ["docA", "docB"],  # 2 golds, 1 found -> rec=0.5, prec=0.2
        "q2": ["docC"],          # 1 gold, 0 found -> rec=0.0, prec=0.0
        "q3": ["docD", "docE", "docF"], # 3 golds, 2 found -> rec=2/3, prec=0.4
    }
    y_pred = {
        "q1": {"answer": ["docA", "doc1", "doc2", "doc3", "doc4"]},
        "q2": {"answer": ["doc5", "doc6", "doc7", "doc8", "doc9"]},
        "q3": {"answer": ["docD", "docE", "doc10", "doc11", "doc12"]},
    }
    metrics = assert_official_equivalence(y_pred, y_true)
    expected_recall = (0.5 + 0.0 + (2.0 / 3.0)) / 3.0
    expected_precision = (0.2 + 0.0 + 0.4) / 3.0
    assert metrics["recall"] == pytest.approx(expected_recall)
    assert metrics["precision"] == pytest.approx(expected_precision)


def test_official_scorer_parity_edge_cases_empty_overflow_duplicates():
    """Verify official scorer rules: empty gives 0, >5 gives 0, duplicates affect precision."""
    y_true = {
        "q_empty": ["docA"],
        "q_overflow": ["docB"],
        "q_duplicates": ["docC"],
        "q_valid": ["docD"],
    }
    y_pred = {
        "q_empty": {"answer": []},  # len == 0 -> 0.0 score
        "q_overflow": {"answer": ["docB", "doc1", "doc2", "doc3", "doc4", "doc5"]},  # len == 6 -> 0.0 score
        "q_duplicates": {"answer": ["docC", "docC", "docX"]},  # 3 items, set has 2 -> rec=1.0, prec=1/3
        "q_valid": {"answer": ["docD", "docE"]},  # 2 items -> rec=1.0, prec=0.5
    }

    # Format for official eval_retrieval directly
    off_pred = {k: {"answer": v["answer"]} for k, v in y_pred.items()}
    off_res = official_eval_retrieval(off_pred, y_true)

    # Format for internal evaluate_predictions
    int_res = evaluate_predictions({k: v["answer"] for k, v in y_pred.items()}, y_true)

    assert int_res["recall_at_5"] == pytest.approx(off_res["recall"])
    assert int_res["precision_at_5"] == pytest.approx(off_res["precision"])

    # q_empty: 0, q_overflow: 0, q_duplicates: 1.0, q_valid: 1.0 -> mean recall = (0 + 0 + 1 + 1)/4 = 0.5
    assert int_res["recall_at_5"] == pytest.approx(0.5)


# -----------------------------------------------------------------------------
# 2. Detailed Metric Computations: MRR, MAP, nDCG@5, Top-K Recall & Precision
# -----------------------------------------------------------------------------

def test_detailed_ranking_metrics_analytical_verification():
    """Verify MRR, MAP, and nDCG@5 against manual analytical calculations."""
    y_true = {
        "q1": ["docA", "docB"],
    }
    # Predictions: rank 1: docX, rank 2: docA (hit), rank 3: docY, rank 4: docB (hit), rank 5: docZ
    y_pred = {
        "q1": ["docX", "docA", "docY", "docB", "docZ"],
    }

    res = evaluate_predictions(y_pred, y_true)

    # Recall at cutoffs
    assert res["recall@1"] == pytest.approx(0.0)      # 0/2
    assert res["recall@3"] == pytest.approx(0.5)      # 1/2 (docA)
    assert res["recall@5"] == pytest.approx(1.0)      # 2/2 (docA, docB)

    # Precision at cutoffs
    assert res["precision@1"] == pytest.approx(0.0)   # 0/1
    assert res["precision@3"] == pytest.approx(1.0 / 3.0)  # 1/3
    assert res["precision@5"] == pytest.approx(2.0 / 5.0)  # 2/5

    # MRR: first hit is at rank 2 -> RR = 1/2 = 0.5
    assert res["mrr"] == pytest.approx(0.5)
    assert res["MRR"] == pytest.approx(0.5)

    # MAP: hits at rank 2 (P@2 = 1/2) and rank 4 (P@4 = 2/4 = 0.5)
    # AP = (1/2 + 2/4) / 2 = 1.0 / 2 = 0.5
    assert res["map"] == pytest.approx(0.5)
    assert res["MAP"] == pytest.approx(0.5)

    # nDCG@5:
    # DCG@5 = 1/log2(2+1) + 1/log2(4+1) = 1/log2(3) + 1/log2(5)
    dcg = 1.0 / np.log2(3.0) + 1.0 / np.log2(5.0)
    # IDCG@5 = 1/log2(1+1) + 1/log2(2+1) = 1/log2(2) + 1/log2(3) = 1.0 + 1/log2(3)
    idcg = 1.0 / np.log2(2.0) + 1.0 / np.log2(3.0)
    expected_ndcg = dcg / idcg

    assert res["ndcg@5"] == pytest.approx(expected_ndcg, rel=1e-5)
    assert res["nDCG@5"] == pytest.approx(expected_ndcg, rel=1e-5)


def test_candidate_recall_cutoffs_monotonicity():
    """Verify Candidate Recall values at 20, 50, 100, 150, 200."""
    candidates = {
        "q1": [f"doc_{i}" for i in range(250)],
        "q2": [f"doc_{i}" for i in range(250)],
    }
    # q1 gold is at rank 30 (found in @50, @100, @150, @200, but not @20)
    # q2 gold is at rank 120 (found in @150, @200, but not @20, @50, @100)
    ground_truth = {
        "q1": ["doc_30"],
        "q2": ["doc_120"],
    }

    metrics = compute_candidate_cutoffs(candidates, ground_truth, cutoffs=[20, 50, 100, 150, 200])

    assert metrics["candidate_recall@20"] == pytest.approx(0.0)      # neither found
    assert metrics["candidate_recall@50"] == pytest.approx(0.5)      # q1 found
    assert metrics["candidate_recall@100"] == pytest.approx(0.5)     # q1 found
    assert metrics["candidate_recall@150"] == pytest.approx(1.0)     # both found
    assert metrics["candidate_recall@200"] == pytest.approx(1.0)     # both found


# -----------------------------------------------------------------------------
# 3. Strict Fold Isolation Guarantees
# -----------------------------------------------------------------------------

def test_random_5fold_split_isolation_and_partitioning():
    """Verify 5-fold generator produces valid partitions and strict isolation."""
    queries = [{"query_id": f"q_{i}"} for i in range(100)]
    qrels = [{"query_id": f"q_{i}", "doc_id": f"doc_{i}"} for i in range(100)]

    folds = generate_random_5fold_split(queries, seed=42)
    assert len(folds) == 5

    # Check isolation verification passes
    report = verify_fold_isolation(folds, qrels)
    assert report["is_isolated"] is True
    assert report["total_queries"] == 100

    # Ensure all validation sets are mutually exclusive and cover exactly all 100 queries
    val_union = set()
    for f in folds:
        val_set = set(f["val_query_ids"])
        train_set = set(f["train_query_ids"])
        assert len(val_set & train_set) == 0  # 0 leakage within fold
        assert len(val_set & val_union) == 0  # 0 overlap with prior folds
        val_union.update(val_set)

    assert val_union == {f"q_{i}" for i in range(100)}


def test_verify_fold_isolation_catches_simulated_leakage():
    """Verify that verify_fold_isolation raises AssertionError if validation query leaks into train."""
    queries = [{"query_id": f"q_{i}"} for i in range(20)]
    folds = generate_random_5fold_split(queries, seed=42)

    # Intentionally contaminate fold 0 train set with one of its val query IDs
    val_id_to_leak = folds[0]["val_query_ids"][0]
    folds[0]["train_query_ids"].append(val_id_to_leak)

    with pytest.raises(AssertionError, match="queries in both train and val"):
        verify_fold_isolation(folds)


def test_document_disjoint_split_zero_gold_document_overlap():
    """Verify document-disjoint split guarantees zero gold document overlap."""
    queries = [{"query_id": f"q_{i}"} for i in range(50)]
    # Group queries into documents: doc_0 has q_0..q_4, doc_1 has q_5..q_9, etc.
    qrels = []
    for i in range(50):
        doc_id = f"doc_{i // 5}"
        qrels.append({"query_id": f"q_{i}", "doc_id": doc_id})

    split = generate_document_disjoint_split(queries, qrels, val_ratio=0.2, seed=42)

    # Verify isolation
    report = verify_document_disjoint_isolation(split, qrels)
    assert report["is_disjoint"] is True
    assert report["train_queries"] > 0
    assert report["val_queries"] > 0


def test_verify_document_disjoint_catches_shared_doc_leakage():
    """Verify that verify_document_disjoint_isolation detects document contamination."""
    qrels = [
        {"query_id": "q1", "doc_id": "docA"},
        {"query_id": "q2", "doc_id": "docA"},  # q1 and q2 share docA
    ]
    # Faulty split placing q1 in train and q2 in val
    split = {
        "train_query_ids": ["q1"],
        "val_query_ids": ["q2"],
    }

    with pytest.raises(AssertionError, match="Document-disjoint split violation"):
        verify_document_disjoint_isolation(split, qrels)


def test_question_memory_strictly_excludes_validation_queries():
    """Verify TrainQuestionMemory only indexes training queries and respects exclude_qid."""
    train_queries = {"q1": "hóa đơn điện tử", "q2": "thuế giá trị gia tăng"}
    train_qrels = {"q1": ["doc1"], "q2": ["doc2"]}

    memory = TrainQuestionMemory(min_similarity=0.5, use_dense=False)
    memory.fit(train_queries, train_qrels)

    assert memory.training_query_ids == {"q1", "q2"}

    # Searching with query 'q1' should match 'q1' if not excluded
    matches_no_exclude = memory.search("hóa đơn điện tử", top_k=5)
    assert len(matches_no_exclude) > 0
    assert any(m["doc_id"] == "doc1" for m in matches_no_exclude)

    # Searching with exclude_qid="q1" should exclude q1 from candidate recommendations
    matches_excluded = memory.search("hóa đơn điện tử", top_k=5, exclude_qid="q1")
    assert not any(m.get("query_id") == "q1" for m in matches_excluded)


# -----------------------------------------------------------------------------
# 4. End-to-End OOF Runner Execution on Mock Synthetic Dataset
# -----------------------------------------------------------------------------

def test_oof_runner_end_to_end_mock_dataset(tmp_path: Path):
    """Run full OOFRunner on a self-contained synthetic dataset and verify output artifacts."""
    data_dir = tmp_path / "data"
    index_dir = tmp_path / "indexes"
    output_dir = tmp_path / "cv_output"
    splits_dir = tmp_path / "splits"

    data_dir.mkdir(parents=True)
    index_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    splits_dir.mkdir(parents=True)

    # 1. Create synthetic corpus: 20 documents
    docs = []
    chunks = []
    for i in range(20):
        did = f"doc_{i}"
        title = f"Nghị định số {100 + i}/2023/NĐ-CP"
        text = f"Quy định chi tiết về quản lý tài chính và thuế vụ doanh nghiệp điều khoản {i}"
        docs.append({
            "doc_id": did,
            "title": title,
            "legal_number": f"{100 + i}/2023/NĐ-CP",
            "doc_type": "nghị định",
            "text": text,
        })
        chunks.append({
            "chunk_id": f"c_{i}_micro",
            "doc_id": did,
            "granularity": "micro",
            "chapter": "",
            "section": "",
            "article": f"Điều {i + 1}",
            "clause": "",
            "point": "",
            "text_norm": text,
            "text_raw": text,
            "legal_number": f"{100 + i}/2023/NĐ-CP",
        })

    docs_df = pd.DataFrame(docs)
    chunks_df = pd.DataFrame(chunks)
    docs_df.to_parquet(data_dir / "documents.parquet")
    chunks_df.to_parquet(data_dir / "chunks.parquet")

    # 2. Create synthetic queries: 15 queries (3 per fold)
    queries = []
    qrels = []
    for i in range(15):
        qid = f"q_{i}"
        target_doc = f"doc_{i % 20}"
        queries.append({
            "query_id": qid,
            "question_norm": f"quy định quản lý tài chính điều {i % 20 + 1}",
            "question_raw": f"Quy định quản lý tài chính điều {i % 20 + 1}?",
        })
        qrels.append({
            "query_id": qid,
            "doc_id": target_doc,
        })

    queries_df = pd.DataFrame(queries)
    qrels_df = pd.DataFrame(qrels)
    queries_df.to_parquet(data_dir / "queries_train.parquet")
    qrels_df.to_parquet(data_dir / "qrels_train.parquet")

    # 3. Create BM25 micro index
    bm25 = BM25MicroRetriever()
    bm25.fit(chunks, show_progress=False)
    bm25.save(index_dir / "bm25")

    # 4. Generate splits
    splits = generate_random_5fold_split(queries, seed=42)
    splits_path = splits_dir / "random_5fold.json"
    with open(splits_path, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    # 5. Execute OOFRunner
    runner = OOFRunner(
        data_dir=data_dir,
        index_dir=index_dir,
        splits_path=splits_path,
        output_dir=output_dir,
        num_folds=5,
        candidate_k=10,
        rerank_k=5,
        use_reranker=False,
        smoke=False,
    )
    cv_report = runner.run()

    # 6. Verify report structure and values
    assert cv_report["total_evaluated_queries"] == 15
    assert cv_report["official_scorer_parity_verified"] is True
    assert len(cv_report["folds"]) == 5
    assert "mean_recall@5" in cv_report
    assert "mean_precision@5" in cv_report
    assert "mean_mrr" in cv_report
    assert "mean_map" in cv_report
    assert "mean_ndcg@5" in cv_report
    assert "mean_candidate@50" in cv_report

    # 7. Verify saved parquet artifacts
    pred_parquet = output_dir / "oof_predictions.parquet"
    feat_parquet = output_dir / "oof_features.parquet"
    report_json = output_dir / "cv_report.json"
    pred_json = output_dir / "oof_predictions.json"

    assert pred_parquet.exists()
    assert feat_parquet.exists()
    assert report_json.exists()
    assert pred_json.exists()

    df_preds = pd.read_parquet(pred_parquet)
    assert len(df_preds) == 15
    assert set(df_preds.columns) == {"query_id", "answer", "fold"}

    df_feats = pd.read_parquet(feat_parquet)
    assert len(df_feats) > 0
    assert "query_id" in df_feats.columns
    assert "doc_id" in df_feats.columns
    assert "label" in df_feats.columns
    assert "fold" in df_feats.columns
    assert "bm25_score" in df_feats.columns
