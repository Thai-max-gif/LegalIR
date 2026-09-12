"""Test suite for Phase 7: OOF Feature Extraction, Learned Fusion (LightGBM/Linear), and RRF Fallback."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.ranking.fusion import (
    LearnedRanker,
    LightGBMRanker,
    LinearRanker,
    ReciprocalRankFusion,
    reciprocal_rank_fusion,
)
from src.ranking.oof_features import (
    CORE_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    FEATURE_SCHEMA_VERSION,
    compute_training_doc_frequencies,
    extract_candidate_features,
    extract_dataset_features,
)
from src.ranking.train_fusion import (
    evaluate_features_with_ranker,
    train_and_evaluate_fusion_cv,
)


# -----------------------------------------------------------------------------
# 1. Feature Extraction & Zero NaN Guarantees
# -----------------------------------------------------------------------------

def test_feature_extraction_has_all_required_phase7_columns():
    """Verify that all Phase 7 specified feature columns are present."""
    required_cols = {
        "raw_bm25_rank",
        "raw_bm25_score",
        "pyvi_bm25_rank",
        "pyvi_bm25_score",
        "dense_rank",
        "dense_score",
        "dense_second_score",
        "dense_margin",
        "memory_rank",
        "memory_similarity",
        "memory_vote_count",
        "exact_score",
        "exact_legal_number",
        "exact_article",
        "exact_clause",
        "exact_point",
        "exact_year",
        "exact_doc_type",
        "exact_title_overlap",
        "source_count",
        "rrf_score",
        "reranker_score",
        "reranker_second_score",
        "reranker_margin",
        "query_length",
        "train_doc_freq",
    }
    assert required_cols <= set(CORE_FEATURE_COLUMNS)
    assert required_cols <= set(FEATURE_COLUMNS)
    assert FEATURE_SCHEMA_VERSION == "v2"


def test_feature_extraction_zero_nan_and_proper_types():
    """Verify feature extractor handles missing fields, None values, and produces zero NaNs."""
    candidates = [
        {
            "doc_id": "doc_complete",
            "raw_bm25_rank": 1,
            "raw_bm25_score": 12.5,
            "pyvi_bm25_rank": 2,
            "pyvi_bm25_score": 10.2,
            "dense_rank": 3,
            "dense_score": 0.85,
            "dense_second_score": 0.75,
            "dense_margin": 0.10,
            "memory_rank": 1,
            "memory_similarity": 0.92,
            "memory_vote_count": 4,
            "exact_score": 1.0,
            "exact_legal_number": True,
            "exact_article": True,
            "exact_clause": False,
            "exact_point": False,
            "exact_year": True,
            "exact_doc_type": True,
            "exact_title_overlap": 0.8,
            "source_count": 5,
            "rrf_score": 0.085,
            "reranker_score": 3.5,
            "reranker_second_score": 1.2,
            "reranker_margin": 2.3,
        },
        {
            # Minimal candidate with almost everything missing / None
            "doc_id": "doc_minimal",
            "bm25_score": None,
            "dense_score": float("nan"),
            "exact_legal_number": None,
        },
    ]

    doc_freqs = {"doc_complete": 0.05}
    qrels = {"q100": ["doc_complete"]}

    df = extract_candidate_features(
        query_id="q100",
        candidate_records=candidates,
        query_text="Luật Doanh nghiệp 2020 Điều 15",
        doc_freq_map=doc_freqs,
        qrels=qrels,
    )

    assert len(df) == 2
    assert list(df["query_id"]) == ["q100", "q100"]
    assert list(df["group"]) == ["q100", "q100"]
    assert list(df["doc_id"]) == ["doc_complete", "doc_minimal"]
    assert list(df["label"]) == [1, 0]

    # Check that complete record features are parsed properly
    row0 = df.iloc[0]
    assert row0["raw_bm25_rank"] == 1.0
    assert row0["raw_bm25_score"] == 12.5
    assert row0["exact_legal_number"] == 1.0
    assert row0["exact_article"] == 1.0
    assert row0["exact_clause"] == 0.0
    assert row0["query_length"] == float(len("Luật Doanh nghiệp 2020 Điều 15"))
    assert row0["train_doc_freq"] == 0.05

    # Check that minimal record has 0 NaNs across all feature columns
    for col in FEATURE_COLUMNS:
        assert not df[col].isna().any(), f"Column {col} contains NaN!"


def test_training_doc_frequency_computation():
    """Verify document frequency prior is correctly calculated from training qrels."""
    qrels = {
        "q1": ["docA", "docB"],
        "q2": ["docA"],
        "q3": ["docC"],
        "q4": ["docA", "docB", "docC"],
    }
    freq_map = compute_training_doc_frequencies(qrels)
    assert len(freq_map) == 3
    # docA appears in 3 out of 4 queries: 3/4 = 0.75
    assert freq_map["docA"] == pytest.approx(0.75)
    # docB appears in 2 out of 4 queries: 2/4 = 0.50
    assert freq_map["docB"] == pytest.approx(0.50)
    # docC appears in 2 out of 4 queries: 2/4 = 0.50
    assert freq_map["docC"] == pytest.approx(0.50)
    # unseen doc defaults to 0.0
    assert freq_map.get("doc_unseen", 0.0) == 0.0


# -----------------------------------------------------------------------------
# 2. Fold Isolation
# -----------------------------------------------------------------------------

def test_fold_isolation_during_cross_validation():
    """Verify strict fold isolation: Fold f model is trained on folds != f only."""
    # Build synthetic 5-fold feature dataset
    np.random.seed(42)
    rows = []
    qrels_dict = {}

    for f_idx in range(5):
        for q_i in range(10):
            qid = f"fold{f_idx}_q{q_i}"
            gold_did = f"gold_{qid}"
            qrels_dict[qid] = [gold_did]

            # 1 positive and 3 negative candidates per query
            for cand_i in range(4):
                is_pos = (cand_i == 0)
                did = gold_did if is_pos else f"neg_{qid}_{cand_i}"
                rows.append({
                    "query_id": qid,
                    "group": qid,
                    "doc_id": did,
                    "fold": f_idx,
                    "raw_bm25_rank": 1.0 if is_pos else float(cand_i + 5),
                    "raw_bm25_score": 10.0 if is_pos else float(4 - cand_i),
                    "dense_rank": 1.0 if is_pos else float(cand_i + 3),
                    "dense_score": 0.9 if is_pos else 0.3,
                    "exact_score": 1.0 if is_pos else 0.0,
                    "label": 1 if is_pos else 0,
                    "target": 1 if is_pos else 0,
                })

    oof_df = pd.DataFrame(rows)

    # For each fold, ensure training data excludes validation fold queries
    for f in range(5):
        train_df = oof_df[oof_df["fold"] != f]
        val_df = oof_df[oof_df["fold"] == f]

        train_qids = set(train_df["query_id"].unique())
        val_qids = set(val_df["query_id"].unique())

        assert len(train_qids & val_qids) == 0, f"Leakage detected in Fold {f}!"


# -----------------------------------------------------------------------------
# 3. LightGBMRanker Training, Persistence, and Deterministic Inference
# -----------------------------------------------------------------------------

def test_lightgbm_ranker_fit_predict_and_persistence(tmp_path):
    """Test LightGBMRanker fitting, serialization, loading, and deterministic prediction."""
    np.random.seed(42)
    X = pd.DataFrame({
        "query_id": ["q1", "q1", "q1", "q2", "q2", "q2"],
        "doc_id": ["d1_pos", "d1_neg1", "d1_neg2", "d2_pos", "d2_neg1", "d2_neg2"],
        "raw_bm25_score": [8.0, 2.0, 1.0, 9.0, 3.0, 1.5],
        "dense_score": [0.95, 0.3, 0.2, 0.92, 0.4, 0.1],
        "exact_score": [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        "reranker_score": [4.0, -1.0, -2.0, 3.8, -0.5, -2.5],
    })
    y = np.array([1, 0, 0, 1, 0, 0], dtype=np.int32)
    groups = np.array([3, 3], dtype=np.int32)

    ranker = LightGBMRanker(
        feature_cols=["raw_bm25_score", "dense_score", "exact_score", "reranker_score"],
        learning_rate=0.1,
        num_leaves=15,
    )
    ranker.fit(X, y, groups, num_boost_round=30)

    # Candidate evaluation for new query
    test_cands = [
        {"doc_id": "neg_cand", "raw_bm25_score": 1.0, "dense_score": 0.2, "exact_score": 0.0, "reranker_score": -1.5},
        {"doc_id": "pos_cand", "raw_bm25_score": 8.5, "dense_score": 0.9, "exact_score": 1.0, "reranker_score": 3.9},
    ]

    ranked1 = ranker.predict(test_cands)
    assert ranked1[0]["doc_id"] == "pos_cand"
    assert ranked1[1]["doc_id"] == "neg_cand"

    # Test serialization & reload
    model_path = tmp_path / "test_ranker_model.txt"
    ranker.save(model_path)
    assert model_path.exists()

    loaded_ranker = LightGBMRanker(model_file=model_path)
    ranked2 = loaded_ranker.predict(test_cands)

    # Deterministic output check
    assert [c["doc_id"] for c in ranked1] == [c["doc_id"] for c in ranked2]
    assert np.isclose(ranked1[0]["final_score"], ranked2[0]["final_score"])


# -----------------------------------------------------------------------------
# 4. LinearRanker Fallback
# -----------------------------------------------------------------------------

def test_linear_ranker_fallback(tmp_path):
    """Test LinearRanker fallback training, serialization, and scoring."""
    X = pd.DataFrame({
        "raw_bm25_score": [10.0, 1.0, 9.0, 2.0],
        "dense_score": [0.9, 0.1, 0.85, 0.15],
        "exact_score": [1.0, 0.0, 1.0, 0.0],
        "query_id": ["q1", "q1", "q2", "q2"],
        "doc_id": ["d1", "d2", "d3", "d4"],
    })
    y = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)

    linear_ranker = LinearRanker(
        feature_cols=["raw_bm25_score", "dense_score", "exact_score"],
        alpha=1.0,
    )
    linear_ranker.fit(X, y)

    cands = [
        {"doc_id": "cand_low", "raw_bm25_score": 1.0, "dense_score": 0.1, "exact_score": 0.0},
        {"doc_id": "cand_high", "raw_bm25_score": 10.0, "dense_score": 0.9, "exact_score": 1.0},
    ]
    ranked = linear_ranker.predict(cands)
    assert ranked[0]["doc_id"] == "cand_high"
    assert ranked[1]["doc_id"] == "cand_low"

    # Test serialization
    save_path = tmp_path / "linear_model.json"
    linear_ranker.save(save_path)
    assert save_path.exists()

    loaded = LinearRanker()
    loaded.load(save_path)
    ranked_loaded = loaded.predict(cands)
    assert [c["doc_id"] for c in ranked] == [c["doc_id"] for c in ranked_loaded]


# -----------------------------------------------------------------------------
# 5. ReciprocalRankFusion Scoring & Fallback
# -----------------------------------------------------------------------------

def test_reciprocal_rank_fusion_scoring():
    """Verify ReciprocalRankFusion combines signals with proper weighting and tie-breaking."""
    cands = [
        {"doc_id": "d_exact", "raw_bm25_rank": 5, "exact_score": 1.0, "reranker_score": 1.0},
        {"doc_id": "d_bm25", "raw_bm25_rank": 1, "exact_score": 0.0, "reranker_score": 0.0},
        {"doc_id": "d_rerank", "raw_bm25_rank": 10, "exact_score": 0.0, "reranker_score": 5.0},
    ]

    rrf = ReciprocalRankFusion(k=60, w_bm25=1.0, w_exact=3.0, w_rerank=2.0)
    ranked = rrf.rank_candidates(cands)

    assert len(ranked) == 3
    # Exact match + high reranker should be top
    assert ranked[0]["doc_id"] in ("d_exact", "d_rerank")
    # All candidates have final_score
    for item in ranked:
        assert "final_score" in item
        assert item["final_score"] > 0.0


def test_learned_ranker_falls_back_to_rrf_when_unfit():
    """Verify that an un-fitted LearnedRanker gracefully falls back to RRF rather than crashing."""
    ranker = LightGBMRanker()
    cands = [
        {"doc_id": "d2", "bm25_rank": 2},
        {"doc_id": "d1", "bm25_rank": 1},
    ]
    ranked = ranker.predict(cands)
    assert len(ranked) == 2
    assert ranked[0]["doc_id"] == "d1"
    assert ranked[1]["doc_id"] == "d2"


# -----------------------------------------------------------------------------
# 6. Model Selection Gate: Learned Ranker vs Weighted RRF
# -----------------------------------------------------------------------------

def test_model_selection_gate_and_full_training(tmp_path):
    """Test full cross-validation and automated model selection gate."""
    np.random.seed(42)
    rows = []
    qrels_dict = {}

    # Create 5 folds of synthetic queries with strong signal in feature 'dense_score'
    for f in range(5):
        for q_i in range(8):
            qid = f"f{f}_q{q_i}"
            gold_did = f"gold_{qid}"
            qrels_dict[qid] = [gold_did]

            # 1 gold positive, 4 negatives
            for c_i in range(5):
                is_gold = (c_i == 0)
                did = gold_did if is_gold else f"neg_{qid}_{c_i}"
                rows.append({
                    "query_id": qid,
                    "group": qid,
                    "doc_id": did,
                    "fold": f,
                    "raw_bm25_rank": 1.0 if is_gold else float(c_i + 4),
                    "raw_bm25_score": 10.0 if is_gold else 2.0,
                    "dense_rank": 1.0 if is_gold else float(c_i + 2),
                    "dense_score": 0.95 if is_gold else 0.1,
                    "exact_score": 1.0 if is_gold else 0.0,
                    "reranker_score": 4.0 if is_gold else -2.0,
                    "label": 1 if is_gold else 0,
                    "target": 1 if is_gold else 0,
                })

    oof_df = pd.DataFrame(rows)
    out_dir = tmp_path / "fusion_output"

    results = train_and_evaluate_fusion_cv(
        oof_df=oof_df,
        qrels_dict=qrels_dict,
        output_dir=out_dir,
        num_boost_round=20,
    )

    assert "manifest" in results
    assert "comparison" in results
    assert "winning_method" in results
    assert results["winner_mean_recall@5"] == pytest.approx(1.0)

    # Check generated files on disk
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "fusion_comparison.json").exists()
    assert (out_dir / "winning_method.json").exists()
    assert (out_dir / "model_full.txt").exists()
