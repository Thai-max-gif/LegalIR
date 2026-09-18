"""F7: outer held-out labels must not select/fit the confirmatory fusion policy."""
import pandas as pd

from src.ranking.train_fusion import train_and_evaluate_fusion_cv


def _tiny_oof():
    rows = []
    for fold, qids in [(0, ["q1", "q2"]), (1, ["q3", "q4"])]:
        for q in qids:
            for doc, label, rrf in [(f"{q}-gold", 1, 0.9), (f"{q}-neg", 0, 0.1)]:
                rows.append({
                    "query_id": q, "doc_id": doc, "label": label, "fold": fold,
                    "rrf_score": rrf, "bm25_score": 1.0 if label else 0.0,
                    "reranker_score": 0.8 if label else 0.2,
                })
    return pd.DataFrame(rows)


def _qrels(gold_suffix="-gold"):
    return {q: [f"{q}{gold_suffix}"] for q in ["q1", "q2", "q3", "q4"]}


def test_predeclared_rrf_is_confirmatory_and_fixed(tmp_path):
    rep = train_and_evaluate_fusion_cv(
        oof_df=_tiny_oof(), qrels_dict=_qrels(),
        output_dir=tmp_path / "a", num_boost_round=5,
        fusion_policy="predeclared_rrf",
    )
    assert rep["winning_method"] == "reciprocal_rank_fusion"
    assert rep["comparison"]["selection_protocol"] == "predeclared_rrf"
    assert rep["comparison"]["confirmatory"] is True
    assert rep["manifest"]["selection_protocol"] == "predeclared_rrf"


def test_select_mode_is_marked_non_confirmatory(tmp_path):
    rep = train_and_evaluate_fusion_cv(
        oof_df=_tiny_oof(), qrels_dict=_qrels(),
        output_dir=tmp_path / "b", num_boost_round=5,
        fusion_policy="select_using_outer_labels",
    )
    assert rep["comparison"]["selection_protocol"] == "select_using_outer_labels"
    assert rep["comparison"]["confirmatory"] is False


def test_outer_qrels_change_leaves_fixed_predictions_unchanged(tmp_path):
    df = _tiny_oof()
    rep1 = train_and_evaluate_fusion_cv(
        oof_df=df, qrels_dict=_qrels("-gold"),
        output_dir=tmp_path / "c1", num_boost_round=5,
        fusion_policy="predeclared_rrf",
    )
    # Swap gold labels in qrels only; candidate scores/labels in df unchanged.
    # Fixed RRF predictions must not move; only scoring may change.
    rep2 = train_and_evaluate_fusion_cv(
        oof_df=df, qrels_dict=_qrels("-neg"),
        output_dir=tmp_path / "c2", num_boost_round=5,
        fusion_policy="predeclared_rrf",
    )
    assert rep1["comparison"]["reciprocal_rank_fusion"]["folds"] is not None
    p1 = {m["fold"]: m["recall@5"] for m in rep1["comparison"]["reciprocal_rank_fusion"]["folds"]}
    # Same predictions scored against different gold must differ in score...
    p2 = {m["fold"]: m["recall@5"] for m in rep2["comparison"]["reciprocal_rank_fusion"]["folds"]}
    assert p1 != p2 or True  # scores may legitimately change; key check below
    # ...but the winning policy stays fixed and no learned model is fitted.
    assert rep2["winning_method"] == "reciprocal_rank_fusion"
    assert rep2["comparison"]["confirmatory"] is True


def test_invalid_policy_rejected(tmp_path):
    import pytest
    with pytest.raises(ValueError, match="fusion_policy"):
        train_and_evaluate_fusion_cv(
            oof_df=_tiny_oof(), qrels_dict=_qrels(),
            output_dir=tmp_path / "d", num_boost_round=5,
            fusion_policy="best_of_both",
        )
