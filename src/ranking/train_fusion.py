"""Learned Fusion (Learning-to-Rank) and RRF Cross-Validation Trainer.

Provides:
- Fold-isolated cross-validation over OOF candidate features.
- Model selection gate comparing Learned Ranker vs Weighted RRF on Recall@5.
- Training on all folds and serialization of winning model artifacts.
"""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.core.config import load_pipeline_config
from src.core.hashing import sha256_file
from src.core.paths import ProjectPaths
from src.evaluation.evaluator import evaluate_predictions
from src.ranking.fusion import LearnedRanker, LightGBMRanker, ReciprocalRankFusion
from src.ranking.oof_features import (
    CORE_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    FEATURE_SCHEMA_VERSION,
    extract_candidate_features,
)
from src.ranking.selector import TopKSelector
from src.retrieval.bm25_micro import BM25MicroRetriever
from src.retrieval.exact_matcher import ExactMatcher
from src.retrieval.hybrid_search import HybridSearchEngine
from src.retrieval.question_memory import TrainQuestionMemory


def evaluate_features_with_ranker(
    val_df: pd.DataFrame,
    ranker: Any,
    qrels_dict: Mapping[str, list[str]],
    top_k: int = 5,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Rank candidates for each query in validation DataFrame and evaluate metrics."""
    if val_df.empty:
        return {}, {}

    selector = TopKSelector(max_k=top_k)
    y_pred: dict[str, list[str]] = {}
    grouped = val_df.groupby("query_id")

    for qid, group_df in grouped:
        qid_str = str(qid)
        cands = group_df.to_dict("records")

        if hasattr(ranker, "predict"):
            ranked = ranker.predict(cands, query_id=qid_str)
        elif hasattr(ranker, "rank_candidates"):
            ranked = ranker.rank_candidates(cands)
        else:
            ranked = cands

        top_docs = selector.select(ranked)
        y_pred[qid_str] = top_docs

    val_qrels = {str(qid): qrels_dict.get(str(qid), []) for qid in y_pred.keys()}
    metrics = evaluate_predictions(y_pred=y_pred, y_true=val_qrels)
    return y_pred, metrics


def train_and_evaluate_fusion_cv(
    oof_df: pd.DataFrame,
    qrels_dict: Mapping[str, list[str]],
    output_dir: str | Path | None = None,
    num_boost_round: int = 100,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    feature_cols: list[str] | None = None,
    rrf_weights: Mapping[str, float] | None = None,
    fusion_policy: str = "select_using_outer_labels",
    use_outer_eval_for_early_stop: bool = True,
) -> dict[str, Any]:
    """Execute 5-fold cross-validation for Learned Ranker vs Weighted RRF.

    Strict fold isolation: Fold f model is trained on folds != f only.

    F7 confirmatory protocol:
    - ``select_using_outer_labels`` (default, development only): current behavior;
      fits with outer val eval_set/early stopping and selects the winner on outer
      held-out quality. Results are selection-influenced development measurements,
      NOT an untouched confirmatory score. The report is marked accordingly.
    - ``predeclared_rrf`` (confirmation): fixed RRF policy. No LightGBM fitting,
      no outer-label early stopping, no winner selection. Outer labels are used
      only by the scorer after predictions are finalized.
    """
    if oof_df.empty or "fold" not in oof_df.columns:
        raise ValueError("oof_df must be non-empty and contain a 'fold' column.")

    unique_folds = sorted(oof_df["fold"].unique())
    if len(unique_folds) < 2:
        raise ValueError("Cross-fitted fusion requires at least 2 folds")

    output_dir = Path(output_dir) if output_dir else Path("artifacts/local/training/fusion")
    output_dir.mkdir(parents=True, exist_ok=True)

    available_cols = list(feature_cols) if feature_cols is not None else [
        c for c in FEATURE_COLUMNS if c in oof_df.columns
    ]
    if not available_cols:
        available_cols = [c for c in CORE_FEATURE_COLUMNS if c in oof_df.columns]

    target_col = "label" if "label" in oof_df.columns else "target"

    if fusion_policy not in ("select_using_outer_labels", "predeclared_rrf"):
        raise ValueError(
            f"Unknown fusion_policy {fusion_policy!r}; expected 'select_using_outer_labels' "
            "(development only) or 'predeclared_rrf' (confirmation)."
        )

    learned_fold_metrics = []
    rrf_fold_metrics = []
    trained_fold_models = {}

    print(f"\nEvaluating Fusion across {len(unique_folds)} folds...")
    if fusion_policy == "predeclared_rrf":
        print("[*] F7 fixed policy: predeclared RRF; no outer-label fitting/selection.")

    rrf_baseline = ReciprocalRankFusion(weights=rrf_weights)

    all_learned_preds: dict[str, list[str]] = {}
    all_rrf_preds: dict[str, list[str]] = {}

    for f_idx in unique_folds:
        train_mask = oof_df["fold"] != f_idx
        val_mask = oof_df["fold"] == f_idx

        train_data = oof_df[train_mask]
        val_data = oof_df[val_mask]

        if fusion_policy == "predeclared_rrf":
            # Fixed-policy confirmation: RRF predictions only. Outer labels reach
            # only the scorer after predictions are finalized.
            rrf_p, rrf_m = evaluate_features_with_ranker(val_data, rrf_baseline, qrels_dict)
            rrf_m["fold"] = int(f_idx)
            rrf_fold_metrics.append(rrf_m)
            all_rrf_preds.update(rrf_p)
            print(f"Fold {f_idx}: RRF Recall@5 = {rrf_m['recall@5'] * 100:.2f}% (Prec@5 = {rrf_m['precision@5'] * 100:.2f}%)")
            continue

        # 1. Fit Fold-Isolated Learned Ranker (development only; outer val labels
        # influence early stopping and winner selection -- see report flags).
        # Note: LightGBM requires groups sorted consecutively by query_id
        train_data_sorted = train_data.sort_values("query_id")
        val_data_sorted = val_data.sort_values("query_id")

        train_groups = train_data_sorted.groupby("query_id", sort=False).size().values
        val_groups = val_data_sorted.groupby("query_id", sort=False).size().values

        X_train = train_data_sorted[available_cols]
        y_train = train_data_sorted[target_col].values

        X_val = val_data_sorted[available_cols]
        y_val = val_data_sorted[target_col].values

        ranker = LightGBMRanker(
            feature_cols=available_cols,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
        )
        if use_outer_eval_for_early_stop:
            ranker.fit(
                X=X_train,
                y=y_train,
                groups=train_groups,
                eval_set=(X_val, y_val),
                eval_groups=val_groups,
                num_boost_round=num_boost_round,
                early_stopping_rounds=10,
            )
        else:
            ranker.fit(
                X=X_train,
                y=y_train,
                groups=train_groups,
                num_boost_round=num_boost_round,
            )

        fold_model_file = output_dir / f"model_fold_{f_idx}.txt"
        ranker.save(fold_model_file)
        trained_fold_models[f"fold_{f_idx}"] = str(fold_model_file)

        # 2. Evaluate Learned Ranker on Validation Fold f
        learned_p, learned_m = evaluate_features_with_ranker(val_data, ranker, qrels_dict)
        learned_m["fold"] = int(f_idx)
        learned_fold_metrics.append(learned_m)
        all_learned_preds.update(learned_p)

        # 3. Evaluate Weighted RRF on Validation Fold f
        rrf_p, rrf_m = evaluate_features_with_ranker(val_data, rrf_baseline, qrels_dict)
        rrf_m["fold"] = int(f_idx)
        rrf_fold_metrics.append(rrf_m)
        all_rrf_preds.update(rrf_p)

        print(
            f"Fold {f_idx}: "
            f"Learned Recall@5 = {learned_m['recall@5'] * 100:.2f}% (Prec@5 = {learned_m['precision@5'] * 100:.2f}%) | "
            f"RRF Recall@5 = {rrf_m['recall@5'] * 100:.2f}% (Prec@5 = {rrf_m['precision@5'] * 100:.2f}%)"
        )

    # Concat predictions across all held-out folds and evaluate full OOF metrics.
    # Outer labels reach ONLY the scorer here, never fitting/selection in fixed mode.
    if fusion_policy == "predeclared_rrf":
        all_val_qrels = {str(qid): qrels_dict.get(str(qid), []) for qid in all_rrf_preds.keys()}
        learned_overall = {"recall@5": 0.0, "precision@5": 0.0}
        rrf_overall = evaluate_predictions(y_pred=all_rrf_preds, y_true=all_val_qrels)
    else:
        all_val_qrels = {str(qid): qrels_dict.get(str(qid), []) for qid in all_learned_preds.keys()}
        learned_overall = evaluate_predictions(y_pred=all_learned_preds, y_true=all_val_qrels)
        rrf_overall = evaluate_predictions(y_pred=all_rrf_preds, y_true=all_val_qrels)

    learned_overall_rec5 = float(learned_overall.get("recall@5", 0.0))
    learned_overall_prec5 = float(learned_overall.get("precision@5", 0.0))
    rrf_overall_rec5 = float(rrf_overall.get("recall@5", 0.0))
    rrf_overall_prec5 = float(rrf_overall.get("precision@5", 0.0))

    # Cross-fold summary metrics
    learned_mean_rec5 = float(np.mean([m["recall@5"] for m in learned_fold_metrics])) if learned_fold_metrics else 0.0
    learned_std_rec5 = float(np.std([m["recall@5"] for m in learned_fold_metrics])) if learned_fold_metrics else 0.0
    learned_mean_prec5 = float(np.mean([m["precision@5"] for m in learned_fold_metrics])) if learned_fold_metrics else 0.0

    rrf_mean_rec5 = float(np.mean([m["recall@5"] for m in rrf_fold_metrics])) if rrf_fold_metrics else 0.0
    rrf_std_rec5 = float(np.std([m["recall@5"] for m in rrf_fold_metrics])) if rrf_fold_metrics else 0.0
    rrf_mean_prec5 = float(np.mean([m["precision@5"] for m in rrf_fold_metrics])) if rrf_fold_metrics else 0.0

    if fusion_policy == "predeclared_rrf":
        learned_wins = False
        selection_protocol = "predeclared_rrf"
        confirmatory = True
    else:
        # Model Selection Gate (development only; uses outer held-out quality).
        # Primary criterion: Official Task 1 Recall@5 across concatenated held-out folds
        learned_wins = (
            learned_overall_rec5 > rrf_overall_rec5
            or (np.isclose(learned_overall_rec5, rrf_overall_rec5, atol=1e-6) and learned_overall_prec5 > rrf_overall_prec5)
        )
        selection_protocol = "select_using_outer_labels"
        confirmatory = False

    # 4. Train Final Model on All Folds (skipped for fixed RRF: nothing to fit;
    # fitting on all labels belongs to post-confirmation production, not scoring).
    full_ranker = None
    if fusion_policy == "predeclared_rrf":
        print("\nSkipping final LightGBM fit for predeclared RRF (no learned model).")
    else:
        print("\nTraining Final Fusion Model on All Folds...")
        all_sorted = oof_df.sort_values("query_id")
        all_groups = all_sorted.groupby("query_id", sort=False).size().values
        all_X = all_sorted[available_cols]
        all_y = all_sorted[target_col].values

        full_ranker = LightGBMRanker(
            feature_cols=available_cols,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
        )
        full_ranker.fit(all_X, all_y, all_groups, num_boost_round=num_boost_round)

        full_model_file = output_dir / "model_full.txt"
        full_ranker.save(full_model_file)
        trained_fold_models["full"] = str(full_model_file)

        # If output_dir is not already fusion_final, also export to checkpoints/fusion_final
        fusion_final_dir = output_dir / "fusion_final" if output_dir.name != "fusion_final" else output_dir
        fusion_final_dir.mkdir(parents=True, exist_ok=True)
        full_ranker.save(fusion_final_dir / "model.txt")
        trained_fold_models["fusion_final"] = str(fusion_final_dir / "model.txt")

    if fusion_policy == "predeclared_rrf":
        winning_method = "reciprocal_rank_fusion"
        winning_model_type = "rrf_weighted"
        winner_rec5 = rrf_overall_rec5
        winner_prec5 = rrf_overall_prec5
        gate_decision = "Predeclared fixed RRF policy (no outer-label fitting/selection; confirmatory)"
    elif learned_wins:
        winning_method = "learned_ranker"
        winning_model_type = getattr(full_ranker, "model_type", "lightgbm") if full_ranker is not None else "lightgbm"
        winner_rec5 = learned_overall_rec5
        winner_prec5 = learned_overall_prec5
        gate_decision = f"Learned Ranker selected (+{(learned_overall_rec5 - rrf_overall_rec5) * 100:.4f}% Recall@5 vs RRF)"
    else:
        winning_method = "reciprocal_rank_fusion"
        winning_model_type = "rrf_weighted"
        winner_rec5 = rrf_overall_rec5
        winner_prec5 = rrf_overall_prec5
        gate_decision = f"Weighted RRF selected (+{(rrf_overall_rec5 - learned_overall_rec5) * 100:.4f}% Recall@5 vs Learned)"

    print("\n" + "=" * 70)
    print(">> FUSION MODEL SELECTION GATE SUMMARY (5-Fold Cross-Fitted):")
    if fusion_policy == "predeclared_rrf":
        print("   Policy: PREDECLARED RRF (confirmatory; no outer-label fitting/selection)")
    else:
        print("   Policy: SELECT_USING_OUTER_LABELS (development only; NOT confirmatory)")
    print(f"   Learned Ranker Full OOF Recall@5 : {learned_overall_rec5 * 100:.4f}% (Mean across folds: {learned_mean_rec5 * 100:.4f}% +/- {learned_std_rec5 * 100:.4f}%)")
    print(f"   Weighted RRF   Full OOF Recall@5 : {rrf_overall_rec5 * 100:.4f}% (Mean across folds: {rrf_mean_rec5 * 100:.4f}% +/- {rrf_std_rec5 * 100:.4f}%)")
    print(f"   Winning Method                   : {winning_method} ({gate_decision})")
    print(f"   Winning Model Type               : {winning_model_type}")
    print("=" * 70)

    # 5. Export artifacts & manifests
    comparison_report = {
        "winning_method": winning_method,
        "winning_model_type": winning_model_type,
        "selection_protocol": selection_protocol,
        "confirmatory": confirmatory,
        "fusion_policy": fusion_policy,
        "use_outer_eval_for_early_stop": bool(use_outer_eval_for_early_stop),
        "winner_mean_recall@5": winner_rec5,
        "winner_mean_precision@5": winner_prec5,
        "gate_decision": gate_decision,
        "learned_ranker": {
            "overall_recall@5": learned_overall_rec5,
            "overall_precision@5": learned_overall_prec5,
            "mean_recall@5": learned_mean_rec5,
            "std_recall@5": learned_std_rec5,
            "mean_precision@5": learned_mean_prec5,
            "folds": learned_fold_metrics,
        },
        "reciprocal_rank_fusion": {
            "overall_recall@5": rrf_overall_rec5,
            "overall_precision@5": rrf_overall_prec5,
            "mean_recall@5": rrf_mean_rec5,
            "std_recall@5": rrf_std_rec5,
            "mean_precision@5": rrf_mean_prec5,
            "folds": rrf_fold_metrics,
        },
    }

    with open(output_dir / "fusion_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison_report, f, indent=2)

    manifest = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "feature_training_stage": "post_rerank",
        "feature_columns": available_cols,
        "total_oof_rows": len(oof_df),
        "winning_method": winning_method,
        "winning_model_type": winning_model_type,
        "selection_protocol": selection_protocol,
        "confirmatory": confirmatory,
        "fusion_policy": fusion_policy,
        "winning_metrics": {
            "mean_recall@5": winner_rec5,
            "mean_precision@5": winner_prec5,
        },
        "models": trained_fold_models,
    }

    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    with open(output_dir / "winning_method.json", "w", encoding="utf-8") as f:
        json.dump({"winning_method": winning_method, "decision": gate_decision}, f, indent=2)

    # Standard production fusion_model.json descriptor
    fusion_descriptor: dict[str, Any] = {
        "schema_version": 1,
        "winning_method": winning_method,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_columns": available_cols,
    }
    if winning_method == "reciprocal_rank_fusion":
        fusion_descriptor["rrf"] = {
            "k": int(getattr(rrf_baseline, "k", 60)),
            "weights": {str(k): float(v) for k, v in getattr(rrf_baseline, "weights", rrf_weights or {}).items()},
        }
    else:
        model_filename = "fusion_model.txt"
        model_payload_path = output_dir / model_filename
        if (output_dir / "model_full.txt").is_file():
            import shutil
            shutil.copy2(output_dir / "model_full.txt", model_payload_path)
        else:
            full_ranker.save(model_payload_path)

        payload_sha = sha256_file(model_payload_path)
        fusion_descriptor["learned_model"] = {
            "file": model_filename,
            "sha256": payload_sha,
        }

    with open(output_dir / "fusion_model.json", "w", encoding="utf-8") as f:
        json.dump(fusion_descriptor, f, indent=2, sort_keys=True)

    return {
        "manifest": manifest,
        "comparison": comparison_report,
        "winning_method": winning_method,
        "winner_mean_recall@5": winner_rec5,
        "fusion_descriptor": fusion_descriptor,
    }


def train_fusion_models(
    config_path: str | Path = "configs/pipeline.yaml",
    output_dir: str | Path | None = None,
    candidate_k: int = 50,
    oof_features_path: str | Path | None = None,
) -> dict[str, Any]:
    """Orchestrate OOF feature extraction and fusion training from canonical datasets."""
    paths = ProjectPaths.from_repo()
    cfg = load_pipeline_config(Path(config_path))

    canonical_dir = paths.canonical
    output_dir = Path(output_dir) if output_dir else paths.repo / "artifacts" / "local" / "training" / "fusion"
    output_dir.mkdir(parents=True, exist_ok=True)

    qrels_path = canonical_dir / "qrels_train.parquet"
    qrels_df = pd.read_parquet(qrels_path)
    qrels_dict = qrels_df.groupby("query_id")["doc_id"].apply(lambda s: [str(x) for x in s]).to_dict()

    if oof_features_path is not None and Path(oof_features_path).exists():
        print(f"Loading precomputed OOF features from {oof_features_path}...")
        oof_df = pd.read_parquet(oof_features_path)
        return train_and_evaluate_fusion_cv(
            oof_df=oof_df,
            qrels_dict=qrels_dict,
            output_dir=output_dir,
        )

    print(f"Loading canonical data for fusion training from {canonical_dir}...")
    docs_df = pd.read_parquet(canonical_dir / "documents.parquet")
    chunks_df = pd.read_parquet(canonical_dir / "chunks.parquet")
    queries_df = pd.read_parquet(canonical_dir / "queries_train.parquet")

    splits_dir = canonical_dir / "splits"
    random_5fold = json.loads((splits_dir / "random_5fold.json").read_text(encoding="utf-8"))

    queries_dict = dict(zip(queries_df["query_id"].astype(str), queries_df["question_norm"]))

    bm25_path = paths.local_indexes / "bm25" / "bm25_micro_index.pkl"
    if bm25_path.exists():
        bm25 = BM25MicroRetriever.load(bm25_path)
    else:
        micro_chunks = chunks_df[chunks_df["granularity"] == "micro"].to_dict(orient="records")
        bm25 = BM25MicroRetriever().fit(micro_chunks)

    exact = ExactMatcher(docs_df.to_dict(orient="records"))

    # Generate features for all 5 folds
    fold_feature_dfs = []
    for f_idx, fold_data in enumerate(random_5fold):
        val_qids = [str(x) for x in fold_data.get("val_query_ids", fold_data.get("val", []))]
        train_qids = [str(x) for x in fold_data.get("train_query_ids", fold_data.get("train", []))]

        fold_train_queries = {qid: queries_dict[qid] for qid in train_qids if qid in queries_dict}
        fold_train_qrels = {qid: qrels_dict[qid] for qid in train_qids if qid in qrels_dict}

        memory = TrainQuestionMemory(min_similarity=0.82)
        memory.fit(fold_train_queries, fold_train_qrels)

        engine = HybridSearchEngine(
            bm25_retriever=bm25,
            exact_matcher=exact,
            question_memory=memory,
            dense_retriever=None,
        )

        print(f"Extracting OOF features for fold {f_idx + 1}/5 ({len(val_qids)} val queries)...")
        for qid in tqdm(val_qids, desc=f"Fold {f_idx} Features", leave=False):
            q_text = queries_dict.get(qid, "")
            gold_set = set(qrels_dict.get(qid, []))
            if not q_text or not gold_set:
                continue

            cands = engine.search_candidates(q_text, exclude_qid=qid, top_k=candidate_k)
            if not cands:
                continue

            df = extract_candidate_features(query_id=qid, candidate_records=cands, query_text=q_text, qrels=gold_set)
            df["fold"] = f_idx
            fold_feature_dfs.append(df)

    oof_df = pd.concat(fold_feature_dfs, ignore_index=True)
    oof_df.to_parquet(output_dir / "oof_features.parquet", index=False)
    print(f"Total OOF feature rows: {len(oof_df)}")

    return train_and_evaluate_fusion_cv(
        oof_df=oof_df,
        qrels_dict=qrels_dict,
        output_dir=output_dir,
    )


def main():
    parser = argparse.ArgumentParser(description="LegalIR LightGBM Fusion Trainer")
    parser.add_argument("--config", type=str, default="configs/pipeline.yaml")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--oof-features", type=str, default=None, help="Path to existing oof_features.parquet")
    args = parser.parse_args()

    train_fusion_models(
        config_path=args.config,
        output_dir=args.output_dir,
        candidate_k=args.candidate_k,
        oof_features_path=args.oof_features,
    )


if __name__ == "__main__":
    main()
