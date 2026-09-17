"""Fusion and Learning-to-Rank (LTR) algorithms for LegalIR candidate scoring.

Provides:
- ReciprocalRankFusion (RRF): Weighted reciprocal rank fusion over multiple retrieval branches.
- LightGBMRanker / LearnedRanker: Gradient boosted trees (LambdaRank) or linear fallback ranker.
- Model serialization, deterministic top-5 ranking, and inference.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

from collections.abc import Mapping
import json
from pathlib import Path
import pickle
from typing import Any
import numpy as np
import pandas as pd

from src.ranking.oof_features import (
    CORE_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    extract_candidate_features,
)


class ReciprocalRankFusion:
    """Weighted Reciprocal Rank Fusion over arbitrary retrieval branch ranks and scores."""

    DEFAULT_BRANCH_WEIGHTS = {
        "bm25": 1.0,
        "bm25_pyvi": 1.0,
        "dense": 1.2,
        "memory": 2.0,
        "exact": 2.5,
        "rerank": 1.8,
    }

    def __init__(
        self,
        k: int = 60,
        w_bm25: float = 1.0,
        w_pyvi: float = 1.0,
        w_exact: float = 2.5,
        w_memory: float = 2.0,
        w_dense: float = 1.2,
        w_rerank: float = 1.8,
        weights: Mapping[str, float] | None = None,
    ):
        self.k = int(k)
        if weights is not None:
            self.weights = dict(self.DEFAULT_BRANCH_WEIGHTS)
            for key, val in weights.items():
                self.weights[str(key)] = float(val)
        else:
            self.weights = {
                "bm25": float(w_bm25),
                "bm25_pyvi": float(w_pyvi),
                "exact": float(w_exact),
                "memory": float(w_memory),
                "dense": float(w_dense),
                "rerank": float(w_rerank),
            }

        # Convenience aliases
        self.w_bm25 = self.weights.get("bm25", 1.0)
        self.w_pyvi = self.weights.get("bm25_pyvi", 1.0)
        self.w_exact = self.weights.get("exact", 2.5)
        self.w_memory = self.weights.get("memory", 2.0)
        self.w_dense = self.weights.get("dense", 1.2)
        self.w_rerank = self.weights.get("rerank", 1.8)

    def rank_candidates(self, candidate_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Compute final fused score combining retrieval ranks + exact matches + reranker scores."""
        if not candidate_records:
            return []

        scored_records = []
        for c in candidate_records:
            score = 0.0

            # 1. Raw / Legal BM25 component
            raw_bm25_r = c.get("raw_bm25_rank", c.get("bm25_rank"))
            if raw_bm25_r is not None and float(raw_bm25_r) < 900.0:
                score += self.w_bm25 / (self.k + float(raw_bm25_r))

            # 2. PyVi BM25 component
            pyvi_bm25_r = c.get("pyvi_bm25_rank", c.get("bm25_pyvi_rank"))
            if pyvi_bm25_r is not None and float(pyvi_bm25_r) < 900.0:
                score += self.w_pyvi / (self.k + float(pyvi_bm25_r))

            # 3. Exact Match component
            exact_sc = float(c.get("exact_score", c.get("exact_match_score", 0.0)) or 0.0)
            if exact_sc > 0.0:
                score += self.w_exact * 0.05 * exact_sc

            # 4. Question Memory component
            mem_r = c.get("memory_rank")
            if mem_r is not None and float(mem_r) < 900.0:
                score += self.w_memory / (self.k + float(mem_r))

            # 5. Dense component
            dense_r = c.get("dense_rank")
            if dense_r is not None and float(dense_r) < 900.0:
                score += self.w_dense / (self.k + float(dense_r))

            # 6. Reranker component (calibrated with sigmoid)
            r_score_val = c.get("reranker_score", c.get("reranker_best_score"))
            if r_score_val is not None:
                r_score = float(r_score_val)
                if r_score > -900.0:
                    if -50.0 < r_score < 50.0:
                        sig_r = 1.0 / (1.0 + np.exp(-r_score))
                    elif r_score >= 50.0:
                        sig_r = 1.0
                    else:
                        sig_r = 0.0
                    score += self.w_rerank * 0.05 * sig_r

            # Fallback: if no individual ranks contributed, use base rrf_score
            if score == 0.0 and "rrf_score" in c and c["rrf_score"] is not None:
                score = float(c["rrf_score"])

            item = dict(c)
            item["final_score"] = float(score)
            scored_records.append(item)

        # Deterministic sort: descending final_score, ascending doc_id
        scored_records.sort(key=lambda x: (-x["final_score"], str(x.get("doc_id", ""))))
        return scored_records

    def predict(self, candidate_records: list[dict[str, Any]], **kwargs) -> list[dict[str, Any]]:
        """Uniform prediction interface matching LearnedRanker."""
        return self.rank_candidates(candidate_records)

    def rank_runs(
        self,
        run_list: list[list[dict[str, Any]]],
        weights: list[float] | None = None,
        key: str = "doc_id",
    ) -> list[dict[str, Any]]:
        """Classic Reciprocal Rank Fusion across multiple retrieval runs."""
        return reciprocal_rank_fusion(run_list, k=self.k, weights=weights, key=key)


class LinearRanker:
    """Linear ranking model fallback (using Scikit-Learn Ridge/Logistic regression)."""

    def __init__(self, feature_cols: list[str] | None = None, alpha: float = 1.0):
        self.alpha = float(alpha)
        self.model = None
        self.feature_cols = list(feature_cols) if feature_cols is not None else list(CORE_FEATURE_COLUMNS)

    def fit(self, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray | None = None, **kwargs):
        from sklearn.linear_model import Ridge
        available_cols = [c for c in self.feature_cols if c in X.columns]
        if not available_cols:
            available_cols = [c for c in CORE_FEATURE_COLUMNS if c in X.columns]
        self.feature_cols = available_cols

        X_mat = X[self.feature_cols].fillna(0.0).values
        y_arr = np.asarray(y, dtype=np.float32)

        self.model = Ridge(alpha=self.alpha, fit_intercept=True)
        self.model.fit(X_mat, y_arr)

    def predict(
        self,
        candidate_records: list[dict[str, Any]],
        query_id: str | None = None,
        query_text: str | None = None,
        doc_freq_map: Mapping[str, float] | None = None,
        **kwargs,
    ) -> list[dict[str, Any]]:
        if self.model is None or not candidate_records:
            return ReciprocalRankFusion().rank_candidates(candidate_records)

        df = extract_candidate_features(
            query_id=query_id,
            candidate_records=candidate_records,
            query_text=query_text,
            doc_freq_map=doc_freq_map,
        )
        for c in self.feature_cols:
            if c not in df.columns:
                df[c] = 0.0

        scores = self.model.predict(df[self.feature_cols].fillna(0.0).values)

        scored_records = []
        for c, sc in zip(candidate_records, scores):
            item = dict(c)
            item["final_score"] = float(sc)
            scored_records.append(item)

        scored_records.sort(key=lambda x: (-x["final_score"], str(x.get("doc_id", ""))))
        return scored_records

    def save(self, file_path: str | Path):
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_type": "linear_ridge",
            "feature_cols": self.feature_cols,
            "alpha": self.alpha,
            "coef": self.model.coef_.tolist() if self.model is not None else None,
            "intercept": float(self.model.intercept_) if self.model is not None else None,
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def load(self, file_path: str | Path):
        from sklearn.linear_model import Ridge
        file_path = Path(file_path)
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.feature_cols = payload.get("feature_cols", list(CORE_FEATURE_COLUMNS))
            self.alpha = float(payload.get("alpha", 1.0))
            if payload.get("coef") is not None:
                self.model = Ridge(alpha=self.alpha, fit_intercept=True)
                self.model.coef_ = np.array(payload["coef"], dtype=np.float32)
                self.model.intercept_ = float(payload.get("intercept", 0.0))


class LightGBMRanker:
    """LightGBM LambdaRank / GBDT ranker with early stopping and robust linear fallback."""

    def __init__(
        self,
        model_file: str | Path | None = None,
        feature_cols: list[str] | None = None,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        min_child_samples: int = 1,
        lambdarank_truncation_level: int = 5,
        strict: bool = False,
    ):
        self.model = None
        self.fallback_model: LinearRanker | None = None
        self.feature_cols = list(feature_cols) if feature_cols is not None else list(FEATURE_COLUMNS)
        self.learning_rate = float(learning_rate)
        self.num_leaves = int(num_leaves)
        self.min_child_samples = int(min_child_samples)
        self.lambdarank_truncation_level = int(lambdarank_truncation_level)
        self.model_type = "lightgbm"
        self.strict = bool(strict)

        if model_file is not None:
            self.load(model_file, strict=self.strict)

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        groups: np.ndarray,
        eval_set: tuple[pd.DataFrame, np.ndarray] | None = None,
        eval_groups: np.ndarray | None = None,
        num_boost_round: int = 100,
        early_stopping_rounds: int = 10,
    ) -> None:
        """Fit LambdaRank LightGBM model with optional early stopping on validation set."""
        available_cols = [c for c in self.feature_cols if c in X.columns]
        if not available_cols:
            available_cols = [c for c in CORE_FEATURE_COLUMNS if c in X.columns]
        self.feature_cols = available_cols

        try:
            import lightgbm as lgb

            train_data = lgb.Dataset(
                X[available_cols].fillna(0.0),
                label=np.asarray(y, dtype=np.float32),
                group=np.asarray(groups, dtype=np.int32),
                free_raw_data=False,
            )

            params = {
                "objective": "lambdarank",
                "lambdarank_truncation_level": self.lambdarank_truncation_level,
                "metric": "ndcg",
                "eval_at": [5],
                "learning_rate": self.learning_rate,
                "num_leaves": max(7, min(self.num_leaves, max(7, len(X)))),
                "min_child_samples": max(1, self.min_child_samples),
                "min_data_in_leaf": max(1, self.min_child_samples),
                "min_data_in_bin": 1,
                "verbose": -1,
                "seed": 42,
            }

            callbacks = []
            valid_sets = [train_data]
            valid_names = ["train"]

            if eval_set is not None and eval_groups is not None:
                X_val, y_val = eval_set
                val_data = lgb.Dataset(
                    X_val[available_cols].fillna(0.0),
                    label=np.asarray(y_val, dtype=np.float32),
                    group=np.asarray(eval_groups, dtype=np.int32),
                    reference=train_data,
                    free_raw_data=False,
                )
                valid_sets.append(val_data)
                valid_names.append("valid")
                if early_stopping_rounds > 0:
                    callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False))

            self.model = lgb.train(
                params,
                train_data,
                num_boost_round=num_boost_round,
                valid_sets=valid_sets,
                valid_names=valid_names,
                callbacks=callbacks if callbacks else None,
            )
            self.model_type = "lightgbm"
            self.fallback_model = None

        except Exception as e:
            if self.strict:
                raise RuntimeError(f"Strict LightGBM training failed: {e}") from e
            # Fallback to linear ranker if LightGBM fails or cannot be imported
            print(f"Warning: LightGBM training failed or unavailable ({e}); falling back to LinearRanker.")
            self.fallback_model = LinearRanker(feature_cols=available_cols)
            self.fallback_model.fit(X, y, groups)
            self.model = None
            self.model_type = "linear_fallback"

    def predict(
        self,
        candidate_records: list[dict[str, Any]],
        query_id: str | None = None,
        query_text: str | None = None,
        doc_freq_map: Mapping[str, float] | None = None,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Predict ranking scores for candidates and return sorted records."""
        if not candidate_records:
            return []

        if self.model is None and self.fallback_model is None:
            if getattr(self, "strict", False):
                raise RuntimeError("LightGBMRanker has no active loaded model or fallback model.")
            rrf = ReciprocalRankFusion()
            return rrf.rank_candidates(candidate_records)

        if self.fallback_model is not None:
            return self.fallback_model.predict(
                candidate_records,
                query_id=query_id,
                query_text=query_text,
                doc_freq_map=doc_freq_map,
                **kwargs,
            )

        df = extract_candidate_features(
            query_id=query_id,
            candidate_records=candidate_records,
            query_text=query_text,
            doc_freq_map=doc_freq_map,
        )
        available_cols = [c for c in self.feature_cols if c in df.columns]
        for c in self.feature_cols:
            if c not in df.columns:
                df[c] = 0.0

        scores = self.model.predict(df[self.feature_cols].fillna(0.0))

        scored_records = []
        for c, sc in zip(candidate_records, scores):
            item = dict(c)
            item["final_score"] = float(sc)
            scored_records.append(item)

        # Deterministic sort: descending final_score, ascending doc_id
        scored_records.sort(key=lambda x: (-x["final_score"], str(x.get("doc_id", ""))))
        return scored_records

    def save(self, file_path: str | Path):
        """Serialize ranker model and metadata to disk."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        meta_path = file_path.with_suffix(".json") if not str(file_path).endswith(".json") else file_path
        model_path = file_path.with_suffix(".txt") if not str(file_path).endswith(".txt") else file_path

        if self.model_type == "lightgbm" and self.model is not None:
            self.model.save_model(str(model_path))
            meta = {
                "model_type": "lightgbm",
                "model_file": str(model_path.name),
                "feature_cols": self.feature_cols,
                "learning_rate": self.learning_rate,
                "num_leaves": self.num_leaves,
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
        elif self.fallback_model is not None:
            self.fallback_model.save(file_path)
        else:
            # Empty ranker metadata
            meta = {
                "model_type": "empty",
                "feature_cols": self.feature_cols,
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

    def load(self, file_path: str | Path, strict: bool = False):
        """Load ranker model and metadata from disk."""
        file_path = Path(file_path)
        if not file_path.exists():
            if strict:
                raise FileNotFoundError(f"Ranker model file not found: {file_path}")
            return

        if file_path.is_dir():
            candidates = [
                file_path / "model.txt",
                file_path / "model_full.txt",
                file_path / "fusion_model.txt",
                file_path / "model.json",
                file_path / "model_full.json",
                file_path / "fusion_model.json",
            ]
            file_path = next((p for p in candidates if p.is_file()), file_path)
            if file_path.is_dir():
                txt_candidates = list(file_path.glob("*.txt"))
                json_candidates = list(file_path.glob("*.json"))
                if txt_candidates:
                    file_path = txt_candidates[0]
                elif json_candidates:
                    file_path = json_candidates[0]

        # 1. Inspect if content is JSON (e.g. linear_ridge metadata or lightgbm config)
        if file_path.is_file():
            try:
                content = file_path.read_text(encoding="utf-8").strip()
                if content.startswith("{") and content.endswith("}"):
                    meta = json.loads(content)
                    if meta.get("model_type") == "linear_ridge":
                        self.fallback_model = LinearRanker()
                        self.fallback_model.load(file_path)
                        self.model = None
                        self.model_type = "linear_ridge"
                        self.feature_cols = list(self.fallback_model.feature_cols)
                        return
                    elif meta.get("model_type") == "lightgbm" and "model_file" in meta:
                        m_target = file_path.parent / meta["model_file"]
                        if m_target.is_file():
                            file_path = m_target
            except Exception:
                pass

        # 2. Attempt LightGBM Booster loading
        if file_path.is_file():
            try:
                import lightgbm as lgb
                self.model = lgb.Booster(model_file=str(file_path))
                self.feature_cols = list(self.model.feature_name())
                self.model_type = "lightgbm"
                self.fallback_model = None
                return
            except Exception as e:
                if strict:
                    raise RuntimeError(f"Could not load LightGBM Booster from {file_path}: {e}") from e
                print(f"Warning: Could not load LightGBM Booster from {file_path}: {e}")


# Unified alias for LearnedRanker
LearnedRanker = LightGBMRanker


def reciprocal_rank_fusion(
    run_list: list[list[dict[str, Any]]],
    k: int = 60,
    weights: list[float] | None = None,
    key: str = "doc_id",
) -> list[dict[str, Any]]:
    """Classic Reciprocal Rank Fusion across multiple retrieval runs."""
    if not run_list:
        return []
    if weights is None:
        weights = [1.0 / len(run_list)] * len(run_list)

    scores: dict[str, float] = {}
    item_map: dict[str, dict[str, Any]] = {}

    for run_idx, run in enumerate(run_list):
        w = weights[run_idx] if run_idx < len(weights) else 1.0
        seen_in_run = set()
        for rank, item in enumerate(run, start=1):
            elem_key = str(item.get(key) or item.get("chunk_id") or "")
            if not elem_key or elem_key in seen_in_run:
                continue
            seen_in_run.add(elem_key)

            if elem_key not in item_map:
                item_map[elem_key] = dict(item)
            scores[elem_key] = scores.get(elem_key, 0.0) + w / (k + rank)

    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    fused = []
    for rank, (elem_key, score) in enumerate(ranked, start=1):
        elem = dict(item_map[elem_key])
        elem["rrf_score"] = float(score)
        elem["rank"] = rank
        fused.append(elem)
    return fused
