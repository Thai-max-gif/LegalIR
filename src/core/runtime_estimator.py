"""
Runtime budget and telemetry estimator for LegalIR.
Uses measured real-hardware telemetry from Colab T4 smoke to project
Kaggle Final and Artifact Factory execution times with explicit uncertainty.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping


def estimate_kaggle_final_runtime(
    telemetry: Mapping[str, Any],
    safety_factor: float = 1.25,
    max_session_hours: float = 9.0,
) -> Dict[str, Any]:
    """
    Project Kaggle Final execution time based on measured hardware telemetry.
    Kaggle Final trains strictly ONE final BGE LoRA adapter (875 steps) on all 7,000 queries,
    reranks 1,000 public candidates under frozen fusion, and validates submission.
    """
    if telemetry.get("is_mock", False) or "Mock" in str(telemetry.get("gpu_name", "")):
        raise ValueError("Mock smoke telemetry cannot authorize final Kaggle runtime projection.")

    stage_timings = telemetry.get("stage_timings", {})
    reranker_training_sec = float(stage_timings.get("reranker_training_sec", 201.85))
    steps = int(telemetry.get("optimizer_steps", 10))
    if steps <= 0:
        raise ValueError(f"Optimizer steps in telemetry must be > 0, got {steps}")

    sec_per_step = reranker_training_sec / steps

    # Coverage-derived steps for 7,000 queries with effective batch 16:
    # 7000 queries / 16 batch * 2 epochs = 875 optimizer steps
    final_optimizer_steps = 875
    projected_training_sec = final_optimizer_steps * sec_per_step

    # Public reranking inference scaling:
    # Scale from measured execution on public queries subset
    subset_public = int(telemetry.get("subset_counts", {}).get("public_queries", 16))
    subset_eval_sec = float(stage_timings.get("prediction_eval_sec", 12.48))
    per_query_inference_sec = subset_eval_sec / max(1, subset_public)
    projected_inference_sec = per_query_inference_sec * 1000  # 1,000 public queries

    stages = {
        "final_reranker_training_sec": round(projected_training_sec, 2),
        "public_reranking_sec": round(projected_inference_sec, 2),
        "verification_and_packaging_sec": 30.0,
    }

    raw_total_sec = sum(stages.values())
    total_projected_sec = raw_total_sec * safety_factor
    total_projected_hours = total_projected_sec / 3600.0

    return {
        "sec_per_step": round(sec_per_step, 3),
        "optimizer_steps": final_optimizer_steps,
        "stages": stages,
        "raw_total_sec": round(raw_total_sec, 2),
        "safety_factor": safety_factor,
        "total_projected_sec": round(total_projected_sec, 2),
        "total_projected_hours": round(total_projected_hours, 2),
        "is_feasible_on_kaggle": total_projected_hours <= max_session_hours,
        "session_budget_hours": max_session_hours,
    }


def estimate_factory_runtime(
    telemetry: Mapping[str, Any],
    safety_factor: float = 1.20,
) -> Dict[str, Any]:
    """
    Project total Artifact Factory execution time across all 5 OOF folds,
    doc-disjoint validation, index construction, and candidate caching.
    """
    stage_timings = telemetry.get("stage_timings", {})
    reranker_training_sec = float(stage_timings.get("reranker_training_sec", 201.85))
    steps = int(telemetry.get("optimizer_steps", 10))
    sec_per_step = reranker_training_sec / max(1, steps)

    # 5 folds (5 * 700 steps = 3,500) + doc-disjoint (~700 steps) = 4,200 steps
    total_validation_steps = 4200
    training_time_sec = total_validation_steps * sec_per_step

    # Heavy retrieval & caching stages
    index_and_cache_sec = 3600.0

    raw_total_sec = training_time_sec + index_and_cache_sec
    total_projected_sec = raw_total_sec * safety_factor
    total_projected_hours = total_projected_sec / 3600.0

    return {
        "sec_per_step": round(sec_per_step, 3),
        "total_validation_steps": total_validation_steps,
        "training_time_sec": round(training_time_sec, 2),
        "index_and_cache_sec": index_and_cache_sec,
        "raw_total_sec": round(raw_total_sec, 2),
        "total_projected_hours": round(total_projected_hours, 2),
        "is_single_kaggle_session_feasible": total_projected_hours <= 9.0,
        "requires_resumable_sessions": True,
    }
