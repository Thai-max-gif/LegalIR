"""OOF evaluation metric aggregation, score promotion policy, and production lock."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from src.core.hashing import sha256_string

HEX_40_RE = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)
HEX_64_RE = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)


def compare_score_promotion(
    candidate: Dict[str, float],
    baseline: Dict[str, float],
    max_candidate_recall_drop: float = 0.02,
) -> Tuple[bool, str]:
    """
    Evaluate candidate configuration against baseline according to strict promotion rules:
    1. Candidate Recall@50/150 must not materially regress.
    2. Primary: Recall@5 must improve.
    3. Tie-break: If Recall@5 ties, Precision@5 must improve.
    """
    c_cand_rec = candidate.get("candidate_recall@150", 1.0)
    b_cand_rec = baseline.get("candidate_recall@150", 1.0)
    if b_cand_rec - c_cand_rec > max_candidate_recall_drop:
        return False, f"Candidate recall regressed by {b_cand_rec - c_cand_rec:.4f} > {max_candidate_recall_drop}"

    c_rec5 = candidate.get("recall@5", 0.0)
    b_rec5 = baseline.get("recall@5", 0.0)
    c_prec5 = candidate.get("precision@5", 0.0)
    b_prec5 = baseline.get("precision@5", 0.0)

    if c_rec5 > b_rec5:
        return True, f"Recall@5 improved ({c_rec5:.4f} > {b_rec5:.4f})"
    elif c_rec5 == b_rec5:
        if c_prec5 > b_prec5:
            return True, f"Recall@5 tied and Precision@5 improved ({c_prec5:.4f} > {b_prec5:.4f})"
        else:
            return False, f"Recall@5 tied but Precision@5 did not improve ({c_prec5:.4f} <= {b_prec5:.4f})"
    else:
        return False, f"Recall@5 regressed ({c_rec5:.4f} < {b_rec5:.4f})"


def aggregate_oof_metrics(fold_metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """Compute macro-average OOF validation metrics across all folds."""
    if not fold_metrics_list:
        return {}

    keys = fold_metrics_list[0].keys()
    agg: Dict[str, float] = {}
    for k in keys:
        vals = [m[k] for m in fold_metrics_list if k in m]
        if vals:
            agg[k] = round(sum(vals) / len(vals), 6)
    return agg


def create_production_lock(
    output_path: Union[str, Path],
    metrics: Dict[str, Any],
    config: Dict[str, Any],
    runtime_commit: str,
    dataset_sha256: Optional[str] = None,
    strict: bool = False,
) -> None:
    """Freeze the approved production configuration into an immutable production_lock.json."""
    output_p = Path(output_path)
    output_p.parent.mkdir(parents=True, exist_ok=True)

    runtime_commit = str(runtime_commit).strip()
    dataset_sha = str(dataset_sha256 or "").strip()

    if strict:
        if not HEX_40_RE.match(runtime_commit):
            raise ValueError(
                f"runtime_commit must be a real 40-char git commit SHA, got: '{runtime_commit}'"
            )
        if not HEX_64_RE.match(dataset_sha):
            raise ValueError(
                f"dataset_sha256 must be a real 64-char SHA-256 digest, got: '{dataset_sha}'"
            )
    else:
        if not dataset_sha:
            dataset_sha = "0" * 64

    config_str = json.dumps(config, sort_keys=True)
    lock_data = {
        "status": "LOCKED",
        "runtime_commit": runtime_commit,
        "dataset_sha256": dataset_sha,
        "config_sha256": sha256_string(config_str),
        "metrics": metrics,
        "config": config,
    }

    with open(output_p, "w", encoding="utf-8") as f:
        json.dump(lock_data, f, indent=2, sort_keys=True)
