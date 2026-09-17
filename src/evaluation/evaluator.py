from collections.abc import Iterable as IterableABC
from collections.abc import Mapping
from typing import Any, Iterable
import numpy as np


DEFAULT_CANDIDATE_CUTOFFS = (20, 50, 100, 150, 200)
FINAL_RANKING_METRICS = ("recall_at_1", "recall_at_3", "recall_at_5", "precision_at_5")
ALL_EVALUATION_METRICS = (
    "recall_at_1",
    "recall_at_3",
    "recall_at_5",
    "precision_at_1",
    "precision_at_3",
    "precision_at_5",
    "mrr",
    "map",
    "ndcg_at_5",
)


def _stringify_id(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _normalize_ids(value: Any) -> list[str]:
    """Normalize a prediction or qrel value to an ordered list of IDs."""
    if isinstance(value, Mapping):
        for key in ("answer", "doc_ids", "doc_id", "document_id"):
            if key in value:
                return _normalize_ids(value[key])
        return []
    if isinstance(value, (str, bytes)):
        return [_stringify_id(value)]
    if value is None:
        return []

    try:
        values = list(value)
    except TypeError:
        values = [value]
    return [_stringify_id(item) for item in values if item is not None]


def _normalize_query_values(values: Mapping[Any, Any]) -> dict[str, list[str]]:
    return {str(query_id): _normalize_ids(value) for query_id, value in values.items()}


def _candidate_doc_id(candidate: Any) -> str | None:
    if isinstance(candidate, Mapping):
        candidate = candidate.get("doc_id", candidate.get("document_id"))
    elif isinstance(candidate, (tuple, list)):
        candidate = candidate[0] if candidate else None
    if candidate is None:
        return None
    return _stringify_id(candidate)


def normalize_candidate_cutoffs(cutoffs: Iterable[int] | int | str | None = None) -> list[int]:
    """Return validated, de-duplicated candidate cutoffs in caller order."""
    if cutoffs is None:
        values: Iterable[int] = DEFAULT_CANDIDATE_CUTOFFS
    elif isinstance(cutoffs, (str, bytes)) or not isinstance(cutoffs, IterableABC):
        values = (cutoffs,)
    else:
        values = cutoffs

    normalized: list[int] = []
    for cutoff in values:
        if isinstance(cutoff, bool):
            raise ValueError("candidate cutoffs must be positive integers")
        try:
            cutoff_int = int(cutoff)
        except (TypeError, ValueError) as exc:
            raise ValueError("candidate cutoffs must be positive integers") from exc
        if cutoff_int <= 0:
            raise ValueError("candidate cutoffs must be positive integers")
        if cutoff_int not in normalized:
            normalized.append(cutoff_int)
    if not normalized:
        raise ValueError("at least one candidate cutoff is required")
    return normalized


def evaluate_predictions(
    y_pred: dict[str, Any],
    y_true: dict[str, Any],
    candidate_pools: dict[str, Any] | None = None,
    runtimes: dict[str, float] | list[float] | float | None = None,
    cutoffs: Iterable[int] | None = None,
) -> dict[str, Any]:
    """
    Exact Codabench retrieval evaluation function with full metric reporting:
    - Official Codabench Recall & Precision
    - Recall@1, Recall@3, Recall@5
    - Precision@1, Precision@3, Precision@5
    - Candidate Recall@20, @50, @100, @150, @200 (if candidate_pools provided)
    - MRR, MAP, nDCG@5
    - Runtime per query (if runtimes provided)
    """
    normalized_pred = _normalize_query_values(y_pred)
    normalized_true = _normalize_query_values(y_true)

    recalls = []
    precisions = []
    r1_list = []
    r3_list = []
    r5_list = []
    p1_list = []
    p3_list = []
    p5_list = []
    mrr_list = []
    map_list = []
    ndcg5_list = []

    for qid, gold_list in normalized_true.items():
        gold_set = set(gold_list)
        if not gold_set:
            continue

        preds = normalized_pred.get(qid, [])
        pred_set = set(preds)

        # Codabench rule: if 0 < len(preds) <= 5, compute set intersection ratio, else 0
        if 0 < len(preds) <= 5:
            rec = len(gold_set & pred_set) / len(gold_set)
            prec = len(gold_set & pred_set) / len(preds)

            top1 = set(preds[:1])
            top3 = set(preds[:3])
            top5 = set(preds[:5])
            r1 = len(gold_set & top1) / len(gold_set)
            r3 = len(gold_set & top3) / len(gold_set)
            r5 = rec
            p1 = len(gold_set & top1) / len(preds[:1]) if len(preds[:1]) > 0 else 0.0
            p3 = len(gold_set & top3) / len(preds[:3]) if len(preds[:3]) > 0 else 0.0
            p5 = prec

            # MRR (Mean Reciprocal Rank @ 5)
            rr = 0.0
            for rank, doc_id in enumerate(preds[:5], start=1):
                if doc_id in gold_set:
                    rr = 1.0 / rank
                    break

            # MAP (Mean Average Precision @ 5)
            ap = 0.0
            hits = 0
            for rank, doc_id in enumerate(preds[:5], start=1):
                if doc_id in gold_set:
                    hits += 1
                    ap += hits / rank
            map_score = ap / len(gold_set) if len(gold_set) > 0 else 0.0

            # nDCG@5 (Normalized Discounted Cumulative Gain @ 5)
            dcg = 0.0
            for rank, doc_id in enumerate(preds[:5], start=1):
                if doc_id in gold_set:
                    dcg += 1.0 / np.log2(rank + 1.0)
            idcg = sum(1.0 / np.log2(i + 1.0) for i in range(1, min(len(gold_set), 5) + 1))
            ndcg_score = float(dcg / idcg) if idcg > 0.0 else 0.0
        else:
            rec = 0.0
            prec = 0.0
            r1 = 0.0
            r3 = 0.0
            r5 = 0.0
            p1 = 0.0
            p3 = 0.0
            p5 = 0.0
            rr = 0.0
            map_score = 0.0
            ndcg_score = 0.0

        recalls.append(rec)
        precisions.append(prec)
        r1_list.append(r1)
        r3_list.append(r3)
        r5_list.append(r5)
        p1_list.append(p1)
        p3_list.append(p3)
        p5_list.append(p5)
        mrr_list.append(rr)
        map_list.append(map_score)
        ndcg5_list.append(ndcg_score)

    mean_rec = float(np.mean(recalls)) if recalls else 0.0
    mean_prec = float(np.mean(precisions)) if precisions else 0.0
    mean_r1 = float(np.mean(r1_list)) if r1_list else 0.0
    mean_r3 = float(np.mean(r3_list)) if r3_list else 0.0
    mean_r5 = float(np.mean(r5_list)) if r5_list else 0.0
    mean_p1 = float(np.mean(p1_list)) if p1_list else 0.0
    mean_p3 = float(np.mean(p3_list)) if p3_list else 0.0
    mean_p5 = float(np.mean(p5_list)) if p5_list else 0.0
    mean_mrr = float(np.mean(mrr_list)) if mrr_list else 0.0
    mean_map = float(np.mean(map_list)) if map_list else 0.0
    mean_ndcg5 = float(np.mean(ndcg5_list)) if ndcg5_list else 0.0

    metrics: dict[str, Any] = {
        # Codabench-compatible aggregate names.
        "recall": mean_rec,
        "precision": mean_prec,
        # Machine-friendly final ranking metric names.
        "recall_at_1": mean_r1,
        "recall_at_3": mean_r3,
        "recall_at_5": mean_r5,
        "precision_at_1": mean_p1,
        "precision_at_3": mean_p3,
        "precision_at_5": mean_prec,
        # Public report aliases used by the benchmark acceptance contract.
        "recall@1": mean_r1,
        "recall@3": mean_r3,
        "recall@5": mean_r5,
        "precision@1": mean_p1,
        "precision@3": mean_p3,
        "precision@5": mean_prec,
        "Recall@1": mean_r1,
        "Recall@3": mean_r3,
        "Recall@5": mean_r5,
        "Precision@1": mean_p1,
        "Precision@3": mean_p3,
        "Precision@5": mean_prec,
        # IR Ranking metrics
        "mrr": mean_mrr,
        "MRR": mean_mrr,
        "mrr@5": mean_mrr,
        "map": mean_map,
        "MAP": mean_map,
        "map@5": mean_map,
        "ndcg_at_5": mean_ndcg5,
        "ndcg@5": mean_ndcg5,
        "nDCG@5": mean_ndcg5,
        "total_evaluated_queries": len(recalls),
    }

    # Candidate Recall cutoffs and Oracle metrics if candidate_pools provided
    if candidate_pools is not None:
        cand_cutoffs = normalize_candidate_cutoffs(cutoffs)
        cand_metrics = compute_candidate_cutoffs(candidate_pools, y_true, cutoffs=cand_cutoffs)
        metrics.update(cand_metrics)
        for k, v in cand_metrics.items():
            num = k.split("@")[-1]
            metrics[f"cand@{num}"] = v
            metrics[f"Candidate Recall@{num}"] = v

        oracle_metrics = compute_top5_oracle(candidate_pools, y_true)
        metrics.update(oracle_metrics)
        metrics["oracle_recall@5"] = oracle_metrics.get("candidate_pool_oracle@5", 0.0)
        metrics["Oracle Recall@5"] = oracle_metrics.get("candidate_pool_oracle@5", 0.0)

    # Runtime metrics if runtimes provided
    if runtimes is not None:
        num_q = max(1, len(recalls))
        if isinstance(runtimes, Mapping):
            times = [float(v) for v in runtimes.values()]
            total_sec = float(sum(times))
            mean_sec = float(np.mean(times)) if times else 0.0
        elif isinstance(runtimes, (list, tuple, np.ndarray)):
            times = [float(v) for v in runtimes]
            total_sec = float(sum(times))
            mean_sec = float(np.mean(times)) if times else 0.0
        elif isinstance(runtimes, (int, float)):
            total_sec = float(runtimes)
            mean_sec = total_sec / num_q
        else:
            total_sec = 0.0
            mean_sec = 0.0

        metrics["runtime_per_query_seconds"] = mean_sec
        metrics["runtime_per_query_ms"] = mean_sec * 1000.0
        metrics["runtime_per_query"] = mean_sec
        metrics["total_runtime_seconds"] = total_sec
        metrics["queries_per_second"] = (num_q / total_sec) if total_sec > 0 else 0.0

    return metrics


def compute_candidate_recall(
    candidates: dict[str, Any],
    ground_truths: dict[str, Any],
    k: int = 50,
) -> float:
    """Compute candidate recall at ``k`` before final reranking/fusion."""
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k <= 0:
        raise ValueError("candidate recall cutoff must be a positive integer")

    normalized_candidates = {str(qid): value for qid, value in candidates.items()}
    normalized_truths = _normalize_query_values(ground_truths)
    recalls = []
    for qid, gold_list in normalized_truths.items():
        gold_set = set(gold_list)
        if not gold_set:
            continue

        raw_candidates = normalized_candidates.get(qid, [])
        if isinstance(raw_candidates, (str, bytes, Mapping)):
            raw_candidates = _normalize_ids(raw_candidates)
        else:
            try:
                raw_candidates = list(raw_candidates)
            except TypeError:
                raw_candidates = _normalize_ids(raw_candidates)
        candidate_ids = [
            doc_id
            for doc_id in (_candidate_doc_id(candidate) for candidate in raw_candidates[:k])
            if doc_id is not None
        ]
        recalls.append(len(gold_set & set(candidate_ids)) / len(gold_set))

    return float(np.mean(recalls)) if recalls else 0.0


def compute_candidate_cutoffs(
    candidates: dict[str, Any],
    ground_truths: dict[str, Any],
    cutoffs: Iterable[int] | None = None,
) -> dict[str, float]:
    """Compute candidate recall for diagnostic cutoffs (e.g. 20, 50, 100, 150, 200)."""
    normalized_cutoffs = normalize_candidate_cutoffs(cutoffs)
    return {
        f"candidate_recall@{cutoff}": compute_candidate_recall(
            candidates,
            ground_truths,
            k=cutoff,
        )
        for cutoff in normalized_cutoffs
    }


def compute_top5_oracle(
    candidates: dict[str, Any] | None,
    ground_truths: dict[str, Any],
) -> dict[str, float]:
    """Compute the top-five oracle feasibility metrics.

    1. Corpus capacity ceiling: min(5, |Gq|) / |Gq|
       Theoretical maximum Recall@5 attainable even with perfect candidate retrieval and ranking.
    2. Candidate pool oracle: min(5, |Gq intersect Cq|) / |Gq|
       Maximum Recall@5 attainable given the actual retrieved candidate pools.
    """
    normalized_truths = _normalize_query_values(ground_truths)
    normalized_candidates = {str(qid): value for qid, value in (candidates or {}).items()}

    corpus_ceilings = []
    oracle_recalls = []

    for qid, gold_list in normalized_truths.items():
        gold_set = set(gold_list)
        if not gold_set:
            continue

        corpus_ceil = min(5, len(gold_set)) / len(gold_set)
        corpus_ceilings.append(corpus_ceil)

        if candidates is not None:
            raw_cands = normalized_candidates.get(qid, [])
            if isinstance(raw_cands, (str, bytes, Mapping)):
                raw_cands = _normalize_ids(raw_cands)
            else:
                try:
                    raw_cands = list(raw_cands)
                except TypeError:
                    raw_cands = _normalize_ids(raw_cands)
            cand_ids = {
                doc_id
                for doc_id in (_candidate_doc_id(c) for c in raw_cands)
                if doc_id is not None
            }
            oracle_rec = min(5, len(gold_set & cand_ids)) / len(gold_set)
            oracle_recalls.append(oracle_rec)

    res = {
        "corpus_capacity_ceiling@5": float(np.mean(corpus_ceilings)) if corpus_ceilings else 0.0,
    }
    if candidates is not None:
        res["candidate_pool_oracle@5"] = float(np.mean(oracle_recalls)) if oracle_recalls else 0.0
    return res
