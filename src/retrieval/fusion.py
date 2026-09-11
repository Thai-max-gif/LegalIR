from __future__ import annotations

from collections import defaultdict
from typing import Iterable


def reciprocal_rank_fusion(rankings: dict[str, Iterable[str]], weights: dict[str, float], k: int = 60, limit: int | None = None) -> list[tuple[str, float]]:
    scores: dict[str, float] = defaultdict(float)
    for branch, ranked_ids in rankings.items():
        if branch not in weights:
            continue
        for rank, item_id in enumerate(ranked_ids, start=1):
            scores[str(item_id)] += float(weights[branch]) / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]


def aggregate_chunks_to_documents(scored_chunks: Iterable[tuple[str, float]], chunk_to_document: dict[str, str], policy: str = "max", limit: int | None = None) -> list[tuple[str, float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for chunk_id, score in scored_chunks:
        if str(chunk_id) in chunk_to_document:
            grouped[chunk_to_document[str(chunk_id)]].append(float(score))
    if policy == "max":
        reduced = ((doc_id, max(scores)) for doc_id, scores in grouped.items())
    elif policy == "mean":
        reduced = ((doc_id, sum(scores) / len(scores)) for doc_id, scores in grouped.items())
    else:
        raise ValueError(f"Unknown aggregation policy: {policy}")
    return sorted(reduced, key=lambda item: (-item[1], item[0]))[:limit]
