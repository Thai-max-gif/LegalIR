from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable


def ranking_metrics(predictions: dict[str, list[str]], qrels: dict[str, set[str]], k: int = 5) -> dict[str, float | int]:
    values = {"recall": [], "mrr": [], "ndcg": [], "hit1": []}
    evaluated = 0
    for query_id, relevant in qrels.items():
        if not relevant:
            continue
        ranked = list(dict.fromkeys(predictions.get(query_id, [])))[:k]
        hits = [idx for idx, doc_id in enumerate(ranked, 1) if doc_id in relevant]
        values["recall"].append(len(hits) / len(relevant))
        values["mrr"].append(1 / hits[0] if hits else 0.0)
        values["hit1"].append(float(bool(hits and hits[0] == 1)))
        dcg = sum(1 / math.log2(rank + 1) for rank in hits)
        ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
        values["ndcg"].append(dcg / ideal if ideal else 0.0)
        evaluated += 1
    if not evaluated:
        raise ValueError("No qrels with relevance available")
    return {f"eval_{name}_at_{k}": sum(v) / evaluated for name, v in values.items()} | {"evaluated_queries": evaluated}


def qrels_from_rows(rows: Iterable[dict]) -> dict[str, set[str]]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if float(row.get("relevance", 1)) > 0:
            grouped[str(row["query_id"])].add(str(row["document_id"]))
    return grouped
