"""Out-of-Fold (OOF) feature extraction for candidate ranking and learned fusion.

Extracts dense, lexical (raw legal + PyVi), memory, exact matching, reranker,
and prior features for each query-candidate pair to feed downstream ranking.
"""

from collections.abc import Mapping, Sequence
from typing import Any
import numpy as np
import pandas as pd

from src.retrieval.types import CandidateRecord

FEATURE_SCHEMA_VERSION = "v2"

# Phase 7 Core Feature Columns
CORE_FEATURE_COLUMNS: list[str] = [
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
]

# Legacy and auxiliary feature columns for backward compatibility
AUXILIARY_FEATURE_COLUMNS: list[str] = [
    "bm25_score",
    "bm25_rank",
    "bm25_inv_rank",
    "bm25_best_score",
    "bm25_second_score",
    "bm25_mean_score",
    "exact_match_score",
    "exact_title",
    "memory_score",
    "memory_inv_rank",
    "memory_lexical_similarity",
    "memory_dense_similarity",
    "dense_inv_rank",
    "dense_best_score",
    "reranker_best_score",
    "evidence_chunk_count",
]

FEATURE_COLUMNS: list[str] = CORE_FEATURE_COLUMNS + [
    c for c in AUXILIARY_FEATURE_COLUMNS if c not in CORE_FEATURE_COLUMNS
]


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Safely convert value to float, replacing None, NaN, and Inf with default."""
    if val is None:
        return default
    try:
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def _safe_int(val: Any, default: int = 0) -> int:
    """Safely convert value to int."""
    if val is None:
        return default
    try:
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return default
        return int(f)
    except (TypeError, ValueError):
        return default


def compute_training_doc_frequencies(
    qrels: Mapping[str, Sequence[str]] | pd.DataFrame | list[dict[str, Any]],
) -> dict[str, float]:
    """Compute normalized document frequency prior from training fold qrels only.

    Args:
        qrels: Mapping of query_id to list of relevant doc_ids, or qrels DataFrame.

    Returns:
        dict mapping doc_id to normalized document frequency (occurrences / total queries).
    """
    counts: dict[str, int] = {}
    total_queries = 0

    if isinstance(qrels, pd.DataFrame):
        if "query_id" in qrels.columns and "doc_id" in qrels.columns:
            grouped = qrels.groupby("query_id")["doc_id"].unique()
            total_queries = len(grouped)
            for docs in grouped:
                for did in docs:
                    did_str = str(did)
                    counts[did_str] = counts.get(did_str, 0) + 1
    elif isinstance(qrels, Mapping):
        total_queries = len(qrels)
        for qid, docs in qrels.items():
            if isinstance(docs, (str, bytes)):
                docs_iter = [docs]
            else:
                docs_iter = list(docs)
            for did in set(docs_iter):
                if did is not None:
                    did_str = str(did)
                    counts[did_str] = counts.get(did_str, 0) + 1
    elif isinstance(qrels, Sequence):
        temp_map: dict[str, set[str]] = {}
        for r in qrels:
            if isinstance(r, Mapping):
                qid = str(r.get("query_id", ""))
                did = str(r.get("doc_id", ""))
                if qid and did:
                    temp_map.setdefault(qid, set()).add(did)
        total_queries = len(temp_map)
        for qid, docs in temp_map.items():
            for did in docs:
                counts[did] = counts.get(did, 0) + 1

    if total_queries <= 0:
        return {}

    return {did: float(cnt) / float(total_queries) for did, cnt in counts.items()}


def extract_candidate_features(
    query_id: str | list[CandidateRecord] | None = None,
    candidate_records: list[CandidateRecord | dict[str, Any]] | None = None,
    query_text: str | None = None,
    doc_freq_map: Mapping[str, float] | None = None,
    qrels: Mapping[str, Sequence[str]] | set[str] | Sequence[str] | None = None,
) -> pd.DataFrame:
    """Converts candidate records for a query into a tabular feature DataFrame.

    Supports multiple calling signatures:
        extract_candidate_features("q1", candidate_records)
        extract_candidate_features(candidate_records)
        extract_candidate_features(query_id="q1", candidate_records=..., query_text="...", ...)
    """
    if isinstance(query_id, list):
        candidate_records = query_id
        query_id = None

    if candidate_records is None:
        candidate_records = []

    qid_str = str(query_id) if query_id is not None else None
    q_len = float(len(query_text.strip())) if query_text is not None else 0.0

    # Build gold doc ID set if qrels provided
    gold_set: set[str] = set()
    if qrels is not None:
        if isinstance(qrels, (set, list, tuple)):
            gold_set = {str(x) for x in qrels if x is not None}
        elif isinstance(qrels, Mapping):
            if qid_str is not None and qid_str in qrels:
                gold_docs = qrels[qid_str]
                if isinstance(gold_docs, (str, bytes)):
                    gold_set = {str(gold_docs)}
                else:
                    gold_set = {str(x) for x in gold_docs if x is not None}

    rows = []
    for c in candidate_records:
        did = str(c.get("doc_id", ""))
        bm25_r = c.get("raw_bm25_rank", c.get("bm25_rank"))
        pyvi_r = c.get("pyvi_bm25_rank", c.get("bm25_pyvi_rank"))
        dense_r = c.get("dense_rank")
        mem_r = c.get("memory_rank")

        # Rank values (default 999.0 for unretrieved branches)
        raw_bm25_rank_val = _safe_float(bm25_r, default=999.0) if bm25_r is not None else 999.0
        pyvi_bm25_rank_val = _safe_float(pyvi_r, default=999.0) if pyvi_r is not None else 999.0
        dense_rank_val = _safe_float(dense_r, default=999.0) if dense_r is not None else 999.0
        mem_rank_val = _safe_float(mem_r, default=999.0) if mem_r is not None else 999.0

        # BM25 scores
        raw_bm25_score_val = _safe_float(c.get("raw_bm25_score", c.get("bm25_raw_score", c.get("bm25_score", 0.0))))
        pyvi_bm25_score_val = _safe_float(c.get("pyvi_bm25_score", c.get("bm25_pyvi_score", c.get("bm25_pyvi_best_score", 0.0))))

        # Dense scores & margin
        dense_score_val = _safe_float(c.get("dense_score", c.get("dense_best_score", 0.0)))
        dense_second_val = _safe_float(c.get("dense_second_score", 0.0))
        dense_margin_val = _safe_float(c.get("dense_margin", dense_score_val - dense_second_val))

        # Memory scores
        mem_sim_val = _safe_float(
            c.get("memory_similarity", c.get("memory_dense_similarity", c.get("memory_lexical_similarity", c.get("memory_score", 0.0))))
        )
        mem_votes_val = _safe_float(c.get("memory_vote_count", 0.0))

        # Exact match features
        exact_score_val = _safe_float(c.get("exact_score", c.get("exact_match_score", 0.0)))
        exact_legal_num = 1.0 if bool(c.get("exact_legal_number", False)) else 0.0
        exact_art = 1.0 if bool(c.get("exact_article", False)) else 0.0
        exact_cls = 1.0 if bool(c.get("exact_clause", False)) else 0.0
        exact_pt = 1.0 if bool(c.get("exact_point", False)) else 0.0
        exact_yr = 1.0 if bool(c.get("exact_year", False)) else 0.0
        exact_doc_t = 1.0 if bool(c.get("exact_doc_type", False)) else 0.0
        exact_title_ov = _safe_float(c.get("exact_title_overlap", 1.0 if c.get("exact_title") else 0.0))

        # Source count & RRF score
        src_cnt = _safe_float(c.get("source_count", 1.0), default=1.0)
        rrf_sc = _safe_float(c.get("rrf_score", 0.0))

        # Reranker scores & margin
        rerank_sc = _safe_float(c.get("reranker_score", c.get("reranker_best_score", -999.0)), default=-999.0)
        rerank_second_sc = _safe_float(c.get("reranker_second_score", -999.0), default=-999.0)
        if "reranker_margin" in c and c["reranker_margin"] is not None:
            rerank_margin_val = _safe_float(c["reranker_margin"])
        elif rerank_sc > -900.0 and rerank_second_sc > -900.0:
            rerank_margin_val = rerank_sc - rerank_second_sc
        else:
            rerank_margin_val = 0.0

        # Prior frequency
        if doc_freq_map is not None:
            doc_freq_val = _safe_float(doc_freq_map.get(did, 0.0))
        else:
            doc_freq_val = _safe_float(c.get("train_doc_freq", 0.0))

        # Candidate query length fallback
        c_q_len = q_len if q_len > 0 else _safe_float(c.get("query_length", 0.0))

        # Determine target label
        if gold_set:
            label_val = 1 if did in gold_set else 0
        elif "label" in c and c["label"] is not None:
            label_val = _safe_int(c["label"])
        elif "target" in c and c["target"] is not None:
            label_val = _safe_int(c["target"])
        else:
            label_val = 0

        row = {
            "query_id": qid_str,
            "group": qid_str,
            "doc_id": did,
            # Phase 7 Core Features
            "raw_bm25_rank": raw_bm25_rank_val,
            "raw_bm25_score": raw_bm25_score_val,
            "pyvi_bm25_rank": pyvi_bm25_rank_val,
            "pyvi_bm25_score": pyvi_bm25_score_val,
            "dense_rank": dense_rank_val,
            "dense_score": dense_score_val,
            "dense_second_score": dense_second_val,
            "dense_margin": dense_margin_val,
            "memory_rank": mem_rank_val,
            "memory_similarity": mem_sim_val,
            "memory_vote_count": mem_votes_val,
            "exact_score": exact_score_val,
            "exact_legal_number": exact_legal_num,
            "exact_article": exact_art,
            "exact_clause": exact_cls,
            "exact_point": exact_pt,
            "exact_year": exact_yr,
            "exact_doc_type": exact_doc_t,
            "exact_title_overlap": exact_title_ov,
            "source_count": src_cnt,
            "rrf_score": rrf_sc,
            "reranker_score": rerank_sc,
            "reranker_second_score": rerank_second_sc,
            "reranker_margin": rerank_margin_val,
            "query_length": c_q_len,
            "train_doc_freq": doc_freq_val,
            # Legacy / Auxiliary Features
            "bm25_score": raw_bm25_score_val,
            "bm25_rank": raw_bm25_rank_val,
            "bm25_inv_rank": (1.0 / (60.0 + float(raw_bm25_rank_val))) if raw_bm25_rank_val < 900.0 else 0.0,
            "bm25_best_score": _safe_float(c.get("bm25_best_score", raw_bm25_score_val)),
            "bm25_second_score": _safe_float(c.get("bm25_second_score", 0.0)),
            "bm25_mean_score": _safe_float(c.get("bm25_mean_score", raw_bm25_score_val)),
            "exact_match_score": exact_score_val,
            "exact_title": 1.0 if bool(c.get("exact_title", False)) else 0.0,
            "memory_score": _safe_float(c.get("memory_score", mem_sim_val)),
            "memory_inv_rank": (1.0 / (60.0 + float(mem_rank_val))) if mem_rank_val < 900.0 else 0.0,
            "memory_lexical_similarity": _safe_float(c.get("memory_lexical_similarity", mem_sim_val)),
            "memory_dense_similarity": _safe_float(c.get("memory_dense_similarity", mem_sim_val)),
            "dense_inv_rank": (1.0 / (60.0 + float(dense_rank_val))) if dense_rank_val < 900.0 else 0.0,
            "dense_best_score": _safe_float(c.get("dense_best_score", dense_score_val)),
            "reranker_best_score": _safe_float(c.get("reranker_best_score", rerank_sc)),
            "evidence_chunk_count": _safe_float(c.get("evidence_chunk_count", 0.0)),
            # Targets
            "label": label_val,
            "target": label_val,
        }
        rows.append(row)

    if not rows:
        cols = ["query_id", "group", "doc_id", "label", "target"] + FEATURE_COLUMNS
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(rows)
    # Ensure no NaN exists in any feature column
    for col in FEATURE_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)
    return df


def extract_dataset_features(
    queries: Mapping[str, str],
    candidates_by_query: Mapping[str, Sequence[CandidateRecord | dict[str, Any]]],
    qrels: Mapping[str, Sequence[str]] | None = None,
    doc_freq_map: Mapping[str, float] | None = None,
    fold_id: int | None = None,
) -> pd.DataFrame:
    """Extract candidate features for an entire collection of queries and candidates."""
    all_dfs = []
    for qid, cands in candidates_by_query.items():
        qid_str = str(qid)
        q_text = queries.get(qid_str, queries.get(qid, ""))
        feat_df = extract_candidate_features(
            query_id=qid_str,
            candidate_records=list(cands),
            query_text=q_text,
            doc_freq_map=doc_freq_map,
            qrels=qrels,
        )
        if not feat_df.empty:
            if fold_id is not None:
                feat_df["fold"] = int(fold_id)
            all_dfs.append(feat_df)

    if not all_dfs:
        cols = ["query_id", "group", "doc_id", "label", "target"] + FEATURE_COLUMNS
        if fold_id is not None:
            cols.append("fold")
        return pd.DataFrame(columns=cols)

    return pd.concat(all_dfs, ignore_index=True)
