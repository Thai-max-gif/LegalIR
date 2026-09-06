"""
Public candidate reranking and frozen fusion top-5 prediction selector.
Loads:
- public_candidates.parquet
- public_evidence.parquet
- final trained adapter
- production_lock.json
- frozen fusion artifact
"""

from __future__ import annotations

import collections
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.ranking.fusion import LightGBMRanker, ReciprocalRankFusion
from src.ranking.reranker import CrossEncoderReranker
from src.retrieval.static_cache import StaticCacheReader


def normalize_public_queries(raw_queries: Mapping[str, Any]) -> Dict[str, str]:
    """Normalize public query dictionary to dict[qid, question_string]."""
    res = {}
    for qid, val in raw_queries.items():
        qid_str = str(qid)
        if isinstance(val, dict):
            res[qid_str] = str(val.get("question", "") or "")
        else:
            res[qid_str] = str(val or "")
    return res


def load_public_evidence_map(public_evidence_path: Union[str, Path]) -> Dict[Tuple[str, str], str]:
    """Load pre-materialized public evidence mapping (query_id, doc_id) -> evidence_text."""
    evidence_p = Path(public_evidence_path)
    if not evidence_p.is_file():
        return {}

    df = pd.read_parquet(evidence_p)
    res = {}
    for _, row in df.iterrows():
        qid = str(row["query_id"])
        did = str(row["doc_id"])
        ev_text = str(row.get("evidence_text", ""))
        res[(qid, did)] = ev_text
    return res


def _load_fusion_ranker(
    production_lock_path: Union[str, Path],
    fusion_model_path: Optional[Union[str, Path]] = None,
) -> Any:
    """Load winning fusion ranker (RRF or Learned Ranker) under frozen lock contract."""
    lock_p = Path(production_lock_path)
    if not lock_p.is_file():
        raise FileNotFoundError(f"Missing production lock at {lock_p}")

    with open(lock_p, "r", encoding="utf-8") as f:
        lock_data = json.load(f)

    cfg = lock_data.get("config", {})
    fusion_cfg = cfg.get("fusion", {})
    default_weights = {
        "bm25": 1.0,
        "bm25_pyvi": 1.0,
        "dense": 1.2,
        "memory": 2.0,
        "exact": 2.5,
        "rerank": 1.8,
    }

    desc_p: Optional[Path] = None
    if fusion_model_path is not None:
        desc_p = Path(fusion_model_path)
        if not desc_p.is_file():
            raise FileNotFoundError(f"Specified fusion model path does not exist: {desc_p}")
    else:
        # Check standard locations in bundle
        candidates = [
            lock_p.parent / "fusion_model.json",
            lock_p.parent / "fusion" / "fusion_model.json",
        ]
        desc_p = next((p for p in candidates if p.is_file()), None)

    if desc_p is not None and desc_p.is_file():
        with open(desc_p, "r", encoding="utf-8") as f:
            desc = json.load(f)

        if desc.get("model_type") in ("linear_ridge", "lightgbm"):
            ranker = LightGBMRanker(model_file=desc_p, strict=True)
            if ranker.model is None and ranker.fallback_model is None:
                raise RuntimeError(f"Failed to load learned fusion ranker from {desc_p}")
            return ranker

        method = desc.get("winning_method") or desc.get("method") or fusion_cfg.get("method", "reciprocal_rank_fusion")
        if method in ("reciprocal_rank_fusion", "rrf"):
            rrf_data = desc.get("rrf", {})
            k = int(rrf_data.get("k", fusion_cfg.get("k", 60)))
            weights = rrf_data.get("weights", fusion_cfg.get("weights", default_weights))
            return ReciprocalRankFusion(k=k, weights=weights)
        elif method in ("learned_ranker", "lightgbm"):
            learned_info = desc.get("learned_model", {})
            payload_name = learned_info.get("file") or desc.get("model_file", "fusion_model.txt")
            payload_candidates = [
                desc_p.parent / payload_name,
                desc_p.parent / "fusion" / payload_name,
                lock_p.parent / payload_name,
                lock_p.parent / "fusion" / payload_name,
                Path(payload_name),
            ]
            target_payload = next((p for p in payload_candidates if p.is_file()), None)
            if target_payload is None or not target_payload.is_file():
                if desc.get("model_type") is not None:
                    target_payload = desc_p
                else:
                    raise FileNotFoundError(f"Learned fusion payload '{payload_name}' not found.")

            feat_cols = desc.get("feature_columns") or fusion_cfg.get("feature_columns")
            ranker = LightGBMRanker(model_file=target_payload, feature_cols=feat_cols, strict=True)
            if ranker.model is None and ranker.fallback_model is None:
                raise RuntimeError(f"Failed to load learned fusion ranker from {target_payload}")
            return ranker
        else:
            raise ValueError(f"Unknown winning fusion method: {method}")

    # Fallback to config definition
    method = fusion_cfg.get("method", "reciprocal_rank_fusion")
    if method in ("reciprocal_rank_fusion", "rrf"):
        k = int(fusion_cfg.get("k", 60))
        weights = fusion_cfg.get("weights", default_weights)
        return ReciprocalRankFusion(k=k, weights=weights)
    elif method in ("learned_ranker", "lightgbm"):
        model_file = fusion_cfg.get("model_file", "fusion_model.txt")
        candidates = [
            lock_p.parent / model_file,
            lock_p.parent / "fusion" / model_file,
            Path(model_file),
        ]
        target_payload = next((p for p in candidates if p.is_file()), None)
        if target_payload is None or not target_payload.is_file():
            raise FileNotFoundError(f"Learned fusion model file '{model_file}' not found.")
        feat_cols = fusion_cfg.get("feature_columns")
        ranker = LightGBMRanker(model_file=target_payload, feature_cols=feat_cols, strict=True)
        if ranker.model is None and ranker.fallback_model is None:
            raise RuntimeError(f"Failed to load learned fusion ranker from {target_payload}")
        return ranker
    else:
        raise ValueError(f"Unknown fusion method in production lock: {method}")


def rerank_and_fuse_public_predictions(
    public_candidates_path: Union[str, Path],
    production_lock_path: Union[str, Path],
    adapter_dir: Optional[Union[str, Path]] = None,
    public_evidence_path: Optional[Union[str, Path]] = None,
    fusion_model_path: Optional[Union[str, Path]] = None,
    top_k: int = 5,
    rerank_k: int = 50,
    device: str = "auto",
    adapter_scores: Optional[Dict[str, Dict[str, float]]] = None,
    public_queries_dict: Optional[Mapping[str, Any]] = None,
) -> Dict[str, List[str]]:
    """
    Rerank public candidates combining static retrieval branch ranks and reranker scores
    under the frozen production configuration from production_lock.json.
    """
    lock_p = Path(production_lock_path)
    if not lock_p.is_file():
        raise FileNotFoundError(f"Missing production lock at {lock_p}")

    with open(lock_p, "r", encoding="utf-8") as f:
        lock_data = json.load(f)

    cfg = lock_data.get("config", {})
    fusion_cfg = cfg.get("fusion", {})
    eff_top_k = int(fusion_cfg.get("top_k", top_k))
    eff_rerank_k = int(fusion_cfg.get("rerank_k", rerank_k))

    ranker = _load_fusion_ranker(
        production_lock_path=lock_p,
        fusion_model_path=fusion_model_path,
    )

    clean_queries = normalize_public_queries(public_queries_dict) if public_queries_dict is not None else {}

    # Read public candidates
    reader = StaticCacheReader(public_candidates_path)
    qids = reader.get_query_ids()

    # Evidence map
    evidence_map = load_public_evidence_map(public_evidence_path) if public_evidence_path else {}

    # Load final adapter if provided and no adapter_scores passed
    reranker = None
    if adapter_dir is not None and Path(adapter_dir).is_dir() and adapter_scores is None:
        reranker_cfg = cfg.get("reranker", {})
        base_model = reranker_cfg.get("model_name", "BAAI/bge-reranker-v2-m3")
        print(f"[*] Loading final adapter from {adapter_dir} for public reranking ...")
        reranker = CrossEncoderReranker(
            model_name=base_model,
            adapter_path=adapter_dir,
            device=device,
        )
        reranker.ensure_loaded()

    computed_adapter_scores: Dict[str, Dict[str, float]] = collections.defaultdict(dict)
    if adapter_scores is not None:
        computed_adapter_scores = adapter_scores

    predictions: Dict[str, List[str]] = {}

    for qid in qids:
        cands = reader.get_query_candidates(qid)
        if not cands:
            continue

        q_text = clean_queries.get(qid, "")

        # Group by doc_id and compute branch ranks / static RRF
        doc_data: Dict[str, Dict[str, Any]] = collections.defaultdict(lambda: {
            "raw_bm25_rank": None,
            "raw_bm25_score": 0.0,
            "pyvi_bm25_rank": None,
            "pyvi_bm25_score": 0.0,
            "dense_rank": None,
            "dense_score": 0.0,
            "exact_score": 0.0,
            "branches": set(),
            "static_rrf": 0.0,
        })

        weights = getattr(ranker, "weights", {
            "bm25": 1.0,
            "bm25_pyvi": 1.0,
            "dense": 1.2,
            "memory": 2.0,
            "exact": 2.5,
            "rerank": 1.8,
        })
        k_val = float(getattr(ranker, "k", 60))

        for c in cands:
            did = str(c.doc_id)
            dinfo = doc_data[did]
            b = str(c.branch)
            dinfo["branches"].add(b)

            if b in ("bm25", "bm25_legal"):
                if dinfo["raw_bm25_rank"] is None or c.rank < dinfo["raw_bm25_rank"]:
                    dinfo["raw_bm25_rank"] = int(c.rank)
                    dinfo["raw_bm25_score"] = float(c.score)
                w = float(weights.get("bm25", 1.0))
                dinfo["static_rrf"] += w / (k_val + float(c.rank))
            elif b in ("bm25_pyvi", "pyvi"):
                if dinfo["pyvi_bm25_rank"] is None or c.rank < dinfo["pyvi_bm25_rank"]:
                    dinfo["pyvi_bm25_rank"] = int(c.rank)
                    dinfo["pyvi_bm25_score"] = float(c.score)
                w = float(weights.get("bm25_pyvi", 1.0))
                dinfo["static_rrf"] += w / (k_val + float(c.rank))
            elif b == "dense":
                if dinfo["dense_rank"] is None or c.rank < dinfo["dense_rank"]:
                    dinfo["dense_rank"] = int(c.rank)
                    dinfo["dense_score"] = float(c.score)
                w = float(weights.get("dense", 1.2))
                dinfo["static_rrf"] += w / (k_val + float(c.rank))
            elif b == "exact":
                dinfo["exact_score"] = max(dinfo["exact_score"], float(c.score))
                w = float(weights.get("exact", 2.5))
                dinfo["static_rrf"] += w * 0.05 * float(c.score)

        # Select top candidates to rerank
        sorted_by_static = sorted(doc_data.items(), key=lambda x: x[1]["static_rrf"], reverse=True)
        docs_to_rerank = [d for d, _ in sorted_by_static[:eff_rerank_k]]

        # Score with reranker if available and not precomputed
        if reranker is not None and qid not in computed_adapter_scores and q_text:
            pairs_to_score = []
            valid_rerank_docs = []
            for did in docs_to_rerank:
                ev_text = evidence_map.get((qid, did), "")
                if ev_text:
                    pairs_to_score.append((q_text, ev_text))
                    valid_rerank_docs.append(did)

            if pairs_to_score:
                scores = reranker.score_pairs(pairs_to_score, batch_size=16, max_length=512)
                for did, s in zip(valid_rerank_docs, scores):
                    computed_adapter_scores[qid][did] = float(s)

        # Calculate reranker second score and margin for candidate features
        q_r_scores = computed_adapter_scores.get(qid, {})
        valid_scores = sorted([s for s in q_r_scores.values() if s > -900.0], reverse=True)
        second_r_score = valid_scores[1] if len(valid_scores) > 1 else (valid_scores[0] if valid_scores else -999.0)

        # Build candidate records matching OOF schema
        cand_records: List[Dict[str, Any]] = []
        for did, dinfo in doc_data.items():
            r_score = q_r_scores.get(did, -999.0)
            margin = (r_score - second_r_score) if r_score > -900.0 else 0.0

            cand_records.append({
                "doc_id": did,
                "raw_bm25_rank": dinfo["raw_bm25_rank"],
                "raw_bm25_score": dinfo["raw_bm25_score"],
                "pyvi_bm25_rank": dinfo["pyvi_bm25_rank"],
                "pyvi_bm25_score": dinfo["pyvi_bm25_score"],
                "dense_rank": dinfo["dense_rank"],
                "dense_score": dinfo["dense_score"],
                "exact_score": dinfo["exact_score"],
                "source_count": len(dinfo["branches"]),
                "rrf_score": dinfo["static_rrf"],
                "reranker_score": r_score,
                "reranker_second_score": second_r_score,
                "reranker_margin": margin,
                "query_length": float(len(q_text)),
            })

        if isinstance(ranker, ReciprocalRankFusion):
            ranked = ranker.rank_candidates(cand_records)
        elif hasattr(ranker, "predict"):
            ranked = ranker.predict(cand_records, query_id=qid, query_text=q_text)
        else:
            ranked = cand_records

        # Deterministic sort: descending final_score, ascending doc_id
        ranked.sort(key=lambda x: (-float(x.get("final_score", 0.0)), str(x.get("doc_id", ""))))

        seen_docs = set()
        unique_preds = []
        for r in ranked:
            did = str(r["doc_id"])
            if did and did not in seen_docs:
                seen_docs.add(did)
                unique_preds.append(did)
            if len(unique_preds) == eff_top_k:
                break

        predictions[qid] = unique_preds

    return predictions
