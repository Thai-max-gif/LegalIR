#!/usr/bin/env python3
"""
CLI script to build static label-free candidate caches for train and public queries.
Uses the proven legacy retrievers (Legal BM25, PyVi BM25, Exact Matcher, Dense Macro DEk21).
Zero qrels accepted or accessed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.core.memory import (
    check_memory_guard,
    format_memory_report,
    release_memory,
    take_memory_snapshot,
)
from src.data.canonical import resolve_duplicate_groups_path, verify_canonical_dataset
from src.retrieval.bm25_micro import BM25MicroRetriever
from src.retrieval.bm25_pyvi import BM25PyViRetriever
from src.retrieval.build_indexes import enrich_chunks_with_doc_metadata
from src.retrieval.dense_macro import DenseMacroRetriever
from src.retrieval.exact_matcher import ExactMatcher
from src.retrieval.static_cache import StaticCacheWriter, StaticCandidateRecord


def load_or_build_retrievers(
    dataset_dir: Path,
    indexes_dir: Optional[Path] = None,
    candidate_k: int = 150,
    max_corpus_chunks: Optional[int] = None,
) -> Dict[str, Any]:
    """Load prebuilt index files if available, otherwise fit on canonical dataset."""
    chunks_path = dataset_dir / "chunks.parquet"
    docs_path = dataset_dir / "documents.parquet"

    print(f"[*] Loading chunks from {chunks_path} ...")
    chunks_df = pd.read_parquet(chunks_path)
    docs_df = pd.read_parquet(docs_path) if docs_path.is_file() else None

    # Micro chunks
    type_col = "granularity" if "granularity" in chunks_df.columns else "chunk_type"
    micro_df = chunks_df[chunks_df[type_col] == "micro"] if type_col in chunks_df.columns else chunks_df
    macro_df = chunks_df[chunks_df[type_col] == "macro"] if type_col in chunks_df.columns else chunks_df

    if max_corpus_chunks is not None:
        micro_df = micro_df.head(max_corpus_chunks)
        macro_df = macro_df.head(max_corpus_chunks)

    if docs_df is not None:
        micro_df = enrich_chunks_with_doc_metadata(micro_df, docs_df)

    retrievers = {}

    # 1. BM25 Micro (Legal)
    bm25_idx_path = (indexes_dir / "bm25" / "bm25_micro_index.pkl") if indexes_dir else None
    if bm25_idx_path and bm25_idx_path.is_file():
        print(f"[*] Loading prebuilt BM25 Legal index from {bm25_idx_path} ...")
        bm25_legal = BM25MicroRetriever.load(bm25_idx_path)
    else:
        print("[*] Fitting BM25 Legal index on micro chunks ...")
        bm25_legal = BM25MicroRetriever()
        bm25_legal.fit(micro_df.to_dict("records"), show_progress=False)
    retrievers["bm25_legal"] = bm25_legal

    # 2. BM25 PyVi
    bm25_pyvi_idx_path = (indexes_dir / "bm25_pyvi" / "bm25_pyvi_index.pkl") if indexes_dir else None
    if bm25_pyvi_idx_path and bm25_pyvi_idx_path.is_file():
        print(f"[*] Loading prebuilt BM25 PyVi index from {bm25_pyvi_idx_path} ...")
        bm25_pyvi = BM25PyViRetriever.load(bm25_pyvi_idx_path)
    else:
        print("[*] Fitting BM25 PyVi index on micro chunks ...")
        bm25_pyvi = BM25PyViRetriever()
        bm25_pyvi.fit(micro_df.to_dict("records"), show_progress=False)
    retrievers["bm25_pyvi"] = bm25_pyvi

    # 3. Exact Matcher
    print("[*] Initializing Exact Matcher ...")
    docs_records = docs_df.to_dict("records") if docs_df is not None else []
    exact_matcher = ExactMatcher(documents=docs_records, chunks=micro_df)
    retrievers["exact"] = exact_matcher

    # 4. Dense Macro (DEk21)
    dense_emb_path = (indexes_dir / "dense" / "corpus_embeddings.npy") if indexes_dir else None
    if dense_emb_path and dense_emb_path.is_file():
        print(f"[*] Loading precomputed Dense Macro embeddings from {dense_emb_path} ...")
        macro_chunks = macro_df.to_dict("records")
        retrievers["dense"] = DenseMacroRetriever.from_arrays(
            embeddings_path=dense_emb_path,
            chunk_ids=[str(c["chunk_id"]) for c in macro_chunks],
            doc_ids=[str(c["doc_id"]) for c in macro_chunks],
        )
    else:
        # Fallback or lightweight dense initialization
        print("[*] Initializing DenseMacroRetriever ...")
        retrievers["dense"] = DenseMacroRetriever()

    return retrievers


def run_retrieval_and_cache(
    queries: List[tuple[str, str]],
    retrievers: Dict[str, Any],
    output_path: Path,
    candidate_k: int = 150,
) -> None:
    """Run retrieval for queries across all static branches and stream records to cache."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = StaticCacheWriter(output_path, batch_size=5000)

    bm25_legal: BM25MicroRetriever = retrievers.get("bm25_legal")
    bm25_pyvi: BM25PyViRetriever = retrievers.get("bm25_pyvi")
    exact: ExactMatcher = retrievers.get("exact")
    dense: DenseMacroRetriever = retrievers.get("dense")

    print(f"[*] Caching {len(queries)} queries to {output_path} (candidate_k={candidate_k}) ...")
    start_time = time.time()

    for idx, (qid, q_text) in enumerate(queries):
        if not q_text:
            continue

        # 1. BM25 Legal
        if bm25_legal is not None:
            res = bm25_legal.retrieve(q_text, top_k=candidate_k)
            for rank, item in enumerate(res, start=1):
                writer.write_record(
                    StaticCandidateRecord(
                        query_id=qid,
                        branch="bm25_legal",
                        rank=rank,
                        doc_id=item["doc_id"],
                        score=float(item["score"]),
                        best_chunk_id=item.get("bm25_best_chunk_id"),
                        second_score=float(item.get("bm25_second_score", 0.0)),
                        mean_score=float(item.get("bm25_mean_score", 0.0)),
                    )
                )

        # 2. BM25 PyVi
        if bm25_pyvi is not None:
            res = bm25_pyvi.retrieve(q_text, top_k=candidate_k)
            for rank, item in enumerate(res, start=1):
                writer.write_record(
                    StaticCandidateRecord(
                        query_id=qid,
                        branch="bm25_pyvi",
                        rank=rank,
                        doc_id=item["doc_id"],
                        score=float(item["score"]),
                        best_chunk_id=item.get("bm25_pyvi_best_chunk_id"),
                        second_score=float(item.get("bm25_pyvi_second_score", 0.0)),
                        mean_score=float(item.get("bm25_pyvi_mean_score", 0.0)),
                    )
                )

        # 3. Exact Matcher
        if exact is not None:
            res = exact.search(q_text, top_k=candidate_k)
            for rank, item in enumerate(res, start=1):
                writer.write_record(
                    StaticCandidateRecord(
                        query_id=qid,
                        branch="exact",
                        rank=rank,
                        doc_id=item["doc_id"],
                        score=float(item.get("score", 1.0)),
                    )
                )

        # 4. Dense Macro
        if dense is not None and getattr(dense, "embeddings", None) is not None:
            res = dense.retrieve(q_text, top_k=candidate_k)
            for rank, item in enumerate(res, start=1):
                writer.write_record(
                    StaticCandidateRecord(
                        query_id=qid,
                        branch="dense",
                        rank=rank,
                        doc_id=item["doc_id"],
                        score=float(item["score"]),
                        best_chunk_id=item.get("dense_best_chunk_id"),
                        second_score=float(item.get("dense_second_score", 0.0)),
                        mean_score=float(item.get("dense_mean_score", 0.0)),
                    )
                )

        if (idx + 1) % 500 == 0:
            elapsed = time.time() - start_time
            print(f"    - Processed {idx + 1}/{len(queries)} queries ({elapsed:.1f}s) ...")

    writer.close()
    elapsed = time.time() - start_time
    print(f"[+] Finished caching {len(queries)} queries in {elapsed:.1f}s ({writer.total_written} records).")


def main():
    parser = argparse.ArgumentParser(description="Build static retrieval candidate cache.")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="kaggle_dataset",
        help="Path to canonical dataset directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/factory/static_cache",
        help="Path to save static cache parquets",
    )
    parser.add_argument(
        "--indexes-dir",
        type=str,
        default="artifacts/task1/indexes",
        help="Optional directory containing prebuilt indexes",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=150,
        help="Number of candidates to cache per branch",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Optional max queries to process (for smoke/testing)",
    )
    parser.add_argument(
        "--max-corpus-chunks",
        type=int,
        default=None,
        help="Optional max corpus chunks to index (for fast smoke/testing)",
    )
    args = parser.parse_args()

    dataset_p = Path(args.dataset_dir)
    if not dataset_p.is_dir():
        if (REPO_ROOT / "artifacts" / "task1" / "data").is_dir():
            dataset_p = REPO_ROOT / "artifacts" / "task1" / "data"
        elif (REPO_ROOT / args.dataset_dir).is_dir():
            dataset_p = REPO_ROOT / args.dataset_dir

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    indexes_p = Path(args.indexes_dir) if args.indexes_dir else None

    print(f"[*] Starting static cache builder for {dataset_p} -> {out_dir}")
    is_valid, ident, errors = verify_canonical_dataset(dataset_p)
    if not is_valid:
        print(f"[!] Dataset verification failed: {errors}")
        sys.exit(1)

    snap = take_memory_snapshot()
    print(format_memory_report(snap, stage="Pre-build Static Cache"))

    # Load train queries (label-free: qrels_train.parquet is NOT loaded)
    train_queries_p = dataset_p / "queries_train.parquet"
    train_df = pd.read_parquet(train_queries_p)
    q_text_col = (
        "question_norm"
        if "question_norm" in train_df.columns
        else ("question_raw" if "question_raw" in train_df.columns else "text")
    )
    train_queries = list(zip(train_df["query_id"].astype(str), train_df[q_text_col].astype(str)))
    if args.max_queries:
        train_queries = train_queries[: args.max_queries]

    # Load public queries
    public_p = dataset_p / "public-official.json"
    with open(public_p, "r", encoding="utf-8") as f:
        public_dict = json.load(f)
    public_queries = [(str(qid), str(qtext)) for qid, qtext in public_dict.items()]
    if args.max_queries:
        public_queries = public_queries[: args.max_queries]

    # Load retrievers
    retrievers = load_or_build_retrievers(
        dataset_dir=dataset_p,
        indexes_dir=indexes_p,
        candidate_k=args.candidate_k,
        max_corpus_chunks=args.max_corpus_chunks,
    )

    # Build train cache
    train_cache_p = out_dir / "static_candidates_train.parquet"
    run_retrieval_and_cache(
        queries=train_queries,
        retrievers=retrievers,
        output_path=train_cache_p,
        candidate_k=args.candidate_k,
    )

    # Build public cache
    public_cache_p = out_dir / "static_candidates_public.parquet"
    run_retrieval_and_cache(
        queries=public_queries,
        retrievers=retrievers,
        output_path=public_cache_p,
        candidate_k=args.candidate_k,
    )

    # Memory cleanup
    retrievers.clear()
    release_memory()
    check_memory_guard(stage="Post-build Static Cache")
    snap_post = take_memory_snapshot()
    print(format_memory_report(snap_post, stage="Post-build Static Cache"))
    print("[+] Static candidate cache build successfully completed.")


if __name__ == "__main__":
    main()
