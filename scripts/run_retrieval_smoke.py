"""GPU inference smoke on a deterministic, closed LegalIR v2 corpus subset.

This command deliberately does not train LoRA or create a released index.  It
validates that the four mounted canonical tables can travel through
BM25 -> dense -> RRF -> BGE -> chunk-to-document -> document metrics.
"""
from __future__ import annotations

import argparse
import gc
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pyarrow.parquet as pq
import torch

from src.config import dotted_override, load_config, write_resolved_config
from src.data.dataset_validator import write_validation_report
from src.data.paths import required_table
from src.retrieval.fusion import aggregate_chunks_to_documents, reciprocal_rank_fusion
from src.utils.metrics import qrels_from_rows, ranking_metrics
from src.utils.runtime import NvmlPeakSampler, atomic_json, hardware_report, report_status


def first(columns: list[str], choices: list[str]) -> str:
    found = next((choice for choice in choices if choice in columns), None)
    if not found:
        raise ValueError(f"None of {choices} appears in {columns}")
    return found


def segment(text: str) -> str:
    from pyvi import ViTokenizer
    return ViTokenizer.tokenize(str(text))


def resolve_revision(model_id: str, configured: str | None) -> str:
    if configured and configured not in {"auto", "auto_resolve_for_diagnostic", "main"}:
        return configured
    from huggingface_hub import HfApi
    return HfApi().model_info(model_id).sha


def choose_subset(root: Path, query_limit: int, chunk_limit: int) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, set[str]]]:
    query_path, qrel_path, chunk_path = (required_table(root, name) for name in ("queries_train.parquet", "qrels_train.parquet", "chunks.parquet"))
    qrel_table = pq.read_table(qrel_path)
    qrel_doc = first(qrel_table.column_names, ["document_id", "doc_id"])
    qrels_all = qrels_from_rows([
        {"query_id": row["query_id"], "document_id": row[qrel_doc], "relevance": row.get("relevance", 1)}
        for row in qrel_table.to_pylist()
    ])
    selected_qids = list(qrels_all)[:query_limit]
    query_table = pq.read_table(query_path)
    query_text = first(query_table.column_names, ["query_text", "question_norm", "question_raw"])
    queries = [
        {"query_id": str(row["query_id"]), "query_text": str(row[query_text])}
        for row in query_table.to_pylist() if str(row["query_id"]) in set(selected_qids)
    ]
    if len(queries) != len(selected_qids):
        found = {row["query_id"] for row in queries}
        selected_qids = [qid for qid in selected_qids if qid in found]
    positives = {doc for qid in selected_qids for doc in qrels_all[qid]}
    columns = pq.ParquetFile(chunk_path).schema_arrow.names
    doc_col, text_col = first(columns, ["document_id", "doc_id"]), first(columns, ["chunk_text", "text_norm", "text_raw"])
    # Stream instead of loading the 1.15M-row corpus.  Keep one positive chunk
    # per positive document plus deterministic non-positive context chunks.
    positive_chunks: dict[str, dict[str, str]] = {}
    negatives: list[dict[str, str]] = []
    for batch in pq.ParquetFile(chunk_path).iter_batches(columns=["chunk_id", doc_col, text_col], batch_size=50_000):
        ids, docs, texts = (batch.column(i).to_pylist() for i in range(3))
        for chunk_id, doc_id, text in zip(ids, docs, texts):
            doc_id, text = str(doc_id), str(text or "")
            if not text:
                continue
            row = {"chunk_id": str(chunk_id), "document_id": doc_id, "chunk_text": text}
            if doc_id in positives and doc_id not in positive_chunks:
                positive_chunks[doc_id] = row
            elif doc_id not in positives and len(negatives) < chunk_limit:
                negatives.append(row)
        if len(positive_chunks) == len(positives) and len(negatives) >= chunk_limit:
            break
    missing = sorted(positives - set(positive_chunks))
    if missing:
        raise ValueError(f"No non-empty chunk found for {len(missing)} qrel documents; examples: {missing[:10]}")
    corpus = list(positive_chunks.values()) + negatives[:chunk_limit]
    if len(corpus) < 2:
        raise ValueError("Closed smoke corpus is too small")
    qrels = {qid: qrels_all[qid] for qid in selected_qids}
    return queries, corpus, qrels


def bm25_rank(corpus: list[dict[str, str]], queries: list[dict[str, str]], top_k: int) -> dict[str, list[str]]:
    import bm25s
    tokenized = bm25s.tokenize([segment(row["chunk_text"]) for row in corpus], stopwords=None)
    retriever = bm25s.BM25(); retriever.index(tokenized)
    ids, _ = retriever.retrieve(bm25s.tokenize([segment(row["query_text"]) for row in queries], stopwords=None), k=min(top_k, len(corpus)))
    return {query["query_id"]: [corpus[int(index)]["chunk_id"] for index in ranked] for query, ranked in zip(queries, ids)}


def dense_rank(corpus: list[dict[str, str]], queries: list[dict[str, str]], model_id: str, revision: str, device: str, top_k: int) -> dict[str, list[str]]:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_id, revision=revision, device=device)
    corpus_vectors = model.encode([segment(row["chunk_text"]) for row in corpus], batch_size=32, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
    query_vectors = model.encode([segment(row["query_text"]) for row in queries], batch_size=32, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
    scores = query_vectors @ corpus_vectors.T
    order = np.argsort(-scores, axis=1, kind="stable")[:, :min(top_k, len(corpus))]
    del model; gc.collect(); torch.cuda.empty_cache()
    return {query["query_id"]: [corpus[int(index)]["chunk_id"] for index in ranked] for query, ranked in zip(queries, order)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()
    config = load_config(args.config, dotted_override([f"data.dataset_root={args.dataset_root}", *args.set]))
    out = Path(config["experiment"]["output_dir"]); out.mkdir(parents=True, exist_ok=True)
    write_resolved_config(config, out)
    validation = write_validation_report(args.dataset_root, out / "dataset_validation_report.json")
    if validation["status"] != "PASS":
        report = {"status": "FAIL", "dataset_validation": validation, "checks": [{"name": "dataset_validation", "status": "FAIL"}]}
        atomic_json(out / "kaggle_retrieval_smoke_report.json", report); return 2
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        device_indices = [index for index in range(torch.cuda.device_count())]
        dense_index, rerank_index = device_indices[0], device_indices[-1]
        dense_device, rerank_device = f"cuda:{dense_index}", f"cuda:{rerank_index}"
        sampled_indices = sorted({dense_index, rerank_index})
        samplers = {index: NvmlPeakSampler(index) for index in sampled_indices}
        for sampler in samplers.values(): sampler.start()
        # Kaggle's Torch 2.10 build rejects both explicit string and integer
        # arguments for reset_peak_memory_stats. Select each device as the
        # current device, then use the no-argument API.
        original_device = torch.cuda.current_device()
        for index in sampled_indices:
            with torch.cuda.device(index):
                torch.cuda.reset_peak_memory_stats()
        queries, corpus, qrels = choose_subset(Path(args.dataset_root), config["data"]["smoke_validation_limit"], config["retrieval"]["smoke_corpus_chunk_limit"])
        dense_revision = resolve_revision(config["model"]["dense_model_id"], config["model"].get("dense_model_revision"))
        reranker_revision = resolve_revision(config["model"]["reranker_id"], config["model"].get("reranker_revision"))
        lexical = bm25_rank(corpus, queries, config["retrieval"]["bm25_k"])
        dense = dense_rank(corpus, queries, config["model"]["dense_model_id"], dense_revision, dense_device, config["retrieval"]["dense_k"])
        fused = {qid: [item for item, _ in reciprocal_rank_fusion({"bm25": lexical[qid], "dense": dense[qid]}, config["retrieval"]["rrf_weights"], config["retrieval"]["rrf_k"], config["retrieval"]["rerank_budget"])] for qid in lexical}
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(config["model"]["reranker_id"], revision=reranker_revision)
        reranker = AutoModelForSequenceClassification.from_pretrained(config["model"]["reranker_id"], revision=reranker_revision, num_labels=1, torch_dtype=torch.float16).to(rerank_device).eval()
        rows = {row["chunk_id"]: row for row in corpus}; chunk_map = {row["chunk_id"]: row["document_id"] for row in corpus}; query_map = {row["query_id"]: row["query_text"] for row in queries}
        predictions: dict[str, list[str]] = {}
        with torch.inference_mode():
            for qid, chunk_ids in fused.items():
                pairs = [(query_map[qid], rows[chunk_id]["chunk_text"]) for chunk_id in chunk_ids]
                scores: list[float] = []
                for start in range(0, len(pairs), 8):
                    inputs = tokenizer(pairs[start:start + 8], padding="max_length", truncation=True, max_length=config["model"]["max_length"], return_tensors="pt").to(rerank_device)
                    scores.extend(reranker(**inputs).logits.reshape(-1).float().cpu().tolist())
                predictions[qid] = [doc for doc, _ in aggregate_chunks_to_documents(zip(chunk_ids, scores), chunk_map, config["retrieval"]["document_aggregation"], config["retrieval"]["top_k"])]
        memory = {}
        for index, sampler in samplers.items():
            with torch.cuda.device(index):
                torch.cuda.synchronize()
                memory[f"cuda:{index}"] = {
                    "allocated": torch.cuda.max_memory_allocated(),
                    "reserved": torch.cuda.max_memory_reserved(),
                    "nvml": sampler.stop(),
                }
        torch.cuda.set_device(original_device)
        result = {"dataset_validation": validation, "scope": {"kind": "closed_corpus_diagnostic", "queries": len(queries), "chunks": len(corpus), "note": "not a released dense index or full-corpus quality result"}, "models": {"dense": {"id": config["model"]["dense_model_id"], "revision": dense_revision, "pooling": "mean", "preprocessing": "PyVi word segmentation"}, "reranker": {"id": config["model"]["reranker_id"], "revision": reranker_revision}}, "metrics": {"candidate": ranking_metrics({qid: [chunk_map[c] for c in fused[qid]] for qid in fused}, qrels, config["retrieval"]["top_k"]), "reranked": ranking_metrics(predictions, qrels, config["retrieval"]["top_k"])}, "vram": memory, "training": {"status": "NOT_RUN", "reason": "canonical v2 release has no published reranker pair/negative artifact; no mining or relabeling was performed"}, "checks": [{"name": "dataset_validation", "status": "PASS"}, {"name": "bm25_dense_rrf_bge_document_dedup", "status": "PASS"}, {"name": "lora_training", "status": "NOT_RUN", "reason": "released supervision pairs absent"}]}
        result["status"] = report_status(result["checks"])
        atomic_json(out / "kaggle_retrieval_smoke_report.json", result)
        print(f"Retrieval smoke {result['status']}: {out / 'kaggle_retrieval_smoke_report.json'}")
        return 0
    except Exception as exc:
        report = {"status": "FAIL", "error": repr(exc), "traceback": traceback.format_exc(), "hardware": hardware_report(), "checks": [{"name": "retrieval_smoke", "status": "FAIL", "reason": repr(exc)}]}
        atomic_json(out / "kaggle_retrieval_smoke_report.json", report)
        print(f"Retrieval smoke FAIL: {exc!r}")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
