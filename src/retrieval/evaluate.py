from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch

from src.retrieval.fusion import aggregate_chunks_to_documents
from src.utils.metrics import qrels_from_rows, ranking_metrics


def _first(columns: list[str], names: list[str]) -> str | None:
    return next((name for name in names if name in columns), None)


def rerank_candidate_table(model: Any, tokenizer: Any, dataset_root: str | Path, candidates_path: str | Path, qrels_path: str | Path, batch_size: int, max_length: int, aggregation: str, top_k: int, device: torch.device) -> dict[str, Any]:
    root = Path(dataset_root)
    candidate_table = pq.read_table(candidates_path)
    required = {"query_id"}
    if not required.issubset(candidate_table.column_names):
        return {"status": "BLOCKED", "reason": "candidate table lacks query_id", "columns": candidate_table.column_names}
    rows = candidate_table.to_pylist()
    item_col = _first(candidate_table.column_names, ["chunk_id", "document_id", "doc_id", "candidate_id"])
    query_col = _first(candidate_table.column_names, ["query_text", "question_norm", "question_raw", "query"])
    text_col = _first(candidate_table.column_names, ["chunk_text", "text_norm", "text_raw", "document_text", "passage_text", "candidate_text"])
    if not item_col or not query_col or not text_col:
        return {"status": "BLOCKED", "reason": "candidate table needs candidate id, query text, and candidate text", "columns": candidate_table.column_names}
    model.eval()
    scored: dict[str, list[tuple[str, float]]] = {}
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            block = rows[start:start + batch_size]
            inputs = tokenizer([str(x[query_col]) for x in block], [str(x[text_col]) for x in block], padding="max_length", truncation=True, max_length=max_length, return_tensors="pt").to(device)
            scores = model(**inputs).logits.reshape(-1).float().cpu().tolist()
            for row, score in zip(block, scores):
                scored.setdefault(str(row["query_id"]), []).append((str(row[item_col]), score))
    chunk_map = {}
    if item_col == "chunk_id":
        chunks_path = root / "canonical" / "chunks.parquet"
        if not chunks_path.is_file():
            chunks_path = root / "chunks.parquet"
        if not chunks_path.is_file():
            return {"status": "BLOCKED", "reason": "chunks.parquet not found for chunk-to-document aggregation"}
        chunk_columns = pq.ParquetFile(chunks_path).schema_arrow.names
        document_col = _first(chunk_columns, ["document_id", "doc_id"])
        if not document_col:
            return {"status": "BLOCKED", "reason": "chunks.parquet lacks document_id/doc_id", "columns": chunk_columns}
        chunks = pq.read_table(chunks_path, columns=["chunk_id", document_col])
        chunk_map = {str(a): str(b) for a, b in zip(chunks["chunk_id"].to_pylist(), chunks[document_col].to_pylist())}
    predictions = {}
    for qid, values in scored.items():
        values.sort(key=lambda x: (-x[1], x[0]))
        if item_col == "chunk_id":
            predictions[qid] = [doc for doc, _ in aggregate_chunks_to_documents(values, chunk_map, aggregation, top_k)]
        else:
            predictions[qid] = list(dict.fromkeys(item for item, _ in values))[:top_k]
    qrel_table = pq.read_table(qrels_path)
    qrel_doc_col = _first(qrel_table.column_names, ["document_id", "doc_id"])
    if not qrel_doc_col or "query_id" not in qrel_table.column_names:
        return {"status": "BLOCKED", "reason": "qrels need query_id and document_id/doc_id", "columns": qrel_table.column_names}
    qrels = qrels_from_rows([
        {"query_id": row["query_id"], "document_id": row[qrel_doc_col], "relevance": row.get("relevance", 1)}
        for row in qrel_table.to_pylist()
    ])
    return {"status": "PASS", "metrics": ranking_metrics(predictions, qrels, top_k), "prediction_count": len(predictions), "candidate_scope": str(candidates_path)}
