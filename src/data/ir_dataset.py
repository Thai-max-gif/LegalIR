from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset


def inspect_pair_schema(path: str | Path) -> dict[str, Any]:
    columns = pq.ParquetFile(path).schema_arrow.names
    lower = {name.lower(): name for name in columns}
    positive = next((lower[x] for x in ["positive_text", "pos_text", "positive", "pos_passage"] if x in lower), None)
    negative = next((lower[x] for x in ["negative_text", "neg_text", "negative", "neg_passage"] if x in lower), None)
    label = next((lower[x] for x in ["label", "relevance", "score"] if x in lower), None)
    query = next((lower[x] for x in ["query_text", "question_norm", "question_raw", "query"] if x in lower), None)
    mode = "pairwise" if query and positive and negative else "bce" if query and label else None
    return {"columns": columns, "query": query, "positive": positive, "negative": negative, "label": label, "mode": mode}


class RerankerDataset(Dataset):
    def __init__(self, path: str | Path, mode: str = "auto", limit: int | None = None):
        self.path = str(path)
        self.schema = inspect_pair_schema(path)
        self.mode = self.schema["mode"] if mode == "auto" else mode
        if self.mode not in {"pairwise", "bce"}:
            raise ValueError(f"Cannot infer supervision format from {self.schema['columns']}")
        if self.mode == "pairwise" and (not self.schema["positive"] or not self.schema["negative"]):
            raise ValueError("Pairwise loss requires query, positive text, and negative text columns")
        if self.mode == "bce" and not self.schema["label"]:
            raise ValueError("BCE loss requires a numeric label/relevance column")
        table = pq.read_table(path)
        self.rows = table.slice(0, min(limit, table.num_rows) if limit else table.num_rows).to_pylist()
        if not self.rows:
            raise ValueError("No training rows available")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        q = str(row[self.schema["query"]])
        if self.mode == "pairwise":
            positive, negative = str(row[self.schema["positive"]]), str(row[self.schema["negative"]])
            if positive == negative:
                raise ValueError(f"Pairwise row {index} has identical positive and negative text")
            return {"query": q, "positive": positive, "negative": negative}
        passage = self.schema["positive"] or self.schema["negative"] or next((x for x in self.schema["columns"] if "text" in x.lower() and x != self.schema["query"]), None)
        if not passage:
            raise ValueError("BCE pairs need a passage text column")
        return {"query": q, "passage": str(row[passage]), "label": float(row[self.schema["label"]])}


@dataclass
class PairCollator:
    tokenizer: Any
    max_length: int
    mode: str

    def _tokenize(self, queries: list[str], passages: list[str]) -> dict[str, torch.Tensor]:
        return self.tokenizer(queries, passages, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        if self.mode == "pairwise":
            qs = [item["query"] for item in batch]
            return {"positive": self._tokenize(qs, [item["positive"] for item in batch]), "negative": self._tokenize(qs, [item["negative"] for item in batch])}
        return {"inputs": self._tokenize([item["query"] for item in batch], [item["passage"] for item in batch]), "labels": torch.tensor([item["label"] for item in batch], dtype=torch.float32)}
