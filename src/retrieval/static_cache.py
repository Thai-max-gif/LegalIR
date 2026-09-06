"""Static retrieval candidate cache schema, streaming Parquet writer, and memory-bounded reader."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq


STATIC_CACHE_SCHEMA = pa.schema([
    ("query_id", pa.string()),
    ("branch", pa.string()),
    ("rank", pa.int32()),
    ("doc_id", pa.string()),
    ("score", pa.float32()),
    ("best_chunk_id", pa.string()),
    ("second_score", pa.float32()),
    ("mean_score", pa.float32()),
    ("extra_json", pa.string()),
])


@dataclass
class StaticCandidateRecord:
    """A single candidate document retrieved by a static retrieval branch."""

    query_id: str
    branch: str
    rank: int
    doc_id: str
    score: float
    best_chunk_id: Optional[str] = None
    second_score: Optional[float] = None
    mean_score: Optional[float] = None
    extra_json: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class StaticCacheWriter:
    """
    Streams static retrieval candidates to normalized Parquet in bounded batches.
    Guaranteed label-free: takes no qrels, ground-truth, or fold labels.
    Persists a companion query index (.index.json) to allow memory-bounded access.
    """

    def __init__(self, output_path: Union[str, Path], batch_size: int = 5000):
        self.output_path = Path(output_path)
        self.batch_size = batch_size
        self.buffer: List[StaticCandidateRecord] = []
        self.writer: Optional[pq.ParquetWriter] = None
        self.total_written = 0
        self.query_index: Dict[str, Dict[str, int]] = {}  # query_id -> {start, count}

    def _init_writer(self) -> None:
        if self.writer is None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self.writer = pq.ParquetWriter(str(self.output_path), STATIC_CACHE_SCHEMA)

    def write_record(self, record: StaticCandidateRecord) -> None:
        qid = record.query_id
        if qid not in self.query_index:
            self.query_index[qid] = {"start": self.total_written + len(self.buffer), "count": 1}
        else:
            self.query_index[qid]["count"] += 1

        self.buffer.append(record)
        if len(self.buffer) >= self.batch_size:
            self.flush()

    def write_records(self, records: List[StaticCandidateRecord]) -> None:
        for r in records:
            self.write_record(r)

    def flush(self) -> None:
        if not self.buffer:
            return
        self._init_writer()

        pydict = {
            "query_id": [r.query_id for r in self.buffer],
            "branch": [r.branch for r in self.buffer],
            "rank": [r.rank for r in self.buffer],
            "doc_id": [r.doc_id for r in self.buffer],
            "score": [float(r.score) for r in self.buffer],
            "best_chunk_id": [r.best_chunk_id for r in self.buffer],
            "second_score": [float(r.second_score) if r.second_score is not None else None for r in self.buffer],
            "mean_score": [float(r.mean_score) if r.mean_score is not None else None for r in self.buffer],
            "extra_json": [r.extra_json for r in self.buffer],
        }

        table = pa.Table.from_pydict(pydict, schema=STATIC_CACHE_SCHEMA)
        self.writer.write_table(table)
        self.total_written += len(self.buffer)
        self.buffer.clear()

    def close(self) -> None:
        self.flush()
        if self.writer is not None:
            self.writer.close()
            self.writer = None

        # Write companion query index
        index_path = self.output_path.with_suffix(".index.json")
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(self.query_index, f)


class StaticCacheReader:
    """
    Memory-bounded reader for static retrieval candidates.
    Never materializes the entire cache table into Pandas memory.
    Uses PyArrow Dataset predicate pushdown filtering.
    """

    def __init__(self, cache_path: Union[str, Path]):
        self.cache_path = Path(cache_path)
        if not self.cache_path.is_file():
            raise FileNotFoundError(f"Static cache file not found: {self.cache_path}")

        self.dataset = ds.dataset(str(self.cache_path), format="parquet")
        self._query_ids: Optional[List[str]] = None

        # Load companion index if present
        index_p = self.cache_path.with_suffix(".index.json")
        self.query_index: Optional[Dict[str, Dict[str, int]]] = None
        if index_p.is_file():
            try:
                with open(index_p, "r", encoding="utf-8") as f:
                    self.query_index = json.load(f)
            except Exception:
                self.query_index = None

    def get_query_ids(self) -> List[str]:
        """Return list of distinct query IDs in cache without full table load."""
        if self.query_index is not None:
            return list(self.query_index.keys())

        if self._query_ids is None:
            # Read single column into Arrow table and extract unique values
            q_col = pq.read_table(str(self.cache_path), columns=["query_id"])["query_id"]
            unique_qids = pc.unique(q_col).to_pylist()
            self._query_ids = unique_qids
        return self._query_ids

    def get_query_candidates(
        self, query_id: str, branch: Optional[str] = None
    ) -> List[StaticCandidateRecord]:
        """
        Fetch candidate records for a single query using PyArrow dataset filtering.
        Memory bounded: reads ONLY matching row groups/rows.
        """
        filt = (pc.field("query_id") == str(query_id))
        if branch is not None:
            filt = filt & (pc.field("branch") == str(branch))

        # Read only filtered slice
        table_slice = self.dataset.to_table(filter=filt)
        if len(table_slice) == 0:
            return []

        # Convert the small slice to pydict
        data = table_slice.to_pydict()
        records: List[StaticCandidateRecord] = []
        num_rows = len(data["query_id"])

        best_chunks = data.get("best_chunk_id", [None] * num_rows)
        second_scores = data.get("second_score", [None] * num_rows)
        mean_scores = data.get("mean_score", [None] * num_rows)
        extra_jsons = data.get("extra_json", [""] * num_rows)

        for i in range(num_rows):
            sec_sc = second_scores[i]
            mean_sc = mean_scores[i]
            records.append(
                StaticCandidateRecord(
                    query_id=data["query_id"][i],
                    branch=data["branch"][i],
                    rank=int(data["rank"][i]),
                    doc_id=data["doc_id"][i],
                    score=float(data["score"][i]),
                    best_chunk_id=best_chunks[i],
                    second_score=float(sec_sc) if sec_sc is not None else None,
                    mean_score=float(mean_sc) if mean_sc is not None else None,
                    extra_json=extra_jsons[i],
                )
            )

        return records
