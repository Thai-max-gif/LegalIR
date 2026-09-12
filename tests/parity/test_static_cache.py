import pytest
import pyarrow.parquet as pq
from pathlib import Path
from src.retrieval.static_cache import (
    StaticCandidateRecord,
    StaticCacheWriter,
    StaticCacheReader,
    STATIC_CACHE_SCHEMA,
)


def test_static_cache_writer_and_reader(tmp_path):
    out_file = tmp_path / "test_cache.parquet"
    writer = StaticCacheWriter(str(out_file), batch_size=2)

    records = [
        StaticCandidateRecord(
            query_id="q1",
            branch="bm25_legal",
            rank=1,
            doc_id="doc_100",
            score=12.5,
            best_chunk_id="chunk_1",
        ),
        StaticCandidateRecord(
            query_id="q1",
            branch="dense",
            rank=1,
            doc_id="doc_200",
            score=0.89,
            best_chunk_id="chunk_2",
        ),
        StaticCandidateRecord(
            query_id="q2",
            branch="exact",
            rank=1,
            doc_id="doc_300",
            score=1.0,
            best_chunk_id=None,
        ),
    ]

    for r in records:
        writer.write_record(r)
    writer.close()

    assert out_file.is_file()

    # Read back
    reader = StaticCacheReader(str(out_file))
    q1_cands = reader.get_query_candidates("q1")
    assert len(q1_cands) == 2
    assert q1_cands[0].doc_id == "doc_100"
    assert q1_cands[0].branch == "bm25_legal"
    assert q1_cands[0].score == pytest.approx(12.5)

    q2_cands = reader.get_query_candidates("q2")
    assert len(q2_cands) == 1
    assert q2_cands[0].doc_id == "doc_300"
    assert q2_cands[0].best_chunk_id is None

    # Check all query IDs
    qids = reader.get_query_ids()
    assert set(qids) == {"q1", "q2"}


def test_static_cache_writer_no_qrels_accepted():
    writer = StaticCacheWriter("dummy.parquet")
    # Verify no attribute or method accepts qrels
    assert not hasattr(writer, "qrels")
    assert not hasattr(writer, "gold_docs")


def test_static_cache_reader_never_materializes_full_dataframe(tmp_path, monkeypatch):
    out_file = tmp_path / "bounded_cache.parquet"
    writer = StaticCacheWriter(str(out_file), batch_size=5)
    for i in range(10):
        writer.write_record(
            StaticCandidateRecord(
                query_id=f"q_{i}",
                branch="bm25",
                rank=1,
                doc_id=f"doc_{i}",
                score=1.0,
            )
        )
    writer.close()

    reader = StaticCacheReader(str(out_file))

    # Assert that reader does not store or produce a full pandas dataframe
    assert not hasattr(reader, "_df")
    cands = reader.get_query_candidates("q_3")
    assert len(cands) == 1
    assert cands[0].doc_id == "doc_3"


def test_static_cache_cli_writes_nonempty_train_and_public_cache(tmp_path):
    import subprocess
    import sys

    # Skip in CI if canonical dataset Parquet files are not present
    dataset_p = Path("kaggle_dataset")
    if not (dataset_p / "chunks.parquet").is_file():
        pytest.skip("Full canonical dataset not present on disk.")

    out_dir = tmp_path / "cli_cache"
    cmd = [
        sys.executable,
        "scripts/build_static_cache.py",
        "--dataset-dir",
        "kaggle_dataset",
        "--output-dir",
        str(out_dir),
        "--max-queries",
        "3",
        "--max-corpus-chunks",
        "200",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"STDOUT: {res.stdout}\nSTDERR: {res.stderr}"

    train_p = out_dir / "static_candidates_train.parquet"
    public_p = out_dir / "static_candidates_public.parquet"

    assert train_p.is_file()
    assert public_p.is_file()

    reader_train = StaticCacheReader(train_p)
    assert len(reader_train.get_query_ids()) == 3
    reader_pub = StaticCacheReader(public_p)
    assert len(reader_pub.get_query_ids()) == 3
