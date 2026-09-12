import json
import pytest
import pandas as pd
from pathlib import Path

from src.evidence.macro_store import MacroEvidenceStore
from src.evidence.selector import LazyPositiveLocalizer, LazyEvidencePackBuilder
from src.training.positive_localizer import PositiveLocalizer
from src.ranking.evidence_pack import EvidencePackBuilder


def test_legacy_positive_localizer_parity_on_official_sample():
    chunks_p = Path("kaggle_dataset/chunks.parquet")
    queries_p = Path("kaggle_dataset/queries_train.parquet")
    qrels_p = Path("kaggle_dataset/qrels_train.parquet")

    if not (chunks_p.is_file() and queries_p.is_file() and qrels_p.is_file()):
        pytest.skip("Official canonical dataset not present.")

    # Load 100 queries and qrels
    queries_df = pd.read_parquet(queries_p).head(100)
    q_col = "question_norm" if "question_norm" in queries_df.columns else "text"
    q_dict = dict(zip(queries_df["query_id"].astype(str), queries_df[q_col].astype(str)))

    qrels_df = pd.read_parquet(qrels_p)
    qrels_dict = {}
    for _, row in qrels_df.iterrows():
        qid = str(row["query_id"])
        did = str(row["doc_id"])
        if qid in q_dict:
            qrels_dict.setdefault(qid, []).append(did)

    # Initialize store and lazy localizer
    store = MacroEvidenceStore(chunks_p)
    lazy_localizer = LazyPositiveLocalizer(store)

    # Test 100 pairs
    count = 0
    for qid, gold_docs in qrels_dict.items():
        q_text = q_dict[qid]
        for gd in gold_docs:
            lazy_res = lazy_localizer.localize(q_text, gd)

            # Build localizer on just that doc's macro chunks
            doc_chunks = store.get_doc_chunks(gd)
            legacy_localizer = PositiveLocalizer([c.to_dict() for c in doc_chunks])
            legacy_res = legacy_localizer.localize(q_text, gd)

            lazy_cid = lazy_res["chunk_id"] if isinstance(lazy_res, dict) else (lazy_res.chunk_id if lazy_res else None)
            legacy_cid = legacy_res["chunk_id"] if isinstance(legacy_res, dict) else (legacy_res.chunk_id if legacy_res else None)

            assert lazy_cid == legacy_cid, f"Mismatch on qid={qid}, doc={gd}: lazy={lazy_cid}, legacy={legacy_cid}"
            count += 1
            if count >= 100:
                break
        if count >= 100:
            break

    assert count >= 50, f"Expected at least 50 pairs verified, got {count}"


def test_legacy_evidence_pack_parity_on_official_sample():
    chunks_p = Path("kaggle_dataset/chunks.parquet")
    queries_p = Path("kaggle_dataset/queries_train.parquet")
    docs_p = Path("kaggle_dataset/documents.parquet")

    if not (chunks_p.is_file() and queries_p.is_file() and docs_p.is_file()):
        pytest.skip("Official canonical dataset not present.")

    queries_df = pd.read_parquet(queries_p).head(100)
    q_col = "question_norm" if "question_norm" in queries_df.columns else "text"
    queries = list(zip(queries_df["query_id"].astype(str), queries_df[q_col].astype(str)))

    docs_df = pd.read_parquet(docs_p).head(100)
    doc_meta = {str(r["doc_id"]): dict(r) for r in docs_df.to_dict("records")}

    store = MacroEvidenceStore(chunks_p)
    lazy_builder = LazyEvidencePackBuilder(store, doc_metadata=doc_meta)

    # Legacy builder with doc metadata
    legacy_builder = EvidencePackBuilder(macro_chunks=[], doc_metadata=doc_meta)

    count = 0
    for qid, q_text in queries:
        for did in list(doc_meta.keys())[:1]:  # check first doc for each query
            doc_chunks = [c.to_dict() for c in store.get_doc_chunks(did)]
            lazy_pack = lazy_builder.build_pack(q_text, did)
            legacy_pack = legacy_builder.build_pack(q_text, did, candidate_chunks=doc_chunks)

            assert lazy_pack == legacy_pack, f"Pack mismatch on query {qid}, doc {did}"
            count += 1

    assert count >= 50, f"Expected at least 50 packs verified, got {count}"
