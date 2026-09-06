import json
import pytest
import pandas as pd
from pathlib import Path

from src.bundle.builder import ProductionBundleBuilder
from src.bundle.verifier import verify_production_bundle
from src.core.hashing import sha256_string


def make_valid_bundle_files(tmp_path):
    d = tmp_path / "src_artifacts"
    d.mkdir(parents=True, exist_ok=True)

    runtime_commit = "a" * 40
    dataset_fingerprint = "b" * 64
    config = {"reranker": {"base_model_name": "BAAI/bge-reranker-v2-m3"}}
    config_sha = sha256_string(json.dumps(config, sort_keys=True))

    lock_data = {
        "status": "LOCKED",
        "runtime_commit": runtime_commit,
        "dataset_sha256": dataset_fingerprint,
        "config_sha256": config_sha,
        "config": config,
    }
    (d / "production_lock.json").write_text(json.dumps(lock_data), encoding="utf-8")

    (d / "dataset_provenance.json").write_text(json.dumps({
        "dataset_fingerprint": dataset_fingerprint,
        "doc_count": 8532,
        "train_query_count": 7000,
        "public_query_count": 1000,
    }), encoding="utf-8")

    (d / "static_cache_provenance.json").write_text(json.dumps({
        "label_free": True,
        "qrels_used": False,
    }), encoding="utf-8")

    (d / "validation_summary.json").write_text(json.dumps({
        "folds": [f"fold_{i}" for i in range(5)],
        "doc_disjoint": {"recall@5": 0.82},
        "leakage_violations": 0,
        "duplicate_negative_violations": 0,
    }), encoding="utf-8")

    (d / "fusion_model.json").write_text(json.dumps({
        "schema_version": 1,
        "winning_method": "reciprocal_rank_fusion",
        "feature_schema_version": "v2",
        "feature_columns": ["raw_bm25_rank"],
        "rrf": {"k": 60, "weights": {"bm25": 1.0}},
    }), encoding="utf-8")

    # Final training pairs (7000 queries, positives and negatives)
    pairs_data = []
    for i in range(7000):
        qid = f"q_{i}"
        pairs_data.append({"query_id": qid, "doc_id": f"doc_{i}_pos", "label": 1.0})
        pairs_data.append({"query_id": qid, "doc_id": f"doc_{i}_neg", "label": 0.0})
    pd.DataFrame(pairs_data).to_parquet(d / "final_training_pairs.parquet", index=False)

    # Public candidates (1000 queries)
    pub_cands = []
    for i in range(1000):
        qid = f"pub_{i}"
        for r in range(5):
            pub_cands.append({"query_id": qid, "doc_id": f"doc_pub_{i}_{r}", "branch": "bm25", "rank": r + 1, "score": 10.0 - r})
    pd.DataFrame(pub_cands).to_parquet(d / "public_candidates.parquet", index=False)

    # Public evidence
    pub_ev = []
    for i in range(1000):
        qid = f"pub_{i}"
        for r in range(5):
            pub_ev.append({"query_id": qid, "doc_id": f"doc_pub_{i}_{r}", "evidence_text": f"evidence for {i} {r}"})
    pd.DataFrame(pub_ev).to_parquet(d / "public_evidence.parquet", index=False)

    return d, runtime_commit, dataset_fingerprint, config_sha


def build_bundle_from_source(bundle_dir, src_dir, runtime_commit, dataset_fingerprint, config_sha):
    builder = ProductionBundleBuilder(
        bundle_dir=bundle_dir,
        runtime_commit=runtime_commit,
        dataset_fingerprint=dataset_fingerprint,
        config_sha256=config_sha,
        strict_mandatory_check=True,
    )
    for f in [
        "final_training_pairs.parquet",
        "public_candidates.parquet",
        "public_evidence.parquet",
        "production_lock.json",
        "fusion_model.json",
        "static_cache_provenance.json",
        "validation_summary.json",
        "dataset_provenance.json",
    ]:
        builder.add_file(f, src_dir / f)
    builder.freeze()


def test_bundle_rejects_lock_runtime_mismatch(tmp_path):
    src_dir, rc, df, cs = make_valid_bundle_files(tmp_path)
    # Modify production lock to have different runtime commit
    lock_data = json.loads((src_dir / "production_lock.json").read_text(encoding="utf-8"))
    lock_data["runtime_commit"] = "f" * 40
    (src_dir / "production_lock.json").write_text(json.dumps(lock_data), encoding="utf-8")

    bundle_dir = tmp_path / "bundle"
    build_bundle_from_source(bundle_dir, src_dir, rc, df, cs)

    is_valid, errors = verify_production_bundle(bundle_dir, strict_mandatory=True)
    assert not is_valid
    assert any("runtime_commit mismatch" in e.lower() for e in errors)


def test_bundle_rejects_lock_dataset_mismatch(tmp_path):
    src_dir, rc, df, cs = make_valid_bundle_files(tmp_path)
    lock_data = json.loads((src_dir / "production_lock.json").read_text(encoding="utf-8"))
    lock_data["dataset_sha256"] = "f" * 64
    (src_dir / "production_lock.json").write_text(json.dumps(lock_data), encoding="utf-8")

    bundle_dir = tmp_path / "bundle"
    build_bundle_from_source(bundle_dir, src_dir, rc, df, cs)

    is_valid, errors = verify_production_bundle(bundle_dir, strict_mandatory=True)
    assert not is_valid
    assert any("dataset_fingerprint mismatch" in e.lower() or "dataset_sha256 mismatch" in e.lower() for e in errors)


def test_bundle_rejects_lock_config_mismatch(tmp_path):
    src_dir, rc, df, cs = make_valid_bundle_files(tmp_path)
    lock_data = json.loads((src_dir / "production_lock.json").read_text(encoding="utf-8"))
    lock_data["config_sha256"] = "f" * 64
    (src_dir / "production_lock.json").write_text(json.dumps(lock_data), encoding="utf-8")

    bundle_dir = tmp_path / "bundle"
    build_bundle_from_source(bundle_dir, src_dir, rc, df, cs)

    is_valid, errors = verify_production_bundle(bundle_dir, strict_mandatory=True)
    assert not is_valid
    assert any("config_sha256 mismatch" in e.lower() for e in errors)


def test_bundle_rejects_incomplete_5fold_validation(tmp_path):
    src_dir, rc, df, cs = make_valid_bundle_files(tmp_path)
    # Only 4 folds in summary
    val_data = json.loads((src_dir / "validation_summary.json").read_text(encoding="utf-8"))
    val_data["folds"] = ["fold_0", "fold_1", "fold_2", "fold_3"]
    (src_dir / "validation_summary.json").write_text(json.dumps(val_data), encoding="utf-8")

    bundle_dir = tmp_path / "bundle"
    build_bundle_from_source(bundle_dir, src_dir, rc, df, cs)

    is_valid, errors = verify_production_bundle(bundle_dir, strict_mandatory=True)
    assert not is_valid
    assert any("5/5" in e or "folds" in e.lower() for e in errors)


def test_bundle_rejects_missing_doc_disjoint(tmp_path):
    src_dir, rc, df, cs = make_valid_bundle_files(tmp_path)
    val_data = json.loads((src_dir / "validation_summary.json").read_text(encoding="utf-8"))
    del val_data["doc_disjoint"]
    (src_dir / "validation_summary.json").write_text(json.dumps(val_data), encoding="utf-8")

    bundle_dir = tmp_path / "bundle"
    build_bundle_from_source(bundle_dir, src_dir, rc, df, cs)

    is_valid, errors = verify_production_bundle(bundle_dir, strict_mandatory=True)
    assert not is_valid
    assert any("doc_disjoint" in e.lower() for e in errors)


def test_bundle_rejects_public_qid_gap(tmp_path):
    src_dir, rc, df, cs = make_valid_bundle_files(tmp_path)
    # Public candidates only has 999 queries instead of 1000
    pub_cands = []
    for i in range(999):
        qid = f"pub_{i}"
        pub_cands.append({"query_id": qid, "doc_id": f"doc_{i}", "branch": "bm25", "rank": 1, "score": 10.0})
    pd.DataFrame(pub_cands).to_parquet(src_dir / "public_candidates.parquet", index=False)

    bundle_dir = tmp_path / "bundle"
    build_bundle_from_source(bundle_dir, src_dir, rc, df, cs)

    is_valid, errors = verify_production_bundle(bundle_dir, strict_mandatory=True)
    assert not is_valid
    assert any("public" in e.lower() and ("qid" in e.lower() or "count" in e.lower() or "gap" in e.lower() or "1000" in e) for e in errors)


def test_bundle_rejects_final_pair_coverage_gap(tmp_path):
    src_dir, rc, df, cs = make_valid_bundle_files(tmp_path)
    # Only 6999 queries instead of 7000
    pairs_data = []
    for i in range(6999):
        qid = f"q_{i}"
        pairs_data.append({"query_id": qid, "doc_id": f"doc_{i}_pos", "label": 1.0})
        pairs_data.append({"query_id": qid, "doc_id": f"doc_{i}_neg", "label": 0.0})
    pd.DataFrame(pairs_data).to_parquet(src_dir / "final_training_pairs.parquet", index=False)

    bundle_dir = tmp_path / "bundle"
    build_bundle_from_source(bundle_dir, src_dir, rc, df, cs)

    is_valid, errors = verify_production_bundle(bundle_dir, strict_mandatory=True)
    assert not is_valid
    assert any("training_pairs" in e.lower() or "pair coverage" in e.lower() or "7000" in e for e in errors)


def test_bundle_rejects_fusion_payload_hash_mismatch(tmp_path):
    src_dir, rc, df, cs = make_valid_bundle_files(tmp_path)
    # Learned ranker with mismatched payload hash
    (src_dir / "fusion_model.txt").write_text("actual model content", encoding="utf-8")
    (src_dir / "fusion_model.json").write_text(json.dumps({
        "schema_version": 1,
        "winning_method": "learned_ranker",
        "feature_schema_version": "v2",
        "feature_columns": ["raw_bm25_rank"],
        "learned_model": {
            "file": "fusion_model.txt",
            "sha256": "9" * 64,  # wrong hash
        },
    }), encoding="utf-8")

    bundle_dir = tmp_path / "bundle"
    builder = ProductionBundleBuilder(
        bundle_dir=bundle_dir,
        runtime_commit=rc,
        dataset_fingerprint=df,
        config_sha256=cs,
        strict_mandatory_check=True,
    )
    for f in [
        "final_training_pairs.parquet",
        "public_candidates.parquet",
        "public_evidence.parquet",
        "production_lock.json",
        "fusion_model.json",
        "static_cache_provenance.json",
        "validation_summary.json",
        "dataset_provenance.json",
    ]:
        builder.add_file(f, src_dir / f)
    builder.add_file("fusion_model.txt", src_dir / "fusion_model.txt")
    builder.freeze()

    is_valid, errors = verify_production_bundle(bundle_dir, strict_mandatory=True)
    assert not is_valid
    assert any("digest mismatch" in e.lower() or "hash mismatch" in e.lower() for e in errors)
