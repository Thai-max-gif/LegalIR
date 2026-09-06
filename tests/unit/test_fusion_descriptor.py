import json
import pytest
import pandas as pd
from pathlib import Path

from src.core.hashing import sha256_file
from src.ranking.train_fusion import train_and_evaluate_fusion_cv
from src.bundle.builder import ProductionBundleBuilder
from src.bundle.verifier import verify_production_bundle
from src.production.public_rerank import rerank_and_fuse_public_predictions
from src.retrieval.static_cache import StaticCacheWriter, StaticCandidateRecord


def test_fusion_descriptor_matches_cross_fitted_winner(tmp_path):
    # Construct synthetic OOF features with 2 folds
    oof_data = []
    for f in [0, 1]:
        for q in range(4):
            qid = f"f{f}_q{q}"
            for d in range(3):
                did = f"doc_{q}_{d}"
                oof_data.append({
                    "query_id": qid,
                    "doc_id": did,
                    "fold": f,
                    "raw_bm25_rank": float(d + 1),
                    "raw_bm25_score": 10.0 - d,
                    "dense_rank": float(d + 1),
                    "dense_score": 0.9 - 0.1 * d,
                    "reranker_score": 2.0 - d,
                    "label": 1.0 if d == 0 else 0.0,
                })
    oof_df = pd.DataFrame(oof_data)
    qrels = {f"f{f}_q{q}": [f"doc_{q}_0"] for f in [0, 1] for q in range(4)}

    output_dir = tmp_path / "fusion_out"
    res = train_and_evaluate_fusion_cv(
        oof_df=oof_df,
        qrels_dict=qrels,
        output_dir=output_dir,
        num_boost_round=10,
    )

    desc_file = output_dir / "fusion_model.json"
    assert desc_file.is_file(), "train_and_evaluate_fusion_cv must generate fusion_model.json"

    with open(desc_file, "r", encoding="utf-8") as f:
        desc = json.load(f)

    assert desc["schema_version"] == 1
    assert desc["winning_method"] == res["winning_method"]
    assert "feature_columns" in desc

    if desc["winning_method"] == "reciprocal_rank_fusion":
        assert "rrf" in desc
        assert "k" in desc["rrf"]
        assert "weights" in desc["rrf"]
        assert "learned_model" not in desc
    else:
        assert "learned_model" in desc
        assert "file" in desc["learned_model"]
        assert "sha256" in desc["learned_model"]
        payload_file = output_dir / desc["learned_model"]["file"]
        assert payload_file.is_file()
        assert sha256_file(payload_file) == desc["learned_model"]["sha256"]


def test_rrf_descriptor_roundtrip_preserves_exact_ranking(tmp_path):
    desc = {
        "schema_version": 1,
        "winning_method": "reciprocal_rank_fusion",
        "feature_schema_version": "v2",
        "feature_columns": ["raw_bm25_rank", "dense_rank"],
        "rrf": {
            "k": 60,
            "weights": {"bm25": 1.5, "dense": 2.0, "rerank": 3.0},
        },
    }
    desc_p = tmp_path / "fusion_model.json"
    desc_p.write_text(json.dumps(desc), encoding="utf-8")

    cands_p = tmp_path / "candidates.parquet"
    writer = StaticCacheWriter(str(cands_p))
    writer.write_record(StaticCandidateRecord("q1", "bm25_legal", 1, "docA", 10.0))
    writer.write_record(StaticCandidateRecord("q1", "dense", 1, "docB", 0.9))
    writer.close()

    lock_p = tmp_path / "production_lock.json"
    lock_data = {"status": "LOCKED", "config": {}}
    lock_p.write_text(json.dumps(lock_data), encoding="utf-8")

    preds = rerank_and_fuse_public_predictions(
        public_candidates_path=cands_p,
        production_lock_path=lock_p,
        fusion_model_path=desc_p,
        adapter_scores={"q1": {"docA": 1.0, "docB": 1.0}},
        top_k=2,
    )
    assert preds["q1"][0] == "docB"  # dense weight 2.0 > bm25 weight 1.5


def test_learned_descriptor_requires_hashed_payload(tmp_path):
    bundle_dir = tmp_path / "bundle"
    desc = {
        "schema_version": 1,
        "winning_method": "learned_ranker",
        "feature_schema_version": "v2",
        "feature_columns": ["raw_bm25_rank"],
        "learned_model": {
            "file": "fusion_model.txt",
            "sha256": "0" * 64,  # Intentional invalid hash
        },
    }
    (tmp_path / "fusion_model.json").write_text(json.dumps(desc), encoding="utf-8")
    (tmp_path / "fusion_model.txt").write_text("dummy model content", encoding="utf-8")

    builder = ProductionBundleBuilder(
        bundle_dir=bundle_dir,
        runtime_commit="1" * 40,
        dataset_fingerprint="2" * 64,
        config_sha256="3" * 64,
        strict_mandatory_check=False,
    )
    builder.add_file("fusion_model.json", tmp_path / "fusion_model.json")
    builder.add_file("fusion_model.txt", tmp_path / "fusion_model.txt")
    builder.freeze()

    is_valid, errors = verify_production_bundle(bundle_dir, strict_mandatory=False)
    assert not is_valid
    assert any("fusion" in e.lower() or "sha" in e.lower() or "mismatch" in e.lower() for e in errors)


def test_bundle_fusion_descriptor_is_consumable_by_public_inference(tmp_path):
    desc = {
        "schema_version": 1,
        "winning_method": "reciprocal_rank_fusion",
        "feature_schema_version": "v2",
        "feature_columns": ["raw_bm25_rank", "dense_rank"],
        "rrf": {
            "k": 60,
            "weights": {"bm25": 1.0, "dense": 1.0, "rerank": 1.0},
        },
    }
    desc_p = tmp_path / "fusion_model.json"
    desc_p.write_text(json.dumps(desc), encoding="utf-8")

    cands_p = tmp_path / "candidates.parquet"
    writer = StaticCacheWriter(str(cands_p))
    writer.write_record(StaticCandidateRecord("q1", "bm25_legal", 1, "docA", 10.0))
    writer.close()

    lock_p = tmp_path / "production_lock.json"
    lock_data = {"status": "LOCKED", "config": {}}
    lock_p.write_text(json.dumps(lock_data), encoding="utf-8")

    preds = rerank_and_fuse_public_predictions(
        public_candidates_path=cands_p,
        production_lock_path=lock_p,
        fusion_model_path=desc_p,
        top_k=1,
    )
    assert preds["q1"] == ["docA"]
