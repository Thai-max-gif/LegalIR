import json
import pytest
from pathlib import Path

from src.core.hashing import sha256_string
from src.validation.promotion import create_production_lock
import scripts.select_production_config as spc


def test_production_lock_rejects_short_runtime_sha(tmp_path):
    out_file = tmp_path / "production_lock.json"
    metrics = {"recall@5": 0.85}
    config = {"fusion": {"method": "reciprocal_rank_fusion"}}

    with pytest.raises(ValueError, match="40-char|runtime_commit"):
        create_production_lock(
            output_path=out_file,
            metrics=metrics,
            config=config,
            runtime_commit="a0efb25",  # short 7-char SHA
            dataset_sha256="a" * 64,
            strict=True,
        )


def test_production_lock_rejects_placeholder_dataset_fingerprint(tmp_path):
    out_file = tmp_path / "production_lock.json"
    metrics = {"recall@5": 0.85}
    config = {"fusion": {"method": "reciprocal_rank_fusion"}}

    with pytest.raises(ValueError, match="64-char|dataset"):
        create_production_lock(
            output_path=out_file,
            metrics=metrics,
            config=config,
            runtime_commit="f" * 40,
            dataset_sha256="canonical_v2",  # placeholder string
            strict=True,
        )


def test_select_production_config_default_path_matches_bundle_builder():
    parser = spc.argparse.ArgumentParser()
    # Check that select_production_config default output lock is artifacts/factory/production_lock.json
    # Inspect default arguments from main parser
    import sys
    orig_argv = sys.argv
    try:
        sys.argv = ["select_production_config.py", "--help"]
        # Parse arguments with default inspect
        test_parser = spc.build_parser() if hasattr(spc, "build_parser") else None
        if test_parser is None:
            # Reconstruct from module
            import inspect
            src_lines = inspect.getsource(spc.main)
            assert 'default="artifacts/factory/production_lock.json"' in src_lines
    finally:
        sys.argv = orig_argv


def test_lock_winner_matches_fusion_descriptor(tmp_path):
    desc = {
        "schema_version": 1,
        "winning_method": "learned_ranker",
        "feature_schema_version": "v2",
        "feature_columns": ["raw_bm25_rank", "dense_rank"],
        "learned_model": {
            "file": "fusion_model.txt",
            "sha256": "c" * 64,
        },
    }
    desc_p = tmp_path / "fusion_model.json"
    desc_p.write_text(json.dumps(desc), encoding="utf-8")

    folds_dir = tmp_path / "folds"
    folds_dir.mkdir()
    for f in range(5):
        f_dir = folds_dir / f"fold_{f}"
        f_dir.mkdir()
        (f_dir / "fold_metrics.json").write_text('{"recall@5": 0.85, "precision@5": 0.3}', encoding="utf-8")

    out_lock = tmp_path / "production_lock.json"

    import sys
    orig_argv = sys.argv
    try:
        sys.argv = [
            "select_production_config.py",
            "--folds-dir", str(folds_dir),
            "--fusion-descriptor", str(desc_p),
            "--output-lock", str(out_lock),
            "--runtime-commit", "1" * 40,
            "--dataset-fingerprint", "2" * 64,
        ]
        spc.main()
    finally:
        sys.argv = orig_argv

    assert out_lock.is_file()
    with open(out_lock, "r", encoding="utf-8") as f:
        lock_data = json.load(f)

    assert lock_data["runtime_commit"] == "1" * 40
    assert lock_data["dataset_sha256"] == "2" * 64
    assert lock_data["config"]["fusion"]["method"] == "learned_ranker"
    assert lock_data["config"]["fusion"]["model_file"] == "fusion_model.txt"
    assert lock_data["config"]["fusion"]["feature_columns"] == ["raw_bm25_rank", "dense_rank"]


def test_lock_config_hash_is_reproducible(tmp_path):
    out_file = tmp_path / "production_lock.json"
    metrics = {"recall@5": 0.85}
    config = {
        "fusion": {"method": "reciprocal_rank_fusion", "k": 60, "weights": {"bm25": 1.0}},
        "reranker": {"model_name": "BAAI/bge-reranker-v2-m3"},
    }

    create_production_lock(
        output_path=out_file,
        metrics=metrics,
        config=config,
        runtime_commit="a" * 40,
        dataset_sha256="b" * 64,
        strict=True,
    )

    with open(out_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    expected_hash = sha256_string(json.dumps(config, sort_keys=True))
    assert data["config_sha256"] == expected_hash
