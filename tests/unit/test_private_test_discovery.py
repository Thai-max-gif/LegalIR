import os
from pathlib import Path
import pytest
from src.pipeline.kaggle_train import discover_public_test_file, validate_official_task1_identity
from src.evaluation.submission import validate_submission


def test_public_and_private_test_discovery(tmp_path: Path):
    # Setup test directory with both files
    data_dir = tmp_path / "dataset"
    data_dir.mkdir()
    pub_file = data_dir / "public-official.json"
    priv_file = data_dir / "private-official.json"
    pub_file.write_text('{"q1": {"question": "test public", "answer": null}}', encoding="utf-8")
    priv_file.write_text('{"q2": {"question": "test private", "answer": null}}', encoding="utf-8")

    # Default discovery should yield public-official.json
    disc_default = discover_public_test_file(data_dir=data_dir)
    assert disc_default == pub_file.resolve()

    # With phase set to private, should yield private-official.json
    os.environ["LEGALIR_TEST_PHASE"] = "private"
    try:
        disc_priv = discover_public_test_file(data_dir=data_dir)
        assert disc_priv == priv_file.resolve()
    finally:
        del os.environ["LEGALIR_TEST_PHASE"]

    # With explicit environment variable
    os.environ["LEGALIR_TEST_FILE"] = str(priv_file)
    try:
        disc_env = discover_public_test_file(data_dir=data_dir)
        assert disc_env == priv_file.resolve()
    finally:
        del os.environ["LEGALIR_TEST_FILE"]


def test_validate_submission_with_private_test(tmp_path: Path):
    test_json = tmp_path / "private-official.json"
    q_dict = {f"q_{i}": {"question": f"Cau hoi {i}?", "answer": None} for i in range(2080)}
    import json
    test_json.write_text(json.dumps(q_dict), encoding="utf-8")

    # Valid predictions matching all 2080 queries
    preds = {f"q_{i}": {"answer": ["d1", "d2"]} for i in range(2080)}
    res = validate_submission(preds, test_json=test_json, raise_on_error=False)
    assert res["is_valid"] is True
    assert res["total_queries"] == 2080

    # Incomplete predictions should fail
    incomplete = {f"q_{i}": {"answer": ["d1"]} for i in range(100)}
    res_bad = validate_submission(incomplete, test_json=test_json, raise_on_error=False)
    assert res_bad["is_valid"] is False
    assert any("mismatch" in e for e in res_bad["errors"])
