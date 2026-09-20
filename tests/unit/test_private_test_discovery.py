import json
import os
from pathlib import Path
import pytest
from src.pipeline.kaggle_train import discover_public_test_file, validate_official_task1_identity
from src.evaluation.submission import validate_submission, validate_submission_zip, package_submission


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


def test_validate_submission_1000_public_queries(tmp_path: Path):
    pub_json = tmp_path / "public-official.json"
    q_dict = {f"pub_{i}": {"question": f"Cau hoi {i}?", "answer": None} for i in range(1000)}
    pub_json.write_text(json.dumps(q_dict), encoding="utf-8")
    corpus_docs = {f"doc_{j}" for j in range(100)}

    # Valid predictions matching all 1000 queries with 5 unique valid docs
    preds = {f"pub_{i}": {"answer": [f"doc_{j % 100}" for j in range(i, i + 5)]} for i in range(1000)}
    res = validate_submission(preds, test_json=pub_json, corpus_doc_ids=corpus_docs, raise_on_error=False)
    assert res["is_valid"] is True
    assert res["total_queries"] == 1000

    # Incomplete (missing query)
    incomp = dict(preds)
    incomp.pop("pub_0")
    res_incomp = validate_submission(incomp, test_json=pub_json, raise_on_error=False)
    assert res_incomp["is_valid"] is False
    assert any("mismatch" in e for e in res_incomp["errors"])

    # Duplicate document IDs
    dup_preds = dict(preds)
    dup_preds["pub_1"] = {"answer": ["doc_0", "doc_0", "doc_1", "doc_2", "doc_3"]}
    res_dup = validate_submission(dup_preds, test_json=pub_json, raise_on_error=False)
    assert res_dup["is_valid"] is False
    assert any("duplicate" in e for e in res_dup["errors"])

    # Invalid document ID not in corpus
    inv_preds = dict(preds)
    inv_preds["pub_2"] = {"answer": ["doc_0", "doc_1", "doc_2", "doc_3", "unknown_doc"]}
    res_inv = validate_submission(inv_preds, test_json=pub_json, corpus_doc_ids=corpus_docs, raise_on_error=False)
    assert res_inv["is_valid"] is False
    assert any("unknown document" in e for e in res_inv["errors"])


def test_validate_submission_2080_private_queries(tmp_path: Path):
    priv_json = tmp_path / "private-official.json"
    q_dict = {f"priv_{i}": {"question": f"Cau hoi {i}?", "answer": None} for i in range(2080)}
    priv_json.write_text(json.dumps(q_dict), encoding="utf-8")
    corpus_docs = {f"doc_{j}" for j in range(100)}

    # Valid predictions matching all 2080 queries with exactly 5 unique valid docs
    preds = {f"priv_{i}": {"answer": [f"doc_{j % 100}" for j in range(i, i + 5)]} for i in range(2080)}
    res = validate_submission(preds, test_json=priv_json, corpus_doc_ids=corpus_docs, raise_on_error=False)
    assert res["is_valid"] is True
    assert res["total_queries"] == 2080

    # Package zip and validate zip format
    sub_json = tmp_path / "submission.json"
    sub_zip = tmp_path / "submission.zip"
    package_submission(preds, sub_json, sub_zip)
    zip_res = validate_submission_zip(sub_zip)
    assert zip_res["is_valid"] is True

    # Incomplete (missing query)
    incomp = dict(preds)
    incomp.pop("priv_0")
    res_incomp = validate_submission(incomp, test_json=priv_json, raise_on_error=False)
    assert res_incomp["is_valid"] is False
    assert any("mismatch" in e for e in res_incomp["errors"])

    # Duplicate document IDs
    dup_preds = dict(preds)
    dup_preds["priv_1"] = {"answer": ["doc_0", "doc_0", "doc_1", "doc_2", "doc_3"]}
    res_dup = validate_submission(dup_preds, test_json=priv_json, raise_on_error=False)
    assert res_dup["is_valid"] is False
    assert any("duplicate" in e for e in res_dup["errors"])

    # Invalid document ID
    inv_preds = dict(preds)
    inv_preds["priv_2"] = {"answer": ["doc_0", "doc_1", "doc_2", "doc_3", "unknown_doc"]}
    res_inv = validate_submission(inv_preds, test_json=priv_json, corpus_doc_ids=corpus_docs, raise_on_error=False)
    assert res_inv["is_valid"] is False
    assert any("unknown document" in e for e in res_inv["errors"])

    # Non-5 length answer (e.g. empty or >5)
    len_preds = dict(preds)
    len_preds["priv_3"] = {"answer": ["doc_0", "doc_1", "doc_2", "doc_3", "doc_4", "doc_5"]}
    res_len = validate_submission(len_preds, test_json=priv_json, raise_on_error=False)
    assert res_len["is_valid"] is False
    assert any("length must be between 1 to 5" in e for e in res_len["errors"])
