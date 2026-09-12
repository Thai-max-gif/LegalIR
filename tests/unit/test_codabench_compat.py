import json
import zipfile
from pathlib import Path
import pytest

from src.evaluation.codabench_compat import assert_official_equivalence
from src.evaluation.submission import package_submission, validate_submission


def test_internal_metrics_equal_official_scorer():
    truth = {"q1": ["1", "2"], "q2": ["3"]}
    pred = {"q1": {"answer": ["2", "9"]}, "q2": {"answer": ["3"]}}
    metrics = assert_official_equivalence(pred, truth)
    assert metrics["recall"] == pytest.approx(0.75)
    assert metrics["precision"] == pytest.approx(0.75)


def test_packaged_zip_contains_exact_json_bytes(tmp_path: Path):
    pred = {"q1": {"answer": ["1"]}}
    json_path = tmp_path / "submission.json"
    zip_path = tmp_path / "submission.zip"
    package_submission(pred, json_path, zip_path)

    assert zip_path.exists()
    with zipfile.ZipFile(zip_path) as archive:
        assert archive.namelist() == ["submission.json"]
        assert archive.read("submission.json") == json_path.read_bytes()


def test_validator_rejects_extra_and_missing_query():
    with pytest.raises(ValueError, match="query keys"):
        validate_submission({"q1": {"answer": ["1"]}, "extra": {"answer": ["1"]}}, {"q1"}, {"1"})

    with pytest.raises(ValueError, match="query keys"):
        validate_submission({"q1": {"answer": ["1"]}}, {"q1", "q2"}, {"1"})


def test_validator_rejects_empty_and_overflow_answers():
    with pytest.raises(ValueError, match="1 to 5"):
        validate_submission({"q1": {"answer": []}}, {"q1"}, {"1"})

    with pytest.raises(ValueError, match="1 to 5"):
        validate_submission({"q1": {"answer": ["1", "2", "3", "4", "5", "6"]}}, {"q1"}, {"1", "2", "3", "4", "5", "6"})


def test_validator_rejects_non_corpus_ids():
    with pytest.raises(ValueError, match="unknown document IDs"):
        validate_submission({"q1": {"answer": ["999999"]}}, {"q1"}, {"1", "2"})


def test_validator_rejects_duplicate_doc_ids():
    with pytest.raises(ValueError, match="duplicate document IDs"):
        validate_submission({"q1": {"answer": ["1", "1"]}}, {"q1"}, {"1", "2"})
