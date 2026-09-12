import os
import json
import zipfile
import pytest
import pandas as pd
from src.ranking.selector import TopKSelector

def test_submission_format_compliance(tmp_path):
    # Mock submission
    sub = {
        "1001": {"answer": ["740", "2113"]},
        "1002": {"answer": ["280282"]}
    }
    sub_file = tmp_path / "submission.json"
    with open(sub_file, "w", encoding="utf-8") as f:
        json.dump(sub, f)

    zip_file = tmp_path / "submission.zip"
    with zipfile.ZipFile(zip_file, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(sub_file, arcname="submission.json")

    # Verify zip content
    with zipfile.ZipFile(zip_file, "r") as zf:
        namelist = zf.namelist()
        assert namelist == ["submission.json"]

        with zf.open("submission.json") as jf:
            loaded = json.load(jf)
            assert len(loaded) == 2
            for qid, v in loaded.items():
                assert "answer" in v
                assert isinstance(v["answer"], list)
                assert 1 <= len(v["answer"]) <= 5
                assert len(set(v["answer"])) == len(v["answer"])


def test_selector_accepts_scored_candidates_and_returns_unique_string_ids():
    selector = TopKSelector(max_k=5)

    result = selector.select([("doc_1", 10.0), ("doc_2", 8.0), ("doc_1", 7.0), ("doc_3", 6.0)])

    assert result == ["doc_1", "doc_2", "doc_3"]
    assert all(isinstance(doc_id, str) for doc_id in result)
    assert 1 <= len(result) <= 5
