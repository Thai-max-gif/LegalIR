import json
import zipfile
from pathlib import Path
import pytest

from src.dataset.source_reader import iter_official_contexts


def test_reader_streams_sorted_contexts_without_extracting(tmp_path: Path):
    archive = tmp_path / "selected-contexts.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("context_2.json", json.dumps({"id": 2, "passage": "B"}))
        zf.writestr("context_1.json", json.dumps({"id": 1, "passage": "A"}))
    rows = list(iter_official_contexts(archive))
    assert [row["id"] for row in rows] == ["1", "2"]
    assert not (tmp_path / "selected-contexts").exists()


def test_reader_rejects_duplicate_document_ids(tmp_path: Path):
    archive = tmp_path / "selected-contexts.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("context_1.json", json.dumps({"id": 1, "passage": "A"}))
        zf.writestr("nested/context_2.json", json.dumps({"id": 1, "passage": "B"}))
    with pytest.raises(ValueError, match="duplicate document ID 1"):
        list(iter_official_contexts(archive))


def test_reader_rejects_empty_archive(tmp_path: Path):
    archive = tmp_path / "empty.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("notes.txt", "nothing")
    with pytest.raises(ValueError, match="no context_.*json members"):
        list(iter_official_contexts(archive))
