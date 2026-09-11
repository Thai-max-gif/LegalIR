import json

from src.retrieval.index_identity import validate_index_identity


def test_identity_requires_preprocessing(tmp_path):
    path = tmp_path / "index.json"
    path.write_text(json.dumps({"model_id": "x", "model_revision": "y", "pooling": "mean", "embedding_dimension": 768}))
    assert validate_index_identity(path, "x", "y")["status"] == "BLOCKED"
