"""Query-embedding cache must validate order, content, and encoder identity."""
import json

import numpy as np
import pytest

from src.pipeline.kaggle_train import (
    load_validated_train_query_embeddings,
    write_train_query_embedding_cache,
)


def test_roundtrip_valid(tmp_path):
    qids = ["q1", "q2"]
    texts = ["hello", "world"]
    embs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    write_train_query_embedding_cache(tmp_path, qids, texts, embs, "ENC", "rev1")
    loaded = load_validated_train_query_embeddings(tmp_path, qids, texts, "ENC", "rev1")
    assert np.allclose(loaded, embs)


def test_rejects_missing_sidecar(tmp_path):
    qids = ["q1"]
    np.save(str(tmp_path / "train_query_embeddings.npy"), np.ones((1, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="sidecar"):
        load_validated_train_query_embeddings(tmp_path, qids, ["t"], "ENC", "rev1")


def test_rejects_text_change(tmp_path):
    qids = ["q1", "q2"]
    write_train_query_embedding_cache(tmp_path, qids, ["a", "b"],
                                      np.ones((2, 2), dtype=np.float32), "ENC", "rev1")
    with pytest.raises(ValueError, match="text"):
        load_validated_train_query_embeddings(tmp_path, qids, ["a", "CHANGED"], "ENC", "rev1")


def test_rejects_encoder_change(tmp_path):
    qids = ["q1"]
    write_train_query_embedding_cache(tmp_path, qids, ["a"],
                                      np.ones((1, 2), dtype=np.float32), "ENC", "rev1")
    with pytest.raises(ValueError, match="encoder"):
        load_validated_train_query_embeddings(tmp_path, qids, ["a"], "ENC", "rev2")


def test_remaps_reordered_ids(tmp_path):
    qids = ["q1", "q2"]
    texts = ["a", "b"]
    embs = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    write_train_query_embedding_cache(tmp_path, qids, texts, embs, "ENC", "rev1")
    # Same set, different order: rows must follow requested order.
    loaded = load_validated_train_query_embeddings(tmp_path, ["q2", "q1"], ["b", "a"], "ENC", "rev1")
    assert np.allclose(loaded[0], [0.0, 2.0])
    assert np.allclose(loaded[1], [1.0, 0.0])


def test_rejects_id_set_change(tmp_path):
    write_train_query_embedding_cache(tmp_path, ["q1", "q2"], ["a", "b"],
                                      np.ones((2, 2), dtype=np.float32), "ENC", "rev1")
    with pytest.raises(ValueError, match="query-id"):
        load_validated_train_query_embeddings(tmp_path, ["q1", "qX"], ["a", "b"], "ENC", "rev1")
