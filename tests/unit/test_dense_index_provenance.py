"""F3: dense document embeddings must bind immutable encoder provenance."""
import json

import numpy as np
import pandas as pd
import pytest

from src.models.bootstrap import MODEL_REGISTRY
from src.retrieval.dense_macro import (
    DenseMacroRetriever,
    chunk_ids_digest,
    is_mutable_revision,
    pinned_dense_revision,
)

PINNED = MODEL_REGISTRY["CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2"]["revision"]
MODEL = "CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2"


def _tiny_retriever(**kw):
    r = DenseMacroRetriever(model_name=MODEL, dimension=4, use_pyvi=False, device="cpu", **kw)
    r.chunk_ids = ["c1", "c2"]
    r.doc_ids = ["d1", "d1"]
    r.embeddings = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]], dtype=np.float32)
    return r


def test_mutable_revisions_rejected():
    assert is_mutable_revision(None)
    assert is_mutable_revision("")
    assert is_mutable_revision("main")
    assert is_mutable_revision("master")
    assert not is_mutable_revision(PINNED)


def test_pinned_revision_matches_registry():
    assert pinned_dense_revision(MODEL) == PINNED


def test_save_load_round_trip_retains_pin(tmp_path):
    r = _tiny_retriever(revision=PINNED)
    r.save(tmp_path / "idx")
    manifest = json.loads((tmp_path / "idx" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["model_revision"] == PINNED
    assert manifest["chunk_ids_sha256"] == chunk_ids_digest(["c1", "c2"])
    loaded = DenseMacroRetriever.load(tmp_path / "idx", model_name=MODEL, revision=PINNED, device="cpu", use_pyvi=False)
    assert loaded.revision == PINNED
    assert loaded.chunk_ids == ["c1", "c2"]
    assert loaded.embeddings.shape == (2, 4)


def test_save_refuses_mutable_or_missing_revision(tmp_path):
    with pytest.raises(ValueError, match="immutable"):
        _tiny_retriever(revision="main").save(tmp_path / "a")
    # Unknown model with no registry pin must fail closed instead of relabeling.
    unknown = DenseMacroRetriever(model_name="UNKNOWN/model", dimension=4, use_pyvi=False, device="cpu")
    unknown.chunk_ids = ["c1"]
    unknown.doc_ids = ["d1"]
    unknown.embeddings = np.array([[1.0, 0, 0, 0]], dtype=np.float32)
    with pytest.raises(ValueError, match="immutable"):
        unknown.save(tmp_path / "b")


def test_load_rejects_legacy_main_manifest(tmp_path):
    r = _tiny_retriever(revision=PINNED)
    out = r.save(tmp_path / "idx")
    manifest_path = out / "manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["model_revision"] = "main"
    data["revision"] = "main"
    manifest_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="mutable|missing"):
        DenseMacroRetriever.load(out, model_name=MODEL, revision=PINNED, device="cpu", use_pyvi=False)


def test_load_rejects_missing_revision_manifest(tmp_path):
    r = _tiny_retriever(revision=PINNED)
    out = r.save(tmp_path / "idx")
    manifest_path = out / "manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data.pop("model_revision", None)
    data.pop("revision", None)
    manifest_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="missing/mutable"):
        DenseMacroRetriever.load(out, model_name=MODEL, revision=PINNED, device="cpu", use_pyvi=False)


def test_load_rejects_wrong_model_and_revision(tmp_path):
    r = _tiny_retriever(revision=PINNED)
    out = r.save(tmp_path / "idx")
    with pytest.raises(ValueError, match="model mismatch"):
        DenseMacroRetriever.load(out, model_name="OTHER/model", revision=PINNED, device="cpu", use_pyvi=False)
    with pytest.raises(ValueError, match="revision mismatch"):
        DenseMacroRetriever.load(out, model_name=MODEL, revision="0" * 40, device="cpu", use_pyvi=False)
    with pytest.raises(ValueError, match="mutable"):
        DenseMacroRetriever.load(out, model_name=MODEL, revision="main", device="cpu", use_pyvi=False)


def test_load_rejects_changed_corpus_and_preprocessing(tmp_path):
    r = _tiny_retriever(revision=PINNED)
    out = r.save(tmp_path / "idx")
    # Corrupt chunk identity.
    df = pd.read_parquet(out / "chunks_meta.parquet")
    df.loc[0, "chunk_id"] = "CHANGED"
    df.to_parquet(out / "chunks_meta.parquet", index=False)
    with pytest.raises(ValueError, match="chunk-identity|row"):
        DenseMacroRetriever.load(out, model_name=MODEL, revision=PINNED, device="cpu", use_pyvi=False)

    r2 = _tiny_retriever(revision=PINNED)
    out2 = r2.save(tmp_path / "idx2")
    with pytest.raises(ValueError, match="preprocessing"):
        DenseMacroRetriever.load(out2, model_name=MODEL, revision=PINNED, device="cpu", use_pyvi=True)


def test_load_rejects_row_dim_mismatch(tmp_path):
    r = _tiny_retriever(revision=PINNED)
    out = r.save(tmp_path / "idx")
    # Truncate embeddings rows.
    arr = np.load(str(out / "embeddings.npy"))
    np.save(str(out / "embeddings.npy"), arr[:1])
    with pytest.raises(ValueError, match="row"):
        DenseMacroRetriever.load(out, model_name=MODEL, revision=PINNED, device="cpu", use_pyvi=False)


def test_load_rejects_nonfinite(tmp_path):
    r = _tiny_retriever(revision=PINNED)
    out = r.save(tmp_path / "idx")
    arr = np.load(str(out / "embeddings.npy")).astype(np.float32)
    arr[0, 0] = float("inf")
    np.save(str(out / "embeddings.npy"), arr.astype(np.float16))
    with pytest.raises(ValueError, match="non-finite"):
        DenseMacroRetriever.load(out, model_name=MODEL, revision=PINNED, device="cpu", use_pyvi=False)


def test_query_cache_rejects_dim_and_nonfinite(tmp_path):
    from src.pipeline.kaggle_train import (
        load_validated_train_query_embeddings,
        write_train_query_embedding_cache,
    )

    qids = ["q1"]
    texts = ["a"]
    write_train_query_embedding_cache(tmp_path, qids, texts,
                                      np.ones((1, 4), dtype=np.float32), MODEL, PINNED)
    with pytest.raises(ValueError, match="dim"):
        load_validated_train_query_embeddings(tmp_path, qids, texts, MODEL, PINNED, expected_dim=8)
    write_train_query_embedding_cache(tmp_path, qids, texts,
                                      np.array([[float("inf")] * 4], dtype=np.float32), MODEL, PINNED)
    with pytest.raises(ValueError, match="non-finite"):
        load_validated_train_query_embeddings(tmp_path, qids, texts, MODEL, PINNED, expected_dim=4)
