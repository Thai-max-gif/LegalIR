from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.retrieval.dense_macro import DenseMacroRetriever


def test_dense_macro_retriever_initialization():
    retriever = DenseMacroRetriever(
        model_name="CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2",
        dimension=768,
        use_pyvi=True,
    )

    assert retriever.dimension == 768
    assert retriever.model_name == "CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2"
    text = "Thời hạn cấp đăng ký xe máy là bao lâu?"
    norm = retriever.preprocess_text(text)
    assert len(norm) > 0


def test_preprocess_text_calls_pyvi_and_segments_legal_phrases(monkeypatch):
    from pyvi import ViTokenizer

    calls = []
    original_tokenize = ViTokenizer.tokenize

    def recording_tokenize(text):
        calls.append(text)
        return original_tokenize(text)

    monkeypatch.setattr(ViTokenizer, "tokenize", recording_tokenize)
    retriever = DenseMacroRetriever(use_pyvi=True)

    segmented = retriever.preprocess_text("Thời hạn cấp đăng ký xe máy là bao lâu?")

    assert calls == ["Thời hạn cấp đăng ký xe máy là bao lâu?"]
    assert "đăng_ký" in segmented


def test_preprocess_text_requires_pyvi_when_enabled(monkeypatch):
    retriever = DenseMacroRetriever(use_pyvi=True)
    monkeypatch.setitem(sys.modules, "pyvi", None)

    with pytest.raises(ImportError, match="PyVi is required"):
        retriever.preprocess_text("đăng ký xe")


def test_exact_dense_search_aggregates_chunks_to_documents(tmp_path: Path):
    emb_path = tmp_path / "embeddings.npy"
    # 3 vectors: dim=2
    np.save(emb_path, np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float16))

    retriever = DenseMacroRetriever.from_arrays(
        embeddings_path=emb_path,
        chunk_ids=["a1", "a2", "b1"],
        doc_ids=["A", "A", "B"],
        query_encoder=lambda _: np.array([[1.0, 0.0]], dtype=np.float32),
    )

    res = retriever.retrieve("query text", top_k=2)
    assert len(res) == 2
    assert res[0]["doc_id"] == "A"
    assert res[0]["dense_best_chunk_id"] == "a1"
    assert res[0]["dense_best_score"] == 1.0
    assert res[0]["dense_second_score"] > 0
    assert res[1]["doc_id"] == "B"


def test_encode_corpus_stores_macro_metadata_and_searches_documents():
    def encode(texts):
        return np.array(
            [[1.0, 0.0] if "alpha" in text else [0.0, 1.0] for text in texts],
            dtype=np.float32,
        )

    retriever = DenseMacroRetriever.from_arrays(query_encoder=encode)
    embeddings = retriever.encode_corpus(
        [
            {"chunk_id": "a1", "doc_id": "A", "text_norm": "alpha"},
            {"chunk_id": "a2", "doc_id": "A", "text_norm": "alpha details"},
            {"chunk_id": "b1", "doc_id": "B", "text_norm": "beta"},
        ]
    )

    assert embeddings.shape == (3, 2)
    assert retriever.chunk_ids == ["a1", "a2", "b1"]
    assert retriever.doc_ids == ["A", "A", "B"]
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0)
    assert retriever.search("alpha question", top_k=1)[0]["doc_id"] == "A"


def test_encode_corpus_keeps_only_search_metadata():
    retriever = DenseMacroRetriever.from_arrays(
        query_encoder=lambda texts: np.ones((len(texts), 2), dtype=np.float32)
    )
    retriever.encode_corpus(
        [
            {
                "chunk_id": "c1",
                "doc_id": "d1",
                "article": "Điều 1",
                "text_norm": "nội dung không lưu lại",
            }
        ]
    )

    assert retriever.corpus == [{"chunk_id": "c1", "doc_id": "d1", "article": "Điều 1"}]


def test_encode_texts_mean_pools_attention_mask_and_normalizes():
    class DummyTokenizer:
        def __call__(self, texts, **kwargs):
            return {
                "input_ids": torch.zeros((1, 3), dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 0]], dtype=torch.long),
            }

    class DummyModel:
        def __call__(self, **kwargs):
            return SimpleNamespace(
                last_hidden_state=torch.tensor(
                    [[[1.0, 0.0], [3.0, 2.0], [100.0, 100.0]]]
                )
            )

    retriever = DenseMacroRetriever(dimension=2, use_pyvi=False, device="cpu")
    retriever.tokenizer = DummyTokenizer()
    retriever.model = DummyModel()

    embedding = retriever.encode_texts(["ignored"], batch_size=1)
    expected = np.array([[2.0, 1.0]], dtype=np.float32)
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    assert np.allclose(embedding, expected)
