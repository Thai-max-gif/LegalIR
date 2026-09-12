import numpy as np
from src.retrieval.question_memory import QuestionMemory


def test_dense_memory_votes_without_validation_rows():
    rows = [
        {"query_id": "train1", "question_norm": "quy định về đất đai", "doc_ids": ["100"]},
        {"query_id": "train2", "question_norm": "quy định về bất động sản", "doc_ids": ["200"]},
    ]

    # Mock encoder: returns 2D vectors
    def mock_encoder(texts):
        vecs = []
        for t in texts:
            if "đất đai" in t:
                vecs.append([1.0, 0.0])
            elif "bất động sản" in t:
                vecs.append([0.0, 1.0])
            else:
                vecs.append([0.5, 0.5])
        return np.array(vecs, dtype=np.float32)

    memory = QuestionMemory(
        train_queries=rows,
        min_similarity=0.70,
        dense_encoder=mock_encoder,
        dense_min_similarity=0.80,
    )

    assert memory.training_query_ids == frozenset({"train1", "train2"})
    assert "val" not in memory.qid_to_docs

    res = memory.retrieve("thông tin luật đất đai", top_k=5)
    assert "100" in res
    assert res["100"]["dense_similarity"] >= 0.8
    assert res["100"]["vote_count"] >= 1


def test_question_memory_dual_signal():
    from src.retrieval.question_memory import TrainQuestionMemory

    memory = TrainQuestionMemory(min_similarity=0.8, use_dense=False)
    train_queries = {
        "q1": "thời hạn cấp đăng ký xe máy",
        "q2": "thủ tục đăng ký kinh doanh",
    }
    train_qrels = {"q1": ["doc_100"], "q2": ["doc_200"]}
    memory.fit(train_queries, train_qrels)

    hits = memory.search("thời hạn cấp đăng ký xe máy của người nước ngoài", top_k=5)
    assert any(hit["doc_id"] == "doc_100" for hit in hits)


def test_train_question_memory_dense_signal_accepts_precomputed_query_embedding():
    from src.retrieval.question_memory import TrainQuestionMemory

    def encoder(texts):
        return np.array(
            [[1.0, 0.0] if "đất" in text else [0.0, 1.0] for text in texts],
            dtype=np.float32,
        )

    memory = TrainQuestionMemory(
        min_similarity=0.95,
        dense_min_similarity=0.8,
        dense_encoder=encoder,
    )
    memory.fit(
        {"q1": "quy định về đất đai", "q2": "thủ tục kinh doanh"},
        {"q1": ["doc_100"], "q2": ["doc_200"]},
    )

    hits = memory.query("không trùng từ", top_k=5, q_emb=np.array([1.0, 0.0]))
    assert hits[0]["doc_id"] == "doc_100"
    assert hits[0]["dense_similarity"] == 1.0
    assert hits[0]["vote_count"] == 1


def test_train_question_memory_use_dense_lazily_loads_dek21(monkeypatch):
    import src.retrieval.question_memory as question_memory_module
    from src.retrieval.question_memory import TrainQuestionMemory

    class FakeDenseMacroRetriever:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def encode_texts(self, texts):
            return np.array(
                [[1.0, 0.0] if "đất" in text or "không" in text else [0.0, 1.0] for text in texts],
                dtype=np.float32,
            )

    monkeypatch.setattr(question_memory_module, "DenseMacroRetriever", FakeDenseMacroRetriever)
    memory = TrainQuestionMemory(
        min_similarity=0.95,
        dense_min_similarity=0.8,
    )
    memory.fit(
        {"q1": "quy định về đất đai", "q2": "thủ tục kinh doanh"},
        {"q1": ["doc_100"], "q2": ["doc_200"]},
    )

    assert memory.use_dense is True
    assert memory.dense_embeddings is not None
    assert memory.dense_encoder.kwargs["model_name"] == memory.DEFAULT_MODEL_NAME
    hits = memory.search("không trùng từ", top_k=5)
    assert hits[0]["doc_id"] == "doc_100"
    assert hits[0]["dense_similarity"] == 1.0
