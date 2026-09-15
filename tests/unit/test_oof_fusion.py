import numpy as np
import pandas as pd
from src.ranking.oof_features import FEATURE_COLUMNS, extract_candidate_features, FEATURE_SCHEMA_VERSION
from src.ranking.fusion import LightGBMRanker
from src.pipeline.oof_runner import OOFRunner


def test_oof_question_memory_reuses_encoder_but_isolates_fold_state(tmp_path):
    """Shared encoder construction must not share fold-local memory indexes."""
    runner = OOFRunner(output_dir=tmp_path)
    first = runner._build_question_memory(min_similarity=0.82)
    second = runner._build_question_memory(min_similarity=0.82)

    assert first is not second
    assert first.dense_encoder is second.dense_encoder
    assert runner.dense_device is None

    first.fit(
        [("q1", "first training question", None)],
        {"q1": ["d1"]},
        dense_embeddings=[[1.0, 0.0]],
        encode_dense=False,
    )
    second.fit(
        [("q2", "second training question", None)],
        {"q2": ["d2"]},
        dense_embeddings=[[0.0, 1.0]],
        encode_dense=False,
    )

    assert first.training_query_ids == frozenset({"q1"})
    assert second.training_query_ids == frozenset({"q2"})
    assert first.qid_to_docs == {"q1": ["d1"]}
    assert second.qid_to_docs == {"q2": ["d2"]}


def test_oof_features_include_all_required_signals():
    assert FEATURE_SCHEMA_VERSION == "v2"
    required = {
        "memory_vote_count",
        "memory_dense_similarity",
        "exact_legal_number",
        "exact_title",
        "reranker_best_score",
        "reranker_second_score",
        "reranker_margin",
        "evidence_chunk_count",
        "source_count",
        "bm25_best_score",
    }
    assert required <= set(FEATURE_COLUMNS)


def test_feature_rows_keep_query_group_identity():
    cands = [
        {"doc_id": "1", "bm25_score": 2.0, "exact_legal_number": True},
        {"doc_id": "2", "bm25_score": 1.0, "dense_best_score": 0.9},
    ]
    df = extract_candidate_features("q1", cands)
    assert len(df) == 2
    assert df.loc[0, "query_id"] == "q1"
    assert df.loc[0, "doc_id"] == "1"
    assert df.loc[0, "exact_legal_number"] == 1.0
    assert df.loc[1, "doc_id"] == "2"


def test_lightgbm_ranker_fit_predict(tmp_path):
    X = pd.DataFrame({
        "bm25_score": [5.0, 1.0, 4.0, 0.5],
        "dense_score": [0.9, 0.2, 0.8, 0.1],
        "reranker_best_score": [3.0, -1.0, 2.5, -2.0],
        "query_id": ["q1", "q1", "q2", "q2"],
        "doc_id": ["d1", "d2", "d3", "d4"],
    })
    y = np.array([1, 0, 1, 0])
    groups = np.array([2, 2])

    ranker = LightGBMRanker()
    ranker.fit(X, y, groups)

    cands = [
        {"doc_id": "d2", "bm25_score": 1.0, "dense_score": 0.2, "reranker_best_score": -1.0},
        {"doc_id": "d1", "bm25_score": 5.0, "dense_score": 0.9, "reranker_best_score": 3.0},
    ]
    ranked = ranker.predict(cands)
    assert [c["doc_id"] for c in ranked] == ["d1", "d2"]
