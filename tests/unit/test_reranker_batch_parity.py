"""Parity for eval-only batching (dedup + length bucketing) and honest-training flag."""
import os

from src.pipeline.oof_runner import _honest_fold_training
from src.ranking.reranker import CrossEncoderReranker


def _make_reranker():
    r = CrossEncoderReranker.__new__(CrossEncoderReranker)
    r.batch_size = 4
    r.max_length = 64
    r.score_fn = None
    return r


def _fake_scorer(self, pairs, batch_size, max_length):
    # Deterministic content-based pseudo-score: simulates a model statelessly.
    return [float(len(q) * 31 + len(p) * 7) for q, p in pairs]


def test_dedup_and_order_preserved(monkeypatch):
    monkeypatch.setattr(
        CrossEncoderReranker, "_score_unique_pairs", _fake_scorer
    )
    r = _make_reranker()
    pairs = [
        ("short q", "a much longer passage text here"),
        ("a considerably longer query string", "tiny"),
        ("short q", "a much longer passage text here"),  # duplicate
        ("mid query here", "mid passage here"),
        ("a considerably longer query string", "tiny"),  # duplicate
    ]
    out = r.score_pairs(pairs, batch_size=2)
    expected = [float(len(q) * 31 + len(p) * 7) for q, p in pairs]
    assert out == expected
    assert out[0] == out[2]
    assert out[1] == out[4]


def test_empty_and_single(monkeypatch):
    monkeypatch.setattr(
        CrossEncoderReranker, "_score_unique_pairs", _fake_scorer
    )
    r = _make_reranker()
    assert r.score_pairs([]) == []
    assert r.score_pairs([("q", "p")]) == [float(len("q") * 31 + len("p") * 7)]


def test_honest_fold_training_flag_restores_env(monkeypatch):
    monkeypatch.delenv("LEGALIR_DISABLE_WARM_START", raising=False)
    with _honest_fold_training():
        assert os.environ["LEGALIR_DISABLE_WARM_START"] == "1"
    assert "LEGALIR_DISABLE_WARM_START" not in os.environ

    monkeypatch.setenv("LEGALIR_DISABLE_WARM_START", "keep-me")
    with _honest_fold_training():
        assert os.environ["LEGALIR_DISABLE_WARM_START"] == "1"
    assert os.environ["LEGALIR_DISABLE_WARM_START"] == "keep-me"
