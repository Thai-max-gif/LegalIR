import pytest

from src.config import load_config
from src.utils.metrics import ranking_metrics


def test_config_loads_base():
    config = load_config("configs/legalir_kaggle_t4_smoke.yaml")
    assert config["training"]["max_steps"] == 10
    assert config["lora"]["r"] == 16


def test_ranking_deduplicates_and_calculates_known_case():
    metrics = ranking_metrics({"q": ["d1", "d1", "d2"]}, {"q": {"d2"}}, 5)
    assert metrics["eval_recall_at_5"] == 1.0
    assert metrics["eval_mrr_at_5"] == 0.5
    assert metrics["eval_hit1_at_5"] == 0.0


def test_ranking_requires_qrels():
    with pytest.raises(ValueError):
        ranking_metrics({}, {}, 5)
