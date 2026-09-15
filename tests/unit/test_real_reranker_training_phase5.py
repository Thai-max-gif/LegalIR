import json
from pathlib import Path
import pytest
import pandas as pd
import torch
import torch.nn as nn
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

from src.ranking.reranker import CrossEncoderReranker
from src.training.hard_negative_miner import HardNegativeMiner
from src.training.losses import (
    ListwiseCrossEntropyLoss,
    PairwiseLogisticLoss,
    PairwiseMarginRankingLoss,
    PointwiseBCELoss,
    get_loss_function,
)
from src.training.trainer import (
    RerankerTrainer,
    find_target_modules,
    setup_peft_model,
)
from src.training.train_reranker import train_reranker


# ==============================================================================
# 1. Ranking Loss Function Tests
# ==============================================================================

def test_pointwise_bce_loss():
    loss_fn = PointwiseBCELoss()
    # High logit for positive label 1.0 -> low loss
    good_logits = torch.tensor([3.0], requires_grad=True)
    labels = torch.tensor([1.0])
    loss_good = loss_fn(good_logits, labels)

    # Low logit for positive label 1.0 -> high loss
    bad_logits = torch.tensor([-3.0], requires_grad=True)
    loss_bad = loss_fn(bad_logits, labels)

    assert loss_good.item() < loss_bad.item()

    # Backpropagation check
    loss_good.backward()
    assert good_logits.grad is not None
    assert torch.abs(good_logits.grad).item() > 0


def test_pairwise_logistic_loss():
    loss_fn = PairwiseLogisticLoss(temperature=1.0)

    # When pos_score > neg_score -> loss is low
    pos_score_good = torch.tensor([2.0], requires_grad=True)
    neg_score_good = torch.tensor([-2.0], requires_grad=True)
    loss_good = loss_fn(pos_score_good, neg_score_good)

    # When pos_score < neg_score -> loss is high
    pos_score_bad = torch.tensor([-2.0], requires_grad=True)
    neg_score_bad = torch.tensor([2.0], requires_grad=True)
    loss_bad = loss_fn(pos_score_bad, neg_score_bad)

    assert loss_good.item() < loss_bad.item()

    # Backward pass check
    loss_good.backward()
    assert pos_score_good.grad is not None
    assert neg_score_good.grad is not None


def test_pairwise_margin_ranking_loss():
    loss_fn = PairwiseMarginRankingLoss(margin=1.0)

    # Margin exceeded (diff = 2.0 > margin 1.0) -> loss is 0.0
    pos_score_good = torch.tensor([3.0], requires_grad=True)
    neg_score_good = torch.tensor([1.0], requires_grad=True)
    loss_good = loss_fn(pos_score_good, neg_score_good)
    assert loss_good.item() == 0.0

    # Margin violated (diff = -1.0 < margin 1.0) -> loss = 1 - (-1) = 2.0
    pos_score_bad = torch.tensor([1.0], requires_grad=True)
    neg_score_bad = torch.tensor([2.0], requires_grad=True)
    loss_bad = loss_fn(pos_score_bad, neg_score_bad)
    assert abs(loss_bad.item() - 2.0) < 1e-5

    loss_bad.backward()
    assert pos_score_bad.grad is not None


def test_listwise_cross_entropy_loss():
    loss_fn = ListwiseCrossEntropyLoss(temperature=1.0)

    # Candidate 0 is positive with high score -> low loss
    good_scores = torch.tensor([[4.0, -1.0, -2.0, -3.0]], requires_grad=True)
    loss_good = loss_fn(good_scores, target_idx=0)

    # Candidate 0 is positive with low score -> high loss
    bad_scores = torch.tensor([[-2.0, 4.0, 3.0, 2.0]], requires_grad=True)
    loss_bad = loss_fn(bad_scores, target_idx=0)

    assert loss_good.item() < loss_bad.item()

    loss_good.backward()
    assert good_scores.grad is not None


def test_loss_function_factory():
    assert isinstance(get_loss_function("bce"), PointwiseBCELoss)
    assert isinstance(get_loss_function("pairwise_logistic"), PairwiseLogisticLoss)
    assert isinstance(get_loss_function("pairwise_margin"), PairwiseMarginRankingLoss)
    assert isinstance(get_loss_function("listwise_ce"), ListwiseCrossEntropyLoss)

    with pytest.raises(ValueError):
        get_loss_function("unknown_unsupported_loss")


# ==============================================================================
# 2. Hard-Negative Miner Tests
# ==============================================================================

def test_hard_negative_miner_excludes_golds_and_duplicates():
    blacklist = {"q1": {"dup_gold_1", "dup_gold_2"}}
    miner = HardNegativeMiner(false_negative_blacklist=blacklist)

    candidates = [
        {"doc_id": "gold_1", "source": "bm25", "rank": 1, "score": 10.0},
        {"doc_id": "dup_gold_1", "source": "bm25", "rank": 2, "score": 9.0},
        {"doc_id": "neg_1", "source": "bm25", "rank": 3, "score": 8.0},
        {"doc_id": "neg_2", "source": "dense", "rank": 4, "score": 7.0},
    ]

    mined = miner.mine_negatives("q1", candidates, ["gold_1"], max_negatives=5, return_records=True)
    mined_ids = [m["doc_id"] for m in mined]

    assert "gold_1" not in mined_ids
    assert "dup_gold_1" not in mined_ids
    assert mined_ids == ["neg_1", "neg_2"]

    stats = miner.get_stats()
    assert stats["excluded_golds_count"] == 1
    assert stats["excluded_duplicates_count"] == 1


def test_hard_negative_miner_multi_band():
    miner = HardNegativeMiner()
    candidates_by_source = {
        "exact": [{"doc_id": "exact_1", "score": 1.0}, {"doc_id": "gold_1", "score": 1.0}],
        "bm25": [{"doc_id": "bm25_1", "score": 15.0}, {"doc_id": "bm25_2", "score": 12.0}],
        "dense": [{"doc_id": "dense_1", "score": 0.85}, {"doc_id": "dense_2", "score": 0.80}],
        "medium_neg": [{"doc_id": "med_1", "score": 0.1}],
    }

    mined = miner.mine_multi_band_negatives(
        query_id="q1",
        candidates_by_source=candidates_by_source,
        gold_doc_ids=["gold_1"],
        per_source_limits={"exact": 1, "bm25": 2, "dense": 2, "medium_neg": 1},
        max_total=5,
    )

    mined_ids = [m["doc_id"] for m in mined]
    assert "gold_1" not in mined_ids
    assert "exact_1" in mined_ids
    assert "bm25_1" in mined_ids
    assert "dense_1" in mined_ids
    assert all("negative_source" in m for m in mined)


# ==============================================================================
# 3. Real End-to-End Supervised LoRA Training & Weight Verification Test
# ==============================================================================

@pytest.fixture
def tiny_bert_fixture(tmp_path: Path):
    """Creates and saves a tiny BERT model and tokenizer for fast unit testing."""
    config = BertConfig(
        vocab_size=300,
        hidden_size=32,
        num_attention_heads=2,
        num_hidden_layers=2,
        intermediate_size=64,
        max_position_embeddings=128,
        num_labels=1,
    )
    model = BertForSequenceClassification(config)

    # Create dummy vocab for tokenizer
    vocab_file = tmp_path / "vocab.txt"
    vocab_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + [f"tok_{i}" for i in range(295)]
    vocab_file.write_text("\n".join(vocab_tokens) + "\n", encoding="utf-8")

    tokenizer = BertTokenizerFast(vocab_file=str(vocab_file))
    model_dir = tmp_path / "tiny_bert"
    model.save_pretrained(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    return str(model_dir), model, tokenizer


def test_dataloader_workers_preserve_sampler_order(tiny_bert_fixture):
    """Exercise worker-backed iteration for train/group/validation loaders."""
    _, model, tokenizer = tiny_bert_fixture
    train_pairs = [
        {"query_id": "q1", "query_text": "tok_1", "doc_id": "d1", "evidence_text": "tok_2", "label": 1.0},
        {"query_id": "q1", "query_text": "tok_1", "doc_id": "d2", "evidence_text": "tok_3", "label": 0.0},
        {"query_id": "q2", "query_text": "tok_4", "doc_id": "d3", "evidence_text": "tok_5", "label": 1.0},
        {"query_id": "q2", "query_text": "tok_4", "doc_id": "d4", "evidence_text": "tok_6", "label": 0.0},
        {"query_id": "q3", "query_text": "tok_7", "doc_id": "d5", "evidence_text": "tok_8", "label": 1.0},
        {"query_id": "q3", "query_text": "tok_7", "doc_id": "d6", "evidence_text": "tok_9", "label": 0.0},
    ]
    val_pairs = [
        {"query_id": "qv", "query_text": "tok_1", "doc_id": "dv1", "evidence_text": "tok_2", "label": 1.0},
        {"query_id": "qv", "query_text": "tok_1", "doc_id": "dv2", "evidence_text": "tok_3", "label": 0.0},
    ]
    common_config = {
        "batch_size": 2,
        "max_length": 32,
        "use_lora": False,
        "num_workers": 1,
    }
    trainer = RerankerTrainer(
        model=model,
        tokenizer=tokenizer,
        train_data=train_pairs,
        val_data=val_pairs,
        config={**common_config, "loss_type": "bce"},
        device="cpu",
    )

    assert trainer.num_workers == 1
    assert trainer.train_loader.num_workers == 1
    assert trainer.train_loader.persistent_workers is True
    assert trainer.train_loader.pin_memory is False
    assert trainer.val_loader is not None
    assert trainer.val_loader.num_workers == 1
    assert trainer.val_loader.persistent_workers is True

    expected_indices = list(iter(trainer.train_sampler))
    expected_qids = [trainer.train_dataset.records[idx]["query_id"] for idx in expected_indices]
    train_iterator = iter(trainer.train_loader)
    try:
        train_batches = list(train_iterator)
        actual_qids = [qid for batch in train_batches for qid in batch["query_ids"]]
    finally:
        shutdown = getattr(train_iterator, "_shutdown_workers", None)
        if shutdown is not None:
            shutdown()
    assert actual_qids == expected_qids

    val_iterator = iter(trainer.val_loader)
    try:
        val_batches = list(val_iterator)
    finally:
        shutdown = getattr(val_iterator, "_shutdown_workers", None)
        if shutdown is not None:
            shutdown()
    assert [qid for batch in val_batches for qid in batch["query_ids"]] == ["qv", "qv"]

    group_trainer = RerankerTrainer(
        model=model,
        tokenizer=tokenizer,
        train_data=train_pairs,
        config={**common_config, "loss_type": "listwise"},
        device="cpu",
    )
    assert group_trainer.train_loader.num_workers == 1
    assert group_trainer.train_loader.persistent_workers is True
    group_indices = list(iter(group_trainer.train_sampler))
    group_expected_qids = [group_trainer.train_dataset.items[idx]["query_id"] for idx in group_indices]
    group_iterator = iter(group_trainer.train_loader)
    try:
        group_batches = list(group_iterator)
    finally:
        shutdown = getattr(group_iterator, "_shutdown_workers", None)
        if shutdown is not None:
            shutdown()
    assert [qid for batch in group_batches for qid in batch["query_ids"]] == group_expected_qids

    zero_trainer = RerankerTrainer(
        model=model,
        tokenizer=tokenizer,
        train_data=train_pairs,
        config={**common_config, "num_workers": 0, "loss_type": "bce"},
        device="cpu",
    )
    assert zero_trainer.train_loader.num_workers == 0
    assert zero_trainer.train_loader.persistent_workers is False


def test_train_reranker_propagates_num_workers(tmp_path: Path, monkeypatch):
    """Runtime worker override reaches the trainer without changing score settings."""
    import importlib

    train_module = importlib.import_module("src.training.train_reranker")
    captured: dict[str, object] = {}

    class StubTrainer:
        batch_size = 2
        gradient_accumulation_steps = 1

        def __init__(self, *, config, **kwargs):
            captured["config"] = config

        def train(self, output_dir):
            return {"status": "completed", "global_steps": 1}

    monkeypatch.setattr(train_module, "RerankerTrainer", StubTrainer)
    pairs_path = tmp_path / "pairs.parquet"
    pd.DataFrame(
        [
            {"query_id": "q1", "label": 1.0},
            {"query_id": "q1", "label": 0.0},
        ]
    ).to_parquet(pairs_path)

    train_module.train_reranker(
        pairs_file=pairs_path,
        output_dir=tmp_path / "checkpoint",
        config_path={"output_dir": "artifacts/local/test/checkpoint", "batch_size": 2},
        base_model_name="mock",
        num_workers=3,
        enforce_full_coverage_steps=False,
    )

    assert captured["config"]["num_workers"] == 3


def test_target_module_inspection_and_peft_setup(tiny_bert_fixture):
    model_dir, model, tokenizer = tiny_bert_fixture

    modules = find_target_modules(model, ["query", "value"])
    assert len(modules) > 0
    assert "query" in modules or "value" in modules

    peft_model, meta = setup_peft_model(model, lora_r=4, lora_alpha=8, target_modules=["query", "value"])
    assert meta["trainable_params"] > 0
    assert meta["trainable_params"] < meta["total_params"]
    assert meta["trainable_percent"] < 100.0


def test_real_supervised_reranker_training_pointwise(tmp_path: Path, tiny_bert_fixture):
    model_dir, model, tokenizer = tiny_bert_fixture
    out_dir = tmp_path / "checkpoint_pointwise"

    train_pairs = [
        {"query_id": "q1", "query_text": "tok_1 tok_2", "doc_id": "d1", "evidence_text": "tok_1 tok_2 tok_3", "label": 1.0},
        {"query_id": "q1", "query_text": "tok_1 tok_2", "doc_id": "d2", "evidence_text": "tok_99 tok_98", "label": 0.0},
        {"query_id": "q2", "query_text": "tok_5 tok_6", "doc_id": "d3", "evidence_text": "tok_5 tok_6 tok_7", "label": 1.0},
        {"query_id": "q2", "query_text": "tok_5 tok_6", "doc_id": "d4", "evidence_text": "tok_88 tok_87", "label": 0.0},
    ] * 4

    val_pairs = [
        {"query_id": "q3", "query_text": "tok_1 tok_2", "doc_id": "d1", "evidence_text": "tok_1 tok_2", "label": 1.0},
        {"query_id": "q3", "query_text": "tok_1 tok_2", "doc_id": "d2", "evidence_text": "tok_90 tok_91", "label": 0.0},
    ]

    training_cfg = {
        "learning_rate": 5e-3,
        "batch_size": 4,
        "max_steps": 10,
        "use_lora": True,
        "lora_r": 4,
        "lora_alpha": 8,
        "target_modules": ["query", "value"],
        "loss_type": "bce",
        "max_length": 64,
        "fp16": False,
    }

    trainer = RerankerTrainer(
        model=model,
        tokenizer=tokenizer,
        train_data=train_pairs,
        val_data=val_pairs,
        config=training_cfg,
        device="cpu",
    )

    report = trainer.train(output_dir=out_dir)

    # 1. Check training status and parameter update proof
    assert report["status"] == "completed"
    assert report["global_steps"] == 10
    assert report["trainable_params"] > 0
    assert report["param_diff"] > 0.0, "Trainable weights must have changed after optimizer steps!"
    assert report["weight_norm_change"] >= 0.0
    assert len(report["loss_history_sample"]) > 0

    # 2. Check saved files
    assert (out_dir / "adapter_config.json").is_file()
    assert (out_dir / "training_manifest.json").is_file()

    manifest_data = json.loads((out_dir / "training_manifest.json").read_text(encoding="utf-8"))
    assert manifest_data["param_diff"] > 0.0
    assert manifest_data["trainable_params"] == report["trainable_params"]

    # 3. Test reloading adapter in CrossEncoderReranker
    reranker = CrossEncoderReranker(
        model_name=model_dir,
        adapter_path=out_dir,
        device="cpu",
        local_files_only=True,
    )
    scores = reranker.score_pairs([("tok_1 tok_2", "tok_1 tok_2 tok_3"), ("tok_1 tok_2", "tok_99")])
    assert len(scores) == 2
    assert isinstance(scores[0], float)
    assert isinstance(scores[1], float)


def test_real_supervised_reranker_training_pairwise(tmp_path: Path, tiny_bert_fixture):
    model_dir, _, tokenizer = tiny_bert_fixture
    out_dir = tmp_path / "checkpoint_pairwise"

    # Reload fresh base model
    base_model = BertForSequenceClassification.from_pretrained(model_dir)

    train_pairs = [
        {"query_id": "q1", "query_text": "tok_1 tok_2", "doc_id": "d1", "evidence_text": "tok_1 tok_2 tok_3", "label": 1.0},
        {"query_id": "q1", "query_text": "tok_1 tok_2", "doc_id": "d2", "evidence_text": "tok_99 tok_98", "label": 0.0},
        {"query_id": "q2", "query_text": "tok_5 tok_6", "doc_id": "d3", "evidence_text": "tok_5 tok_6 tok_7", "label": 1.0},
        {"query_id": "q2", "query_text": "tok_5 tok_6", "doc_id": "d4", "evidence_text": "tok_88 tok_87", "label": 0.0},
    ] * 4

    training_cfg = {
        "learning_rate": 5e-3,
        "batch_size": 2,
        "max_steps": 6,
        "use_lora": True,
        "lora_r": 4,
        "lora_alpha": 8,
        "target_modules": ["query", "value"],
        "loss_type": "pairwise_logistic",
        "max_length": 64,
        "fp16": False,
    }

    trainer = RerankerTrainer(
        model=base_model,
        tokenizer=tokenizer,
        train_data=train_pairs,
        config=training_cfg,
        device="cpu",
    )

    report = trainer.train(output_dir=out_dir)
    assert report["status"] == "completed"
    assert report["param_diff"] > 0.0
    assert (out_dir / "adapter_config.json").is_file()


def test_real_supervised_reranker_training_listwise(tmp_path: Path, tiny_bert_fixture):
    model_dir, _, tokenizer = tiny_bert_fixture
    out_dir = tmp_path / "checkpoint_listwise"

    base_model = BertForSequenceClassification.from_pretrained(model_dir)

    train_pairs = [
        {"query_id": "q1", "query_text": "tok_1 tok_2", "doc_id": "d1", "evidence_text": "tok_1 tok_2 tok_3", "label": 1.0},
        {"query_id": "q1", "query_text": "tok_1 tok_2", "doc_id": "d2", "evidence_text": "tok_99 tok_98", "label": 0.0},
        {"query_id": "q1", "query_text": "tok_1 tok_2", "doc_id": "d3", "evidence_text": "tok_77 tok_76", "label": 0.0},
        {"query_id": "q2", "query_text": "tok_5 tok_6", "doc_id": "d4", "evidence_text": "tok_5 tok_6 tok_7", "label": 1.0},
        {"query_id": "q2", "query_text": "tok_5 tok_6", "doc_id": "d5", "evidence_text": "tok_88 tok_87", "label": 0.0},
        {"query_id": "q2", "query_text": "tok_5 tok_6", "doc_id": "d6", "evidence_text": "tok_66 tok_65", "label": 0.0},
    ] * 3

    training_cfg = {
        "learning_rate": 5e-3,
        "batch_size": 2,
        "max_steps": 6,
        "use_lora": True,
        "lora_r": 4,
        "lora_alpha": 8,
        "target_modules": ["query", "value"],
        "loss_type": "listwise_ce",
        "max_length": 64,
        "fp16": False,
    }

    trainer = RerankerTrainer(
        model=base_model,
        tokenizer=tokenizer,
        train_data=train_pairs,
        config=training_cfg,
        device="cpu",
    )

    report = trainer.train(output_dir=out_dir)
    assert report["status"] == "completed"
    assert report["param_diff"] > 0.0
    assert (out_dir / "adapter_config.json").is_file()
