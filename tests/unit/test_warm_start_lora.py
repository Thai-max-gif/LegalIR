import pytest
import torch
import torch.nn as nn
from pathlib import Path
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast
from src.training.trainer import setup_peft_model, RerankerTrainer


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

    vocab_file = tmp_path / "vocab.txt"
    vocab_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + [f"tok_{i}" for i in range(295)]
    vocab_file.write_text("\n".join(vocab_tokens) + "\n", encoding="utf-8")

    tokenizer = BertTokenizerFast(vocab_file=str(vocab_file))
    model_dir = tmp_path / "tiny_bert"
    model.save_pretrained(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    return str(model_dir), model, tokenizer


def test_setup_peft_model_standard(tiny_bert_fixture):
    _, model, _ = tiny_bert_fixture
    peft_model, meta = setup_peft_model(
        model=model,
        lora_r=4,
        lora_alpha=8,
        target_modules=["query", "value"],
        pretrained_adapter=None,
    )
    assert meta["trainable_params"] > 0
    assert meta["trainable_percent"] < 100.0
    assert not meta.get("warm_start", False)


def test_setup_peft_model_fallback_on_mock(tiny_bert_fixture):
    # For a tiny mock model (hidden_size=32 < 256), warm-start adapter should be gracefully ignored
    _, model, _ = tiny_bert_fixture
    peft_model, meta = setup_peft_model(
        model=model,
        lora_r=4,
        lora_alpha=8,
        target_modules=["query", "value"],
        pretrained_adapter="dangphuc2109/legalir-task1-reranker",
    )
    # Still works and initializes cleanly
    assert meta["trainable_params"] > 0
    assert meta["trainable_percent"] < 100.0


def test_setup_peft_model_local_adapter_save_and_reload(tmp_path: Path, tiny_bert_fixture):
    _, model, _ = tiny_bert_fixture
    # Create and save an initial adapter
    peft_model, meta = setup_peft_model(
        model=model,
        lora_r=4,
        lora_alpha=8,
        target_modules=["query", "value"],
    )
    save_dir = tmp_path / "saved_adapter"
    peft_model.save_pretrained(save_dir)
    assert (save_dir / "adapter_config.json").exists()

    # Now simulate a model with hidden_size >= 256
    model.config.hidden_size = 256
    reloaded_model, reloaded_meta = setup_peft_model(
        model=model,
        pretrained_adapter=str(save_dir),
        allow_warm_start=True,
    )
    assert reloaded_meta.get("warm_start") is True
    assert reloaded_meta["trainable_params"] > 0
    # Ensure all lora parameters require grad
    for name, param in reloaded_model.named_parameters():
        if "lora_" in name:
            assert param.requires_grad is True
