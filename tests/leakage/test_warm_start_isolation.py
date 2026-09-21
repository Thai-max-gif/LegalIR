import ast
import json
from pathlib import Path
import pytest
import torch
import torch.nn as nn
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

from src.training.trainer import setup_peft_model
from src.training.train_reranker import train_reranker
from src.evaluation.submission import validate_submission


@pytest.fixture
def mock_model_fixture(tmp_path: Path):
    config = BertConfig(
        vocab_size=300,
        hidden_size=256,  # >= 256 so warm-start is eligible
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
    model_dir = tmp_path / "mock_bert"
    model.save_pretrained(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    # Save a valid dummy LoRA adapter
    peft_model, _ = setup_peft_model(model, lora_r=4, lora_alpha=8, target_modules=["query", "value"])
    adapter_dir = tmp_path / "test_adapter"
    peft_model.save_pretrained(str(adapter_dir))

    return model, tokenizer, adapter_dir


def test_setup_peft_model_default_refuses_warm_start(mock_model_fixture):
    model, _, adapter_dir = mock_model_fixture
    # When allow_warm_start is omitted (default) or False, it MUST NOT warm-start
    peft_model, meta = setup_peft_model(
        model=model,
        pretrained_adapter=str(adapter_dir),
        # allow_warm_start omitted -> default False
    )
    assert not meta.get("warm_start", False), "setup_peft_model must refuse warm-start by default"


def test_setup_peft_model_explicit_opt_in_allows_warm_start(mock_model_fixture):
    model, _, adapter_dir = mock_model_fixture
    peft_model, meta = setup_peft_model(
        model=model,
        pretrained_adapter=str(adapter_dir),
        allow_warm_start=True,
    )
    assert meta.get("warm_start") is True, "setup_peft_model must allow warm-start when allow_warm_start=True"


def test_oof_runner_source_has_no_warm_start():
    oof_runner_path = Path("src/pipeline/oof_runner.py")
    tree = ast.parse(oof_runner_path.read_text(encoding="utf-8"))
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func_name = ""
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
            if func_name == "train_reranker":
                for kw in node.keywords:
                    if kw.arg == "allow_warm_start":
                        # If present, it must be False
                        val = getattr(kw.value, "value", None)
                        assert val is False, f"oof_runner.py calls train_reranker with allow_warm_start={val}!"


def test_validate_submission_exact_answer_count():
    preds = {
        "q1": {"answer": ["d1", "d2", "d3", "d4"]},  # 4 docs
        "q2": {"answer": ["d1", "d2", "d3", "d4", "d5"]}, # 5 docs
    }
    # Without exact_answer_count, 4 docs is accepted (1 <= len <= 5)
    res = validate_submission(preds, raise_on_error=False)
    assert res["is_valid"] is True

    # With exact_answer_count=5, 4 docs MUST BE REJECTED
    res_strict = validate_submission(preds, exact_answer_count=5, raise_on_error=False)
    assert res_strict["is_valid"] is False
    assert any("answer length must be exactly 5" in err for err in res_strict["errors"])


def test_train_reranker_forces_cold_start_for_folds(tmp_path: Path, monkeypatch):
    import sys
    import pandas as pd
    from src.training.train_reranker import train_reranker
    tr_mod = sys.modules["src.training.train_reranker"]

    pairs_file = tmp_path / "pairs.parquet"
    df = pd.DataFrame([
        {"query_id": "q1", "query_text": "t1", "doc_id": "d1", "evidence_text": "e1", "label": 1.0},
        {"query_id": "q1", "query_text": "t1", "doc_id": "d2", "evidence_text": "e2", "label": 0.0},
    ])
    df.to_parquet(pairs_file)

    captured_cfg = {}

    class MockTrainer:
        def __init__(self, *args, **kwargs):
            nonlocal captured_cfg
            captured_cfg = dict(kwargs.get("config", {}))
        def train(self, *args, **kwargs):
            return {
                "status": "completed",
                "global_steps": 1,
                "warm_start": False,
            }

    monkeypatch.setattr(tr_mod, "RerankerTrainer", MockTrainer)

    # When fold=0 is passed, even if allow_warm_start=True is passed or in config, it MUST be False
    train_reranker(
        pairs_file=pairs_file,
        output_dir=tmp_path / "out",
        base_model_name="mock",
        max_steps=1,
        fold=0,
        allow_warm_start=True,
    )
    assert captured_cfg.get("allow_warm_start") is False, "train_reranker must force allow_warm_start=False when fold is provided!"
