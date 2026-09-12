"""Phase 8 Test Suite: Strict Parameter Budget Auditor (<4B Budget Rule) & Compliance.

Tests:
1. PyTorch nn.Module parameter counting (full & trainable).
2. PEFT/LoRA adapter parameter counting without reducing base architecture.
3. Hugging Face PretrainedConfig & offline config dictionary counting.
4. Compliance pass for baseline models (<4B).
5. Hard failure with ParameterBudgetExceededError when parameters >= 4B.
6. parameter_audit.json schema, fields, and round-trip serialization.
7. Standalone CLI entrypoint (scripts/audit_parameters.py) exit codes.
8. Integration into LegalIRPipeline.audit_parameters.
"""

import json
import subprocess
import sys
from pathlib import Path
import pytest
import torch
import torch.nn as nn
from transformers import BertConfig, RobertaConfig, XLMRobertaConfig

from src.models.parameter_audit import (
    DEFAULT_PIPELINE_MODELS,
    KNOWN_PARAM_COUNTS,
    MAX_PARAMETER_BUDGET,
    ParameterBudgetExceededError,
    audit_model_parameters,
    audit_system_parameters,
    count_parameters,
    count_parameters_from_config,
    estimate_transformer_parameters,
    extract_models_from_config,
    validate_parameter_budget,
)
from src.pipeline.predict import LegalIRPipeline


class MockTwoLayerModel(nn.Module):
    def __init__(self, in_dim: int = 100, hidden_dim: int = 200, out_dim: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)  # 100*200 + 200 = 20,200
        self.fc2 = nn.Linear(hidden_dim, out_dim)  # 200*10 + 10 = 2,010
        # Total = 22,210 params

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class MockLargeModel(nn.Module):
    """Mock model representing a large model with 4.5 billion parameters."""
    def __init__(self, params_count: int = 4_500_000_000):
        super().__init__()
        self._mock_params_count = params_count

    def parameters(self):
        # Return a single mock parameter with 4.5B elements on meta device
        p = torch.nn.Parameter(torch.empty(self._mock_params_count, device="meta"))
        return iter([p])


def test_parameter_calculation_pytorch_module():
    """Test precise parameter counting on standard PyTorch nn.Modules and state_dicts."""
    model = MockTwoLayerModel(in_dim=100, hidden_dim=200, out_dim=10)
    expected_total = (100 * 200 + 200) + (200 * 10 + 10)  # 22,210

    # 1. Full parameter count
    total = count_parameters(model, trainable_only=False)
    assert total == expected_total

    # 2. Trainable-only parameter count with partial freezing
    for p in model.fc1.parameters():
        p.requires_grad = False

    trainable = count_parameters(model, trainable_only=True)
    expected_trainable = 200 * 10 + 10  # 2,010
    assert trainable == expected_trainable
    assert count_parameters(model, trainable_only=False) == expected_total

    # 3. State dict parameter count
    state_dict = model.state_dict()
    assert count_parameters(state_dict) == expected_total


def test_parameter_calculation_peft_lora_model():
    """Test that PEFT/LoRA adapter models preserve full base model parameter count."""
    try:
        from peft import LoraConfig, get_peft_model
        from transformers import BertForSequenceClassification

        cfg = BertConfig(
            vocab_size=1000,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=128,
            num_labels=1,
        )
        base = BertForSequenceClassification(cfg)
        base_params = sum(p.numel() for p in base.parameters())

        lora_cfg = LoraConfig(r=4, lora_alpha=8, target_modules=["query", "value"])
        peft_model = get_peft_model(base, lora_cfg)

        # Rule check: LoRA does NOT reduce parameter count for competition budget
        total_counted = count_parameters(peft_model, trainable_only=False)
        trainable_counted = count_parameters(peft_model, trainable_only=True)

        assert total_counted >= base_params
        assert trainable_counted < base_params

        # Test audit report on PEFT model
        report = audit_model_parameters(peft_model, name="test_peft_model", role="reranker")
        assert report["is_peft_lora"] is True
        assert report["parameters"] == total_counted
        assert report["trainable_parameters"] == trainable_counted
        assert report["base_parameters"] == base_params
        assert report["adapter_parameters"] == trainable_counted
    except ImportError:
        pytest.skip("peft not available in test environment")


def test_parameter_calculation_hf_configs():
    """Test parameter calculation on Hugging Face PretrainedConfigs and offline dicts."""
    # 1. BertConfig
    bert_cfg = BertConfig(
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
    )
    p_bert = count_parameters_from_config(bert_cfg)
    assert p_bert == 109_482_240  # Exact bert-base-uncased architecture parameter count

    # 2. RobertaConfig
    roberta_cfg = RobertaConfig()
    p_roberta = count_parameters_from_config(roberta_cfg)
    assert p_roberta == 124_644_864  # Exact roberta-base architecture parameter count

    # 3. Raw dictionary config
    raw_dict = {
        "model_type": "bert",
        "vocab_size": 30522,
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
    }
    p_dict = count_parameters_from_config(raw_dict)
    assert p_dict > 100_000_000

    # 4. Analytical transformer parameter estimation
    est = estimate_transformer_parameters(raw_dict)
    assert abs(est - 109_482_240) < 5_000  # within 0.005%


def test_compliance_pass_baseline_models(tmp_path: Path):
    """Test compliance pass for the baseline system (DEk21 + BGE-reranker < 4B)."""
    audit_file = tmp_path / "parameter_audit.json"
    report = audit_system_parameters(
        models=DEFAULT_PIPELINE_MODELS,
        output_json=audit_file,
        raise_on_violation=True,
    )

    assert report["is_compliant"] is True
    assert report["verdict"] == "PASS"
    assert report["budget_limit"] == MAX_PARAMETER_BUDGET
    assert report["total_learned_parameters"] < MAX_PARAMETER_BUDGET

    # Baseline models total ~702.7M parameters
    assert 650_000_000 < report["total_learned_parameters"] < 800_000_000
    assert report["budget_utilization_pct"] < 25.0  # <25% of 4B budget
    assert report["headroom_parameters"] > 3_000_000_000

    # Test validate_parameter_budget helper
    assert validate_parameter_budget(models=DEFAULT_PIPELINE_MODELS, raise_on_violation=True) is True
    assert validate_parameter_budget(total_params=report["total_learned_parameters"], raise_on_violation=True) is True


def test_hard_failure_parameter_budget_exceeded():
    """Test that a model or system exceeding 4B parameters raises ParameterBudgetExceededError."""
    large_model = MockLargeModel(params_count=4_500_000_000)

    # 1. validate_parameter_budget hard failure
    with pytest.raises(ParameterBudgetExceededError, match="exceeds"):
        validate_parameter_budget(total_params=4_000_000_000, raise_on_violation=True)

    with pytest.raises(ParameterBudgetExceededError, match="exceeds"):
        validate_parameter_budget(total_params=5_200_000_000, raise_on_violation=True)

    # 2. audit_system_parameters hard failure on oversized model
    with pytest.raises(ParameterBudgetExceededError, match="exceeds"):
        audit_system_parameters(
            models=[large_model],
            output_json=None,
            raise_on_violation=True,
        )

    # 3. Soft check with raise_on_violation=False
    soft_report = audit_system_parameters(
        models=[large_model],
        output_json=None,
        raise_on_violation=False,
    )
    assert soft_report["is_compliant"] is False
    assert soft_report["verdict"] == "FAIL"
    assert soft_report["total_learned_parameters"] == 4_500_000_000


def test_parameter_audit_json_export_and_serialization(tmp_path: Path):
    """Test parameter_audit.json schema, required fields, and roundtrip JSON loading."""
    out_json = tmp_path / "parameter_audit.json"
    report = audit_system_parameters(
        models=DEFAULT_PIPELINE_MODELS,
        output_json=out_json,
        raise_on_violation=True,
    )

    assert out_json.exists()
    with open(out_json, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    # Validate top-level schema
    required_keys = [
        "is_compliant",
        "verdict",
        "total_learned_parameters",
        "total_parameters_billions",
        "budget_limit",
        "budget_limit_billions",
        "budget_utilization_pct",
        "headroom_parameters",
        "headroom_billions",
        "models",
        "audit_timestamp_utc",
        "rule_compliance",
    ]
    for k in required_keys:
        assert k in loaded, f"Missing required key in parameter_audit.json: {k}"

    assert loaded["is_compliant"] is True
    assert loaded["verdict"] == "PASS"
    assert loaded["budget_limit"] == 4_000_000_000
    assert loaded["rule_compliance"]["passed"] is True

    # Validate per-model breakdown
    for model_name, m_info in loaded["models"].items():
        assert "parameters" in m_info
        assert "parameters_billions" in m_info
        assert "role" in m_info
        assert m_info["parameters"] > 0


def test_extract_models_from_config(tmp_path: Path):
    """Test extracting model definitions from pipeline and model configuration files."""
    # 1. Pipeline yaml
    pipeline_cfg = tmp_path / "test_pipeline.yaml"
    pipeline_cfg.write_text(
        """
retrieval:
  dense_macro:
    model_name: "CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2"
ranking:
  reranker:
    enabled: true
    model_name: "BAAI/bge-reranker-v2-m3"
""",
        encoding="utf-8",
    )
    models = extract_models_from_config(pipeline_cfg)
    assert len(models) == 2
    assert models[0]["name"] == "CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2"
    assert models[1]["name"] == "BAAI/bge-reranker-v2-m3"

    # 2. Disabled reranker in pipeline yaml
    pipeline_cfg_no_rrk = tmp_path / "test_pipeline_no_rrk.yaml"
    pipeline_cfg_no_rrk.write_text(
        """
retrieval:
  dense_macro:
    model_name: "CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2"
ranking:
  reranker:
    enabled: false
    model_name: "BAAI/bge-reranker-v2-m3"
""",
        encoding="utf-8",
    )
    models_no_rrk = extract_models_from_config(pipeline_cfg_no_rrk)
    assert len(models_no_rrk) == 1
    assert models_no_rrk[0]["name"] == "CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2"


def test_pipeline_audit_parameters_method(tmp_path: Path):
    """Test LegalIRPipeline.audit_parameters method integration."""
    class DummyRetriever:
        model_name = "CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2"

    class DummyEngine:
        dense_retriever = DummyRetriever()

    class DummyReranker:
        model_name = "BAAI/bge-reranker-v2-m3"

    pipeline = LegalIRPipeline(
        hybrid_engine=DummyEngine(),
        reranker=DummyReranker(),
    )

    out_file = tmp_path / "pipeline_parameter_audit.json"
    report = pipeline.audit_parameters(output_json=out_file, raise_on_violation=True)

    assert report["is_compliant"] is True
    assert report["verdict"] == "PASS"
    assert out_file.exists()


def test_cli_audit_parameters(tmp_path: Path):
    """Test scripts/audit_parameters.py standalone CLI."""
    out_json = tmp_path / "cli_parameter_audit.json"
    cmd = [
        sys.executable,
        "scripts/audit_parameters.py",
        "--config",
        "configs/pipeline.yaml",
        "--output-json",
        str(out_json),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert "COMPLIANCE VERDICT" in res.stdout
    assert "PASS" in res.stdout
    assert out_json.exists()
