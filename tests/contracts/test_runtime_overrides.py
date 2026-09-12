"""
Tests for runtime profile override validation.
Ensures hardware/runtime profiles cannot mutate score-affecting algorithm settings.
"""

from __future__ import annotations

import pytest
from src.release.fingerprints import validate_runtime_overrides, ProtectedKeyViolationError


@pytest.fixture
def base_algorithm_config() -> dict:
    return {
        "dataset": {
            "name": "legalir-task1-clean-data",
            "version": "v2",
        },
        "retrieval": {
            "top_k_candidates": 100,
            "branch_weights": {"dense": 0.6, "bm25": 0.4},
        },
        "models": {
            "reranker": "BAAI/bge-reranker-v2-m3",
            "reranker_revision": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
            "dense": "CODE4LIFEOFFICIAL/huydang-dek21-embedding-v2",
        },
        "training": {
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "loss_type": "bce",
            "learning_rate": 2e-5,
            "batch_size": 8,
            "gradient_accumulation_steps": 2,
            "precision": "bf16",
        },
        "runtime": {
            "device": "cuda:0",
            "num_workers": 4,
        },
    }


def test_allowed_runtime_overrides(base_algorithm_config):
    """Runtime profile can modify non-score-affecting throughput parameters."""
    runtime_profile = {
        "training": {
            "batch_size": 16,
            "gradient_accumulation_steps": 1,
            "precision": "fp16",
        },
        "runtime": {
            "device": "cuda:0",
            "num_workers": 2,
        },
    }
    merged = validate_runtime_overrides(base_algorithm_config, runtime_profile)
    assert merged["training"]["batch_size"] == 16
    assert merged["training"]["gradient_accumulation_steps"] == 1
    assert merged["training"]["precision"] == "fp16"
    assert merged["runtime"]["num_workers"] == 2
    # Base algorithm parameters remain unchanged
    assert merged["training"]["lora_r"] == 16
    assert merged["retrieval"]["top_k_candidates"] == 100


def test_rejects_model_id_override(base_algorithm_config):
    """Runtime profile must fail if it tries to alter base model IDs."""
    runtime_profile = {
        "models": {
            "reranker": "bert-base-uncased",
        }
    }
    with pytest.raises(ProtectedKeyViolationError, match="models.reranker"):
        validate_runtime_overrides(base_algorithm_config, runtime_profile)


def test_rejects_lora_shape_override(base_algorithm_config):
    """Runtime profile must fail if it tries to alter LoRA architecture."""
    runtime_profile = {
        "training": {
            "lora_r": 32,
        }
    }
    with pytest.raises(ProtectedKeyViolationError, match="training.lora_r"):
        validate_runtime_overrides(base_algorithm_config, runtime_profile)


def test_rejects_retrieval_weights_override(base_algorithm_config):
    """Runtime profile must fail if it tries to alter retrieval branch weights."""
    runtime_profile = {
        "retrieval": {
            "branch_weights": {"dense": 0.8, "bm25": 0.2},
        }
    }
    with pytest.raises(ProtectedKeyViolationError, match="retrieval.branch_weights"):
        validate_runtime_overrides(base_algorithm_config, runtime_profile)


def test_rejects_loss_type_override(base_algorithm_config):
    """Runtime profile must fail if it alters loss formulation."""
    runtime_profile = {
        "training": {
            "loss_type": "margin_mse",
        }
    }
    with pytest.raises(ProtectedKeyViolationError, match="training.loss_type"):
        validate_runtime_overrides(base_algorithm_config, runtime_profile)
