"""
Tests for configuration layering and canonical fingerprint stability.
Verifies that algorithm configs combined with runtime profiles yield predictable,
valid configurations and maintain cryptographic stability.
"""

from __future__ import annotations

from pathlib import Path
import yaml

from src.release.fingerprints import (
    fingerprint_structured_config,
    validate_runtime_overrides,
)

CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "configs"


def test_algorithm_config_exists_and_fingerprints_stably():
    """Algorithm config must exist and its canonical fingerprint must remain deterministic."""
    algo_path = CONFIG_DIR / "algorithm" / "legalir_v2.yaml"
    assert algo_path.is_file(), f"Missing {algo_path}"
    hash1 = fingerprint_structured_config(algo_path)
    hash2 = fingerprint_structured_config(algo_path)
    assert hash1 == hash2
    assert len(hash1) == 64


def test_merge_algorithm_with_runtime_profiles():
    """Algorithm config merges cleanly with all three runtime profiles."""
    algo_data = yaml.safe_load((CONFIG_DIR / "algorithm" / "legalir_v2.yaml").read_text())

    for runtime_name in ["kaggle_t4x2.yaml", "colab_t4.yaml", "colab_a100.yaml"]:
        runtime_path = CONFIG_DIR / "runtime" / runtime_name
        assert runtime_path.is_file(), f"Missing {runtime_path}"
        runtime_data = yaml.safe_load(runtime_path.read_text())

        resolved = validate_runtime_overrides(algo_data, runtime_data)
        assert "runtime" in resolved
        assert "retrieval" in resolved
        assert "ranking" in resolved
        # Score-affecting settings preserved
        assert resolved["ranking"]["reranker"]["model_name"] == "BAAI/bge-reranker-v2-m3"
        assert resolved["ranking"]["selector"]["max_k"] == 5
