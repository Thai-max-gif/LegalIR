"""
Tests for Gate Chain validation.
Ensures A100 production run strictly enforces valid upstream PASS reports
with matching Git SHA, dataset hash, and algorithm config hash.
"""

from __future__ import annotations

import pytest
from pathlib import Path

from src.release.fingerprints import (
    verify_prior_gate_reports,
    GateChainValidationError,
)

SAMPLE_SHA = "718efb7ba4565fa5b863f05927122484f8e58c2f"
SAMPLE_DATASET_HASH = "a" * 64
SAMPLE_CONFIG_HASH = "b" * 64


@pytest.fixture
def valid_kaggle_report() -> dict:
    return {
        "stage": "KAGGLE_T4X2",
        "verdict": "PASS",
        "git_sha": SAMPLE_SHA,
        "dataset_manifest_sha256": SAMPLE_DATASET_HASH,
        "algorithm_config_sha256": SAMPLE_CONFIG_HASH,
        "gpu_count": 2,
        "real_models_only": True,
        "cpu_fallback_used": False,
        "model_fallback_used": False,
    }


@pytest.fixture
def valid_colab_t4_report() -> dict:
    return {
        "stage": "COLAB_SINGLE_T4",
        "verdict": "PASS",
        "git_sha": SAMPLE_SHA,
        "dataset_manifest_sha256": SAMPLE_DATASET_HASH,
        "algorithm_config_sha256": SAMPLE_CONFIG_HASH,
        "cuda_device_count": 1,
        "real_models_only": True,
        "single_gpu_production_path": True,
    }


def test_valid_gate_chain_passes(valid_kaggle_report, valid_colab_t4_report):
    """Matching PASS reports satisfy the A100 production gate chain."""
    result = verify_prior_gate_reports(
        kaggle_report=valid_kaggle_report,
        colab_t4_report=valid_colab_t4_report,
        expected_sha=SAMPLE_SHA,
        expected_dataset_hash=SAMPLE_DATASET_HASH,
        expected_config_hash=SAMPLE_CONFIG_HASH,
    )
    assert result.is_valid


def test_missing_kaggle_report_fails(valid_colab_t4_report):
    """A100 cannot run without Kaggle PASS report."""
    with pytest.raises(GateChainValidationError, match="Kaggle T4x2 report missing"):
        verify_prior_gate_reports(
            kaggle_report=None,
            colab_t4_report=valid_colab_t4_report,
            expected_sha=SAMPLE_SHA,
            expected_dataset_hash=SAMPLE_DATASET_HASH,
            expected_config_hash=SAMPLE_CONFIG_HASH,
        )


def test_missing_colab_t4_report_fails(valid_kaggle_report):
    """A100 cannot run without Colab T4 PASS report."""
    with pytest.raises(GateChainValidationError, match="Colab T4 report missing"):
        verify_prior_gate_reports(
            kaggle_report=valid_kaggle_report,
            colab_t4_report=None,
            expected_sha=SAMPLE_SHA,
            expected_dataset_hash=SAMPLE_DATASET_HASH,
            expected_config_hash=SAMPLE_CONFIG_HASH,
        )


def test_non_pass_verdict_fails(valid_kaggle_report, valid_colab_t4_report):
    """Reports with verdict != 'PASS' fail verification."""
    valid_kaggle_report["verdict"] = "FAIL"
    with pytest.raises(GateChainValidationError, match="did not pass"):
        verify_prior_gate_reports(
            kaggle_report=valid_kaggle_report,
            colab_t4_report=valid_colab_t4_report,
            expected_sha=SAMPLE_SHA,
            expected_dataset_hash=SAMPLE_DATASET_HASH,
            expected_config_hash=SAMPLE_CONFIG_HASH,
        )


def test_mismatched_sha_fails(valid_kaggle_report, valid_colab_t4_report):
    """A report for an older or different SHA is rejected."""
    valid_colab_t4_report["git_sha"] = "0" * 40
    with pytest.raises(GateChainValidationError, match="Git SHA mismatch"):
        verify_prior_gate_reports(
            kaggle_report=valid_kaggle_report,
            colab_t4_report=valid_colab_t4_report,
            expected_sha=SAMPLE_SHA,
            expected_dataset_hash=SAMPLE_DATASET_HASH,
            expected_config_hash=SAMPLE_CONFIG_HASH,
        )


def test_mismatched_dataset_hash_fails(valid_kaggle_report, valid_colab_t4_report):
    """A report generated on a different dataset version is rejected."""
    valid_kaggle_report["dataset_manifest_sha256"] = "f" * 64
    with pytest.raises(GateChainValidationError, match="Dataset hash mismatch"):
        verify_prior_gate_reports(
            kaggle_report=valid_kaggle_report,
            colab_t4_report=valid_colab_t4_report,
            expected_sha=SAMPLE_SHA,
            expected_dataset_hash=SAMPLE_DATASET_HASH,
            expected_config_hash=SAMPLE_CONFIG_HASH,
        )


def test_mismatched_config_hash_fails(valid_kaggle_report, valid_colab_t4_report):
    """A report generated with different algorithm config is rejected."""
    valid_kaggle_report["algorithm_config_sha256"] = "e" * 64
    with pytest.raises(GateChainValidationError, match="Algorithm config hash mismatch"):
        verify_prior_gate_reports(
            kaggle_report=valid_kaggle_report,
            colab_t4_report=valid_colab_t4_report,
            expected_sha=SAMPLE_SHA,
            expected_dataset_hash=SAMPLE_DATASET_HASH,
            expected_config_hash=SAMPLE_CONFIG_HASH,
        )


def test_mock_devices_rejected_in_gate_chain(valid_kaggle_report, valid_colab_t4_report):
    """A report containing mock hardware devices is rejected."""
    valid_kaggle_report["devices"] = ["Mock Tesla T4", "Mock Tesla T4"]
    with pytest.raises(GateChainValidationError, match="contains mock hardware devices"):
        verify_prior_gate_reports(
            kaggle_report=valid_kaggle_report,
            colab_t4_report=valid_colab_t4_report,
            expected_sha=SAMPLE_SHA,
            expected_dataset_hash=SAMPLE_DATASET_HASH,
            expected_config_hash=SAMPLE_CONFIG_HASH,
        )


def test_mock_gpu_rejected_in_gate_chain(valid_kaggle_report, valid_colab_t4_report):
    """A Colab report with mock `gpu`/`gpu_name` is rejected even if verdict is PASS."""
    valid_colab_t4_report["gpu"] = "Mock Tesla T4"
    with pytest.raises(GateChainValidationError, match="contains mock hardware devices"):
        verify_prior_gate_reports(
            kaggle_report=valid_kaggle_report,
            colab_t4_report=valid_colab_t4_report,
            expected_sha=SAMPLE_SHA,
            expected_dataset_hash=SAMPLE_DATASET_HASH,
            expected_config_hash=SAMPLE_CONFIG_HASH,
        )
