"""
Tests for DeviceContract and Hardware Profile verification.
Verifies Kaggle Dual-T4, Colab Single-T4, and Colab Single-A100 hardware contracts.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from src.release.contracts import (
    DeviceContract,
    HardwareProfile,
    KAGGLE_T4X2_CONTRACT,
    COLAB_T4_CONTRACT,
    COLAB_A100_CONTRACT,
    verify_device_contract,
)


def test_kaggle_t4x2_contract_properties():
    """Kaggle dual-T4 profile requires >= 2 CUDA devices and separate placement."""
    assert KAGGLE_T4X2_CONTRACT.min_cuda_devices == 2
    assert KAGGLE_T4X2_CONTRACT.dense_device == "cuda:0"
    assert KAGGLE_T4X2_CONTRACT.reranker_device == "cuda:1"
    assert KAGGLE_T4X2_CONTRACT.accelerator_contains == "T4"


def test_colab_t4_contract_properties():
    """Colab single-T4 profile tests the exact single-GPU topology used by A100."""
    assert COLAB_T4_CONTRACT.min_cuda_devices == 1
    assert COLAB_T4_CONTRACT.dense_device == "cuda:0"
    assert COLAB_T4_CONTRACT.reranker_device == "cuda:0"
    assert COLAB_T4_CONTRACT.accelerator_contains == "T4"


def test_colab_a100_contract_properties():
    """Colab A100 profile requires 1 A100 GPU on cuda:0."""
    assert COLAB_A100_CONTRACT.min_cuda_devices == 1
    assert COLAB_A100_CONTRACT.dense_device == "cuda:0"
    assert COLAB_A100_CONTRACT.reranker_device == "cuda:0"
    assert COLAB_A100_CONTRACT.accelerator_contains == "A100"


def test_verify_device_contract_accepts_valid_a100():
    """A single A100 device satisfies the A100 contract."""
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.device_count", return_value=1), \
         patch("torch.cuda.get_device_name", return_value="NVIDIA A100-SXM4-40GB"):
        profile = verify_device_contract(COLAB_A100_CONTRACT)
        assert profile.is_valid
        assert profile.device_count == 1
        assert "A100" in profile.device_names[0]


def test_verify_device_contract_rejects_t4_for_a100_unless_debug():
    """A T4 GPU must be rejected by the A100 contract unless debug mode is enabled."""
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.device_count", return_value=1), \
         patch("torch.cuda.get_device_name", return_value="Tesla T4"):
        with pytest.raises(RuntimeError, match="Hardware mismatch.*A100"):
            verify_device_contract(COLAB_A100_CONTRACT, allow_debug=False)

        # In debug mode, it should return a non-PASS verdict flag
        profile = verify_device_contract(COLAB_A100_CONTRACT, allow_debug=True)
        assert not profile.is_authoritative
        assert profile.verdict == "DEBUG_ONLY"


def test_verify_device_contract_rejects_insufficient_gpus_for_kaggle_t4x2():
    """Kaggle dual-T4 contract fails when only 1 GPU is visible."""
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.device_count", return_value=1), \
         patch("torch.cuda.get_device_name", return_value="Tesla T4"):
        with pytest.raises(RuntimeError, match="requires >= 2 CUDA devices"):
            verify_device_contract(KAGGLE_T4X2_CONTRACT)


def test_verify_device_contract_fails_on_cpu():
    """CUDA gates fail closed when torch.cuda.is_available() is False."""
    with patch("torch.cuda.is_available", return_value=False):
        with pytest.raises(RuntimeError, match="CUDA not available"):
            verify_device_contract(COLAB_A100_CONTRACT)
