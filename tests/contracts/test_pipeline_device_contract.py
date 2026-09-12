"""
Tests for DeviceContract resolution and hardware validation in kaggle_train pipeline.
Ensures single-GPU A100 and single-GPU T4 contracts run without throwing >= 2 GPU errors.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.release.contracts import (
    COLAB_A100_CONTRACT,
    COLAB_T4_CONTRACT,
    KAGGLE_T4X2_CONTRACT,
)
from src.pipeline.kaggle_train import resolve_pipeline_device_allocation


def test_resolve_devices_with_a100_contract():
    """A100 contract resolves to 1 GPU, dense=cuda:0, reranker=cuda:0."""
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.device_count", return_value=1), \
         patch("torch.cuda.get_device_name", return_value="NVIDIA A100-SXM4-40GB"):
        dense_dev, rerank_dev = resolve_pipeline_device_allocation(
            device_contract=COLAB_A100_CONTRACT,
            run_mode="full",
        )
        assert dense_dev == "cuda:0"
        assert rerank_dev == "cuda:0"


def test_resolve_devices_with_colab_t4_contract():
    """Colab T4 contract resolves to 1 GPU, dense=cuda:0, reranker=cuda:0."""
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.device_count", return_value=1), \
         patch("torch.cuda.get_device_name", return_value="Tesla T4"):
        dense_dev, rerank_dev = resolve_pipeline_device_allocation(
            device_contract=COLAB_T4_CONTRACT,
            run_mode="full",
        )
        assert dense_dev == "cuda:0"
        assert rerank_dev == "cuda:0"


def test_resolve_devices_with_kaggle_t4x2_contract():
    """Kaggle dual-T4 contract resolves to dense=cuda:0, reranker=cuda:1."""
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.device_count", return_value=2), \
         patch("torch.cuda.get_device_name", return_value="Tesla T4"):
        dense_dev, rerank_dev = resolve_pipeline_device_allocation(
            device_contract=KAGGLE_T4X2_CONTRACT,
            run_mode="full",
        )
        assert dense_dev == "cuda:0"
        assert rerank_dev == "cuda:1"


def test_kaggle_t4x2_contract_rejects_single_gpu():
    """Kaggle dual-T4 contract must raise RuntimeError when only 1 GPU is available."""
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.device_count", return_value=1), \
         patch("torch.cuda.get_device_name", return_value="Tesla T4"):
        with pytest.raises(RuntimeError, match="requires >= 2 CUDA devices"):
            resolve_pipeline_device_allocation(
                device_contract=KAGGLE_T4X2_CONTRACT,
                run_mode="full",
            )
