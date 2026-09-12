"""
Unit tests for precision contract and mixed-precision resolution.
Verifies FP16 (with GradScaler), BF16 (autocast only, no GradScaler), and FP32 modes.
"""

from __future__ import annotations

import torch
import pytest

from src.training.trainer import resolve_precision_config


def test_resolve_fp16():
    """FP16 on CUDA enables autocast with float16 and enables GradScaler."""
    cuda_device = torch.device("cuda:0" if torch.cuda.is_available() else "cuda")
    precision, dtype, amp_enabled, scaler_enabled = resolve_precision_config(
        {"precision": "fp16"}, device=cuda_device
    )
    assert precision == "fp16"
    assert dtype == torch.float16
    assert amp_enabled is True
    assert scaler_enabled is True


def test_resolve_bf16():
    """BF16 on CUDA enables autocast with bfloat16 but DISABLES GradScaler."""
    cuda_device = torch.device("cuda:0" if torch.cuda.is_available() else "cuda")
    precision, dtype, amp_enabled, scaler_enabled = resolve_precision_config(
        {"precision": "bf16"}, device=cuda_device
    )
    assert precision == "bf16"
    assert dtype == torch.bfloat16
    assert amp_enabled is True
    assert scaler_enabled is False


def test_resolve_fp32():
    """FP32 disables autocast and disables GradScaler."""
    cuda_device = torch.device("cuda:0" if torch.cuda.is_available() else "cuda")
    precision, dtype, amp_enabled, scaler_enabled = resolve_precision_config(
        {"precision": "fp32"}, device=cuda_device
    )
    assert precision == "fp32"
    assert dtype is None
    assert amp_enabled is False
    assert scaler_enabled is False


def test_resolve_cpu_disables_amp_and_scaler():
    """CPU execution disables both AMP and GradScaler regardless of config."""
    cpu_device = torch.device("cpu")
    precision, dtype, amp_enabled, scaler_enabled = resolve_precision_config(
        {"precision": "bf16"}, device=cpu_device
    )
    assert precision == "bf16"
    assert amp_enabled is False
    assert scaler_enabled is False


def test_backward_compatibility_fp16_boolean():
    """Legacy fp16: true boolean maps to fp16."""
    cuda_device = torch.device("cuda:0" if torch.cuda.is_available() else "cuda")
    precision, dtype, amp_enabled, scaler_enabled = resolve_precision_config(
        {"fp16": True}, device=cuda_device
    )
    assert precision == "fp16"
    assert dtype == torch.float16
    assert amp_enabled is True
    assert scaler_enabled is True

    precision, dtype, amp_enabled, scaler_enabled = resolve_precision_config(
        {"fp16": False}, device=cuda_device
    )
    assert precision == "fp32"
    assert amp_enabled is False
    assert scaler_enabled is False
