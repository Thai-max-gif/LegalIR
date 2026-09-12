"""
Hardware and Release Execution Contracts for LegalIR.
Defines DeviceContract, HardwareProfile, and authoritative gate invariants.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional


@dataclasses.dataclass(frozen=True)
class DeviceContract:
    """Device contract defining required GPU count, accelerator family, and component placement."""
    min_cuda_devices: int
    dense_device: str
    reranker_device: str
    accelerator_contains: Optional[str] = None
    profile_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


KAGGLE_T4X2_CONTRACT = DeviceContract(
    min_cuda_devices=2,
    dense_device="cuda:0",
    reranker_device="cuda:1",
    accelerator_contains="T4",
    profile_name="kaggle_t4x2",
)

COLAB_T4_CONTRACT = DeviceContract(
    min_cuda_devices=1,
    dense_device="cuda:0",
    reranker_device="cuda:0",
    accelerator_contains="T4",
    profile_name="colab_t4",
)

COLAB_A100_CONTRACT = DeviceContract(
    min_cuda_devices=1,
    dense_device="cuda:0",
    reranker_device="cuda:0",
    accelerator_contains="A100",
    profile_name="colab_a100",
)


@dataclasses.dataclass(frozen=True)
class HardwareProfile:
    """Observed hardware state and contract verification result."""
    is_valid: bool
    is_authoritative: bool
    device_count: int
    device_names: list[str]
    verdict: str
    details: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def verify_device_contract(
    contract: DeviceContract,
    allow_debug: bool = False,
) -> HardwareProfile:
    """
    Verify that the current runtime environment strictly satisfies a DeviceContract.
    Fails closed if CUDA is unavailable, device count is insufficient, or GPU family mismatches.
    """
    try:
        import torch
    except ImportError as e:
        raise RuntimeError("PyTorch is required to verify device contracts.") from e

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA not available. Contract '{contract.profile_name}' requires at least "
            f"{contract.min_cuda_devices} CUDA device(s)."
        )

    count = torch.cuda.device_count()
    if count < contract.min_cuda_devices:
        raise RuntimeError(
            f"Insufficient CUDA devices: contract '{contract.profile_name}' requires >= {contract.min_cuda_devices} "
            f"CUDA devices, but found {count}."
        )

    device_names = [torch.cuda.get_device_name(i) for i in range(count)]

    # Check accelerator family (e.g., "T4", "A100")
    if contract.accelerator_contains:
        mismatches = [
            name for name in device_names[: contract.min_cuda_devices]
            if contract.accelerator_contains not in name
        ]
        if mismatches:
            err_msg = (
                f"Hardware mismatch for contract '{contract.profile_name}': expected GPU containing "
                f"'{contract.accelerator_contains}', but detected {device_names}."
            )
            if allow_debug:
                return HardwareProfile(
                    is_valid=True,
                    is_authoritative=False,
                    device_count=count,
                    device_names=device_names,
                    verdict="DEBUG_ONLY",
                    details=f"Overridden in debug mode: {err_msg}",
                )
            raise RuntimeError(err_msg)

    return HardwareProfile(
        is_valid=True,
        is_authoritative=True,
        device_count=count,
        device_names=device_names,
        verdict="PASS",
        details=f"Satisfied {contract.profile_name} contract on {device_names}",
    )
