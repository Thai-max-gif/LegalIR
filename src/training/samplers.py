"""Training batch samplers, microbatch factorization, and T4 throughput probing."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class MicrobatchFactorization:
    """Factorization of effective batch size into microbatch and gradient accumulation."""

    microbatch_size: int
    gradient_accumulation_steps: int

    @property
    def effective_batch_size(self) -> int:
        return self.microbatch_size * self.gradient_accumulation_steps

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


def get_effective_batch_factorizations(
    target_effective_batch: int = 16,
) -> List[MicrobatchFactorization]:
    """
    Get ordered list of candidate microbatch factorizations for Tesla T4 GPU.
    Ordered from largest microbatch (fastest execution) to smaller fallbacks:
    8x2 -> 4x4 -> 2x8 -> 1x16.
    """
    candidates = [
        (8, 2),
        (4, 4),
        (2, 8),
        (1, 16),
    ]
    return [
        MicrobatchFactorization(mb, ga)
        for mb, ga in candidates
        if mb * ga == target_effective_batch
    ]


def validate_factorization(
    factorization: MicrobatchFactorization,
    expected_effective_batch: int = 16,
) -> bool:
    """Assert that factorization maintains the exact target effective batch size."""
    return factorization.effective_batch_size == expected_effective_batch


def probe_factorization_step(
    model: Any,
    tokenizer: Any,
    factorization: MicrobatchFactorization,
    sample_pairs: List[Tuple[str, str, float]],
    device: str = "cpu",
    max_length: int = 512,
    learning_rate: float = 5e-5,
) -> Dict[str, Any]:
    """
    Perform a real forward and backward optimizer step on real query/passage pairs
    at max_length=512 to verify memory and throughput.
    """
    if not validate_factorization(factorization, expected_effective_batch=16):
        raise ValueError(
            f"Effective batch size must remain 16, got {factorization.effective_batch_size}"
        )

    import torch
    import torch.nn as nn
    from torch.optim import AdamW

    dev = torch.device(device)
    model.to(dev)
    model.train()

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    mb = factorization.microbatch_size
    ga = factorization.gradient_accumulation_steps
    eff = factorization.effective_batch_size

    # Prepare full effective batch of size mb * ga
    full_pairs = list(sample_pairs)
    if len(full_pairs) < eff:
        full_pairs = (full_pairs * (eff // max(1, len(full_pairs)) + 1))[:eff]
    else:
        full_pairs = full_pairs[:eff]

    t0 = time.perf_counter()

    # Capture initial weights for param diff verification
    initial_weights = [p.clone().detach() for p in model.parameters() if p.requires_grad]

    optimizer.zero_grad()
    total_loss = 0.0

    # Execute all gradient accumulation microbatches
    for acc_idx in range(ga):
        batch_slice = full_pairs[acc_idx * mb : (acc_idx + 1) * mb]
        queries = [str(p[0]) for p in batch_slice]
        passages = [str(p[1]) for p in batch_slice]
        labels = torch.tensor([float(p[2]) for p in batch_slice], dtype=torch.float32, device=dev).unsqueeze(1)

        encoded = tokenizer(
            queries,
            passages,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        encoded = {k: v.to(dev) for k, v in encoded.items()}

        outputs = model(**encoded)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        loss = criterion(logits, labels)
        scaled_loss = loss / ga
        scaled_loss.backward()
        total_loss += float(loss.item())

    # Gradient clipping as in production training
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # Optimizer step after complete accumulation
    optimizer.step()
    step_duration = time.perf_counter() - t0

    # Verify parameter difference
    diffs = [
        torch.norm(p.detach() - init).item()
        for p, init in zip([p for p in model.parameters() if p.requires_grad], initial_weights)
    ]
    param_diff = sum(diffs) / max(1, len(diffs))

    peak_vram = 0
    if dev.type == "cuda" and torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated(dev)

    return {
        "status": "PASS",
        "factorization": factorization.to_dict(),
        "effective_batch_size": factorization.effective_batch_size,
        "loss": float(total_loss / max(1, ga)),
        "param_diff": float(param_diff),
        "seconds_per_step": float(step_duration),
        "peak_vram_bytes": peak_vram,
        "oom_occurred": False,
    }
