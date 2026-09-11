from __future__ import annotations

import torch
import torch.nn.functional as F


def pairwise_logistic_loss(positive_scores: torch.Tensor, negative_scores: torch.Tensor) -> torch.Tensor:
    if positive_scores.shape != negative_scores.shape:
        raise ValueError("positive and negative score shapes differ")
    return F.softplus(-(positive_scores - negative_scores)).mean()


def pointwise_bce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logits = logits.reshape(-1)
    labels = labels.to(dtype=logits.dtype).reshape(-1)
    if logits.shape != labels.shape:
        raise ValueError("BCE logits and labels must have same flattened shape")
    return F.binary_cross_entropy_with_logits(logits, labels)
