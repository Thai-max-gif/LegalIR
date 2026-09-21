import math
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class PointwiseBCELoss(nn.Module):
    """Pointwise Binary Cross-Entropy with Logits loss for reranker scoring."""

    def __init__(self, pos_weight: float | None = None, reduction: str = "mean", **kwargs: Any):
        super().__init__()
        weight_tensor = torch.tensor([pos_weight], dtype=torch.float32) if pos_weight is not None else None
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=weight_tensor, reduction=reduction)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        logits: Tensor of shape (batch_size,) or (batch_size, 1)
        labels: Tensor of shape (batch_size,) or (batch_size, 1) with float values in {0.0, 1.0}
        """
        logits = logits.view(-1).float()
        labels = labels.view(-1).float()
        # The loss module itself is never .to(device) by the trainer, so the
        # CPU-created pos_weight must follow the logits device here (GPU runs
        # crashed with cuda:0 vs cpu mismatch otherwise).
        pos_weight = self.loss_fn.pos_weight
        if pos_weight is not None and pos_weight.device != logits.device:
            pos_weight = pos_weight.to(logits.device)
            self.loss_fn.pos_weight = pos_weight
        return self.loss_fn(logits, labels)


class PairwiseLogisticLoss(nn.Module):
    """
    Pairwise Logistic Loss for ranking:
    L(s+, s-) = log(1 + exp(-(s+ - s-))) = softplus(-(s+ - s-))
    """

    def __init__(self, temperature: float = 1.0, reduction: str = "mean", **kwargs: Any):
        super().__init__()
        self.temperature = max(1e-6, float(temperature))
        self.reduction = reduction

    def forward(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        pos_scores: Tensor of shape (batch_size,) or (batch_size, 1)
        neg_scores: Tensor of shape (batch_size,) or (batch_size, 1)
        """
        pos = pos_scores.view(-1).float()
        neg = neg_scores.view(-1).float()
        diff = (pos - neg) / self.temperature
        loss = F.softplus(-diff)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class PairwiseMarginRankingLoss(nn.Module):
    """
    Pairwise Margin Ranking Loss:
    L(s+, s-) = max(0, margin - (s+ - s-))
    """

    def __init__(self, margin: float = 1.0, reduction: str = "mean", **kwargs: Any):
        super().__init__()
        self.margin = float(margin)
        self.reduction = reduction

    def forward(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor,
    ) -> torch.Tensor:
        pos = pos_scores.view(-1).float()
        neg = neg_scores.view(-1).float()
        diff = pos - neg
        loss = F.relu(self.margin - diff)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class ListwiseCrossEntropyLoss(nn.Module):
    """
    Listwise Softmax Cross-Entropy Loss over 1 positive + N negatives.
    Target is index 0 (positive score).
    L(s) = -log( exp(s+ / T) / sum_i exp(s_i / T) )
    """

    def __init__(self, temperature: float = 1.0, reduction: str = "mean", **kwargs: Any):
        super().__init__()
        self.temperature = max(1e-6, float(temperature))
        self.reduction = reduction

    def forward(
        self,
        scores: torch.Tensor,
        target_idx: int | torch.Tensor = 0,
    ) -> torch.Tensor:
        """
        scores: Tensor of shape (batch_size, num_candidates) where column 0 is positive score
        target_idx: int or Tensor of target class indices (default 0)
        """
        if scores.dim() == 1:
            scores = scores.unsqueeze(0)

        scaled_scores = scores.float() / self.temperature
        batch_size = scores.size(0)

        if isinstance(target_idx, int):
            targets = torch.full(
                (batch_size,), target_idx, dtype=torch.long, device=scores.device
            )
        else:
            targets = target_idx.to(scores.device).long().view(-1)

        return F.cross_entropy(scaled_scores, targets, reduction=self.reduction)


def get_loss_function(loss_type: str = "bce", **kwargs: Any) -> nn.Module:
    """Factory function for ranking loss objectives."""
    loss_key = str(loss_type).lower().strip()
    if loss_key in ("bce", "pointwise", "bce_with_logits", "binary_cross_entropy"):
        return PointwiseBCELoss(**kwargs)
    elif loss_key in ("pairwise_logistic", "logistic", "pairwise"):
        return PairwiseLogisticLoss(**kwargs)
    elif loss_key in ("pairwise_margin", "margin", "margin_ranking"):
        return PairwiseMarginRankingLoss(**kwargs)
    elif loss_key in ("listwise", "listwise_ce", "listwise_cross_entropy", "softmax"):
        return ListwiseCrossEntropyLoss(**kwargs)
    else:
        raise ValueError(
            f"Unknown loss_type: '{loss_type}'. Supported: 'bce', 'pairwise_logistic', 'pairwise_margin', 'listwise_ce'"
        )
