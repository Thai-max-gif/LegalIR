import torch

from src.reranker.losses import pairwise_logistic_loss, pointwise_bce_loss


def test_pairwise_has_gradient():
    pos = torch.tensor([2.0], requires_grad=True)
    neg = torch.tensor([0.0], requires_grad=True)
    loss = pairwise_logistic_loss(pos, neg)
    loss.backward()
    assert torch.isfinite(loss) and pos.grad is not None and neg.grad is not None


def test_bce_shape_and_gradient():
    logits = torch.tensor([0.0, 1.0], requires_grad=True)
    loss = pointwise_bce_loss(logits, torch.tensor([0.0, 1.0]))
    loss.backward()
    assert torch.isfinite(loss) and logits.grad is not None
