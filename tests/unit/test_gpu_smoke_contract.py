import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required; run this on the T4/A100 target")
def test_cuda_gradient_updates_a_trainable_parameter():
    parameter = torch.nn.Parameter(torch.tensor([1.0], device="cuda"))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    before = parameter.detach().clone()
    (parameter.square().sum()).backward()
    optimizer.step()
    assert not torch.equal(before, parameter.detach())


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two ranks; run in the Kaggle T4×2 launcher")
def test_two_gpu_target_is_visible():
    assert torch.cuda.device_count() == 2
