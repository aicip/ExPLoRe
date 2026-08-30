from __future__ import annotations

import pytest
import torch

from explore_vtla.router import LossCoupledRouter, loss_coupled_reduce, router_balance_loss


def _router():
    torch.manual_seed(1)
    return LossCoupledRouter(dim=8, num_experts=4, hidden_mult=1.5)


def test_router_shapes_and_normalization():
    router = _router()
    x = torch.randn(2, 12, 8)
    out, weights = router(x)
    assert out.shape == x.shape
    assert weights["dispatch"].shape == (2, 12, 4)
    assert torch.allclose(weights["dispatch"].sum(dim=1), torch.ones(2, 4), atol=1e-6)
    assert torch.allclose(weights["combine"].sum(dim=-1), torch.ones(2, 12), atol=1e-6)


def test_zero_reliability_tokens_receive_zero_dispatch():
    router = _router()
    x = torch.randn(2, 6, 8)
    reliability = torch.ones(2, 6)
    reliability[:, 2] = 0
    _, weights = router(x, reliability)
    assert torch.equal(weights["dispatch"][:, 2], torch.zeros_like(weights["dispatch"][:, 2]))


def test_router_rejects_all_unusable_sample():
    router = _router()
    x = torch.randn(2, 6, 8)
    reliability = torch.ones(2, 6)
    reliability[0].zero_()
    with pytest.raises(ValueError):
        router(x, reliability)


def test_loss_coupling_sends_gradient_to_router():
    router = _router()
    x = torch.randn(2, 10, 8)
    _, weights = router(x)
    losses = torch.rand(2, 10, 4)
    total, _ = loss_coupled_reduce(losses, weights["dispatch"])
    total.backward()
    assert router.phi.grad is not None
    assert float(router.phi.grad.abs().sum()) > 0


def test_detached_loss_coupling_does_not_send_router_gradient():
    router = _router()
    x = torch.randn(2, 10, 8)
    _, weights = router(x)
    losses = torch.rand(2, 10, 4, requires_grad=True)
    total, _ = loss_coupled_reduce(losses, weights["dispatch"], detach_router=True)
    total.backward()
    assert router.phi.grad is None
    assert losses.grad is not None


def test_valid_mask_is_renormalized():
    dispatch = torch.tensor([[[0.9], [0.1]]], dtype=torch.float32)
    losses = torch.tensor([[[100.0], [2.0]]])
    mask = torch.tensor([[[0.0], [1.0]]])
    total, per_obj = loss_coupled_reduce(losses, dispatch, mask)
    assert torch.allclose(total, torch.tensor(2.0))
    assert torch.allclose(per_obj, torch.tensor([2.0]))


def test_valid_mask_rejects_objective_with_no_valid_tokens():
    dispatch = torch.full((1, 3, 2), 1 / 3)
    losses = torch.ones_like(dispatch)
    mask = torch.ones_like(dispatch)
    mask[..., 1] = 0
    with pytest.raises(ValueError):
        loss_coupled_reduce(losses, dispatch, mask)


def test_router_balance_loss_is_zero_for_uniform_use():
    combine = torch.full((2, 5, 4), 0.25)
    assert torch.allclose(router_balance_loss(combine), torch.tensor(0.0))
