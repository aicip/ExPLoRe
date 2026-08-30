from __future__ import annotations

import torch

from explore_vtla.counterfactual import action_drop_impacts, counterfactual_action_calibration_loss
from explore_vtla.model import ExPLoReVTLA
from explore_vtla.synthetic import default_synthetic_config, make_synthetic_sequence


def test_action_drop_impacts_are_nonnegative_and_cover_modalities():
    torch.manual_seed(12)
    cfg = default_synthetic_config(hidden_dim=32)
    seq = make_synthetic_sequence(cfg, batch_size=2, cycles=1)
    model = ExPLoReVTLA(cfg)
    impacts = action_drop_impacts(model, seq)
    assert impacts.shape == (len(cfg.modality_order),)
    assert (impacts >= 0).all()


def test_counterfactual_loss_has_router_gradient():
    torch.manual_seed(13)
    cfg = default_synthetic_config(hidden_dim=32)
    seq = make_synthetic_sequence(cfg, batch_size=2, cycles=1)
    model = ExPLoReVTLA(cfg)
    model.train()
    output = model(seq)
    loss, target = counterfactual_action_calibration_loss(model, seq, output)
    assert torch.allclose(target.sum(), torch.tensor(1.0), atol=1e-6)
    loss.backward()
    assert model.router.phi.grad is not None
    assert float(model.router.phi.grad.abs().sum()) > 0
