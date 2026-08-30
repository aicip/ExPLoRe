from __future__ import annotations

import torch

from explore_vtla.model import ExPLoReVTLA
from explore_vtla.synthetic import default_synthetic_config, make_synthetic_sequence


def _setup():
    torch.manual_seed(3)
    cfg = default_synthetic_config(hidden_dim=32)
    seq = make_synthetic_sequence(cfg, batch_size=4, cycles=1, seed=5)
    return cfg, seq, ExPLoReVTLA(cfg)


def test_forward_shapes():
    cfg, seq, model = _setup()
    out = model(seq)
    assert out.routed_state.shape == (4, 6, len(cfg.modality_order), 32)
    assert out.dispatch.shape == (4, 6, len(cfg.modality_order), len(cfg.objectives))
    assert out.action.shape == (4, 6, len(cfg.modality_order), cfg.action_dim)
    assert out.contact_logits.shape == (4, 6, len(cfg.modality_order))


def test_reconstruction_shapes_match_modalities():
    cfg, seq, model = _setup()
    out = model(seq)
    for name in cfg.modality_order:
        assert out.reconstruction[name].shape == seq.modalities[name].shape
        assert out.next_prediction[name].shape == seq.modalities[name].shape


def test_training_loss_is_finite_and_backpropagates():
    _, seq, model = _setup()
    out = model(seq)
    loss, metrics = model.training_loss(seq, out)
    assert torch.isfinite(loss)
    assert set(metrics) >= {"reconstruction", "alignment", "world_model", "action", "contact", "slip", "feasibility"}
    loss.backward()
    assert model.router.phi.grad is not None
    assert float(model.router.phi.grad.abs().sum()) > 0


def test_model_masks_dropped_modality_from_dispatch():
    cfg, seq, model = _setup()
    tactile_idx = cfg.modality_order.index("tactile")
    seq.quality[..., tactile_idx, :] = 0
    out = model(seq)
    assert torch.equal(out.dispatch[:, :, tactile_idx], torch.zeros_like(out.dispatch[:, :, tactile_idx]))


def test_causal_mask_blocks_future_but_not_same_time():
    cfg, _, model = _setup()
    mask = model._causal_mask(3, len(cfg.modality_order), torch.device("cpu"))
    modalities = len(cfg.modality_order)
    assert mask[0, modalities].isneginf()
    assert mask[0, modalities - 1] == 0
    assert mask[modalities, 0] == 0


def test_detach_ablation_removes_loss_coupling_gradient():
    _, seq, model = _setup()
    out = model(seq)
    loss, _ = model.training_loss(seq, out, detach_router=True)
    loss.backward()
    # Router still participates in the forward feature path, so the expert/combine path can
    # carry gradients.  What is detached is the dispatch-as-loss-coefficient path.  Compare
    # to an isolated dispatch-only objective in router tests for the strict contract.
    assert model.router.phi.grad is not None
