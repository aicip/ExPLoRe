from __future__ import annotations

import torch

from explore_vtla.interventions import Intervention, InterventionKind, apply_intervention
from explore_vtla.metrics import (
    brier_score,
    expected_calibration_error,
    normalized_entropy,
    rank_correlation,
    routing_by_modality,
    routing_by_phase,
    routing_faithfulness,
)
from explore_vtla.synthetic import default_synthetic_config, make_synthetic_sequence


def _seq():
    cfg = default_synthetic_config()
    return cfg, make_synthetic_sequence(cfg, batch_size=3, cycles=1)


def test_drop_marks_modality_unavailable():
    cfg, seq = _seq()
    out = apply_intervention(seq, cfg.modality_order, Intervention("tactile", InterventionKind.DROP))
    idx = cfg.modality_order.index("tactile")
    assert torch.equal(out.modalities["tactile"], torch.zeros_like(out.modalities["tactile"]))
    assert torch.equal(out.quality[..., idx, :], torch.zeros_like(out.quality[..., idx, :]))


def test_delay_shifts_signal():
    cfg, seq = _seq()
    original = seq.modalities["vision"].clone()
    out = apply_intervention(seq, cfg.modality_order, Intervention("vision", InterventionKind.DELAY, steps=1))
    assert torch.allclose(out.modalities["vision"][:, 1:], original[:, :-1])


def test_temporal_shuffle_preserves_values():
    cfg, seq = _seq()
    original = seq.modalities["proprio"]
    out = apply_intervention(seq, cfg.modality_order, Intervention("proprio", InterventionKind.TEMPORAL_SHUFFLE), seed=4)
    assert torch.allclose(
        original.sort(dim=1).values,
        out.modalities["proprio"].sort(dim=1).values,
    )


def test_routing_by_modality_normalizes():
    dispatch = torch.rand(2, 4, 3, 5)
    dispatch = dispatch / dispatch.sum(dim=(1, 2), keepdim=True)
    result = routing_by_modality(dispatch)
    assert result.shape == (5, 3)
    assert torch.allclose(result.sum(dim=-1), torch.ones(5), atol=1e-6)


def test_routing_by_phase_normalizes():
    dispatch = torch.rand(2, 4, 3, 5)
    dispatch = dispatch / dispatch.sum(dim=(1, 2), keepdim=True)
    phase = torch.tensor([[0, 1, 0, 1], [0, 1, 0, 1]])
    result = routing_by_phase(dispatch, phase, 2)
    assert result.shape == (5, 2)
    assert torch.allclose(result.sum(dim=-1), torch.ones(5), atol=1e-6)


def test_normalized_entropy_uniform_is_one():
    p = torch.full((2, 4), 0.25)
    assert torch.allclose(normalized_entropy(p), torch.ones(2))


def test_calibration_metrics_perfect_predictions_are_zero():
    p = torch.tensor([0.0, 1.0, 0.0, 1.0])
    y = torch.tensor([0.0, 1.0, 0.0, 1.0])
    assert brier_score(p, y) == 0.0
    assert expected_calibration_error(p, y, bins=4) == 0.0


def test_rank_correlation_and_faithfulness():
    a = torch.tensor([0.1, 0.2, 0.9, 0.4])
    b = torch.tensor([1.0, 2.0, 9.0, 4.0])
    assert abs(rank_correlation(a, b) - 1.0) < 1e-6
    assert abs(routing_faithfulness(a, b) - 1.0) < 1e-6
