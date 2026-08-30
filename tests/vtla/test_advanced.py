from __future__ import annotations

from pathlib import Path

import torch

from explore_vtla.constraint_pressure import (
    PerturbationObservation,
    finite_difference_pressure,
    normalized_violation,
)
from explore_vtla.dataio import load_sequence_npz, save_sequence_npz
from explore_vtla.faithfulness import evaluate_modality_faithfulness
from explore_vtla.model import ExPLoReVTLA
from explore_vtla.provenance import repository_fingerprint
from explore_vtla.synthetic import default_synthetic_config, make_synthetic_sequence
from explore_vtla.uncertainty import mc_action_uncertainty


def test_normalized_violation_both_constraint_directions():
    assert normalized_violation(12, op="<=", threshold=10) == 0.2
    assert normalized_violation(8, op="<=", threshold=10) == 0.0
    assert normalized_violation(8, op=">=", threshold=10) == 0.2


def test_constraint_pressure_orders_strongest_parameter():
    baseline = {"failure_rate": 0.2}
    observations = [
        PerturbationObservation("dropout", 0.1, baseline, {"failure_rate": 0.1}),
        PerturbationObservation("experts", 1.0, baseline, {"failure_rate": 0.19}),
    ]
    result = finite_difference_pressure(
        observations, metric="failure_rate", op="<=", threshold=0.05
    )
    assert result[0].parameter == "dropout"
    assert result[0].direction_to_reduce_violation == 1


def test_npz_roundtrip(tmp_path: Path):
    cfg = default_synthetic_config()
    seq = make_synthetic_sequence(cfg, batch_size=2, cycles=1)
    path = tmp_path / "sequence.npz"
    save_sequence_npz(path, cfg, seq)
    loaded = load_sequence_npz(path, cfg)
    for name in cfg.modality_order:
        torch.testing.assert_close(loaded.modalities[name], seq.modalities[name])
    torch.testing.assert_close(loaded.action, seq.action)
    assert loaded.metadata == seq.metadata


def test_npz_rejects_mismatched_modality_schema(tmp_path: Path):
    cfg = default_synthetic_config()
    seq = make_synthetic_sequence(cfg, batch_size=1, cycles=1)
    path = tmp_path / "sequence.npz"
    save_sequence_npz(path, cfg, seq)
    bad = default_synthetic_config()
    bad = type(bad)(
        modality_dims={"vision": 6, "tactile": 6},
        action_dim=3,
        hidden_dim=48,
        num_heads=4,
        transformer_layers=1,
        max_timesteps=32,
    )
    import pytest

    with pytest.raises(ValueError):
        load_sequence_npz(path, bad)


def test_faithfulness_report_covers_every_modality():
    torch.manual_seed(4)
    cfg = default_synthetic_config(hidden_dim=32)
    seq = make_synthetic_sequence(cfg, batch_size=3, cycles=1)
    model = ExPLoReVTLA(cfg)
    report = evaluate_modality_faithfulness(model, seq)
    assert set(report.routing_importance) == set(cfg.modality_order)
    assert set(report.intervention_impact) == set(cfg.modality_order)
    assert -1.0 <= report.rank_correlation <= 1.0


def test_mc_uncertainty_is_zero_when_dropout_disabled():
    torch.manual_seed(5)
    cfg = default_synthetic_config(hidden_dim=32)
    seq = make_synthetic_sequence(cfg, batch_size=2, cycles=1)
    model = ExPLoReVTLA(cfg)
    result = mc_action_uncertainty(model, seq, samples=3)
    assert result.mean.shape == seq.action.shape
    assert result.std.shape == seq.action.shape
    assert result.mean_scalar_std == 0.0


def test_mc_uncertainty_positive_when_dropout_enabled():
    torch.manual_seed(6)
    base = default_synthetic_config(hidden_dim=32)
    cfg = type(base)(
        modality_dims=base.modality_dims,
        action_dim=base.action_dim,
        hidden_dim=base.hidden_dim,
        num_heads=base.num_heads,
        transformer_layers=base.transformer_layers,
        max_timesteps=base.max_timesteps,
        dropout=0.2,
    )
    seq = make_synthetic_sequence(cfg, batch_size=2, cycles=1)
    model = ExPLoReVTLA(cfg)
    result = mc_action_uncertainty(model, seq, samples=4)
    assert result.mean_scalar_std > 0


def test_repository_fingerprint_is_stable_and_nonempty():
    root = Path(__file__).resolve().parents[2]
    first = repository_fingerprint(root)
    second = repository_fingerprint(root)
    assert first["file_count"] > 0
    assert first["tree_hash"] == second["tree_hash"]
