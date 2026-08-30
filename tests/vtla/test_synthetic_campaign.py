from __future__ import annotations

from dataclasses import asdict

import torch

from explore_vtla.campaign import default_interventions, run_campaign
from explore_vtla.experiment import run_smoke_training
from explore_vtla.synthetic import run_mechanism_benchmark


def test_mechanism_benchmark_learns_known_routing_structure():
    result = run_mechanism_benchmark(seed=11, steps=120)
    assert result.final_loss < result.initial_loss * 0.55
    assert result.specialization_score > 0.55
    assert result.specialization_score - result.detached_specialization_score > 0.20
    assert min(result.target_mass) > 0.45


def test_mechanism_benchmark_is_deterministic():
    a = asdict(run_mechanism_benchmark(seed=13, steps=80))
    b = asdict(run_mechanism_benchmark(seed=13, steps=80))
    assert a == b


def test_smoke_training_reduces_end_to_end_loss():
    _, result = run_smoke_training(seed=29, steps=35)
    assert result.final_loss < result.initial_loss
    assert result.loss_reduction_fraction > 0.10
    assert result.parameter_report["trainable_parameters"] > 0


def test_campaign_contains_failure_oriented_scenarios():
    model, _ = run_smoke_training(seed=17, steps=10)
    from explore_vtla.synthetic import make_synthetic_sequence

    sequence = make_synthetic_sequence(model.config, batch_size=4, cycles=1, seed=17)
    results = run_campaign(model, sequence)
    assert len(results) == 1 + len(default_interventions())
    names = {result.name for result in results}
    assert "nominal" in names
    assert "drop_tactile" in names
    assert "spike_force" in names


def test_campaign_metrics_are_finite():
    model, _ = run_smoke_training(seed=19, steps=8)
    from explore_vtla.synthetic import make_synthetic_sequence

    results = run_campaign(model, make_synthetic_sequence(model.config, batch_size=3, cycles=1))
    for result in results:
        values = torch.tensor(
            [
                result.action_rmse,
                result.contact_accuracy,
                result.feasibility_brier,
                result.feasibility_ece,
                result.mean_action_delta_from_nominal,
            ]
        )
        assert torch.isfinite(values).all()
