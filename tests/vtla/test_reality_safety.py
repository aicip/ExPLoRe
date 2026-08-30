from __future__ import annotations

import torch

from explore_vtla.model import ExPLoReVTLA
from explore_vtla.reality import PredictionErrorMemory, RealityReconciler
from explore_vtla.safety import IndependentSafetyGate, SafetyDecision, SafetyEnvelope
from explore_vtla.synthetic import default_synthetic_config, make_synthetic_sequence


def test_reality_report_covers_all_modalities():
    torch.manual_seed(2)
    cfg = default_synthetic_config(hidden_dim=32)
    seq = make_synthetic_sequence(cfg, batch_size=2, cycles=1)
    out = ExPLoReVTLA(cfg)(seq)
    report = RealityReconciler().report(cfg, seq, out)
    assert set(report.mean_error_by_modality) == set(cfg.modality_order)
    assert all(value >= 0 for value in report.max_error_by_modality.values())


def test_prediction_error_memory_penalizes_high_error():
    memory = PredictionErrorMemory(("vision", "tactile"), decay=0.0, gain=1.0)
    factors = memory.update({"vision": 0.1, "tactile": 2.0})
    assert factors["tactile"] < factors["vision"]


def test_safety_gate_allows_bounded_action():
    gate = IndependentSafetyGate(SafetyEnvelope())
    result = gate.evaluate(
        torch.tensor([0.1, 0.2]),
        force=torch.tensor([1.0, 1.0]),
        feasibility_probability=0.9,
        action_uncertainty=0.1,
    )
    assert result.decision is SafetyDecision.ALLOW


def test_safety_gate_clamps_action_norm():
    gate = IndependentSafetyGate(SafetyEnvelope(max_action_norm=1.0))
    result = gate.evaluate(
        torch.tensor([3.0, 4.0]),
        force=torch.tensor([0.0, 0.0]),
        feasibility_probability=0.9,
        action_uncertainty=0.1,
    )
    assert result.decision is SafetyDecision.CLAMP
    assert abs(float(torch.linalg.vector_norm(result.action)) - 1.0) < 1e-6


def test_safety_gate_holds_on_force_limit():
    gate = IndependentSafetyGate(SafetyEnvelope(max_force_norm=2.0))
    result = gate.evaluate(
        torch.tensor([0.1, 0.2]),
        force=torch.tensor([3.0, 0.0]),
        feasibility_probability=0.9,
        action_uncertainty=0.1,
    )
    assert result.decision is SafetyDecision.HOLD
    assert "force_limit" in result.reasons


def test_safety_gate_holds_on_low_feasibility():
    gate = IndependentSafetyGate(SafetyEnvelope(min_feasibility_probability=0.7))
    result = gate.evaluate(
        torch.tensor([0.1]),
        force=torch.tensor([0.0]),
        feasibility_probability=0.2,
        action_uncertainty=0.1,
    )
    assert result.decision is SafetyDecision.HOLD


def test_safety_gate_fails_closed_on_non_finite_signal():
    gate = IndependentSafetyGate(SafetyEnvelope())
    result = gate.evaluate(
        torch.tensor([float("nan")]),
        force=torch.tensor([0.0]),
        feasibility_probability=0.9,
        action_uncertainty=0.1,
    )
    assert result.decision is SafetyDecision.HOLD
