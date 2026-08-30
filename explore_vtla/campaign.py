"""Failure-oriented robustness campaigns for VTLA models."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from .contracts import VTLASequence
from .interventions import Intervention, InterventionKind, apply_intervention
from .metrics import brier_score, expected_calibration_error
from .model import ExPLoReVTLA


@dataclass(frozen=True)
class ScenarioResult:
    name: str
    action_rmse: float
    contact_accuracy: float
    feasibility_brier: float
    feasibility_ece: float
    mean_action_delta_from_nominal: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _evaluate(model: ExPLoReVTLA, sequence: VTLASequence, nominal_action: torch.Tensor | None) -> ScenarioResult:
    model.eval()
    with torch.no_grad():
        output = model(sequence)
        action = model.aggregate_action(output)
        contact_prob = model.aggregate_probability(output, "contact")
        feasibility_prob = model.aggregate_probability(output, "feasibility")
        action_rmse = torch.sqrt(torch.mean((action - sequence.action) ** 2)).item()
        contact_accuracy = ((contact_prob >= 0.5) == (sequence.contact >= 0.5)).float().mean().item()
        delta = 0.0 if nominal_action is None else torch.mean(torch.abs(action - nominal_action)).item()
        return ScenarioResult(
            name=str(sequence.metadata.get("scenario", "nominal")),
            action_rmse=float(action_rmse),
            contact_accuracy=float(contact_accuracy),
            feasibility_brier=brier_score(feasibility_prob, sequence.feasible),
            feasibility_ece=expected_calibration_error(feasibility_prob, sequence.feasible, bins=8),
            mean_action_delta_from_nominal=float(delta),
        )


def default_interventions() -> tuple[Intervention, ...]:
    return (
        Intervention("tactile", InterventionKind.DROP),
        Intervention("vision", InterventionKind.DROP),
        Intervention("tactile", InterventionKind.DELAY, steps=2),
        Intervention("force", InterventionKind.SPIKE, magnitude=4.0),
        Intervention("tactile", InterventionKind.DRIFT, magnitude=1.5),
        Intervention("vision", InterventionKind.NOISE, magnitude=1.0),
        Intervention("proprio", InterventionKind.TEMPORAL_SHUFFLE),
    )


def run_campaign(
    model: ExPLoReVTLA,
    sequence: VTLASequence,
    interventions: tuple[Intervention, ...] | None = None,
) -> list[ScenarioResult]:
    interventions = interventions or default_interventions()
    nominal = sequence.clone()
    nominal.metadata["scenario"] = "nominal"
    model.eval()
    with torch.no_grad():
        nominal_output = model(nominal)
        nominal_action = model.aggregate_action(nominal_output)
    results = [_evaluate(model, nominal, None)]
    for idx, intervention in enumerate(interventions):
        altered = apply_intervention(sequence, model.config.modality_order, intervention, seed=100 + idx)
        altered.metadata["scenario"] = f"{intervention.kind.value}_{intervention.modality}"
        results.append(_evaluate(model, altered, nominal_action))
    return results
