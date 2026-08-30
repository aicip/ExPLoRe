"""Causal intervention checks for whether routing importance is behaviorally faithful."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from .contracts import VTLASequence
from .interventions import Intervention, InterventionKind, apply_intervention
from .metrics import routing_by_modality, routing_faithfulness
from .model import ExPLoReVTLA


@dataclass(frozen=True)
class FaithfulnessReport:
    objective: str
    routing_importance: dict[str, float]
    intervention_impact: dict[str, float]
    rank_correlation: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _aggregate_action(model: ExPLoReVTLA, sequence: VTLASequence) -> tuple[torch.Tensor, torch.Tensor]:
    output = model(sequence)
    return model.aggregate_action(output), output.dispatch


def evaluate_modality_faithfulness(
    model: ExPLoReVTLA,
    sequence: VTLASequence,
    *,
    objective: str = "action",
) -> FaithfulnessReport:
    if objective not in model.config.objectives:
        raise ValueError(f"unknown objective {objective!r}")
    model.eval()
    with torch.no_grad():
        baseline_action, dispatch = _aggregate_action(model, sequence)
        objective_idx = tuple(model.config.objectives).index(objective)
        importance = routing_by_modality(dispatch)[objective_idx]
        impacts = []
        for idx, name in enumerate(model.config.modality_order):
            altered = apply_intervention(
                sequence,
                model.config.modality_order,
                Intervention(name, InterventionKind.DROP),
                seed=700 + idx,
            )
            action, _ = _aggregate_action(model, altered)
            impacts.append(torch.mean(torch.abs(action - baseline_action)))
        impact_tensor = torch.stack(impacts)
        corr = routing_faithfulness(importance.cpu(), impact_tensor.cpu())
    return FaithfulnessReport(
        objective=objective,
        routing_importance={
            name: float(importance[idx].item()) for idx, name in enumerate(model.config.modality_order)
        },
        intervention_impact={
            name: float(impact_tensor[idx].item()) for idx, name in enumerate(model.config.modality_order)
        },
        rank_correlation=corr,
    )
