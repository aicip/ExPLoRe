"""Intervention-calibrated routing objectives.

This module does not assume that a routing map is automatically an explanation.
It measures the performance damage caused by dropping each modality, then trains
the selected objective's routing mass toward that measured intervention profile.
The intervention targets are detached, so the optimization cannot reduce the
calibration loss by manipulating the counterfactual measurement path itself.
"""

from __future__ import annotations

import torch

from .contracts import VTLASequence
from .interventions import Intervention, InterventionKind, apply_intervention
from .model import ExPLoReVTLA, VTLAOutput


def action_drop_impacts(
    model: ExPLoReVTLA,
    sequence: VTLASequence,
    baseline_output: VTLAOutput | None = None,
) -> torch.Tensor:
    """Return non-negative action-MSE degradation from dropping each modality."""

    model.eval()
    with torch.no_grad():
        baseline_output = baseline_output or model(sequence)
        baseline_action = model.aggregate_action(baseline_output)
        baseline_error = torch.mean((baseline_action - sequence.action) ** 2)
        impacts = []
        for idx, name in enumerate(model.config.modality_order):
            altered = apply_intervention(
                sequence,
                model.config.modality_order,
                Intervention(name, InterventionKind.DROP),
                seed=900 + idx,
            )
            altered_action = model.aggregate_action(model(altered))
            altered_error = torch.mean((altered_action - altered.action) ** 2)
            impacts.append(torch.relu(altered_error - baseline_error))
        return torch.stack(impacts)


def counterfactual_action_calibration_loss(
    model: ExPLoReVTLA,
    sequence: VTLASequence,
    output: VTLAOutput,
    *,
    floor: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match mean action-routing mass to measured modality-drop degradation."""

    was_training = model.training
    impacts = action_drop_impacts(model, sequence, baseline_output=None).detach()
    model.train(was_training)
    target = impacts + float(floor)
    target = target / target.sum().clamp_min(1e-12)
    routing = model.objective_weights(output, "action").mean(dim=(0, 1))
    routing = routing / routing.sum().clamp_min(1e-12)
    # Cross entropy with a detached empirical intervention distribution.
    loss = -(target * torch.log(routing.clamp_min(1e-12))).sum()
    return loss, target
