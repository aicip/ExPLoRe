"""Predictive uncertainty utilities for stochastic VTLA inference."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .contracts import VTLASequence
from .model import ExPLoReVTLA


@dataclass(frozen=True)
class ActionUncertainty:
    mean: torch.Tensor
    std: torch.Tensor
    mean_scalar_std: float


def mc_action_uncertainty(
    model: ExPLoReVTLA,
    sequence: VTLASequence,
    *,
    samples: int = 8,
) -> ActionUncertainty:
    if samples < 2:
        raise ValueError("samples must be >= 2")
    was_training = model.training
    model.train()
    actions = []
    with torch.no_grad():
        for _ in range(samples):
            output = model(sequence)
            actions.append(model.aggregate_action(output))
    model.train(was_training)
    stack = torch.stack(actions, dim=0)
    std = stack.std(dim=0, unbiased=False)
    return ActionUncertainty(
        mean=stack.mean(dim=0),
        std=std,
        mean_scalar_std=float(std.mean().item()),
    )
