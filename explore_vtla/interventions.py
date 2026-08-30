"""Controlled interventions for routing-faithfulness and robustness experiments."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch

from .contracts import VTLASequence


class InterventionKind(str, Enum):
    DROP = "drop"
    NOISE = "noise"
    DELAY = "delay"
    TEMPORAL_SHUFFLE = "temporal_shuffle"
    DRIFT = "drift"
    SPIKE = "spike"


@dataclass(frozen=True)
class Intervention:
    modality: str
    kind: InterventionKind
    magnitude: float = 1.0
    steps: int = 1


def apply_intervention(
    sequence: VTLASequence,
    modality_order: tuple[str, ...],
    intervention: Intervention,
    *,
    seed: int = 0,
) -> VTLASequence:
    if intervention.modality not in sequence.modalities:
        raise ValueError(f"unknown modality {intervention.modality!r}")
    result = sequence.clone()
    idx = modality_order.index(intervention.modality)
    x = result.modalities[intervention.modality]
    generator = torch.Generator(device=x.device).manual_seed(int(seed))

    if intervention.kind is InterventionKind.DROP:
        x.zero_()
        result.quality[..., idx, 0] = 0.0
        result.quality[..., idx, 1:] = 0.0
    elif intervention.kind is InterventionKind.NOISE:
        noise = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
        x.add_(noise * float(intervention.magnitude))
        result.quality[..., idx, 1] *= max(0.0, 1.0 - min(1.0, intervention.magnitude / 3.0))
    elif intervention.kind is InterventionKind.DELAY:
        steps = int(intervention.steps)
        if steps <= 0 or steps >= x.shape[1]:
            raise ValueError("delay steps must be between 1 and T-1")
        x[:, steps:] = x[:, :-steps].clone()
        x[:, :steps] = 0.0
        result.quality[:, :steps, idx, 0] = 0.0
        result.quality[..., idx, 2] *= 0.5
    elif intervention.kind is InterventionKind.TEMPORAL_SHUFFLE:
        perm = torch.randperm(x.shape[1], generator=generator, device=x.device)
        result.modalities[intervention.modality] = x[:, perm].clone()
    elif intervention.kind is InterventionKind.DRIFT:
        ramp = torch.linspace(0.0, float(intervention.magnitude), x.shape[1], device=x.device, dtype=x.dtype)
        x.add_(ramp.view(1, -1, 1))
        result.quality[..., idx, 1] *= 0.65
    elif intervention.kind is InterventionKind.SPIKE:
        middle = x.shape[1] // 2
        x[:, middle, :] += float(intervention.magnitude)
        result.quality[:, middle, idx, 1] *= 0.5
    else:  # pragma: no cover - enum closes this branch
        raise ValueError(f"unsupported intervention {intervention.kind}")
    result.metadata["intervention"] = {
        "modality": intervention.modality,
        "kind": intervention.kind.value,
        "magnitude": intervention.magnitude,
        "steps": intervention.steps,
    }
    return result
