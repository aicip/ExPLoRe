"""Quantitative diagnostics for VTLA routing specialization and collapse."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class RoutingDiagnostics:
    objective_token_entropy: tuple[float, ...]
    mean_objective_token_entropy: float
    expert_combine_utilization: tuple[float, ...]
    min_expert_combine_utilization: float
    max_expert_combine_utilization: float
    expert_utilization_cv: float
    modality_objective_mi_bits: float
    modality_objective_nmi: float
    phase_objective_mi_bits: float | None
    phase_objective_nmi: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _entropy(probabilities: torch.Tensor) -> torch.Tensor:
    p = probabilities.clamp_min(1e-12)
    return -(probabilities * torch.log2(p)).sum()


def _mutual_information(joint_mass: torch.Tensor) -> tuple[float, float]:
    if joint_mass.ndim != 2 or (joint_mass < 0).any():
        raise ValueError("joint_mass must be a non-negative 2-D tensor")
    total = joint_mass.sum()
    if float(total) <= 0:
        return 0.0, 0.0
    joint = joint_mass / total
    px = joint.sum(dim=1, keepdim=True)
    py = joint.sum(dim=0, keepdim=True)
    independent = px * py
    mask = joint > 0
    mi = (joint[mask] * torch.log2(joint[mask] / independent.expand_as(joint)[mask])).sum()
    hx = _entropy(px.squeeze(1))
    hy = _entropy(py.squeeze(0))
    denom = torch.sqrt(hx * hy)
    nmi = torch.zeros_like(mi) if float(denom) <= 1e-12 else mi / denom
    return float(mi.item()), float(nmi.clamp(0.0, 1.0).item())


def routing_diagnostics(
    dispatch: torch.Tensor,
    combine: torch.Tensor,
    phase: torch.Tensor | None = None,
) -> RoutingDiagnostics:
    """Measure token concentration, expert use, and routing dependence.

    ``dispatch`` and ``combine`` must have shape ``[B,T,M,E]``. Dispatch is
    expected to be normalized over ``T*M`` for each sample/expert, while combine
    is expected to be normalized over experts for each token. The function does
    not assume these properties blindly and rejects non-finite/negative input.
    """

    if dispatch.ndim != 4 or combine.shape != dispatch.shape:
        raise ValueError("dispatch and combine must have identical [B,T,M,E] shape")
    if not torch.isfinite(dispatch).all() or not torch.isfinite(combine).all():
        raise ValueError("routing tensors must be finite")
    if (dispatch < 0).any() or (combine < 0).any():
        raise ValueError("routing tensors must be non-negative")
    batch, timesteps, modalities, experts = dispatch.shape
    tokens = timesteps * modalities
    if tokens <= 1 or experts <= 1:
        raise ValueError("routing diagnostics require at least two tokens and experts")

    flat = dispatch.reshape(batch, tokens, experts)
    normalized = flat / flat.sum(dim=1, keepdim=True).clamp_min(1e-12)
    entropy = -(normalized * torch.log(normalized.clamp_min(1e-12))).sum(dim=1)
    entropy = entropy / math.log(tokens)
    per_objective_entropy = entropy.mean(dim=0)

    utilization = combine.mean(dim=(0, 1, 2))
    utilization = utilization / utilization.sum().clamp_min(1e-12)
    mean_util = utilization.mean()
    util_cv = utilization.std(unbiased=False) / mean_util.clamp_min(1e-12)

    modality_joint = dispatch.sum(dim=(0, 1))
    modality_mi, modality_nmi = _mutual_information(modality_joint)

    phase_mi: float | None = None
    phase_nmi: float | None = None
    if phase is not None:
        if phase.shape != (batch, timesteps):
            raise ValueError("phase must have shape [B,T]")
        if (phase < 0).any():
            raise ValueError("phase values must be non-negative")
        values = torch.unique(phase.detach()).sort().values
        phase_joint = torch.zeros(
            len(values), experts, device=dispatch.device, dtype=dispatch.dtype
        )
        for row, value in enumerate(values):
            mask = (phase == value).to(dispatch.dtype).unsqueeze(-1).unsqueeze(-1)
            phase_joint[row] = (dispatch * mask).sum(dim=(0, 1, 2))
        phase_mi, phase_nmi = _mutual_information(phase_joint)

    return RoutingDiagnostics(
        objective_token_entropy=tuple(float(value.item()) for value in per_objective_entropy),
        mean_objective_token_entropy=float(per_objective_entropy.mean().item()),
        expert_combine_utilization=tuple(float(value.item()) for value in utilization),
        min_expert_combine_utilization=float(utilization.min().item()),
        max_expert_combine_utilization=float(utilization.max().item()),
        expert_utilization_cv=float(util_cv.item()),
        modality_objective_mi_bits=modality_mi,
        modality_objective_nmi=modality_nmi,
        phase_objective_mi_bits=phase_mi,
        phase_objective_nmi=phase_nmi,
    )
