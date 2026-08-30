"""Routing, calibration, intervention, and efficiency metrics."""

from __future__ import annotations

import math

import torch


def routing_by_modality(dispatch: torch.Tensor) -> torch.Tensor:
    """Return objective-by-modality routing mass normalized across modalities."""

    if dispatch.ndim != 4:
        raise ValueError("dispatch must have shape [B,T,M,E]")
    mass = dispatch.sum(dim=(0, 1)).transpose(0, 1)  # [E,M]
    return mass / mass.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def routing_by_phase(dispatch: torch.Tensor, phase: torch.Tensor, num_phases: int) -> torch.Tensor:
    if dispatch.ndim != 4 or phase.shape != dispatch.shape[:2]:
        raise ValueError("dispatch/phase shape mismatch")
    experts = dispatch.shape[-1]
    result = torch.zeros(experts, num_phases, device=dispatch.device, dtype=dispatch.dtype)
    for p in range(num_phases):
        mask = (phase == p).to(dispatch.dtype).unsqueeze(-1).unsqueeze(-1)
        result[:, p] = (dispatch * mask).sum(dim=(0, 1, 2))
    return result / result.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def normalized_entropy(probabilities: torch.Tensor, dim: int = -1) -> torch.Tensor:
    count = probabilities.shape[dim]
    if count <= 1:
        return torch.zeros_like(probabilities.sum(dim=dim))
    p = probabilities.clamp_min(1e-12)
    entropy = -(p * torch.log(p)).sum(dim=dim)
    return entropy / math.log(count)


def expected_calibration_error(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    bins: int = 10,
) -> float:
    if probabilities.shape != targets.shape or bins <= 1:
        raise ValueError("probabilities/targets must match and bins > 1")
    probs = probabilities.detach().flatten()
    truth = targets.detach().flatten().to(probs.dtype)
    ece = torch.zeros((), dtype=probs.dtype, device=probs.device)
    edges = torch.linspace(0, 1, bins + 1, device=probs.device, dtype=probs.dtype)
    for idx in range(bins):
        if idx == bins - 1:
            mask = (probs >= edges[idx]) & (probs <= edges[idx + 1])
        else:
            mask = (probs >= edges[idx]) & (probs < edges[idx + 1])
        if mask.any():
            accuracy = truth[mask].mean()
            confidence = probs[mask].mean()
            ece += mask.float().mean() * torch.abs(accuracy - confidence)
    return float(ece.item())


def brier_score(probabilities: torch.Tensor, targets: torch.Tensor) -> float:
    if probabilities.shape != targets.shape:
        raise ValueError("probabilities and targets must match")
    return float(torch.mean((probabilities - targets.to(probabilities.dtype)) ** 2).item())


def _rank(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(len(values), device=values.device, dtype=torch.float32)
    return ranks


def rank_correlation(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.ndim != 1 or b.ndim != 1 or a.numel() != b.numel() or a.numel() < 2:
        raise ValueError("rank correlation expects equal 1-D vectors with at least 2 values")
    ra, rb = _rank(a), _rank(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = torch.sqrt((ra**2).sum() * (rb**2).sum()).clamp_min(1e-12)
    return float(((ra * rb).sum() / denom).item())


def routing_faithfulness(routing_importance: torch.Tensor, intervention_impact: torch.Tensor) -> float:
    return rank_correlation(routing_importance, intervention_impact)


def parameter_report(model: torch.nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total_parameters": int(total), "trainable_parameters": int(trainable)}
