"""Spatiotemporal Soft-MoE routing with ExPLoRe-style differentiable loss coupling."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class ExpertMLP(nn.Module):
    def __init__(self, dim: int, hidden_mult: float, dropout: float) -> None:
        super().__init__()
        hidden = max(dim, int(round(dim * hidden_mult)))
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LossCoupledRouter(nn.Module):
    """Soft-MoE router whose dispatch weights are also per-token loss coefficients.

    ``reliability`` is an external, non-learned trust mask.  Zero-reliability
    tokens cannot receive dispatch mass, preventing unavailable sensor payloads
    from being selected simply because their numeric values happen to minimize a
    loss.  Non-zero reliability scales routing preference but does not overwrite
    the learned router logits.
    """

    def __init__(
        self,
        dim: int,
        num_experts: int,
        hidden_mult: float = 2.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim <= 0 or num_experts <= 1:
            raise ValueError("dim must be positive and num_experts > 1")
        self.dim = int(dim)
        self.num_experts = int(num_experts)
        self.phi = nn.Parameter(torch.empty(dim, num_experts))
        nn.init.kaiming_uniform_(self.phi, a=math.sqrt(5))
        self.scale = nn.Parameter(torch.ones(()))
        self.experts = nn.ModuleList(
            ExpertMLP(dim, hidden_mult, dropout) for _ in range(num_experts)
        )

    def forward(
        self,
        x: torch.Tensor,
        reliability: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if x.ndim != 3:
            raise ValueError("router input must have shape [B,N,D]")
        batch, tokens, dim = x.shape
        if dim != self.dim:
            raise ValueError(f"expected embedding dim {self.dim}, got {dim}")
        if reliability is None:
            reliability = torch.ones(batch, tokens, device=x.device, dtype=x.dtype)
        if reliability.shape != (batch, tokens):
            raise ValueError("reliability must have shape [B,N]")
        if (reliability < 0).any() or (reliability > 1).any():
            raise ValueError("reliability must be in [0,1]")
        if (reliability.sum(dim=1) <= 0).any():
            raise ValueError("each sample must contain at least one usable token")

        x_norm = F.normalize(x, dim=-1, eps=1e-6)
        phi_norm = F.normalize(self.phi, dim=0, eps=1e-6)
        logits = torch.einsum("bnd,de->bne", x_norm, phi_norm) * self.scale

        combine = F.softmax(logits, dim=-1)
        reliability_term = torch.log(reliability.clamp_min(1e-12)).unsqueeze(-1)
        dispatch = F.softmax(logits + reliability_term, dim=1)
        dispatch = torch.where(reliability.unsqueeze(-1) > 0, dispatch, torch.zeros_like(dispatch))
        dispatch = dispatch / dispatch.sum(dim=1, keepdim=True).clamp_min(1e-12)

        slots = torch.einsum("bne,bnd->bed", dispatch, x)
        expert_out = torch.stack(
            [expert(slots[:, idx, :]) for idx, expert in enumerate(self.experts)], dim=1
        )
        output = torch.einsum("bne,bed->bnd", combine, expert_out)
        return output, {"dispatch": dispatch, "combine": combine, "logits": logits}


def loss_coupled_reduce(
    per_token_losses: torch.Tensor,
    dispatch: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    *,
    detach_router: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce ``[B,N,E]`` losses using dispatch weights without mask leakage.

    The weights are re-normalized after applying ``valid_mask`` so an expert
    cannot lower an objective by routing to unlabeled/padded positions.
    """

    if per_token_losses.shape != dispatch.shape:
        raise ValueError("per_token_losses and dispatch must have identical [B,N,E] shape")
    weights = dispatch.detach() if detach_router else dispatch
    if valid_mask is None:
        valid = torch.ones_like(weights)
    else:
        if valid_mask.shape != weights.shape:
            raise ValueError("valid_mask must match loss shape")
        valid = valid_mask.to(dtype=weights.dtype)
    weights = weights * valid
    denom = weights.sum(dim=1, keepdim=True)
    if (denom <= 0).any():
        bad = torch.nonzero(denom.squeeze(1) <= 0, as_tuple=False)
        raise ValueError(f"objective has no valid routed tokens at indices {bad.tolist()}")
    normalized = weights / denom.clamp_min(1e-12)
    per_objective = (normalized * per_token_losses).sum(dim=1).mean(dim=0)
    return per_objective.mean(), per_objective


def router_balance_loss(combine: torch.Tensor) -> torch.Tensor:
    """Squared deviation of average expert use from uniform usage."""

    usage = combine.mean(dim=(0, 1))
    target = torch.full_like(usage, 1.0 / usage.numel())
    return torch.mean((usage - target) ** 2)
