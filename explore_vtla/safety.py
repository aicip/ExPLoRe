"""Independent deterministic safety envelope for learned VTLA actions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch


class SafetyDecision(str, Enum):
    ALLOW = "allow"
    CLAMP = "clamp"
    HOLD = "hold"


@dataclass(frozen=True)
class SafetyEnvelope:
    max_action_norm: float = 1.0
    max_force_norm: float = 8.0
    min_feasibility_probability: float = 0.55
    max_action_uncertainty: float = 0.35

    def __post_init__(self) -> None:
        if self.max_action_norm <= 0 or self.max_force_norm <= 0:
            raise ValueError("norm limits must be positive")
        if not 0 <= self.min_feasibility_probability <= 1:
            raise ValueError("feasibility probability must be in [0,1]")
        if self.max_action_uncertainty < 0:
            raise ValueError("max_action_uncertainty must be non-negative")


@dataclass(frozen=True)
class SafetyResult:
    decision: SafetyDecision
    action: torch.Tensor
    reasons: tuple[str, ...]


class IndependentSafetyGate:
    """Fail-closed action gate whose thresholds are not controlled by the model."""

    def __init__(self, envelope: SafetyEnvelope) -> None:
        self.envelope = envelope

    def evaluate(
        self,
        action: torch.Tensor,
        *,
        force: torch.Tensor,
        feasibility_probability: float,
        action_uncertainty: float,
    ) -> SafetyResult:
        if action.ndim != 1 or force.ndim != 1:
            raise ValueError("action and force must be 1-D vectors")
        if not torch.isfinite(action).all() or not torch.isfinite(force).all():
            return SafetyResult(SafetyDecision.HOLD, torch.zeros_like(action), ("non_finite_signal",))
        reasons: list[str] = []
        if float(torch.linalg.vector_norm(force)) > self.envelope.max_force_norm:
            reasons.append("force_limit")
        if float(feasibility_probability) < self.envelope.min_feasibility_probability:
            reasons.append("low_feasibility")
        if float(action_uncertainty) > self.envelope.max_action_uncertainty:
            reasons.append("high_uncertainty")
        if reasons:
            return SafetyResult(SafetyDecision.HOLD, torch.zeros_like(action), tuple(reasons))

        norm = float(torch.linalg.vector_norm(action))
        if norm > self.envelope.max_action_norm:
            scaled = action * (self.envelope.max_action_norm / max(norm, 1e-12))
            return SafetyResult(SafetyDecision.CLAMP, scaled, ("action_norm_limit",))
        return SafetyResult(SafetyDecision.ALLOW, action.clone(), ())
