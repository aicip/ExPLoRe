"""Typed contracts for VTLA data, evidence authority, and model configuration."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum, IntEnum
from typing import Mapping, Sequence

import torch


class SignalHealth(str, Enum):
    NOMINAL = "nominal"
    DEGRADED = "degraded"
    INVALID = "invalid"
    UNAVAILABLE = "unavailable"


class AuthorityLevel(IntEnum):
    """Highest evidence authority earned by a result."""

    M0_UNIT_VERIFIED = 0
    M1_SYNTHETIC_MECHANISM = 1
    M2_OFFLINE_REAL_DATA = 2
    M3_PHYSICS_SIMULATION = 3
    M4_HARDWARE_IN_LOOP = 4
    M5_REAL_ROBOT = 5
    M6_INDEPENDENT_REPRODUCTION = 6


DEFAULT_MODALITIES = ("vision", "tactile", "force", "proprio", "language")
DEFAULT_OBJECTIVES = (
    "reconstruction",
    "alignment",
    "world_model",
    "action",
    "contact",
    "slip",
    "feasibility",
)


@dataclass(frozen=True)
class VTLAConfig:
    modality_dims: Mapping[str, int]
    action_dim: int
    hidden_dim: int = 64
    num_heads: int = 4
    transformer_layers: int = 2
    expert_hidden_mult: float = 2.0
    objectives: Sequence[str] = DEFAULT_OBJECTIVES
    max_timesteps: int = 128
    dropout: float = 0.0
    router_balance_weight: float = 0.01

    def __post_init__(self) -> None:
        if not self.modality_dims:
            raise ValueError("modality_dims must not be empty")
        for name, dim in self.modality_dims.items():
            if not name or int(dim) <= 0:
                raise ValueError(f"invalid modality specification: {name!r}={dim!r}")
        if self.action_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("action_dim and hidden_dim must be positive")
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if self.transformer_layers <= 0 or self.max_timesteps <= 1:
            raise ValueError("transformer_layers must be positive and max_timesteps > 1")
        if len(self.objectives) < 2 or len(set(self.objectives)) != len(self.objectives):
            raise ValueError("objectives must contain at least two unique names")
        unsupported = set(self.objectives) - set(DEFAULT_OBJECTIVES)
        if unsupported:
            raise ValueError(f"unsupported objectives: {sorted(unsupported)}")

    @property
    def modality_order(self) -> tuple[str, ...]:
        return tuple(self.modality_dims.keys())


@dataclass
class VTLASequence:
    """Tensorized multimodal trajectory.

    Each modality is shaped ``[B, T, F_m]``.  ``quality`` is shaped
    ``[B, T, M, 3]`` and stores availability, confidence, and freshness in
    ``[0, 1]``.  Invalid sensor payload values remain distinguishable from a
    legitimate numeric zero through this side-channel.
    """

    modalities: dict[str, torch.Tensor]
    action: torch.Tensor
    contact: torch.Tensor
    slip: torch.Tensor
    feasible: torch.Tensor
    quality: torch.Tensor
    phase: torch.Tensor | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def validate(
        self,
        modality_order: Sequence[str],
        *,
        modality_dims: Mapping[str, int] | None = None,
        action_dim: int | None = None,
    ) -> "VTLASequence":
        expected = tuple(modality_order)
        if tuple(self.modalities.keys()) != expected:
            raise ValueError(
                f"modalities must be ordered exactly as {expected}, got {tuple(self.modalities.keys())}"
            )
        if self.action.ndim != 3:
            raise ValueError("action must have shape [B,T,A]")
        if not self.action.is_floating_point():
            raise ValueError("action must use a floating-point dtype")
        batch, timesteps, observed_action_dim = self.action.shape
        if action_dim is not None and observed_action_dim != int(action_dim):
            raise ValueError(
                f"action feature dimension must be {int(action_dim)}, got {observed_action_dim}"
            )
        reference_device = self.action.device
        for name, tensor in self.modalities.items():
            if tensor.ndim != 3 or tensor.shape[:2] != (batch, timesteps):
                raise ValueError(f"modality {name!r} must have shape [B,T,F]")
            if modality_dims is not None and tensor.shape[-1] != int(modality_dims[name]):
                raise ValueError(
                    f"modality {name!r} feature dimension must be {int(modality_dims[name])}, "
                    f"got {tensor.shape[-1]}"
                )
            if not tensor.is_floating_point():
                raise ValueError(f"modality {name!r} must use a floating-point dtype")
            if tensor.device != reference_device:
                raise ValueError(f"modality {name!r} must be on device {reference_device}")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"modality {name!r} contains non-finite values")
        for name, tensor in {
            "contact": self.contact,
            "slip": self.slip,
            "feasible": self.feasible,
        }.items():
            if tensor.shape != (batch, timesteps):
                raise ValueError(f"{name} must have shape [B,T]")
            if tensor.device != reference_device:
                raise ValueError(f"{name} must be on device {reference_device}")
            if not tensor.is_floating_point():
                raise ValueError(f"{name} must use a floating-point dtype")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{name} contains non-finite values")
            if (tensor < 0).any() or (tensor > 1).any():
                raise ValueError(f"{name} targets must be in [0,1]")
        if self.quality.shape != (batch, timesteps, len(expected), 3):
            raise ValueError("quality must have shape [B,T,M,3]")
        if self.quality.device != reference_device:
            raise ValueError(f"quality must be on device {reference_device}")
        if not self.quality.is_floating_point():
            raise ValueError("quality must use a floating-point dtype")
        if not torch.isfinite(self.quality).all():
            raise ValueError("quality contains non-finite values")
        if (self.quality < 0).any() or (self.quality > 1).any():
            raise ValueError("quality values must be in [0,1]")
        if self.phase is not None:
            if self.phase.shape != (batch, timesteps):
                raise ValueError("phase must have shape [B,T]")
            if self.phase.device != reference_device:
                raise ValueError(f"phase must be on device {reference_device}")
            if self.phase.is_floating_point():
                if not torch.equal(self.phase, self.phase.round()):
                    raise ValueError("phase values must be integer-valued")
            if (self.phase < 0).any():
                raise ValueError("phase values must be non-negative")
        return self

    @property
    def batch_size(self) -> int:
        return int(self.action.shape[0])

    @property
    def timesteps(self) -> int:
        return int(self.action.shape[1])

    def reliability(self) -> torch.Tensor:
        availability = self.quality[..., 0]
        confidence = self.quality[..., 1]
        freshness = self.quality[..., 2]
        return availability * confidence * freshness

    def clone(self) -> "VTLASequence":
        return replace(
            self,
            modalities={name: tensor.clone() for name, tensor in self.modalities.items()},
            action=self.action.clone(),
            contact=self.contact.clone(),
            slip=self.slip.clone(),
            feasible=self.feasible.clone(),
            quality=self.quality.clone(),
            phase=None if self.phase is None else self.phase.clone(),
            metadata=dict(self.metadata),
        )
