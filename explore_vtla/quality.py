"""Signal health, freshness, reliability, and embedding-drift utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .contracts import SignalHealth

_HEALTH_FACTOR = {
    SignalHealth.NOMINAL: 1.0,
    SignalHealth.DEGRADED: 0.55,
    SignalHealth.INVALID: 0.0,
    SignalHealth.UNAVAILABLE: 0.0,
}


@dataclass(frozen=True)
class SignalQuality:
    health: SignalHealth
    confidence: float
    age_ms: float
    max_age_ms: float

    def reliability(self) -> float:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0,1]")
        if self.age_ms < 0 or self.max_age_ms <= 0:
            raise ValueError("age_ms must be non-negative and max_age_ms positive")
        freshness = math.exp(-self.age_ms / self.max_age_ms)
        return float(_HEALTH_FACTOR[self.health] * self.confidence * freshness)

    def tensor_features(self) -> tuple[float, float, float]:
        # Reuse the same validation contract as reliability().
        self.reliability()
        available = 0.0 if self.health in {SignalHealth.INVALID, SignalHealth.UNAVAILABLE} else 1.0
        freshness = math.exp(-self.age_ms / self.max_age_ms)
        return available, float(self.confidence), float(freshness)


class EmbeddingDriftMonitor:
    """Reference-centroid cosine drift monitor with an explicit warmup phase."""

    def __init__(self, dim: int, warmup: int = 8, threshold: float = 0.25) -> None:
        if dim <= 0 or warmup <= 0 or not 0 <= threshold <= 2:
            raise ValueError("invalid drift monitor configuration")
        self.dim = int(dim)
        self.warmup = int(warmup)
        self.threshold = float(threshold)
        self._count = 0
        self._sum = torch.zeros(dim, dtype=torch.float64)

    @property
    def ready(self) -> bool:
        return self._count >= self.warmup

    @property
    def reference(self) -> torch.Tensor:
        if self._count == 0:
            raise RuntimeError("drift monitor has no reference observations")
        return (self._sum / self._count).to(torch.float32)

    def observe_reference(self, x: torch.Tensor) -> None:
        flat = x.detach().reshape(-1, self.dim).to(torch.float64)
        self._sum += flat.sum(dim=0).cpu()
        self._count += int(flat.shape[0])

    def score(self, x: torch.Tensor) -> torch.Tensor:
        if not self.ready:
            raise RuntimeError("drift monitor is not ready")
        ref = self.reference.to(x.device, x.dtype)
        flat = x.reshape(-1, self.dim)
        similarity = torch.nn.functional.cosine_similarity(flat, ref.expand_as(flat), dim=-1)
        return (1.0 - similarity).reshape(x.shape[:-1])

    def is_drifted(self, x: torch.Tensor) -> torch.Tensor:
        return self.score(x) > self.threshold
