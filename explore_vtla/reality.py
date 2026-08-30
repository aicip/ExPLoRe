"""Prediction-versus-observation reconciliation for world-model outputs."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .contracts import VTLAConfig, VTLASequence
from .model import VTLAOutput


@dataclass(frozen=True)
class RealityReport:
    mean_error_by_modality: dict[str, float]
    max_error_by_modality: dict[str, float]
    mismatch_rate_by_modality: dict[str, float]


class RealityReconciler:
    def __init__(self, threshold: float = 1.0, eps: float = 1e-6) -> None:
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        self.threshold = float(threshold)
        self.eps = float(eps)

    def per_step_error(
        self,
        config: VTLAConfig,
        sequence: VTLASequence,
        output: VTLAOutput,
    ) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {}
        for name in config.modality_order:
            target = sequence.modalities[name]
            pred = output.next_prediction[name]
            scale = target.std(dim=(0, 1), unbiased=False).clamp_min(self.eps)
            error = torch.zeros(target.shape[:2], device=target.device, dtype=target.dtype)
            if target.shape[1] > 1:
                normalized = (pred[:, :-1] - target[:, 1:]) / scale
                error[:, :-1] = torch.sqrt(torch.mean(normalized**2, dim=-1))
            result[name] = error
        return result

    def report(
        self,
        config: VTLAConfig,
        sequence: VTLASequence,
        output: VTLAOutput,
    ) -> RealityReport:
        errors = self.per_step_error(config, sequence, output)
        means: dict[str, float] = {}
        maxima: dict[str, float] = {}
        rates: dict[str, float] = {}
        for name, error in errors.items():
            usable = error[:, :-1] if error.shape[1] > 1 else error
            means[name] = float(usable.mean().item())
            maxima[name] = float(usable.max().item()) if usable.numel() else 0.0
            rates[name] = float((usable > self.threshold).float().mean().item()) if usable.numel() else 0.0
        return RealityReport(means, maxima, rates)


class PredictionErrorMemory:
    """EMA memory that can down-weight repeatedly surprising modalities."""

    def __init__(self, modalities: tuple[str, ...], decay: float = 0.9, gain: float = 0.5) -> None:
        if not 0 <= decay < 1 or gain < 0:
            raise ValueError("invalid memory configuration")
        self.modalities = modalities
        self.decay = float(decay)
        self.gain = float(gain)
        self.ema = {name: 0.0 for name in modalities}

    def update(self, errors: dict[str, float]) -> dict[str, float]:
        factors: dict[str, float] = {}
        for name in self.modalities:
            value = float(errors[name])
            self.ema[name] = self.decay * self.ema[name] + (1 - self.decay) * value
            factors[name] = float(torch.exp(torch.tensor(-self.gain * self.ema[name])).item())
        return factors

    def apply_to_sequence(
        self,
        sequence: VTLASequence,
        errors: dict[str, float],
    ) -> VTLASequence:
        """Return a clone whose confidence reflects accumulated prediction error.

        The learned model never writes these factors itself. They are computed by
        this external memory and applied through the explicit quality side-channel.
        """

        factors = self.update(errors)
        result = sequence.clone()
        for idx, name in enumerate(self.modalities):
            if name not in result.modalities:
                raise ValueError(f"sequence is missing modality {name!r}")
            result.quality[..., idx, 1] *= factors[name]
        result.metadata["prediction_error_reliability_factors"] = dict(factors)
        return result
