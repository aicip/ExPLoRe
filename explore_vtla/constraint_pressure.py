"""Finite-difference constraint pressure for failure-directed experiment planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class PerturbationObservation:
    parameter: str
    delta_parameter: float
    baseline_metrics: Mapping[str, float]
    perturbed_metrics: Mapping[str, float]


@dataclass(frozen=True)
class PressureResult:
    parameter: str
    sensitivity: float
    direction_to_reduce_violation: int


def normalized_violation(value: float, *, op: str, threshold: float, scale: float | None = None) -> float:
    denom = abs(scale if scale is not None else threshold)
    denom = max(denom, 1e-12)
    if op == "<=":
        return max(0.0, value - threshold) / denom
    if op == ">=":
        return max(0.0, threshold - value) / denom
    raise ValueError("op must be '<=' or '>='")


def finite_difference_pressure(
    observations: list[PerturbationObservation],
    *,
    metric: str,
    op: str,
    threshold: float,
    scale: float | None = None,
) -> list[PressureResult]:
    results: list[PressureResult] = []
    for obs in observations:
        if obs.delta_parameter == 0:
            raise ValueError("delta_parameter must be non-zero")
        if metric not in obs.baseline_metrics or metric not in obs.perturbed_metrics:
            raise ValueError(f"metric {metric!r} missing from observation")
        base = normalized_violation(obs.baseline_metrics[metric], op=op, threshold=threshold, scale=scale)
        perturbed = normalized_violation(obs.perturbed_metrics[metric], op=op, threshold=threshold, scale=scale)
        sensitivity = (perturbed - base) / obs.delta_parameter
        direction = -1 if sensitivity > 0 else (1 if sensitivity < 0 else 0)
        results.append(PressureResult(obs.parameter, float(sensitivity), direction))
    return sorted(results, key=lambda item: abs(item.sensitivity), reverse=True)
