"""Constraint-aware candidate selection that refuses infeasible high-score winners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class Constraint:
    metric: str
    op: str
    threshold: float

    def passes(self, metrics: Mapping[str, float]) -> bool:
        if self.metric not in metrics:
            return False
        value = float(metrics[self.metric])
        if self.op == "<=":
            return value <= self.threshold
        if self.op == ">=":
            return value >= self.threshold
        raise ValueError("constraint op must be '<=' or '>='")


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    score: float
    metrics: Mapping[str, float]


@dataclass(frozen=True)
class Selection:
    winner: Candidate | None
    rejected: tuple[tuple[str, tuple[str, ...]], ...]


def select_candidate(candidates: list[Candidate], constraints: tuple[Constraint, ...]) -> Selection:
    feasible: list[Candidate] = []
    rejected: list[tuple[str, tuple[str, ...]]] = []
    for candidate in candidates:
        failures = tuple(
            f"{constraint.metric}{constraint.op}{constraint.threshold}"
            for constraint in constraints
            if not constraint.passes(candidate.metrics)
        )
        if failures:
            rejected.append((candidate.candidate_id, failures))
        else:
            feasible.append(candidate)
    winner = max(feasible, key=lambda candidate: candidate.score) if feasible else None
    return Selection(winner=winner, rejected=tuple(rejected))
