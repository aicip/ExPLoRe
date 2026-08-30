"""Deterministic synthetic end-to-end training and evaluation helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from .campaign import run_campaign
from .counterfactual import counterfactual_action_calibration_loss
from .diagnostics import routing_diagnostics
from .faithfulness import evaluate_modality_faithfulness
from .metrics import parameter_report, routing_by_modality, routing_by_phase
from .model import ExPLoReVTLA
from .reality import RealityReconciler
from .synthetic import default_synthetic_config, make_synthetic_sequence


@dataclass(frozen=True)
class SmokeTrainResult:
    initial_loss: float
    final_loss: float
    loss_reduction_fraction: float
    parameter_report: dict[str, int]
    routing_by_modality: list[list[float]]
    routing_by_phase: list[list[float]]
    reality_mean_error: dict[str, float]
    campaign: list[dict[str, object]]
    faithfulness: dict[str, object]
    routing_diagnostics: dict[str, object]
    counterfactual_weight: float
    counterfactual_interval: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FaithfulnessAblationResult:
    uncalibrated_rank_correlation: float
    calibrated_rank_correlation: float
    improvement: float
    uncalibrated_loss_reduction: float
    calibrated_loss_reduction: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def run_faithfulness_ablation(seed: int = 23, steps: int = 100) -> FaithfulnessAblationResult:
    _, uncalibrated = run_smoke_training(
        seed=seed, steps=steps, counterfactual_weight=0.0
    )
    _, calibrated = run_smoke_training(
        seed=seed, steps=steps, counterfactual_weight=1.0
    )
    uncorr = float(uncalibrated.faithfulness["rank_correlation"])
    corr = float(calibrated.faithfulness["rank_correlation"])
    return FaithfulnessAblationResult(
        uncalibrated_rank_correlation=uncorr,
        calibrated_rank_correlation=corr,
        improvement=corr - uncorr,
        uncalibrated_loss_reduction=uncalibrated.loss_reduction_fraction,
        calibrated_loss_reduction=calibrated.loss_reduction_fraction,
    )


def run_smoke_training(
    seed: int = 23,
    steps: int = 60,
    *,
    counterfactual_weight: float = 1.0,
    counterfactual_interval: int = 5,
) -> tuple[ExPLoReVTLA, SmokeTrainResult]:
    torch.manual_seed(seed)
    config = default_synthetic_config(hidden_dim=32)
    sequence = make_synthetic_sequence(config, batch_size=12, cycles=2, seed=seed)
    model = ExPLoReVTLA(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    model.train()
    initial = None
    final = None
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        output = model(sequence)
        loss, _ = model.training_loss(sequence, output)
        if counterfactual_weight > 0 and step > 0 and step % counterfactual_interval == 0:
            cf_loss, _ = counterfactual_action_calibration_loss(model, sequence, output)
            loss = loss + counterfactual_weight * cf_loss
        if initial is None:
            initial = float(loss.detach().item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        final = float(loss.detach().item())
    assert initial is not None and final is not None

    model.eval()
    with torch.no_grad():
        output = model(sequence)
        mod = routing_by_modality(output.dispatch).cpu().tolist()
        phase = routing_by_phase(output.dispatch, sequence.phase, 6).cpu().tolist()  # type: ignore[arg-type]
        reality = RealityReconciler(threshold=1.0).report(config, sequence, output)
        diagnostics = routing_diagnostics(output.dispatch, output.combine, sequence.phase)
    campaign = [item.to_dict() for item in run_campaign(model, sequence)]
    faithfulness = evaluate_modality_faithfulness(model, sequence, objective="action").to_dict()
    result = SmokeTrainResult(
        initial_loss=initial,
        final_loss=final,
        loss_reduction_fraction=(initial - final) / max(initial, 1e-12),
        parameter_report=parameter_report(model),
        routing_by_modality=mod,
        routing_by_phase=phase,
        reality_mean_error=reality.mean_error_by_modality,
        campaign=campaign,
        faithfulness=faithfulness,
        routing_diagnostics=diagnostics.to_dict(),
        counterfactual_weight=float(counterfactual_weight),
        counterfactual_interval=int(counterfactual_interval),
    )
    return model, result


@dataclass(frozen=True)
class LossCouplingTaskAblationResult:
    seeds: tuple[int, ...]
    steps: int
    coupled_final_loss: tuple[float, ...]
    detached_final_loss: tuple[float, ...]
    coupled_action_rmse: tuple[float, ...]
    detached_action_rmse: tuple[float, ...]
    coupled_final_loss_win_rate: float
    coupled_action_rmse_win_rate: float
    median_detached_minus_coupled_final_loss: float
    median_detached_minus_coupled_action_rmse: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return float(0.5 * (ordered[middle - 1] + ordered[middle]))


def _train_detach_ablation_once(seed: int, steps: int, detach_router: bool) -> tuple[float, float]:
    torch.manual_seed(seed)
    config = default_synthetic_config(hidden_dim=32)
    sequence = make_synthetic_sequence(config, batch_size=12, cycles=2, seed=seed)
    model = ExPLoReVTLA(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    final_loss = 0.0
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        output = model(sequence)
        loss, _ = model.training_loss(sequence, output, detach_router=detach_router)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        final_loss = float(loss.detach().item())
    model.eval()
    with torch.no_grad():
        output = model(sequence)
        action_rmse = torch.sqrt(
            torch.mean((model.aggregate_action(output) - sequence.action) ** 2)
        )
    return final_loss, float(action_rmse.item())


def run_loss_coupling_task_ablation(
    seeds: tuple[int, ...] = (41, 43, 45, 47),
    *,
    steps: int = 60,
) -> LossCouplingTaskAblationResult:
    """Compare coupled and detached loss routing from identical initializations.

    This ablation intentionally distinguishes optimization-loss behavior from
    downstream action error. A routing method does not earn a task-performance
    claim merely because it can lower its own weighted training objective.
    """

    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("ablation requires at least two unique seeds")
    if steps <= 0:
        raise ValueError("steps must be positive")
    coupled_loss: list[float] = []
    detached_loss: list[float] = []
    coupled_rmse: list[float] = []
    detached_rmse: list[float] = []
    for seed in seeds:
        c_loss, c_rmse = _train_detach_ablation_once(int(seed), steps, False)
        d_loss, d_rmse = _train_detach_ablation_once(int(seed), steps, True)
        coupled_loss.append(c_loss)
        detached_loss.append(d_loss)
        coupled_rmse.append(c_rmse)
        detached_rmse.append(d_rmse)

    loss_deltas = [d - c for c, d in zip(coupled_loss, detached_loss)]
    rmse_deltas = [d - c for c, d in zip(coupled_rmse, detached_rmse)]
    return LossCouplingTaskAblationResult(
        seeds=tuple(int(seed) for seed in seeds),
        steps=int(steps),
        coupled_final_loss=tuple(coupled_loss),
        detached_final_loss=tuple(detached_loss),
        coupled_action_rmse=tuple(coupled_rmse),
        detached_action_rmse=tuple(detached_rmse),
        coupled_final_loss_win_rate=sum(delta > 0 for delta in loss_deltas) / len(loss_deltas),
        coupled_action_rmse_win_rate=sum(delta > 0 for delta in rmse_deltas) / len(rmse_deltas),
        median_detached_minus_coupled_final_loss=_median(loss_deltas),
        median_detached_minus_coupled_action_rmse=_median(rmse_deltas),
    )
