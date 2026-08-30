"""Deterministic synthetic trajectory and routing-mechanism benchmarks."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .contracts import VTLAConfig, VTLASequence
from .router import LossCoupledRouter, loss_coupled_reduce

PHASES = ("approach", "contact", "load", "slip", "recovery", "release")


def default_synthetic_config(hidden_dim: int = 48) -> VTLAConfig:
    return VTLAConfig(
        modality_dims={
            "vision": 6,
            "tactile": 6,
            "force": 4,
            "proprio": 5,
            "language": 5,
            "action_context": 3,
            "embodiment": 4,
        },
        action_dim=3,
        hidden_dim=hidden_dim,
        num_heads=4,
        transformer_layers=1,
        max_timesteps=32,
        dropout=0.0,
    )


def make_synthetic_sequence(
    config: VTLAConfig | None = None,
    *,
    batch_size: int = 16,
    cycles: int = 2,
    seed: int = 7,
) -> VTLASequence:
    config = config or default_synthetic_config()
    generator = torch.Generator().manual_seed(seed)
    timesteps = len(PHASES) * cycles
    phase = torch.arange(timesteps).remainder(len(PHASES)).repeat(batch_size, 1)
    phase_f = phase.to(torch.float32) / (len(PHASES) - 1)
    object_code = torch.randn(batch_size, 1, 2, generator=generator).expand(-1, timesteps, -1)

    modalities: dict[str, torch.Tensor] = {}
    for name, dim in config.modality_dims.items():
        x = 0.04 * torch.randn(batch_size, timesteps, dim, generator=generator)
        if name == "vision":
            x[..., 0] += phase_f
            x[..., 1:3] += object_code
            x[..., 3] += (phase == 0).float() + (phase == 5).float()
        elif name == "tactile":
            contact = ((phase >= 1) & (phase <= 4)).float()
            x[..., 0] += 2.0 * contact
            x[..., 1] += 3.0 * (phase == 3).float()
            x[..., 2] += 1.5 * (phase == 2).float()
        elif name == "force":
            x[..., 0] += 2.5 * (phase == 2).float()
            x[..., 1] += 3.5 * (phase == 3).float()
            x[..., 2] += 1.5 * (phase == 4).float()
        elif name == "proprio":
            x[..., 0] += torch.sin(phase_f * torch.pi)
            x[..., 1] += (phase == 4).float()
        elif name == "language":
            x[..., 0] += 1.0
            x[..., 1] += object_code[..., 0]
        elif name == "action_context":
            base_action = torch.stack(
                [
                    torch.cos(phase_f * torch.pi),
                    0.5 * torch.sin(phase_f * torch.pi),
                    torch.where(phase == 3, torch.full_like(phase_f, -0.75), 0.25 * phase_f),
                ],
                dim=-1,
            )
            x[:, 1:, :] += base_action[:, :-1, :]
        elif name == "embodiment":
            embodiment_id = (torch.arange(batch_size) % 2).view(-1, 1)
            x[..., 0] += (embodiment_id == 0).float()
            x[..., 1] += (embodiment_id == 1).float()
            x[..., 2] += 0.5 + 0.1 * embodiment_id.float()
            x[..., 3] += 1.0 - 0.1 * embodiment_id.float()
        modalities[name] = x

    contact = ((phase >= 1) & (phase <= 4)).float()
    slip = (phase == 3).float()
    feasible = torch.ones_like(contact)
    feasible = torch.where(phase == 3, torch.full_like(feasible, 0.0), feasible)
    action = torch.stack(
        [
            torch.cos(phase_f * torch.pi),
            0.5 * torch.sin(phase_f * torch.pi),
            torch.where(phase == 3, torch.full_like(phase_f, -0.75), 0.25 * phase_f),
        ],
        dim=-1,
    )
    quality = torch.ones(batch_size, timesteps, len(config.modality_dims), 3)
    return VTLASequence(
        modalities=modalities,
        action=action,
        contact=contact,
        slip=slip,
        feasible=feasible,
        quality=quality,
        phase=phase,
        metadata={"authority": "M1_SYNTHETIC_MECHANISM", "seed": seed},
    ).validate(config.modality_order)


@dataclass(frozen=True)
class MechanismBenchmarkResult:
    initial_loss: float
    final_loss: float
    specialization_score: float
    detached_specialization_score: float
    target_mass: tuple[float, ...]


def _mechanism_problem(seed: int = 11) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    generator = torch.Generator().manual_seed(seed)
    batch, phases, modalities, experts = 8, 6, 5, 4
    dim = phases + modalities
    phase_ids = torch.arange(phases).view(phases, 1).expand(phases, modalities).reshape(-1)
    modality_ids = torch.arange(modalities).view(1, modalities).expand(phases, modalities).reshape(-1)
    features = torch.zeros(batch, phases * modalities, dim)
    features[..., :phases] = torch.nn.functional.one_hot(phase_ids, phases).float()
    features[..., phases:] = torch.nn.functional.one_hot(modality_ids, modalities).float()
    features += 0.005 * torch.randn(features.shape, generator=generator)

    # Each objective has a physically interpretable low-loss routing target.
    # objective 0: vision on approach/release; 1: tactile on contact/slip;
    # 2: force on load/slip; 3: proprio on recovery.
    preferred = {
        0: {(0, 0), (5, 0)},
        1: {(1, 1), (3, 1)},
        2: {(2, 2), (3, 2)},
        3: {(4, 3)},
    }
    losses = torch.full((batch, phases * modalities, experts), 1.5)
    target_mask = torch.zeros_like(losses)
    for token, (p, m) in enumerate(zip(phase_ids.tolist(), modality_ids.tolist())):
        for expert in range(experts):
            if (p, m) in preferred[expert]:
                losses[:, token, expert] = 0.05
                target_mask[:, token, expert] = 1.0
            elif any(pm[1] == m for pm in preferred[expert]):
                losses[:, token, expert] = 0.65
    losses += 0.01 * torch.rand(losses.shape, generator=generator)
    return features, losses, target_mask, modalities, experts


def _specialization(dispatch: torch.Tensor, target_mask: torch.Tensor) -> float:
    # Target mask can include multiple preferred tokens per expert.
    mass = (dispatch * target_mask).sum(dim=1).mean(dim=0)
    return float(mass.mean().item())


def run_mechanism_benchmark(seed: int = 11, steps: int = 160) -> MechanismBenchmarkResult:
    torch.manual_seed(seed)
    features, losses, target_mask, _, experts = _mechanism_problem(seed)
    router = LossCoupledRouter(features.shape[-1], experts, hidden_mult=1.0, dropout=0.0)
    optimizer = torch.optim.Adam([router.phi, router.scale], lr=0.08)

    with torch.no_grad():
        _, initial_weights = router(features, torch.ones(features.shape[:2]))
        initial_loss, _ = loss_coupled_reduce(losses, initial_weights["dispatch"])

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        _, weights = router(features, torch.ones(features.shape[:2]))
        loss, _ = loss_coupled_reduce(losses, weights["dispatch"])
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        _, final_weights = router(features, torch.ones(features.shape[:2]))
        final_loss, _ = loss_coupled_reduce(losses, final_weights["dispatch"])
        specialization = _specialization(final_weights["dispatch"], target_mask)

    torch.manual_seed(seed)
    detached = LossCoupledRouter(features.shape[-1], experts, hidden_mult=1.0, dropout=0.0)
    with torch.no_grad():
        _, detached_weights = detached(features, torch.ones(features.shape[:2]))
        detached_specialization = _specialization(detached_weights["dispatch"], target_mask)

    target_mass = tuple(
        float((final_weights["dispatch"] * target_mask).sum(dim=1).mean(dim=0)[idx].item())
        for idx in range(experts)
    )
    return MechanismBenchmarkResult(
        initial_loss=float(initial_loss.item()),
        final_loss=float(final_loss.item()),
        specialization_score=specialization,
        detached_specialization_score=detached_specialization,
        target_mass=target_mass,
    )


@dataclass(frozen=True)
class MechanismReplicationResult:
    seeds: tuple[int, ...]
    specialization_deltas: tuple[float, ...]
    final_to_initial_loss_ratios: tuple[float, ...]
    minimum_specialization_delta: float
    median_specialization_delta: float
    maximum_specialization_delta: float
    all_pass: bool


def run_mechanism_replication(
    seeds: tuple[int, ...] = (11, 13, 17, 19),
    *,
    steps: int = 120,
    minimum_delta: float = 0.20,
) -> MechanismReplicationResult:
    """Repeat the known-mechanism benchmark across deterministic seeds.

    This is deliberately stronger than selecting a single favorable run.  A
    release gate can require every declared seed to clear the same specialization
    threshold while also recording the achieved loss reduction.
    """

    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("replication requires at least two unique seeds")
    if steps <= 0 or minimum_delta < 0:
        raise ValueError("steps must be positive and minimum_delta non-negative")
    deltas: list[float] = []
    ratios: list[float] = []
    for seed in seeds:
        result = run_mechanism_benchmark(seed=int(seed), steps=steps)
        delta = result.specialization_score - result.detached_specialization_score
        deltas.append(float(delta))
        ratios.append(float(result.final_loss / max(result.initial_loss, 1e-12)))
    ordered = sorted(deltas)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        median = ordered[middle]
    else:
        median = 0.5 * (ordered[middle - 1] + ordered[middle])
    return MechanismReplicationResult(
        seeds=tuple(int(seed) for seed in seeds),
        specialization_deltas=tuple(deltas),
        final_to_initial_loss_ratios=tuple(ratios),
        minimum_specialization_delta=min(deltas),
        median_specialization_delta=float(median),
        maximum_specialization_delta=max(deltas),
        all_pass=all(delta >= minimum_delta for delta in deltas),
    )
