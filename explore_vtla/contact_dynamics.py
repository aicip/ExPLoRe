"""Deterministic contact dynamics for VTLA mechanism and regression tests.

The model is intentionally small: a point end-effector approaches a compliant
surface in the normal direction and can slide tangentially against a Coulomb
friction limit. It is not a robot or contact-mechanics validator. Its purpose is
to provide an executable causal process in which contact, force, and slip emerge
from state evolution instead of being assigned as arbitrary labels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from .contracts import VTLAConfig, VTLASequence
from .synthetic import default_synthetic_config


@dataclass(frozen=True)
class ContactParameters:
    surface_position_m: float = 0.04
    normal_stiffness_n_m: float = 800.0
    normal_damping_ns_m: float = 12.0
    tangential_stiffness_n_m: float = 180.0
    tangential_damping_ns_m: float = 3.0
    friction_coefficient: float = 0.45
    mass_kg: float = 1.0
    velocity_servo_gain_s: float = 25.0

    def __post_init__(self) -> None:
        positive = (
            self.normal_stiffness_n_m,
            self.normal_damping_ns_m,
            self.tangential_stiffness_n_m,
            self.tangential_damping_ns_m,
            self.mass_kg,
            self.velocity_servo_gain_s,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("contact stiffness, damping, mass, and servo gain must be positive")
        if self.friction_coefficient < 0:
            raise ValueError("friction_coefficient must be non-negative")


@dataclass(frozen=True)
class ContactState:
    time_s: float
    normal_position_m: float
    normal_velocity_m_s: float
    tangential_position_m: float
    tangential_velocity_m_s: float
    penetration_m: float
    normal_force_n: float
    tangential_force_n: float
    contact: bool
    slip: bool


class ContactWorld:
    """Small deterministic compliant-contact plant with a friction limit."""

    def __init__(self, parameters: ContactParameters | None = None) -> None:
        self.parameters = parameters or ContactParameters()
        self.reset()

    def reset(self) -> ContactState:
        self._time_s = 0.0
        self._normal_position_m = 0.0
        self._normal_velocity_m_s = 0.0
        self._tangential_position_m = 0.0
        self._tangential_velocity_m_s = 0.0
        self._tangential_spring_m = 0.0
        self._slip = False
        return self.state

    @property
    def state(self) -> ContactState:
        p = self.parameters
        penetration = max(0.0, self._normal_position_m - p.surface_position_m)
        normal_force = 0.0
        tangential_force = 0.0
        slip = False
        if penetration > 0.0:
            normal_force = (
                p.normal_stiffness_n_m * penetration
                + p.normal_damping_ns_m * max(0.0, self._normal_velocity_m_s)
            )
            trial = (
                p.tangential_stiffness_n_m * self._tangential_spring_m
                + p.tangential_damping_ns_m * self._tangential_velocity_m_s
            )
            limit = p.friction_coefficient * normal_force
            tangential_force = max(-limit, min(limit, trial))
            slip = self._slip
        return ContactState(
            time_s=self._time_s,
            normal_position_m=self._normal_position_m,
            normal_velocity_m_s=self._normal_velocity_m_s,
            tangential_position_m=self._tangential_position_m,
            tangential_velocity_m_s=self._tangential_velocity_m_s,
            penetration_m=penetration,
            normal_force_n=max(0.0, normal_force),
            tangential_force_n=tangential_force,
            contact=penetration > 0.0,
            slip=slip,
        )

    def step(
        self,
        *,
        normal_velocity_command_m_s: float,
        tangential_velocity_command_m_s: float,
        dt_s: float = 0.005,
    ) -> ContactState:
        if dt_s <= 0:
            raise ValueError("dt_s must be positive")
        p = self.parameters
        dt = float(dt_s)
        normal_accel = (
            float(normal_velocity_command_m_s) - self._normal_velocity_m_s
        ) * p.velocity_servo_gain_s
        tangent_accel = (
            float(tangential_velocity_command_m_s) - self._tangential_velocity_m_s
        ) * p.velocity_servo_gain_s
        self._normal_velocity_m_s += normal_accel * dt
        self._tangential_velocity_m_s += tangent_accel * dt
        self._normal_position_m += self._normal_velocity_m_s * dt
        self._tangential_position_m += self._tangential_velocity_m_s * dt

        current = self.state
        if current.contact:
            self._tangential_spring_m += self._tangential_velocity_m_s * dt
            trial = (
                p.tangential_stiffness_n_m * self._tangential_spring_m
                + p.tangential_damping_ns_m * self._tangential_velocity_m_s
            )
            limit = p.friction_coefficient * current.normal_force_n
            self._slip = abs(trial) > limit + 1e-12
            if self._slip and p.tangential_stiffness_n_m > 0:
                sign = 1.0 if trial >= 0 else -1.0
                self._tangential_spring_m = (
                    sign * limit - p.tangential_damping_ns_m * self._tangential_velocity_m_s
                ) / p.tangential_stiffness_n_m
        else:
            self._tangential_spring_m = 0.0
            self._slip = False

        self._time_s += dt
        return self.state


def _command_for_phase(phase: int) -> tuple[float, float]:
    if phase == 0:  # approach
        return 0.20, 0.0
    if phase == 1:  # load
        return 0.04, 0.0
    if phase == 2:  # tangential manipulation
        return 0.0, 0.12
    if phase == 3:  # stronger slide, should induce slip
        return 0.0, 0.35
    if phase == 4:  # recovery
        return -0.04, -0.08
    return -0.20, 0.0  # release


def simulate_contact_episode(
    *,
    steps_per_phase: int = 24,
    dt_s: float = 0.01,
    parameters: ContactParameters | None = None,
) -> tuple[tuple[ContactState, ...], tuple[tuple[float, float], ...], tuple[int, ...]]:
    if steps_per_phase <= 1:
        raise ValueError("steps_per_phase must be greater than one")
    world = ContactWorld(parameters)
    states: list[ContactState] = []
    commands: list[tuple[float, float]] = []
    phases: list[int] = []
    for phase in range(6):
        command = _command_for_phase(phase)
        for _ in range(steps_per_phase):
            state = world.step(
                normal_velocity_command_m_s=command[0],
                tangential_velocity_command_m_s=command[1],
                dt_s=dt_s,
            )
            states.append(state)
            commands.append(command)
            phases.append(phase)
    return tuple(states), tuple(commands), tuple(phases)


def make_contact_world_sequence(
    config: VTLAConfig | None = None,
    *,
    steps_per_phase: int = 24,
    dt_s: float = 0.01,
    parameters: ContactParameters | None = None,
) -> VTLASequence:
    """Convert the deterministic contact plant into the standard VTLA contract."""

    config = config or default_synthetic_config()
    required_dims = {
        "vision": 6,
        "tactile": 6,
        "force": 4,
        "proprio": 5,
        "language": 5,
        "action_context": 3,
        "embodiment": 4,
    }
    if dict(config.modality_dims) != required_dims or config.action_dim != 3:
        raise ValueError("contact-world adapter requires the default synthetic VTLA dimensions")
    params = parameters or ContactParameters()
    states, commands, phases = simulate_contact_episode(
        steps_per_phase=steps_per_phase,
        dt_s=dt_s,
        parameters=params,
    )
    t = len(states)
    modalities = {name: torch.zeros(1, t, dim) for name, dim in config.modality_dims.items()}
    action = torch.zeros(1, t, config.action_dim)
    contact = torch.zeros(1, t)
    slip = torch.zeros(1, t)
    feasible = torch.ones(1, t)
    phase_tensor = torch.tensor(phases, dtype=torch.long).view(1, -1)

    previous_action = torch.zeros(3)
    for idx, (state, command, phase) in enumerate(zip(states, commands, phases)):
        distance = params.surface_position_m - state.normal_position_m
        modalities["vision"][0, idx] = torch.tensor(
            [distance, state.normal_position_m, state.tangential_position_m, float(phase) / 5.0, 1.0, 0.0]
        )
        area_proxy = min(1.0, state.penetration_m * 200.0)
        modalities["tactile"][0, idx] = torch.tensor(
            [
                state.normal_force_n,
                state.tangential_force_n,
                float(state.contact),
                float(state.slip),
                state.penetration_m,
                area_proxy,
            ]
        )
        modalities["force"][0, idx] = torch.tensor(
            [state.normal_force_n, state.tangential_force_n, abs(state.tangential_force_n), float(state.slip)]
        )
        modalities["proprio"][0, idx] = torch.tensor(
            [
                state.normal_position_m,
                state.normal_velocity_m_s,
                state.tangential_position_m,
                state.tangential_velocity_m_s,
                state.time_s,
            ]
        )
        modalities["language"][0, idx] = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0])
        modalities["action_context"][0, idx] = previous_action
        modalities["embodiment"][0, idx] = torch.tensor(
            [
                params.mass_kg,
                params.normal_stiffness_n_m / 1000.0,
                params.friction_coefficient,
                params.tangential_stiffness_n_m / 1000.0,
            ]
        )
        current_action = torch.tensor([command[0], command[1], 1.0 if phase <= 3 else -1.0])
        action[0, idx] = current_action
        previous_action = current_action
        contact[0, idx] = float(state.contact)
        slip[0, idx] = float(state.slip)
        feasible[0, idx] = 0.0 if state.normal_force_n > 10.0 else 1.0

    quality = torch.ones(1, t, len(config.modality_order), 3)
    return VTLASequence(
        modalities=modalities,
        action=action,
        contact=contact,
        slip=slip,
        feasible=feasible,
        quality=quality,
        phase=phase_tensor,
        metadata={
            "authority": "M1_SYNTHETIC_MECHANISM",
            "generator": "deterministic_compliant_contact_world_v1",
            "parameters": asdict(params),
            "dt_s": dt_s,
        },
    ).validate(
        config.modality_order,
        modality_dims=config.modality_dims,
        action_dim=config.action_dim,
    )


def contact_regression_report() -> dict[str, object]:
    states, _, phases = simulate_contact_episode()
    pre_contact_forces = [state.normal_force_n for state in states if not state.contact]
    contact_forces = [state.normal_force_n for state in states if state.contact]
    slip_count = sum(state.slip for state in states)
    first_contact = next((idx for idx, state in enumerate(states) if state.contact), None)
    return {
        "authority": "M1_SYNTHETIC_MECHANISM",
        "model": "deterministic_compliant_contact_world_v1",
        "steps": len(states),
        "phase_count": len(set(phases)),
        "first_contact_step": first_contact,
        "max_pre_contact_force_n": max(pre_contact_forces, default=0.0),
        "max_contact_force_n": max(contact_forces, default=0.0),
        "slip_steps": int(slip_count),
        "invariants": {
            "zero_force_before_contact": max(pre_contact_forces, default=0.0) <= 1e-12,
            "positive_force_after_contact": max(contact_forces, default=0.0) > 0.0,
            "slip_emerges": slip_count > 0,
        },
        "physical_validation": False,
    }
