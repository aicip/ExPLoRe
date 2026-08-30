"""Evidence-authority gates that keep public claims below measured authority."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .contracts import AuthorityLevel


@dataclass(frozen=True)
class ClaimRequirement:
    claim_id: str
    statement: str
    minimum_authority: AuthorityLevel
    metric: str | None = None
    minimum_value: float | None = None


@dataclass(frozen=True)
class ClaimDecision:
    claim_id: str
    allowed: bool
    reason: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


DEFAULT_CLAIMS = (
    ClaimRequirement(
        "loss_coupling_mechanism",
        "Loss-coupled routing specializes on a deterministic synthetic mechanism benchmark.",
        AuthorityLevel.M1_SYNTHETIC_MECHANISM,
        metric="specialization_delta",
        minimum_value=0.20,
    ),
    ClaimRequirement(
        "synthetic_contact_dynamics_regression",
        "The deterministic contact-world generator satisfies its declared causal invariants.",
        AuthorityLevel.M1_SYNTHETIC_MECHANISM,
        metric="contact_regression_invariants_pass",
        minimum_value=1.0,
    ),
    ClaimRequirement(
        "replicated_loss_coupling_mechanism",
        "Loss-coupled specialization clears the declared threshold across all release seeds.",
        AuthorityLevel.M1_SYNTHETIC_MECHANISM,
        metric="minimum_replicated_specialization_delta",
        minimum_value=0.20,
    ),
    ClaimRequirement(
        "synthetic_routing_faithfulness",
        "Intervention-calibrated routing is behaviorally faithful on the deterministic synthetic benchmark.",
        AuthorityLevel.M1_SYNTHETIC_MECHANISM,
        metric="calibrated_faithfulness",
        minimum_value=0.50,
    ),
    ClaimRequirement(
        "synthetic_action_superiority",
        "Loss coupling consistently improves synthetic action prediction over detached loss routing.",
        AuthorityLevel.M1_SYNTHETIC_MECHANISM,
        metric="coupled_action_rmse_win_rate",
        minimum_value=0.75,
    ),
    ClaimRequirement(
        "offline_real_dataset_gain",
        "ExPLoRe-VTLA improves a real offline robotics benchmark.",
        AuthorityLevel.M2_OFFLINE_REAL_DATA,
    ),
    ClaimRequirement(
        "real_robot_effectiveness",
        "ExPLoRe-VTLA improves contact-rich manipulation on real robot hardware.",
        AuthorityLevel.M5_REAL_ROBOT,
    ),
)


def evaluate_claims(
    authority: AuthorityLevel,
    metrics: dict[str, float],
    claims: tuple[ClaimRequirement, ...] = DEFAULT_CLAIMS,
) -> list[ClaimDecision]:
    decisions: list[ClaimDecision] = []
    for claim in claims:
        if authority < claim.minimum_authority:
            decisions.append(
                ClaimDecision(
                    claim.claim_id,
                    False,
                    f"requires {claim.minimum_authority.name}, current authority is {authority.name}",
                )
            )
            continue
        if claim.metric is not None:
            value = metrics.get(claim.metric)
            if value is None:
                decisions.append(ClaimDecision(claim.claim_id, False, f"missing metric {claim.metric}"))
                continue
            if claim.minimum_value is not None and value < claim.minimum_value:
                decisions.append(
                    ClaimDecision(
                        claim.claim_id,
                        False,
                        f"{claim.metric}={value:.6g} below required {claim.minimum_value:.6g}",
                    )
                )
                continue
        decisions.append(ClaimDecision(claim.claim_id, True, "evidence gate satisfied"))
    return decisions
