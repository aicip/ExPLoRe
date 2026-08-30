"""CPU-only ExPLoRe-VTLA quickstart using deterministic synthetic data."""

from __future__ import annotations

import torch

from explore_vtla.model import ExPLoReVTLA
from explore_vtla.safety import IndependentSafetyGate, SafetyEnvelope
from explore_vtla.synthetic import default_synthetic_config, make_synthetic_sequence


def main() -> None:
    torch.manual_seed(7)
    config = default_synthetic_config(hidden_dim=32)
    sequence = make_synthetic_sequence(config, batch_size=2, cycles=1, seed=7)
    model = ExPLoReVTLA(config)
    model.eval()
    with torch.no_grad():
        output = model(sequence)
        action = model.aggregate_action(output)
        feasibility = model.aggregate_probability(output, "feasibility")

    gate = IndependentSafetyGate(SafetyEnvelope())
    result = gate.evaluate(
        action[0, -1],
        force=sequence.modalities["force"][0, -1],
        feasibility_probability=float(feasibility[0, -1]),
        action_uncertainty=0.0,
    )
    print(
        {
            "action": action[0, -1].detach().tolist(),
            "feasibility_probability": float(feasibility[0, -1]),
            "safety_decision": result.decision.value,
            "safety_reasons": result.reasons,
            "authority": "M1 synthetic quickstart only",
        }
    )


if __name__ == "__main__":
    main()
