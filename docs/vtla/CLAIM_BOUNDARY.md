# Claim Boundary

ExPLoRe-VTLA deliberately separates software verification, synthetic mechanism evidence, offline real-data results, simulation, hardware-in-loop results, real-robot results, and independent reproduction.

## Evidence ladder

- **M0**: unit and mathematical verification
- **M1**: deterministic synthetic mechanism evidence
- **M2**: offline real-dataset evidence
- **M3**: physics-simulation evidence
- **M4**: hardware-in-loop evidence
- **M5**: real-robot evidence
- **M6**: independent reproduction

The repository release included here earns **M1 only**.

## Allowed at M1

- The preserved ExPLoRe core path and VTLA extension pass the declared automated software tests in the verified local environment.
- VTLA tensor, routing, reliability, intervention, safety, evidence, and claim contracts are executable and tested.
- Loss-coupled routing can learn a deterministic synthetic modality/phase specialization problem whose preferred routing structure is known by construction.
- The same mechanism threshold can be required across multiple deterministic release seeds.
- A deterministic synthetic contact plant can satisfy its declared causal regression invariants.
- Counterfactual routing calibration can be evaluated against measured modality-drop impact on the synthetic action path.
- Coupled and detached loss routing can be compared on both their optimized objective and a separate action-RMSE outcome without assuming those two results must agree.

## Not allowed at M1

The following statements require evidence not included in this release:

- "ExPLoRe-VTLA improves real robot manipulation."
- "ExPLoRe-VTLA improves Daimon-Infinity / FreeTacMan / DROID benchmark performance."
- "ExPLoRe-VTLA is safe for deployment around people."
- "ExPLoRe-VTLA has validated sim-to-real transfer."
- "ExPLoRe-VTLA matches or exceeds an Amazon production robotics model."
- "Routing maps are causally faithful on real-world data."
- "The synthetic contact plant is physically validated."
- "The v1.0.0 VTLA path performs raw image-patch or raw tactile-taxel routing."

## Negative evidence is first-class

A failed claim gate is an output, not a documentation defect. The release evidence intentionally separates:

- whether coupled routing lowers its weighted training objective, and
- whether it consistently improves action RMSE over a detached-routing control.

If the downstream-performance threshold is not met, the `synthetic_action_superiority` claim remains blocked even when the optimization-loss result is favorable.

Evidence artifacts preserve blocked higher-authority and failed empirical claims so attractive synthetic results cannot silently be promoted into real-world claims.
