# ExPLoRe-VTLA Architecture

## Core hypothesis

ExPLoRe demonstrated a differentiable coupling between Soft-MoE dispatch weights and per-patch multi-objective losses. ExPLoRe-VTLA tests whether the same mechanism can be generalized from static image patches to **temporal multimodal feature tokens**.

The v1.0.0 VTLA path represents a trajectory as `B x T x M` projected tokens, where `M` is an ordered set of modalities. Each modality contributes one feature vector per timestep and has its own declared raw feature dimension before projection into the shared hidden space.

This release therefore does **not** claim raw image-patch or raw tactile-taxel routing. Raw sensors require a documented encoder/preprocessing stage before entering the VTLA contract.

The reference configuration uses:

- vision
- tactile
- force
- proprioception
- language
- previous-action context
- embodiment metadata

The architecture accepts any modality dictionary whose dimensions are declared in `VTLAConfig`.

## Validated sequence contract

`VTLASequence.validate` checks the declared modality set and dimensions, action dimension, batch/time agreement, floating signal tensors, common device, finite/bounded quality channels, probability-valued contact/slip/feasibility labels, and non-negative integer phase labels when phases are present.

This makes malformed or semantically ambiguous trajectories fail before training.

## Quality side-channel

Each token carries an external `availability / confidence / freshness` triplet. Raw sensor values are multiplied by reliability before projection, while the quality triplet is embedded separately. An unavailable sensor is therefore not silently represented as a legitimate numerical zero.

Reliability is also enforced in dispatch normalization. A zero-reliability token receives zero dispatch mass for every objective.

`PredictionErrorMemory` can additionally apply a bounded EMA-derived confidence factor after the world model repeatedly disagrees with observed reality. This memory is external to the learned network: the model cannot directly rewrite its own sensor-health authority.

## Temporal encoder

All modality tokens are flattened in time-major order and passed through a causal Transformer encoder. Tokens at the same timestep can interact; tokens may attend to the past but not future timesteps.

## Loss-coupled router

The router computes normalized token and expert embeddings and then:

```text
logit[b,n,e] = scale * cosine(token[b,n], phi[e])
```

Dispatch weights are normalized across tokens for each expert/objective. Combine weights are normalized across experts for each token.

The external reliability mask participates in dispatch normalization, so invalid/unavailable observations cannot be selected by the loss router.

## Objective channels

Seven expert/objective channels are implemented:

1. reconstruction
2. cross-modal alignment
3. next-state world model
4. action prediction
5. contact prediction
6. slip prediction
7. feasibility prediction

Each objective produces a per-token loss tensor. Dispatch weights are reused as differentiable per-token loss coefficients. Valid masks are renormalized after masking so a router cannot lower an objective simply by selecting unlabeled or structurally invalid positions.

## Routed inference

For policy-relevant outputs, objective routing is not merely logged. Action, contact, slip, and feasibility predictions are aggregated using their corresponding objective-specific routing mass, normalized across modalities at each timestep.

This makes routing behaviorally relevant and therefore testable by intervention.

## Quantitative routing diagnostics

`explore_vtla.diagnostics` reports:

- normalized objective token entropy,
- expert combine utilization,
- minimum/maximum utilization,
- expert-utilization coefficient of variation,
- modality-objective mutual information and normalized MI,
- phase-objective mutual information and normalized MI when phase labels exist.

These metrics quantify concentration, collapse, and specialization. They do not by themselves establish causal importance.

## Reality reconciliation

World-model heads predict the next observation of each modality. `RealityReconciler` compares the prediction to the observed next state after scale normalization and reports mean/max error plus mismatch rate.

`PredictionErrorMemory` supplies a bounded feedback primitive that can reduce future confidence in a modality that repeatedly violates its prediction envelope without overwriting raw observations.

## Counterfactual routing calibration

Routing maps are not automatically accepted as explanations. For the action objective, ExPLoRe-VTLA can measure action-error degradation when each modality is dropped. The measured impact profile is detached and normalized, then used as a target for mean action-routing distribution.

This creates a falsifiable requirement: a modality receiving high action-routing mass should, under the same scenario, cause a comparatively large action degradation when removed.

## Deterministic contact world

`explore_vtla.contact_dynamics` implements a small deterministic normal/tangential contact plant with compliant normal force, tangential stiffness/damping, Coulomb-style friction limiting, and slip. It can emit a valid VTLA trajectory for approach, load, manipulation, slip, recovery, and release phases.

The contact world is a **software mechanism/regression plant**. It is M1 synthetic evidence and is explicitly marked `physical_validation: false`. It is not a calibrated contact solver or M3 physics validation.

## Independent safety boundary

The deterministic `IndependentSafetyGate` sits outside learned routing. It evaluates force norm, action-feasibility probability, predictive uncertainty, finite-valued signals, and action magnitude. The learned model cannot change the configured limits.

Decisions are `ALLOW`, `CLAMP`, or `HOLD`.

This gate is a software safety boundary, not a certified robotic safety controller.
