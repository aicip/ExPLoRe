# ExPLoRe-VTLA

**Reality-coupled loss routing for vision-tactile-language-action learning.**

ExPLoRe-VTLA is an Apache-2.0 research fork of [aicip/ExPLoRe](https://github.com/aicip/ExPLoRe). It preserves the original ExPLoRe masked-image-modeling implementation and extends its central mechanism, differentiable Soft-MoE dispatch weights reused as per-token loss coefficients, into temporally structured multimodal robot-learning experiments.

The extension is verification-first. A routing heatmap is **not** treated as an explanation merely because it looks plausible. The repository includes deterministic mechanism tests, sensor reliability masks, intervention-based routing-faithfulness checks, counterfactual calibration, world-model reconciliation, a deterministic contact-world regression, independent safety gating, failure campaigns, evidence hashing, a whole-release manifest, and explicit claim-authority limits.

## What changed from ExPLoRe

Original ExPLoRe asks:

> Which objective should matter for each image patch?

ExPLoRe-VTLA tests a related embodied-learning question:

> Which objective and modality should matter at this time in a physical interaction, given vision, touch, force, proprioception, language, action context, embodiment, sensor reliability, and the model's recent prediction error?

The new path is additive. Original ExPLoRe code remains under `src/`; VTLA-specific code lives under `explore_vtla/`.

### Important granularity boundary

The v1.0.0 VTLA path operates on **precomputed per-modality feature vectors**, one projected token per modality per timestep. It does not claim raw camera-patch, raw taxel-patch, or production VLA tokenization. Raw images, tactile arrays, and other sensor streams must first be converted into documented features or tokens through an external preprocessing stage. This boundary is intentional and keeps the current release's evidence at the level actually exercised by its tests.

## Implemented VTLA capabilities

- **Temporal multimodal Soft-MoE loss routing** across trajectory tokens.
- **Reliability-aware dispatch**: unavailable inputs cannot receive loss-routing mass, and confidence/freshness are explicit instead of being encoded as fake zero measurements.
- **Strict trajectory contracts** for declared modality dimensions, action dimensions, floating signal types, device consistency, probability labels, quality bounds, and phase labels.
- **Seven objective channels**: reconstruction, cross-modal alignment, next-state world modeling, action prediction, contact prediction, slip prediction, and action feasibility.
- **Action-context and embodiment tokens**, enabling a common contract for temporal policy context and cross-embodiment experiments.
- **Causal objective aggregation**: action/contact/slip/feasibility outputs use the same objective-specific routing weights whose importance is later inspected.
- **Prediction-versus-reality reconciliation** with normalized next-state error and an external EMA error-memory primitive that can reduce future confidence through the explicit quality side-channel.
- **Intervention-calibrated routing**: measured modality-drop degradation can calibrate action-routing mass, rather than assuming routing maps are automatically faithful explanations.
- **Quantitative routing diagnostics** for token entropy, expert utilization, modality-objective mutual information, and phase-objective mutual information.
- **Deterministic contact dynamics regression** with compliant normal contact, tangential friction, and slip, adapted from the HapticSight donor concept. It is synthetic M1 evidence, not physical validation.
- **Deterministic safety gate** independent of learned routing, with force, feasibility, uncertainty, finite-value, and action-norm limits.
- **Failure-oriented interventions**: modality drop, noise, delay, temporal shuffle, drift, and force spikes.
- **Calibration metrics**: Brier score and expected calibration error.
- **Predictive uncertainty** via Monte Carlo stochastic inference when dropout is enabled.
- **Finite-difference constraint pressure** for failure-directed experiment planning.
- **Constrained candidate selection** that refuses high-score candidates when declared gates fail.
- **Portable NPZ trajectory schema** for external dataset conversion without pretending unrun dataset integrations exist.
- **SHA-256 experiment evidence**, source fingerprints, whole-repository release manifests, and explicit claim-authority levels.

## Earned local evidence

The repository ships an M1 synthetic-mechanism evidence bundle under `results/vtla_v1/`. It is intentionally bounded to deterministic synthetic evidence; **no real-dataset or real-robot performance claim is made**.

The release bundle includes:

- a known-answer loss-routing mechanism benchmark,
- multi-seed mechanism replication,
- a deterministic contact-dynamics regression,
- end-to-end synthetic smoke training and routing diagnostics,
- an intervention-calibrated routing-faithfulness ablation,
- a coupled-vs-detached task ablation that separately reports optimization loss and action RMSE,
- claim decisions that preserve failed claims instead of hiding them,
- environment and source provenance.

Run the verifier instead of trusting this README:

```bash
python -m explore_vtla verify-evidence results/vtla_v1
python -m explore_vtla verify-release-manifest .
```

## Quick verification

Minimal CPU verification does not require downloading CLIP or timm. The core ExPLoRe path has a narrow PyTorch fallback for the layer primitives used by the test suite, while normal research environments continue to use timm when installed.

```bash
python -m pytest -q tests
python -m explore_vtla mechanism --steps 160 --min-delta 0.20
python -m explore_vtla mechanism-replicate --seed 31 --runs 4 --steps 160 --min-delta 0.20
python scripts/check_vtla_green.py
```

The original upstream training dependencies remain in `requirements.txt`. Minimal deterministic CI dependencies are in `requirements-vtla-ci.txt`.

## Architecture

```text
vision ─────────┐
tactile ────────┤
force ──────────┤
proprio ────────┤
language ───────┤──> modality projection + quality/freshness encoding
previous action ┤                         │
embodiment ─────┘                         ▼
                                 causal temporal encoder
                                           │
                                  loss-coupled Soft-MoE
                                           │
                 ┌──────────┬──────────┬────┼────┬────────┬──────────┐
                 ▼          ▼          ▼         ▼        ▼          ▼
              recon     alignment   world     action   contact   feasibility
                                      model                + slip
                                           │
                                prediction ↔ observation
                                           │
                          reliability memory / intervention
                                           │
                              uncertainty + diagnostics
                                           │
                              independent safety boundary
```

See [`docs/vtla/ARCHITECTURE.md`](docs/vtla/ARCHITECTURE.md) for the exact tensor contracts and loss path.

## Evidence authority

| Level | Meaning |
|---|---|
| M0 | Unit/mathematical verification |
| M1 | Deterministic synthetic mechanism evidence |
| M2 | Offline real-dataset evidence |
| M3 | Physics-simulation evidence |
| M4 | Hardware-in-loop evidence |
| M5 | Real-robot evidence |
| M6 | Independent reproduction |

The included release is **M1**. Higher-authority claims are explicitly blocked until corresponding evidence exists.

See [`docs/vtla/CLAIM_BOUNDARY.md`](docs/vtla/CLAIM_BOUNDARY.md).

## External datasets

No external dataset is bundled and no benchmark result is invented. `explore_vtla.dataio` provides a strict portable sequence format so real datasets can be converted into the same validated trajectory contract.

Candidate public research datasets include Daimon-Infinity, FreeTacMan, DROID, and other visuo-tactile / robot-manipulation corpora. Their licenses and data terms must be reviewed independently before use.

See [`docs/vtla/DATA_INTEGRATION.md`](docs/vtla/DATA_INTEGRATION.md).

## Upstream compatibility

The original ExPLoRe README is preserved at [`docs/upstream/UPSTREAM_README.md`](docs/upstream/UPSTREAM_README.md). The original MIM implementation, configs, scripts, and tests remain present.

## License and attribution

Apache License 2.0. The upstream ExPLoRe copyright and license are preserved. New ExPLoRe-VTLA code is also released under Apache-2.0.

Donor repositories were used selectively as design references. License-compatible donor material is attributed where required; unrelated donor runtimes were not merged merely to increase code volume. See [`NOTICE`](NOTICE), [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md), and [`docs/vtla/DONOR_PROVENANCE.md`](docs/vtla/DONOR_PROVENANCE.md).

## Claim boundary

This repository does **not** claim:

- real-robot manipulation improvement,
- real-world safety certification,
- production Amazon/AWS deployment,
- Amazon endorsement or affiliation,
- validated sim-to-real transfer,
- improved performance on Daimon-Infinity, FreeTacMan, DROID, or any other real dataset until those experiments are actually run,
- raw-image/raw-taxel VLA capability in this v1.0.0 feature-level path,
- that routing weights are faithful explanations unless intervention evidence supports that claim.

The release is a functioning research framework plus deterministic M0/M1 evidence. Higher-authority claims remain blocked until higher-authority evidence exists.
