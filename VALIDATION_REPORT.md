# Validation Report, ExPLoRe-VTLA v1.0.0

This report is based on executable local verification of the delivered source tree. Its authority is deliberately limited to software verification and deterministic synthetic mechanism evidence.

## Final deterministic test state

- Full repository tests: **246 passed**
- Preserved upstream/core ExPLoRe tests: **170 passed**
- VTLA-specific tests: **76 passed**
- Python compile gate: **PASS**
- VTLA anti-placeholder AST/text gate: **PASS**
- SHA-256 experiment evidence verification: **PASS**

The final release also carries a strict whole-repository `MANIFEST.sha256`; final archive verification is performed after this document and all other release files are frozen.

## Loss-coupling mechanism benchmark

Persisted artifact: `results/vtla_v1/mechanism.json`

For the release seed/configuration:

- initial weighted mechanism loss: **1.3346958160**
- final weighted mechanism loss: **0.0961917415**
- learned preferred-target routing mass: **0.9517468214** mean
- detached/random-start preferred-target mass: **0.0501215346** mean
- specialization delta: **+0.9016252868**
- per-objective preferred-target mass: **0.95364 / 0.95419 / 0.95389 / 0.94527**

This earns **M1 synthetic mechanism** authority. It does not establish real-data benefit.

## Replicated mechanism gate

Persisted artifact: `results/vtla_v1/mechanism_replication.json`

Four deterministic seeds were required to clear the same specialization threshold:

- seeds: **31 / 33 / 35 / 37**
- minimum specialization delta: **0.8834090903**
- median specialization delta: **0.8917596620**
- maximum specialization delta: **0.9016252868**
- all four seeds passed the declared **0.20** threshold

This reduces dependence on a single favorable mechanism seed, while remaining synthetic M1 evidence.

## Deterministic contact-world regression

Persisted artifact: `results/vtla_v1/contact_dynamics_regression.json`

The executable synthetic contact plant produced:

- 144 timesteps across 6 declared phases
- first contact at step **22**
- maximum pre-contact normal force: **0.0 N**
- maximum contact normal force: **14.0799995 N**
- slip detected on **52** timesteps

Declared causal regression invariants all passed:

- zero normal force before contact: **PASS**
- positive normal force after contact: **PASS**
- slip emerges under the stronger tangential command: **PASS**

The artifact explicitly records `physical_validation: false`. This is a deterministic software/contact-mechanism regression, not a calibrated contact solver or M3 physics validation.

## End-to-end synthetic smoke training

Persisted artifact: `results/vtla_v1/smoke_training.json`

- initial coupled training loss: **0.4863524139**
- final coupled training loss: **0.0741030425**
- reduction: **84.7635%**
- reference model parameters: **46,729**
- nominal synthetic contact accuracy: **97.9167%**
- nominal synthetic feasibility Brier score: **0.00102270**
- nominal synthetic feasibility ECE: **0.0199894**

The smoke run also records routing diagnostics rather than relying only on heatmaps:

- mean normalized objective token entropy: **0.9689468**
- expert-utilization coefficient of variation: **0.1396608**
- modality-objective mutual information: **0.025521 bits**
- phase-objective mutual information: **0.140834 bits**

These diagnostics show that the full smoke model does not exhibit extreme global routing sparsity. They are descriptive synthetic diagnostics, not evidence of causal importance.

## Counterfactual routing-faithfulness ablation

Persisted artifact: `results/vtla_v1/faithfulness_ablation.json`

Using equivalent synthetic conditions:

- uncalibrated routing-vs-drop-impact rank correlation: **0.1785714**
- intervention-calibrated correlation: **0.8571429**
- improvement: **+0.6785714**
- uncalibrated loss reduction: **94.0744%**
- calibrated loss reduction: **88.6095%**

The repository therefore does not assume routing weights are faithful explanations. The calibrated path is evaluated against measured behavioral degradation under intervention.

## Coupled-vs-detached task ablation: preserved negative result

Persisted artifact: `results/vtla_v1/loss_coupling_task_ablation.json`

Seeds: **41 / 43 / 45 / 47**, 60 optimization steps each.

Coupled loss routing achieved the lower final weighted training objective on **4/4 seeds (100%)**. However, it achieved lower action RMSE than detached routing on only **1/4 seeds (25%)**.

- coupled final-loss win rate: **1.00**
- coupled action-RMSE win rate: **0.25**
- median `(detached - coupled)` final-loss delta: **+0.1155893**
- median `(detached - coupled)` action-RMSE delta: **-0.0612796**

This distinction is intentional. Lowering the routed objective does **not** earn a downstream task-superiority claim. The `synthetic_action_superiority` claim requires a 0.75 action-RMSE win rate and therefore remains **BLOCKED**.

## Claim gates

Persisted artifact: `results/vtla_v1/claims.json`

Allowed at current M1 authority:

- deterministic loss-coupling specialization
- deterministic contact-world regression invariants
- replicated loss-coupling mechanism threshold
- deterministic synthetic routing-faithfulness after intervention calibration

Blocked:

- synthetic action superiority, because action-RMSE win rate is **0.25 < 0.75**
- offline real-dataset improvement, because M2 evidence is absent
- real-robot effectiveness, because M5 evidence is absent

## Environment limitation

Ruff is not installed in the build sandbox and network access needed to install it is unavailable, so local Ruff execution is **not** claimed. GitHub Actions installs and enforces Ruff across Python 3.10, 3.11, 3.12, and 3.13.

Local executable verification includes Python compilation, all 246 tests, mechanism gating, evidence generation, evidence verification, anti-placeholder scanning, and final release-manifest/archive verification.
