# Testing and Release Gates

## Full test suite

```bash
python -m pytest -q tests
```

The suite contains inherited ExPLoRe tests plus VTLA-specific tests for:

- strict tensor contracts, declared feature/action dimensions, dtype/device consistency, and probability bounds
- signal reliability and freshness
- router normalization and loss-coupling gradients
- invalid-sensor dispatch masking
- causal temporal masking
- every routed objective
- routed action/contact/slip/feasibility inference aggregation
- world-model reality reconciliation and prediction-error reliability memory
- drift detection
- safety allow/clamp/hold behavior
- every intervention type
- routing/calibration metrics and quantitative routing diagnostics
- evidence tamper/stale/unmanifested/path-traversal detection
- whole-release manifest integrity
- authority claim gates, including expected blocked claims
- constraint-aware selection
- NPZ data round trips and manifest/schema rejection
- Monte Carlo uncertainty
- finite-difference constraint pressure
- deterministic mechanism learning
- multi-seed mechanism replication
- counterfactual calibration gradients and faithfulness
- coupled-vs-detached task ablation
- deterministic contact-world regression and VTLA conversion
- robustness campaign execution

## Mechanism gates

Single deterministic known-answer benchmark:

```bash
python -m explore_vtla mechanism --steps 160 --min-delta 0.20
```

Replicated deterministic gate:

```bash
python -m explore_vtla mechanism-replicate --seed 31 --runs 4 --steps 160 --min-delta 0.20
```

These are not accuracy benchmarks. They are M1 mechanism tests in which the preferred modality/phase routing structure is known by construction.

## Release evidence

```bash
python scripts/run_vtla_release.py
python -m explore_vtla verify-evidence results/vtla_v1
```

## Whole-release integrity

After all source, tests, docs, and committed evidence are frozen:

```bash
python scripts/build_release_manifest.py
python -m explore_vtla verify-release-manifest .
```

Strict verification rejects missing, changed, unmanifested, duplicate, or unsafe manifest paths.

## Complete green gate

```bash
python scripts/check_vtla_green.py
```

The script performs an AST/text anti-placeholder scan over the VTLA implementation, Python compilation, the full tests, the deterministic mechanism gate, evidence verification, release-manifest verification when present, and Ruff when Ruff is available locally.

GitHub Actions installs and enforces Ruff on Python 3.10, 3.11, 3.12, and 3.13. A local run must report Ruff as unavailable rather than pretending it passed if the executable is not installed.
