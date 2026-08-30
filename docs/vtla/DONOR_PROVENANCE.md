# Donor Provenance

The VTLA extension was designed after reviewing multiple donor repositories supplied by the fork maintainer. Donors were treated as engineering references, not as code to merge wholesale.

## Directly relevant design donors

### IX-HapticSight, MIT

Relevant ideas:

- explicit signal health and freshness
- tactile/contact state as a first-class modality
- independent safety authority
- failure-oriented sensor handling
- deterministic contact-world regression

VTLA implementation: signal-health/safety mechanisms were reimplemented specifically for tensorized model training. The small normal/tangential contact regression in `explore_vtla/contact_dynamics.py` adapts the donor's contact-world testing concept; the required MIT notice is preserved in [`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md).

### SynapDrive-AI, Apache-2.0

Relevant ideas:

- uncertainty and drift monitoring
- predicted-versus-observed reconciliation
- fail-safe action routing / Simplex-style separation
- replay/evidence discipline

VTLA implementation: selectively reimplemented as `quality.py`, `reality.py`, `uncertainty.py`, and the independent safety boundary.

### IX-Autonomy-Assurance-Case-Runtime, Apache-2.0

Relevant ideas:

- explicit evidence authority
- claim guardrails
- integrity manifests
- scenario/failure campaigns

VTLA implementation: reduced research-specific versions in `claims.py`, `evidence.py`, `campaign.py`, `provenance.py`, and `release_manifest.py`.

## Conceptual donors only

IX-StellaratorForge contributed the failure-directed optimization concept behind finite-difference constraint pressure. Fusion-specific physics, geometry, confinement, and reactor code are not used.

IX-BlackFox, IX-Sally, IX-IntentRealityLoop, IX-main, and IX-PackHunt-Guard were reviewed. Their general governance/reality-loop ideas informed design discussion, but their runtimes were not merged into ExPLoRe-VTLA.

## Why selective reuse matters

The central research object remains ExPLoRe's loss-coupling mechanism. Importing unrelated agent, governance, security, or scientific runtimes would increase code volume without strengthening the hypothesis. ExPLoRe-VTLA therefore keeps donor influence narrow, attributable, and testable.
