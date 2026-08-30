# Changelog

## 1.0.0

- Preserve the original ExPLoRe MIM implementation as the upstream research baseline.
- Add PyTorch-only fallbacks for the narrow timm layer surface needed by core CPU verification.
- Make CLIP teacher import lazy so core tests do not require a model download.
- Add ExPLoRe-VTLA temporal multimodal loss routing over feature-level trajectory tokens.
- Add strict declared-dimension, dtype, device, probability, quality, and phase validation.
- Add sensor reliability/freshness contracts and unavailable-modality dispatch masking.
- Add seven routed objectives and objective-specific inference aggregation.
- Add action-context and embodiment tokens.
- Add world-model reality reconciliation, externally applied prediction-error reliability memory, and drift monitoring.
- Add deterministic intervention suite and routing-faithfulness metrics.
- Add intervention-calibrated action routing.
- Add quantitative routing entropy, utilization, and mutual-information diagnostics.
- Add replicated multi-seed mechanism verification.
- Add coupled-vs-detached task ablation that separates optimization loss from action RMSE and preserves negative results.
- Add deterministic compliant contact dynamics with friction/slip regression, explicitly bounded to M1 synthetic authority.
- Add predictive uncertainty and independent safety gating.
- Add failure campaigns, constraint-aware selection, and finite-difference constraint pressure.
- Add portable external-data NPZ schema with strict manifest validation.
- Harden experiment evidence verification against stale, unmanifested, duplicate, or unsafe artifact paths.
- Add SHA-256 evidence, source fingerprints, claim authority, and a strict whole-repository release manifest.
- Add CI across Python 3.10, 3.11, 3.12, and 3.13.
