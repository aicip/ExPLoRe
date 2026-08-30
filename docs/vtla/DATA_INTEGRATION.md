# External Data Integration

No third-party dataset is bundled with this repository, and no real-data benchmark result is pre-populated.

## Portable sequence schema

`explore_vtla.dataio` provides a compressed NPZ interchange format containing:

- ordered modality tensors `[B,T,F_m]`
- action target `[B,T,A]`
- contact, slip, and feasibility labels `[B,T]`
- quality tensor `[B,T,M,3]`
- optional phase labels `[B,T]`
- schema/version metadata
- declared modality dimensions and action dimension

The loader uses `allow_pickle=False`, validates the schema/version, exact modality order, declared feature dimensions, declared action dimension, tensor shape/batch/time consistency, floating signal types, common device, bounded probability labels, quality bounds, and phase validity before returning a `VTLASequence`.

## Current representation boundary

The v1.0.0 model consumes feature vectors, not raw camera frames or raw tactile taxel grids. A real-data adapter must therefore document the encoder or preprocessing step used to transform source sensor data into each declared `[T,F_m]` feature stream.

That preprocessing is part of the experiment provenance and must not be hidden behind an undocumented converter.

## Recommended adapter workflow

1. Read the source dataset using its official tooling.
2. Align observations and actions in timestamp order.
3. Preserve missing/stale sensors in the quality channel rather than filling them with apparently valid zeros.
4. Convert raw vision/tactile streams to declared features or tokens using a documented preprocessing stage.
5. Export a VTLA NPZ shard.
6. Hash the source split list and conversion configuration.
7. Train/evaluate through the same VTLA model and evidence path.
8. Keep any result at M2 until higher-authority hardware evidence exists.

## Candidate public datasets

Potential research candidates include Daimon-Infinity, FreeTacMan, and DROID. Dataset-specific licenses, noncommercial clauses, download terms, and redistribution rules must be checked before use. This repository intentionally does not pretend that a dataset is integrated until an actual converter and actual experiment have been run.
