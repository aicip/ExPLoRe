"""Portable NPZ trajectory interchange for external dataset adapters.

Dataset-specific download/authentication code is intentionally not embedded.
Researchers can convert a dataset into this explicit schema, after which the
same VTLA loader, validation, evidence, and training paths are used.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .contracts import VTLAConfig, VTLASequence

SCHEMA_VERSION = "explore-vtla-sequence-v1"


def save_sequence_npz(path: str | Path, config: VTLAConfig, sequence: VTLASequence) -> None:
    sequence.validate(
        config.modality_order,
        modality_dims=config.modality_dims,
        action_dim=config.action_dim,
    )
    path = Path(path)
    payload: dict[str, np.ndarray] = {
        "action": sequence.action.detach().cpu().numpy(),
        "contact": sequence.contact.detach().cpu().numpy(),
        "slip": sequence.slip.detach().cpu().numpy(),
        "feasible": sequence.feasible.detach().cpu().numpy(),
        "quality": sequence.quality.detach().cpu().numpy(),
        "manifest_json": np.array(
            json.dumps(
                {
                    "schema": SCHEMA_VERSION,
                    "modalities": list(config.modality_order),
                    "modality_dims": dict(config.modality_dims),
                    "action_dim": config.action_dim,
                    "metadata": sequence.metadata,
                },
                sort_keys=True,
            )
        ),
    }
    if sequence.phase is not None:
        payload["phase"] = sequence.phase.detach().cpu().numpy()
    for name, tensor in sequence.modalities.items():
        payload[f"modality__{name}"] = tensor.detach().cpu().numpy()
    np.savez_compressed(path, **payload)


def load_sequence_npz(path: str | Path, config: VTLAConfig) -> VTLASequence:
    with np.load(Path(path), allow_pickle=False) as data:
        manifest = json.loads(str(data["manifest_json"].item()))
        if manifest.get("schema") != SCHEMA_VERSION:
            raise ValueError("unsupported VTLA sequence schema")
        if tuple(manifest.get("modalities", ())) != config.modality_order:
            raise ValueError("modality order does not match configuration")
        if manifest.get("modality_dims") != dict(config.modality_dims):
            raise ValueError("modality dimensions do not match configuration")
        if int(manifest.get("action_dim", -1)) != config.action_dim:
            raise ValueError("action dimension does not match configuration")
        modalities = {
            name: torch.from_numpy(data[f"modality__{name}"].copy()).to(torch.float32)
            for name in config.modality_order
        }
        phase = torch.from_numpy(data["phase"].copy()).long() if "phase" in data.files else None
        sequence = VTLASequence(
            modalities=modalities,
            action=torch.from_numpy(data["action"].copy()).to(torch.float32),
            contact=torch.from_numpy(data["contact"].copy()).to(torch.float32),
            slip=torch.from_numpy(data["slip"].copy()).to(torch.float32),
            feasible=torch.from_numpy(data["feasible"].copy()).to(torch.float32),
            quality=torch.from_numpy(data["quality"].copy()).to(torch.float32),
            phase=phase,
            metadata=dict(manifest.get("metadata", {})),
        )
    return sequence.validate(
        config.modality_order,
        modality_dims=config.modality_dims,
        action_dim=config.action_dim,
    )
