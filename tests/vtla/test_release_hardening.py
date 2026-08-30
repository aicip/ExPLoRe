from __future__ import annotations

from pathlib import Path

import pytest
import torch

from explore_vtla.contracts import DEFAULT_OBJECTIVES, SignalHealth, VTLAConfig
from explore_vtla.diagnostics import routing_diagnostics
from explore_vtla.evidence import verify_bundle, write_bundle
from explore_vtla.provenance import repository_fingerprint
from explore_vtla.quality import SignalQuality
from explore_vtla.reality import PredictionErrorMemory
from explore_vtla.synthetic import (
    default_synthetic_config,
    make_synthetic_sequence,
    run_mechanism_replication,
)


def test_config_rejects_unknown_objective() -> None:
    with pytest.raises(ValueError, match="unsupported objectives"):
        VTLAConfig(
            modality_dims={"vision": 3},
            action_dim=2,
            objectives=(*DEFAULT_OBJECTIVES, "imaginary_objective"),
        )


def test_sequence_rejects_wrong_declared_modality_dimension() -> None:
    cfg = default_synthetic_config()
    seq = make_synthetic_sequence(cfg, batch_size=2, cycles=1)
    seq.modalities["vision"] = seq.modalities["vision"][..., :-1]
    with pytest.raises(ValueError, match="vision.*feature dimension"):
        seq.validate(
            cfg.modality_order,
            modality_dims=cfg.modality_dims,
            action_dim=cfg.action_dim,
        )


def test_sequence_rejects_wrong_declared_action_dimension() -> None:
    cfg = default_synthetic_config()
    seq = make_synthetic_sequence(cfg, batch_size=2, cycles=1)
    seq.action = seq.action[..., :-1]
    with pytest.raises(ValueError, match="action feature dimension"):
        seq.validate(
            cfg.modality_order,
            modality_dims=cfg.modality_dims,
            action_dim=cfg.action_dim,
        )


def test_sequence_rejects_non_probability_targets() -> None:
    cfg = default_synthetic_config()
    seq = make_synthetic_sequence(cfg, batch_size=2, cycles=1)
    seq.feasible[0, 0] = 1.5
    with pytest.raises(ValueError, match="feasible targets"):
        seq.validate(cfg.modality_order)


def test_signal_quality_tensor_features_use_same_validation() -> None:
    quality = SignalQuality(SignalHealth.NOMINAL, confidence=1.2, age_ms=1.0, max_age_ms=10.0)
    with pytest.raises(ValueError, match="confidence"):
        quality.tensor_features()


def test_evidence_rejects_path_traversal_artifact_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="simple .json basenames"):
        write_bundle(tmp_path, {"../outside.json": {"x": 1}})


def test_evidence_rejects_unmanifested_json(tmp_path: Path) -> None:
    write_bundle(tmp_path, {"a.json": {"x": 1}})
    (tmp_path / "stale.json").write_text("{}\n", encoding="utf-8")
    ok, errors = verify_bundle(tmp_path)
    assert not ok
    assert "unmanifested artifact: stale.json" in errors


def test_evidence_rejects_invalid_digest(tmp_path: Path) -> None:
    (tmp_path / "a.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "SHA256SUMS").write_text("bad  a.json\n", encoding="utf-8")
    ok, errors = verify_bundle(tmp_path)
    assert not ok
    assert "invalid sha256 digest: a.json" in errors


def test_repository_fingerprint_covers_release_workflow() -> None:
    root = Path(__file__).resolve().parents[2]
    fingerprint = repository_fingerprint(root)
    files = fingerprint["files"]
    assert isinstance(files, dict)
    assert ".github/workflows/vtla-quality.yml" in files
    assert "scripts/run_vtla_release.py" in files
    assert "pyproject.toml" in files


def test_routing_diagnostics_uniform_case_has_no_mutual_information() -> None:
    dispatch = torch.full((2, 3, 4, 5), 1.0 / 12.0)
    combine = torch.full((2, 3, 4, 5), 1.0 / 5.0)
    phase = torch.tensor([[0, 1, 2], [0, 1, 2]])
    report = routing_diagnostics(dispatch, combine, phase)
    assert report.modality_objective_mi_bits == pytest.approx(0.0, abs=1e-6)
    assert report.phase_objective_mi_bits == pytest.approx(0.0, abs=1e-6)
    assert report.mean_objective_token_entropy == pytest.approx(1.0, abs=1e-6)
    assert report.expert_utilization_cv == pytest.approx(0.0, abs=1e-6)


def test_routing_diagnostics_detects_modality_objective_dependence() -> None:
    dispatch = torch.zeros(1, 2, 2, 2)
    dispatch[:, :, 0, 0] = 0.5
    dispatch[:, :, 1, 1] = 0.5
    combine = torch.full_like(dispatch, 0.5)
    report = routing_diagnostics(dispatch, combine)
    assert report.modality_objective_mi_bits > 0.9
    assert report.modality_objective_nmi > 0.9


def test_prediction_error_memory_applies_external_reliability_feedback() -> None:
    cfg = default_synthetic_config()
    seq = make_synthetic_sequence(cfg, batch_size=1, cycles=1)
    memory = PredictionErrorMemory(cfg.modality_order, decay=0.0, gain=1.0)
    errors = {name: 0.1 for name in cfg.modality_order}
    errors["tactile"] = 2.0
    adjusted = memory.apply_to_sequence(seq, errors)
    tactile_idx = cfg.modality_order.index("tactile")
    vision_idx = cfg.modality_order.index("vision")
    assert adjusted.reliability()[..., tactile_idx].mean() < adjusted.reliability()[..., vision_idx].mean()
    assert torch.allclose(seq.reliability(), torch.ones_like(seq.reliability()))


def test_mechanism_replication_requires_every_declared_seed_to_pass() -> None:
    result = run_mechanism_replication((11, 13, 17, 19), steps=80, minimum_delta=0.20)
    assert result.all_pass
    assert result.minimum_specialization_delta > 0.20
    assert max(result.final_to_initial_loss_ratios) < 0.25


def test_loss_coupling_task_ablation_reports_separate_training_and_action_metrics() -> None:
    from explore_vtla.experiment import run_loss_coupling_task_ablation

    result = run_loss_coupling_task_ablation((5, 7), steps=8)
    assert len(result.coupled_final_loss) == 2
    assert len(result.detached_action_rmse) == 2
    assert 0.0 <= result.coupled_final_loss_win_rate <= 1.0
    assert 0.0 <= result.coupled_action_rmse_win_rate <= 1.0


def test_release_manifest_detects_tampering_and_unmanifested_files(tmp_path: Path) -> None:
    from explore_vtla.release_manifest import verify_release_manifest, write_release_manifest

    (tmp_path / "a.txt").write_text("alpha\n", encoding="utf-8")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "b.json").write_text("{}\n", encoding="utf-8")
    write_release_manifest(tmp_path)
    assert verify_release_manifest(tmp_path).verified

    (tmp_path / "a.txt").write_text("changed\n", encoding="utf-8")
    tampered = verify_release_manifest(tmp_path)
    assert not tampered.verified
    assert any("hash mismatch: a.txt" == error for error in tampered.errors)

    (tmp_path / "a.txt").write_text("alpha\n", encoding="utf-8")
    (tmp_path / "extra.txt").write_text("extra\n", encoding="utf-8")
    extra = verify_release_manifest(tmp_path)
    assert not extra.verified
    assert any("unmanifested release file: extra.txt" == error for error in extra.errors)


def test_contact_world_regression_has_causal_contact_and_slip() -> None:
    from explore_vtla.contact_dynamics import contact_regression_report

    report = contact_regression_report()
    assert report["first_contact_step"] is not None
    assert report["max_pre_contact_force_n"] == 0.0
    assert report["max_contact_force_n"] > 0.0
    assert report["slip_steps"] > 0
    assert all(report["invariants"].values())
    assert report["physical_validation"] is False


def test_contact_world_sequence_satisfies_vtla_contract_and_contains_failure_targets() -> None:
    from explore_vtla.contact_dynamics import make_contact_world_sequence

    sequence = make_contact_world_sequence()
    assert sequence.contact.sum() > 0
    assert sequence.slip.sum() > 0
    assert (sequence.feasible == 0).sum() > 0
    assert sequence.metadata["generator"] == "deterministic_compliant_contact_world_v1"
