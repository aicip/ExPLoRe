from __future__ import annotations

import pytest
import torch

from explore_vtla.contracts import SignalHealth, VTLAConfig
from explore_vtla.quality import EmbeddingDriftMonitor, SignalQuality
from explore_vtla.synthetic import default_synthetic_config, make_synthetic_sequence


def test_config_rejects_empty_modalities():
    with pytest.raises(ValueError):
        VTLAConfig(modality_dims={}, action_dim=2)


def test_config_rejects_hidden_head_mismatch():
    with pytest.raises(ValueError):
        VTLAConfig(modality_dims={"vision": 3}, action_dim=2, hidden_dim=30, num_heads=8)


def test_synthetic_sequence_validates_and_reliability_is_one():
    cfg = default_synthetic_config()
    seq = make_synthetic_sequence(cfg, batch_size=3, cycles=1)
    seq.validate(cfg.modality_order)
    assert torch.allclose(seq.reliability(), torch.ones_like(seq.reliability()))


def test_sequence_clone_is_independent():
    cfg = default_synthetic_config()
    seq = make_synthetic_sequence(cfg, batch_size=2, cycles=1)
    clone = seq.clone()
    clone.modalities["vision"].zero_()
    assert not torch.equal(clone.modalities["vision"], seq.modalities["vision"])


def test_signal_quality_nominal_fresh_is_high():
    q = SignalQuality(SignalHealth.NOMINAL, confidence=0.9, age_ms=2, max_age_ms=100)
    assert 0.85 < q.reliability() <= 0.9


def test_signal_quality_unavailable_is_zero():
    q = SignalQuality(SignalHealth.UNAVAILABLE, confidence=1.0, age_ms=0, max_age_ms=100)
    assert q.reliability() == 0.0


def test_signal_quality_rejects_invalid_confidence():
    with pytest.raises(ValueError):
        SignalQuality(SignalHealth.NOMINAL, confidence=1.2, age_ms=0, max_age_ms=100).reliability()


def test_embedding_drift_monitor_requires_warmup():
    monitor = EmbeddingDriftMonitor(4, warmup=2)
    monitor.observe_reference(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    with pytest.raises(RuntimeError):
        monitor.score(torch.ones(1, 4))


def test_embedding_drift_detects_orthogonal_shift():
    monitor = EmbeddingDriftMonitor(3, warmup=2, threshold=0.3)
    monitor.observe_reference(torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    assert not bool(monitor.is_drifted(torch.tensor([[1.0, 0.0, 0.0]])).item())
    assert bool(monitor.is_drifted(torch.tensor([[0.0, 1.0, 0.0]])).item())
