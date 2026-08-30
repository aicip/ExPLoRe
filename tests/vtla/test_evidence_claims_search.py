from __future__ import annotations

import json
from pathlib import Path

from explore_vtla.claims import evaluate_claims
from explore_vtla.contracts import AuthorityLevel
from explore_vtla.evidence import verify_bundle, write_bundle
from explore_vtla.search import Candidate, Constraint, select_candidate


def test_evidence_bundle_roundtrip(tmp_path: Path):
    write_bundle(tmp_path, {"a.json": {"x": 1}, "b.json": {"y": [1, 2]}})
    ok, errors = verify_bundle(tmp_path)
    assert ok
    assert errors == ()


def test_evidence_detects_tampering(tmp_path: Path):
    write_bundle(tmp_path, {"a.json": {"x": 1}})
    (tmp_path / "a.json").write_text(json.dumps({"x": 2}))
    ok, errors = verify_bundle(tmp_path)
    assert not ok
    assert any("hash mismatch" in error for error in errors)


def test_claim_gate_allows_synthetic_mechanism_only():
    decisions = evaluate_claims(
        AuthorityLevel.M1_SYNTHETIC_MECHANISM,
        {"specialization_delta": 0.4},
    )
    lookup = {item.claim_id: item.allowed for item in decisions}
    assert lookup["loss_coupling_mechanism"]
    assert not lookup["synthetic_routing_faithfulness"]
    assert not lookup["offline_real_dataset_gain"]
    assert not lookup["real_robot_effectiveness"]


def test_claim_gate_blocks_weak_synthetic_delta():
    decisions = evaluate_claims(
        AuthorityLevel.M1_SYNTHETIC_MECHANISM,
        {"specialization_delta": 0.05},
    )
    assert not decisions[0].allowed


def test_constraint_search_rejects_higher_scoring_infeasible_candidate():
    candidates = [
        Candidate("fast", 0.8, {"latency_ms": 8.0, "failure_rate": 0.02}),
        Candidate("high_score_but_bad", 0.95, {"latency_ms": 40.0, "failure_rate": 0.01}),
    ]
    selection = select_candidate(candidates, (Constraint("latency_ms", "<=", 10.0),))
    assert selection.winner is not None
    assert selection.winner.candidate_id == "fast"
    assert selection.rejected[0][0] == "high_score_but_bad"


def test_constraint_search_returns_no_winner_when_all_fail():
    candidates = [Candidate("a", 1.0, {"latency": 20.0})]
    selection = select_candidate(candidates, (Constraint("latency", "<=", 5.0),))
    assert selection.winner is None
