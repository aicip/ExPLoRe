"""Command-line interface for deterministic VTLA verification artifacts."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict
from pathlib import Path

import torch

from .claims import evaluate_claims
from .contact_dynamics import contact_regression_report
from .contracts import AuthorityLevel
from .evidence import verify_bundle, write_bundle
from .experiment import (
    run_faithfulness_ablation,
    run_loss_coupling_task_ablation,
    run_smoke_training,
)
from .provenance import repository_fingerprint
from .release_manifest import verify_release_manifest, write_release_manifest
from .synthetic import run_mechanism_benchmark, run_mechanism_replication


def _print(data: object) -> None:
    print(json.dumps(data, indent=2, sort_keys=True))


def command_mechanism(args: argparse.Namespace) -> int:
    result = run_mechanism_benchmark(seed=args.seed, steps=args.steps)
    payload = asdict(result)
    payload["specialization_delta"] = (
        result.specialization_score - result.detached_specialization_score
    )
    payload["authority"] = AuthorityLevel.M1_SYNTHETIC_MECHANISM.name
    _print(payload)
    return 0 if payload["specialization_delta"] >= args.min_delta else 2


def command_mechanism_replicate(args: argparse.Namespace) -> int:
    seeds = tuple(args.seed + 2 * idx for idx in range(args.runs))
    result = run_mechanism_replication(
        seeds, steps=args.steps, minimum_delta=args.min_delta
    )
    payload = asdict(result)
    payload["authority"] = AuthorityLevel.M1_SYNTHETIC_MECHANISM.name
    _print(payload)
    return 0 if result.all_pass else 2


def command_smoke(args: argparse.Namespace) -> int:
    _, result = run_smoke_training(seed=args.seed, steps=args.steps)
    payload = result.to_dict()
    payload["authority"] = AuthorityLevel.M1_SYNTHETIC_MECHANISM.name
    _print(payload)
    return 0 if result.loss_reduction_fraction >= args.min_reduction else 2


def command_release_evidence(args: argparse.Namespace) -> int:
    mechanism = run_mechanism_benchmark(seed=args.seed, steps=args.mechanism_steps)
    replication = run_mechanism_replication(
        tuple(args.seed + 2 * idx for idx in range(4)),
        steps=args.mechanism_steps,
        minimum_delta=args.min_delta,
    )
    _, smoke = run_smoke_training(seed=args.seed + 1, steps=args.smoke_steps)
    ablation = run_faithfulness_ablation(seed=args.seed + 2, steps=args.faithfulness_steps)
    task_ablation = run_loss_coupling_task_ablation(
        tuple(args.seed + 10 + 2 * idx for idx in range(4)),
        steps=args.task_ablation_steps,
    )
    specialization_delta = mechanism.specialization_score - mechanism.detached_specialization_score
    contact_regression = contact_regression_report()
    contact_invariants = contact_regression["invariants"]
    contact_pass = float(all(bool(value) for value in contact_invariants.values()))
    metrics = {
        "specialization_delta": specialization_delta,
        "contact_regression_invariants_pass": contact_pass,
        "mechanism_initial_loss": mechanism.initial_loss,
        "mechanism_final_loss": mechanism.final_loss,
        "smoke_loss_reduction_fraction": smoke.loss_reduction_fraction,
        "minimum_replicated_specialization_delta": replication.minimum_specialization_delta,
        "calibrated_faithfulness": ablation.calibrated_rank_correlation,
        "faithfulness_improvement": ablation.improvement,
        "coupled_action_rmse_win_rate": task_ablation.coupled_action_rmse_win_rate,
    }
    claims = [decision.to_dict() for decision in evaluate_claims(AuthorityLevel.M1_SYNTHETIC_MECHANISM, metrics)]
    artifacts = {
        "mechanism.json": {**asdict(mechanism), "specialization_delta": specialization_delta},
        "mechanism_replication.json": asdict(replication),
        "contact_dynamics_regression.json": contact_regression,
        "smoke_training.json": smoke.to_dict(),
        "faithfulness_ablation.json": ablation.to_dict(),
        "loss_coupling_task_ablation.json": task_ablation.to_dict(),
        "claims.json": {"authority": AuthorityLevel.M1_SYNTHETIC_MECHANISM.name, "decisions": claims},
        "environment.json": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "platform": platform.platform(),
            "experimental_robot_validation": False,
            "real_dataset_training_run": False,
        },
        "run_config.json": {
            "mechanism_seed": args.seed,
            "mechanism_steps": args.mechanism_steps,
            "smoke_seed": args.seed + 1,
            "smoke_steps": args.smoke_steps,
            "faithfulness_seed": args.seed + 2,
            "faithfulness_steps": args.faithfulness_steps,
            "task_ablation_steps": args.task_ablation_steps,
            "task_ablation_seeds": list(task_ablation.seeds),
            "minimum_specialization_delta": args.min_delta,
            "replication_seeds": list(replication.seeds),
        },
        "source_provenance.json": repository_fingerprint(Path(__file__).resolve().parents[1]),
        "dataset_manifest.json": {
            "name": "deterministic_contact_routing_synthetic_v1",
            "authority": AuthorityLevel.M1_SYNTHETIC_MECHANISM.name,
            "external_dataset": False,
            "real_robot_data": False,
        },
    }
    write_bundle(args.output, artifacts)
    ok, errors = verify_bundle(args.output)
    _print({"output": str(args.output), "verified": ok, "errors": errors, "metrics": metrics})
    return 0 if (
        ok
        and specialization_delta >= args.min_delta
        and contact_pass == 1.0
        and replication.all_pass
        and ablation.calibrated_rank_correlation >= 0.50
    ) else 2


def command_verify(args: argparse.Namespace) -> int:
    ok, errors = verify_bundle(args.directory)
    _print({"verified": ok, "errors": errors})
    return 0 if ok else 2


def command_write_release_manifest(args: argparse.Namespace) -> int:
    manifest = write_release_manifest(args.root)
    _print({"manifest": str(manifest)})
    return 0


def command_verify_release_manifest(args: argparse.Namespace) -> int:
    result = verify_release_manifest(args.root, strict=not args.allow_unmanifested)
    _print(result.to_dict())
    return 0 if result.verified else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="explore-vtla")
    sub = parser.add_subparsers(dest="command", required=True)

    mechanism = sub.add_parser("mechanism", help="run deterministic loss-routing mechanism benchmark")
    mechanism.add_argument("--seed", type=int, default=11)
    mechanism.add_argument("--steps", type=int, default=160)
    mechanism.add_argument("--min-delta", type=float, default=0.20)
    mechanism.set_defaults(func=command_mechanism)

    replicate = sub.add_parser(
        "mechanism-replicate",
        help="repeat the deterministic mechanism benchmark across declared seeds",
    )
    replicate.add_argument("--seed", type=int, default=11)
    replicate.add_argument("--runs", type=int, default=4)
    replicate.add_argument("--steps", type=int, default=120)
    replicate.add_argument("--min-delta", type=float, default=0.20)
    replicate.set_defaults(func=command_mechanism_replicate)

    smoke = sub.add_parser("smoke", help="run tiny end-to-end VTLA training")
    smoke.add_argument("--seed", type=int, default=23)
    smoke.add_argument("--steps", type=int, default=100)
    smoke.add_argument("--min-reduction", type=float, default=0.10)
    smoke.set_defaults(func=command_smoke)

    evidence = sub.add_parser("release-evidence", help="generate deterministic M1 evidence bundle")
    evidence.add_argument("--output", type=Path, default=Path("results/vtla_v1"))
    evidence.add_argument("--seed", type=int, default=31)
    evidence.add_argument("--mechanism-steps", type=int, default=160)
    evidence.add_argument("--smoke-steps", type=int, default=80)
    evidence.add_argument("--faithfulness-steps", type=int, default=100)
    evidence.add_argument("--task-ablation-steps", type=int, default=60)
    evidence.add_argument("--min-delta", type=float, default=0.20)
    evidence.set_defaults(func=command_release_evidence)

    verify = sub.add_parser("verify-evidence", help="verify an evidence bundle")
    verify.add_argument("directory", type=Path)
    verify.set_defaults(func=command_verify)

    manifest = sub.add_parser("write-release-manifest", help="write MANIFEST.sha256")
    manifest.add_argument("root", type=Path, nargs="?", default=Path("."))
    manifest.set_defaults(func=command_write_release_manifest)

    verify_manifest = sub.add_parser(
        "verify-release-manifest", help="verify MANIFEST.sha256 against the release tree"
    )
    verify_manifest.add_argument("root", type=Path, nargs="?", default=Path("."))
    verify_manifest.add_argument("--allow-unmanifested", action="store_true")
    verify_manifest.set_defaults(func=command_verify_release_manifest)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
