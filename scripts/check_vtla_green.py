"""Deterministic local release gate for ExPLoRe-VTLA."""

from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args: str) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def anti_placeholder_scan() -> None:
    forbidden_text = ("TODO", "FIXME", "PLACEHOLDER", "NotImplementedError")
    failures: list[str] = []
    for path in sorted((ROOT / "explore_vtla").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for token in forbidden_text:
            if token in text:
                failures.append(f"{path.relative_to(ROOT)} contains {token}")
        tree = ast.parse(text, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Pass):
                failures.append(f"{path.relative_to(ROOT)}:{node.lineno} contains bare pass")
    if failures:
        raise SystemExit("anti-placeholder gate failed:\n" + "\n".join(failures))
    print("anti-placeholder gate: PASS")


def main() -> int:
    anti_placeholder_scan()
    run(sys.executable, "-m", "compileall", "-q", "explore_vtla", "src/models")
    run(sys.executable, "-m", "pytest", "-q", "tests")
    run(
        sys.executable,
        "-m",
        "explore_vtla",
        "mechanism",
        "--steps",
        "160",
        "--min-delta",
        "0.20",
    )
    run(
        sys.executable,
        "-m",
        "explore_vtla",
        "verify-evidence",
        "results/vtla_v1",
    )
    if (ROOT / "MANIFEST.sha256").is_file():
        run(sys.executable, "-m", "explore_vtla", "verify-release-manifest", ".")
    else:
        print("INFO: MANIFEST.sha256 not generated yet; final release packaging creates it.")

    if importlib.util.find_spec("ruff") is not None:
        run(
            sys.executable,
            "-m",
            "ruff",
            "check",
            "explore_vtla",
            "tests/vtla",
            "scripts/check_vtla_green.py",
            "scripts/run_vtla_release.py",
            "scripts/build_release_manifest.py",
            "examples/vtla_quickstart.py",
        )
    else:
        print("INFO: ruff module not installed locally; GitHub Actions installs and enforces it.")

    print(
        json.dumps(
            {
                "vtla_quality_gate": "GREEN",
                "authority": "M1_SYNTHETIC_MECHANISM",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
