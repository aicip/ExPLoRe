"""Regenerate the deterministic ExPLoRe-VTLA M1 release evidence bundle."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    output = ROOT / "results" / "vtla_v1"
    command = [
        sys.executable,
        "-m",
        "explore_vtla",
        "release-evidence",
        "--output",
        str(output),
        "--seed",
        "31",
        "--mechanism-steps",
        "160",
        "--smoke-steps",
        "80",
        "--faithfulness-steps",
        "100",
        "--task-ablation-steps",
        "60",
        "--min-delta",
        "0.20",
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
