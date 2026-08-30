"""Canonical JSON evidence bundles with strict SHA-256 integrity verification."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Mapping

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def canonical_json(data: object) -> bytes:
    return (
        json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _validate_artifact_name(filename: str) -> None:
    path = Path(filename)
    if (
        not filename.endswith(".json")
        or path.name != filename
        or filename in {".", ".."}
        or "/" in filename
        or "\\" in filename
    ):
        raise ValueError("evidence artifact names must be simple .json basenames")


def write_bundle(directory: str | Path, artifacts: Mapping[str, object]) -> dict[str, str]:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    expected_names = set(artifacts)
    for filename in expected_names:
        _validate_artifact_name(filename)

    # Prevent stale JSON from surviving a regenerated release bundle and being
    # mistaken for current evidence.
    for path in root.iterdir():
        if path.is_file() and path.suffix == ".json" and path.name not in expected_names:
            path.unlink()

    for filename, payload in sorted(artifacts.items()):
        path = root / filename
        data = canonical_json(payload)
        path.write_bytes(data)
        hashes[filename] = sha256_bytes(data)
    manifest_lines = [f"{digest}  {name}\n" for name, digest in sorted(hashes.items())]
    manifest = "".join(manifest_lines).encode("utf-8")
    (root / "SHA256SUMS").write_bytes(manifest)
    return hashes


def verify_bundle(directory: str | Path) -> tuple[bool, tuple[str, ...]]:
    root = Path(directory)
    manifest = root / "SHA256SUMS"
    if not manifest.is_file():
        return False, ("missing SHA256SUMS",)

    errors: list[str] = []
    listed: set[str] = set()
    for raw in manifest.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            expected, filename = raw.split("  ", 1)
        except ValueError:
            errors.append(f"malformed manifest line: {raw}")
            continue
        if filename in listed:
            errors.append(f"duplicate manifest entry: {filename}")
            continue
        listed.add(filename)
        if not _SHA256_RE.fullmatch(expected):
            errors.append(f"invalid sha256 digest: {filename}")
            continue
        try:
            _validate_artifact_name(filename)
        except ValueError:
            errors.append(f"unsafe artifact name: {filename}")
            continue
        path = root / filename
        if not path.is_file():
            errors.append(f"missing artifact: {filename}")
            continue
        actual = sha256_bytes(path.read_bytes())
        if actual != expected:
            errors.append(f"hash mismatch: {filename}")

    actual_json = {path.name for path in root.glob("*.json") if path.is_file()}
    for filename in sorted(actual_json - listed):
        errors.append(f"unmanifested artifact: {filename}")
    for filename in sorted(listed - actual_json):
        if not (root / filename).is_file():
            # Missing artifacts are already reported above; keep this branch from
            # duplicating the same error.
            continue
    return not errors, tuple(errors)
