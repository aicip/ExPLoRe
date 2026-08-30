"""Whole-repository SHA-256 manifests for release-archive integrity."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path

_MANIFEST_NAME = "MANIFEST.sha256"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_EXCLUDED_PARTS = {
    ".git",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "__pycache__",
    ".venv",
    "venv",
}
_EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".zip"}


@dataclass(frozen=True)
class ManifestVerification:
    verified: bool
    checked_files: int
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _included(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    if path.name == _MANIFEST_NAME:
        return False
    if any(part in _EXCLUDED_PARTS for part in relative.parts):
        return False
    if path.suffix.lower() in _EXCLUDED_SUFFIXES:
        return False
    return path.is_file()


def release_files(root: str | Path) -> tuple[Path, ...]:
    base = Path(root).resolve()
    return tuple(
        path
        for path in sorted(base.rglob("*"))
        if _included(path, base)
    )


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_release_manifest(root: str | Path) -> Path:
    base = Path(root).resolve()
    manifest = base / _MANIFEST_NAME
    lines = [
        f"{_hash_file(path)}  {path.relative_to(base).as_posix()}\n"
        for path in release_files(base)
    ]
    manifest.write_text("".join(lines), encoding="utf-8", newline="\n")
    return manifest


def verify_release_manifest(root: str | Path, *, strict: bool = True) -> ManifestVerification:
    base = Path(root).resolve()
    manifest = base / _MANIFEST_NAME
    if not manifest.is_file():
        return ManifestVerification(False, 0, ("missing MANIFEST.sha256",))

    errors: list[str] = []
    listed: dict[str, str] = {}
    for raw in manifest.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        try:
            expected, relative = raw.split("  ", 1)
        except ValueError:
            errors.append(f"malformed manifest line: {raw}")
            continue
        if not _SHA256_RE.fullmatch(expected):
            errors.append(f"invalid sha256 digest: {relative}")
            continue
        rel_path = Path(relative)
        if rel_path.is_absolute() or ".." in rel_path.parts or relative in listed:
            errors.append(f"unsafe or duplicate manifest path: {relative}")
            continue
        listed[relative] = expected
        path = base / rel_path
        if not path.is_file():
            errors.append(f"missing file: {relative}")
            continue
        actual = _hash_file(path)
        if actual != expected:
            errors.append(f"hash mismatch: {relative}")

    if strict:
        actual_paths = {path.relative_to(base).as_posix() for path in release_files(base)}
        listed_paths = set(listed)
        for relative in sorted(actual_paths - listed_paths):
            errors.append(f"unmanifested release file: {relative}")
        for relative in sorted(listed_paths - actual_paths):
            if not (base / relative).is_file():
                continue
            errors.append(f"manifested path is excluded from release set: {relative}")

    return ManifestVerification(not errors, len(listed), tuple(errors))
