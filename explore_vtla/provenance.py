"""Repository and dataset fingerprints for reproducible evidence binding."""

from __future__ import annotations

import hashlib
from pathlib import Path

DEFAULT_SOURCE_ROOTS = (
    "explore_vtla",
    "src/models",
    "src/utils/losses.py",
    "scripts/check_vtla_green.py",
    "scripts/run_vtla_release.py",
    "scripts/build_release_manifest.py",
    "pyproject.toml",
    "requirements-vtla-ci.txt",
    ".github/workflows/vtla-quality.yml",
)


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_files(start: Path) -> list[Path]:
    if start.is_file():
        return [start]
    return [
        path
        for path in sorted(start.rglob("*"))
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    ]


def repository_fingerprint(
    root: str | Path,
    source_roots: tuple[str, ...] = DEFAULT_SOURCE_ROOTS,
) -> dict[str, object]:
    base = Path(root)
    files: dict[str, str] = {}
    for relative_root in source_roots:
        start = base / relative_root
        if not start.exists():
            continue
        for path in _source_files(start):
            relative = path.relative_to(base).as_posix()
            files[relative] = hash_file(path)
    joined = "".join(f"{name}:{digest}\n" for name, digest in sorted(files.items())).encode()
    return {
        "algorithm": "sha256",
        "file_count": len(files),
        "tree_hash": hashlib.sha256(joined).hexdigest(),
        "files": files,
    }
