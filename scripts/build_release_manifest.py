"""Write the final whole-repository SHA-256 release manifest."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

write_release_manifest = importlib.import_module(
    "explore_vtla.release_manifest"
).write_release_manifest


if __name__ == "__main__":
    print(write_release_manifest(ROOT))
