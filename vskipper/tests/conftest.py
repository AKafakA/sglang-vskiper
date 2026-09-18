"""Bind standalone tool imports to this checkout during remote validation."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for path in reversed((
    ROOT / "vskipper/src",
    ROOT / "python",
    ROOT / "vskipper/src/vskipper/experiments",
    ROOT / "vskipper/src/vskipper/experiments/gates",
    ROOT / "vskipper/src/vskipper/analysis",
    ROOT / "vskipper/tests",
)):
    sys.path.insert(0, str(path))
