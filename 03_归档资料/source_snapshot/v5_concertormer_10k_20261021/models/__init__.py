"""V5 Concertormer adapter; no BasicSR package or custom CUDA imports."""
from pathlib import Path
import sys

# Small pinned pure-Python dependency shipped locally; no environment mutation.
_deps = str(Path(__file__).resolve().parents[1] / "vendor" / "python")
if _deps not in sys.path:
    sys.path.insert(0, _deps)

from .v5_concertormer import V5Concertormer, RawXi, build_model, OFFICIAL_CONFIG

__all__ = ["V5Concertormer", "RawXi", "build_model", "OFFICIAL_CONFIG"]
