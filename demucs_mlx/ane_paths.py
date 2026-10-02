"""Where this package keeps its Neural Engine assets, and how to name it to users."""
from __future__ import annotations

from pathlib import Path

#: The pip distribution whose extras install the Core ML runtime and converter.
DISTRIBUTION = "demucs-mlx"
#: The module that converts the Core ML assets (``python -m <module> convert``).
ANE_MODULE = "demucs_mlx.ane"


def ane_cache_dir() -> Path:
    """Converted Core ML assets and the compiled native bridge."""
    return Path.home() / ".cache" / "demucs-mlx" / "ane"
