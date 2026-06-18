"""Runtime environment tweaks for local scientific Python stacks."""

from __future__ import annotations

import os
from pathlib import Path

from utils.io import ensure_directory


def configure_local_cache_environment() -> None:
    """Route cache-heavy libraries to writable local directories."""

    cache_root = ensure_directory(Path(".runtime_cache"))
    matplotlib_dir = ensure_directory(cache_root / "matplotlib")
    xdg_cache_dir = ensure_directory(cache_root / "xdg")
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_dir.resolve()))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache_dir.resolve()))
