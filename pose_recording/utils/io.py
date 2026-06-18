"""Filesystem helpers."""

from __future__ import annotations

from pathlib import Path


def ensure_directory(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj

