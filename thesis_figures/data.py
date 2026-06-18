"""Shared input and output locations for thesis figure generation."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
THESIS_DATA_ROOT = REPO_ROOT / "thesis_data"
THESIS_OUTPUT_ROOT = REPO_ROOT / "thesis_figures"


def resolve_input_dir(
    relative_dir: str | Path,
    fallback_dir: Path,
    *,
    required: tuple[str, ...] = (),
    input_root: Path = THESIS_DATA_ROOT,
) -> Path:
    """Prefer a thesis_data subdirectory when it contains the required inputs."""
    preferred = input_root / relative_dir
    if not preferred.exists():
        return fallback_dir
    if required and not all((preferred / item).exists() for item in required):
        return fallback_dir
    return preferred
