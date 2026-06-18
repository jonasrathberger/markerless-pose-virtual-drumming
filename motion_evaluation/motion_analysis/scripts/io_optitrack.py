from __future__ import annotations

from pathlib import Path

from io_common import parse_motion_csv
from motion_types import MotionRecording


def load_optitrack_csv(path: str | Path) -> MotionRecording:
    """Load one OptiTrack CSV export."""
    return parse_motion_csv(path=path, source="optitrack")
