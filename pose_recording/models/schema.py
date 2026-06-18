"""Normalized per-frame data structures shared by every backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


LANDMARK_EXPORT_COLUMNS = [
    "session_id",
    "model_name",
    "frame_index",
    "timestamp_monotonic_sec",
    "timestamp_wallclock_iso",
    "image_width",
    "image_height",
    "person_id",
    "landmark_name",
    "landmark_group",
    "side",
    "x_norm",
    "y_norm",
    "z_rel",
    "x_px",
    "y_px",
    "confidence",
    "visibility",
    "tracking_present",
]


@dataclass(frozen=True, slots=True)
class LandmarkSpec:
    """Defines a canonical landmark identifier and metadata."""

    landmark_id: str
    landmark_name: str
    landmark_group: str
    side: str


@dataclass(slots=True)
class LandmarkObservation:
    """Normalized output for one landmark."""

    x_norm: float | None = None
    y_norm: float | None = None
    z_rel: float | None = None
    x_px: float | None = None
    y_px: float | None = None
    confidence: float | None = None
    visibility: float | None = None
    present: bool = False


@dataclass(slots=True)
class FrameResult:
    """Normalized frame output returned by each backend."""

    landmarks: dict[str, LandmarkObservation] = field(default_factory=dict)
    person_id: int | None = 0
    backend_metadata: dict[str, Any] = field(default_factory=dict)

    def present_landmark_count(self) -> int:
        return sum(1 for observation in self.landmarks.values() if observation.present)

