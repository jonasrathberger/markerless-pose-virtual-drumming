"""Adapters from Apple Vision landmark output into drum engine frames."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .schema import PoseFrame, PosePoint

if TYPE_CHECKING:
    from apple_vision_pose.models import Landmark


def apple_vision_landmarks_to_pose_frame(
    landmarks: dict[str, "Landmark"],
    *,
    timestamp_seconds: float,
) -> PoseFrame:
    """Convert Apple Vision landmarks into drum-engine pose coordinates.

    ``x_rel``/``y_rel`` are expected to be shoulder-center, shoulder-scaled
    coordinates projected into the body axis. The x axis is flipped so positive
    x means the musician's right side in the mirrored preview.
    """
    points: dict[str, PosePoint] = {}
    for key, landmark in landmarks.items():
        if not landmark.tracking_present or landmark.x_rel is None or landmark.y_rel is None:
            continue
        points[key] = PosePoint(
            id=key,
            x=-landmark.x_rel,
            y=landmark.y_rel,
            confidence=landmark.confidence,
            tracking_present=landmark.tracking_present,
            x_raw=landmark.x_norm,
            y_raw=landmark.y_norm,
        )
    return PoseFrame(timestamp_seconds=timestamp_seconds, points=points)
