"""Helper functions for later wrist-relative analysis."""

from __future__ import annotations

from typing import Iterable

from .canonical import canonical_id
from .schema import LandmarkObservation


def derive_palm_center(
    landmarks: dict[str, LandmarkObservation],
    side: str,
) -> tuple[float, float, float] | None:
    """Approximate palm center from stable hand landmarks when available."""

    group_name = f"{side}_hand"
    landmark_ids = [
        canonical_id(group_name, side, "wrist"),
        canonical_id(group_name, side, "index_mcp"),
        canonical_id(group_name, side, "middle_mcp"),
        canonical_id(group_name, side, "pinky_mcp"),
    ]
    points: list[tuple[float, float, float]] = []
    for landmark_id in landmark_ids:
        observation = landmarks.get(landmark_id)
        if not observation or not observation.present:
            continue
        if observation.x_norm is None or observation.y_norm is None:
            continue
        points.append(
            (
                observation.x_norm,
                observation.y_norm,
                observation.z_rel or 0.0,
            )
        )
    if not points:
        return None
    count = float(len(points))
    return (
        sum(point[0] for point in points) / count,
        sum(point[1] for point in points) / count,
        sum(point[2] for point in points) / count,
    )


def relative_vector(
    reference: LandmarkObservation | None,
    target: LandmarkObservation | None,
) -> tuple[float, float, float] | None:
    """Compute a target-minus-reference vector in normalized coordinates."""

    if not reference or not target:
        return None
    if not reference.present or not target.present:
        return None
    if reference.x_norm is None or reference.y_norm is None:
        return None
    if target.x_norm is None or target.y_norm is None:
        return None
    return (
        target.x_norm - reference.x_norm,
        target.y_norm - reference.y_norm,
        (target.z_rel or 0.0) - (reference.z_rel or 0.0),
    )


def available_landmarks(
    landmarks: dict[str, LandmarkObservation],
    ids: Iterable[str],
) -> dict[str, LandmarkObservation]:
    """Filter present landmarks from a larger dictionary."""

    filtered: dict[str, LandmarkObservation] = {}
    for landmark_id in ids:
        observation = landmarks.get(landmark_id)
        if observation and observation.present:
            filtered[landmark_id] = observation
    return filtered

