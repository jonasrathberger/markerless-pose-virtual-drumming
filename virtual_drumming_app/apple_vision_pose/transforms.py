"""Coordinate transforms for pose landmarks."""

from __future__ import annotations

import math

from .models import Landmark

LEFT_SHOULDER_KEY = "body:VNHumanBodyPoseObservationJointNameLeftShoulder"
RIGHT_SHOULDER_KEY = "body:VNHumanBodyPoseObservationJointNameRightShoulder"
MIN_SHOULDER_SCALE = 1e-6


def normalize_landmarks_to_shoulder_center(landmarks: dict[str, Landmark]) -> None:
    """Set body-axis shoulder-relative coordinates in-place.

    Existing ``x_norm``/``y_norm`` values stay in image coordinates. The relative
    values use the center point between both shoulders as ``(0, 0)`` and the
    shoulder-to-shoulder distance as one coordinate unit. Coordinates are
    projected onto the current shoulder line and its downward perpendicular, so
    body roll relative to the camera is normalized.
    """
    left_shoulder = landmarks.get(LEFT_SHOULDER_KEY)
    right_shoulder = landmarks.get(RIGHT_SHOULDER_KEY)
    if not _tracked(left_shoulder) or not _tracked(right_shoulder):
        for landmark in landmarks.values():
            landmark.x_rel = None
            landmark.y_rel = None
        return

    origin_x = (left_shoulder.x_norm + right_shoulder.x_norm) * 0.5
    origin_y = (left_shoulder.y_norm + right_shoulder.y_norm) * 0.5
    shoulder_dx = left_shoulder.x_norm - right_shoulder.x_norm
    shoulder_dy = left_shoulder.y_norm - right_shoulder.y_norm
    shoulder_scale = math.hypot(
        shoulder_dx,
        shoulder_dy,
    )
    if shoulder_scale <= MIN_SHOULDER_SCALE:
        for landmark in landmarks.values():
            landmark.x_rel = None
            landmark.y_rel = None
        return

    x_axis_x = shoulder_dx / shoulder_scale
    x_axis_y = shoulder_dy / shoulder_scale
    y_axis_x = -x_axis_y
    y_axis_y = x_axis_x
    if y_axis_y < 0.0:
        y_axis_x = -y_axis_x
        y_axis_y = -y_axis_y

    for landmark in landmarks.values():
        dx = landmark.x_norm - origin_x
        dy = landmark.y_norm - origin_y
        landmark.x_rel = ((dx * x_axis_x) + (dy * x_axis_y)) / shoulder_scale
        landmark.y_rel = ((dx * y_axis_x) + (dy * y_axis_y)) / shoulder_scale


def _tracked(landmark: Landmark | None) -> bool:
    return landmark is not None and landmark.tracking_present
