from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NormalizationResult:
    coordinates: dict[str, np.ndarray]
    origin_xy: np.ndarray
    origin_source: str
    scale: float
    scale_source: str
    rotation_angle_rad: float
    rotation_source: str
    orientation_x_sign: float
    orientation_y_sign: float
    orientation_source: str
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class SimilarityTransform:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray
    warnings: tuple[str, ...]


def midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a + b) / 2.0


def valid_mask(points: np.ndarray) -> np.ndarray:
    return np.isfinite(points).all(axis=1)


def pair_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    distances = np.full(len(a), np.nan, dtype=float)
    mask = valid_mask(a) & valid_mask(b)
    if np.any(mask):
        distances[mask] = np.linalg.norm(a[mask] - b[mask], axis=1)
    return distances


DEFAULT_ORIGIN_PRIORITY = ("shoulders", "hips", "mean_available")
DEFAULT_SCALE_PRIORITY = ("shoulder_width", "hip_width", "torso_length", "knee_width", "body_extent")


def estimate_origin(
    coordinates: dict[str, np.ndarray],
    priority: tuple[str, ...] = DEFAULT_ORIGIN_PRIORITY,
) -> tuple[np.ndarray, str]:
    frame_count = len(next(iter(coordinates.values())))

    for source in priority:
        origin = np.full((frame_count, 2), np.nan, dtype=float)
        if source == "shoulders":
            shoulder_left = coordinates.get("shoulder_left")
            shoulder_right = coordinates.get("shoulder_right")
            if shoulder_left is None or shoulder_right is None:
                continue
            mask = valid_mask(shoulder_left) & valid_mask(shoulder_right)
            origin[mask] = midpoint(shoulder_left[mask], shoulder_right[mask])
            if np.any(mask):
                return fill_missing_rows(origin), "shoulders"

        elif source == "hips":
            hip_left = coordinates.get("hip_left")
            hip_right = coordinates.get("hip_right")
            if hip_left is None or hip_right is None:
                continue
            mask = valid_mask(hip_left) & valid_mask(hip_right)
            origin[mask] = midpoint(hip_left[mask], hip_right[mask])
            if np.any(mask):
                return fill_missing_rows(origin), "hips"

        elif source == "mean_available":
            available = [points for points in coordinates.values() if points.size and np.isfinite(points).any()]
            if available:
                stacked = np.stack(available, axis=0)
                origin = np.nanmean(stacked, axis=0)
                return fill_missing_rows(origin), "mean_available"

    return np.zeros((frame_count, 2), dtype=float), "zeros"


def estimate_scale(
    coordinates: dict[str, np.ndarray],
    priority: tuple[str, ...] = DEFAULT_SCALE_PRIORITY,
) -> tuple[float, str]:
    for source in priority:
        shoulder_left = coordinates.get("shoulder_left")
        shoulder_right = coordinates.get("shoulder_right")
        hip_left = coordinates.get("hip_left")
        hip_right = coordinates.get("hip_right")

        if source == "shoulder_width" and shoulder_left is not None and shoulder_right is not None:
            distances = pair_distance(shoulder_left, shoulder_right)
            value = robust_median_positive(distances)
            if value is not None:
                return value, "shoulder_width"

        elif source == "hip_width" and hip_left is not None and hip_right is not None:
            distances = pair_distance(hip_left, hip_right)
            value = robust_median_positive(distances)
            if value is not None:
                return value, "hip_width"

        elif (
            source == "torso_length"
            and shoulder_left is not None
            and shoulder_right is not None
            and hip_left is not None
            and hip_right is not None
        ):
            shoulder_mid = midpoint(shoulder_left, shoulder_right)
            hip_mid = midpoint(hip_left, hip_right)
            distances = pair_distance(shoulder_mid, hip_mid)
            value = robust_median_positive(distances)
            if value is not None:
                return value, "torso_length"

        elif source == "knee_width":
            knee_left = coordinates.get("knee_left")
            knee_right = coordinates.get("knee_right")
            if knee_left is None or knee_right is None:
                continue
            distances = pair_distance(knee_left, knee_right)
            value = robust_median_positive(distances)
            if value is not None:
                return value, "knee_width"

        elif source == "body_extent":
            all_points = [points for points in coordinates.values() if points.size]
            if all_points:
                stacked = np.stack(all_points, axis=0)
                center = np.nanmean(stacked, axis=0)
                radial = np.linalg.norm(stacked - center[np.newaxis, :, :], axis=2)
                value = robust_median_positive(radial.reshape(-1))
                if value is not None:
                    return value, "body_extent"

    return 1.0, "fallback_unit"


def estimate_rotation_angle(coordinates: dict[str, np.ndarray]) -> tuple[float, str]:
    shoulder_left = coordinates.get("shoulder_left")
    shoulder_right = coordinates.get("shoulder_right")
    if shoulder_left is not None and shoulder_right is not None:
        angle = median_pair_angle(shoulder_left, shoulder_right)
        if angle is not None:
            return angle, "shoulders"

    hip_left = coordinates.get("hip_left")
    hip_right = coordinates.get("hip_right")
    if hip_left is not None and hip_right is not None:
        angle = median_pair_angle(hip_left, hip_right)
        if angle is not None:
            return angle, "hips"

    return 0.0, "none"


def normalize_trial_coordinates(
    coordinates: dict[str, np.ndarray],
    apply_rotation: bool,
    origin_priority: tuple[str, ...] = DEFAULT_ORIGIN_PRIORITY,
    scale_priority: tuple[str, ...] = DEFAULT_SCALE_PRIORITY,
    normalize_axes: bool = True,
) -> NormalizationResult:
    warnings: list[str] = []
    origin_xy, origin_source = estimate_origin(coordinates, origin_priority)
    scale, scale_source = estimate_scale(coordinates, scale_priority)
    if origin_source in {"mean_available", "zeros"}:
        warnings.append(f"Used fallback origin source: {origin_source}.")
    if scale_source in {"body_extent", "fallback_unit"}:
        warnings.append(f"Used fallback scale source: {scale_source}.")
    if not np.isfinite(scale) or scale <= 0:
        warnings.append("Invalid scale estimate, fell back to 1.0.")
        scale = 1.0

    rotation_angle_rad, rotation_source = estimate_rotation_angle(coordinates) if apply_rotation else (0.0, "disabled")
    rotation = np.array(
        [
            [np.cos(-rotation_angle_rad), -np.sin(-rotation_angle_rad)],
            [np.sin(-rotation_angle_rad), np.cos(-rotation_angle_rad)],
        ],
        dtype=float,
    )

    normalized: dict[str, np.ndarray] = {}
    for joint_id, points in coordinates.items():
        shifted = points - origin_xy
        scaled = shifted / scale
        normalized[joint_id] = scaled @ rotation.T if apply_rotation else scaled

    orientation_x_sign, orientation_y_sign, orientation_source = estimate_body_axis_orientation(normalized) if normalize_axes else (1.0, 1.0, "disabled")
    if normalize_axes:
        for joint_id, points in normalized.items():
            oriented = points.copy()
            oriented[:, 0] = oriented[:, 0] * orientation_x_sign
            oriented[:, 1] = oriented[:, 1] * orientation_y_sign
            normalized[joint_id] = oriented

    return NormalizationResult(
        coordinates=normalized,
        origin_xy=origin_xy,
        origin_source=origin_source,
        scale=float(scale),
        scale_source=scale_source,
        rotation_angle_rad=float(rotation_angle_rad),
        rotation_source=rotation_source,
        orientation_x_sign=float(orientation_x_sign),
        orientation_y_sign=float(orientation_y_sign),
        orientation_source=orientation_source,
        warnings=tuple(warnings),
    )


def estimate_body_axis_orientation(coordinates: dict[str, np.ndarray]) -> tuple[float, float, str]:
    """Canonicalize body axes so mirrored image/model coordinates do not leak into plots.

    The normalized 2D space uses anatomical directions rather than source coordinate
    conventions: right shoulder/hip should lie at positive x relative to left, and
    the shoulder midpoint should lie at positive y relative to the hip midpoint.
    """
    x_sign = 1.0
    y_sign = 1.0
    sources: list[str] = []

    horizontal_vector = median_side_vector(coordinates, "shoulder")
    horizontal_source = "shoulders"
    if horizontal_vector is None:
        horizontal_vector = median_side_vector(coordinates, "hip")
        horizontal_source = "hips"
    if horizontal_vector is not None and np.isfinite(horizontal_vector[0]) and horizontal_vector[0] < 0:
        x_sign = -1.0
    if horizontal_vector is not None:
        sources.append(horizontal_source)

    vertical_vector = median_midline_vector(coordinates, "hip", "shoulder")
    vertical_source = "hip_to_shoulder"
    if vertical_vector is None:
        vertical_vector = median_midline_vector(coordinates, "knee", "hip")
        vertical_source = "knee_to_hip"
    if vertical_vector is not None and np.isfinite(vertical_vector[1]) and vertical_vector[1] < 0:
        y_sign = -1.0
    if vertical_vector is not None:
        sources.append(vertical_source)

    return x_sign, y_sign, "+".join(sources) if sources else "none"


def median_side_vector(coordinates: dict[str, np.ndarray], joint_name: str) -> np.ndarray | None:
    left = coordinates.get(f"{joint_name}_left")
    right = coordinates.get(f"{joint_name}_right")
    if left is None or right is None:
        return None
    vectors = right - left
    mask = valid_mask(vectors)
    if not np.any(mask):
        return None
    return np.nanmedian(vectors[mask], axis=0)


def median_midline_vector(coordinates: dict[str, np.ndarray], lower_joint: str, upper_joint: str) -> np.ndarray | None:
    lower_left = coordinates.get(f"{lower_joint}_left")
    lower_right = coordinates.get(f"{lower_joint}_right")
    upper_left = coordinates.get(f"{upper_joint}_left")
    upper_right = coordinates.get(f"{upper_joint}_right")
    if lower_left is None or lower_right is None or upper_left is None or upper_right is None:
        return None
    lower_mid = midpoint(lower_left, lower_right)
    upper_mid = midpoint(upper_left, upper_right)
    vectors = upper_mid - lower_mid
    mask = valid_mask(vectors)
    if not np.any(mask):
        return None
    return np.nanmedian(vectors[mask], axis=0)


def fit_similarity_transform(
    reference_coordinates: dict[str, np.ndarray],
    source_coordinates: dict[str, np.ndarray],
    joint_ids: tuple[str, ...] | None = None,
) -> SimilarityTransform:
    reference_points: list[np.ndarray] = []
    source_points: list[np.ndarray] = []
    selected_joint_ids = tuple(reference_coordinates) if joint_ids is None else joint_ids
    for joint_id in selected_joint_ids:
        reference = reference_coordinates.get(joint_id)
        if reference is None:
            continue
        source = source_coordinates.get(joint_id)
        if source is None:
            continue
        mask = valid_mask(reference) & valid_mask(source)
        if not np.any(mask):
            continue
        reference_points.append(reference[mask])
        source_points.append(source[mask])

    if not reference_points or not source_points:
        return SimilarityTransform(1.0, np.eye(2), np.zeros(2), ("No valid points for similarity alignment.",))

    reference_stack = np.vstack(reference_points)
    source_stack = np.vstack(source_points)
    ref_centroid = reference_stack.mean(axis=0)
    src_centroid = source_stack.mean(axis=0)
    ref_centered = reference_stack - ref_centroid
    src_centered = source_stack - src_centroid
    covariance = src_centered.T @ ref_centered / len(reference_stack)
    u, singular_values, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = u @ vt

    src_var = np.mean(np.sum(src_centered**2, axis=1))
    scale = float(np.sum(singular_values) / src_var) if src_var > 0 else 1.0
    translation = ref_centroid - scale * (src_centroid @ rotation)
    return SimilarityTransform(scale=scale, rotation=rotation, translation=translation, warnings=tuple())


def apply_similarity_transform_to_trial(
    coordinates: dict[str, np.ndarray],
    transform: SimilarityTransform,
) -> dict[str, np.ndarray]:
    transformed: dict[str, np.ndarray] = {}
    for joint_id, points in coordinates.items():
        transformed[joint_id] = transform.scale * (points @ transform.rotation) + transform.translation
    return transformed


def fill_missing_rows(points: np.ndarray) -> np.ndarray:
    filled = points.copy()
    valid_rows = np.isfinite(filled).all(axis=1)
    if not np.any(valid_rows):
        return np.zeros_like(filled)
    valid_indices = np.flatnonzero(valid_rows)
    first_valid = valid_indices[0]
    filled[:first_valid] = filled[first_valid]
    for start, end in zip(valid_indices[:-1], valid_indices[1:]):
        if end == start + 1:
            continue
        delta = filled[end] - filled[start]
        steps = end - start
        for offset in range(1, steps):
            filled[start + offset] = filled[start] + delta * (offset / steps)
    last_valid = valid_indices[-1]
    filled[last_valid + 1 :] = filled[last_valid]
    return filled


def robust_median_positive(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return None
    return float(np.median(values))


def median_pair_angle(a: np.ndarray, b: np.ndarray) -> float | None:
    mask = valid_mask(a) & valid_mask(b)
    if not np.any(mask):
        return None
    vectors = b[mask] - a[mask]
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    return float(np.median(angles))
