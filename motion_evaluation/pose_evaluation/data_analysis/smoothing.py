from __future__ import annotations

from dataclasses import dataclass

import numpy as np


try:
    from scipy.signal import savgol_filter
except Exception:  # pragma: no cover - fallback when scipy is unavailable
    savgol_filter = None


@dataclass(frozen=True)
class SmoothingResult:
    smoothed: dict[str, np.ndarray]
    velocities: dict[str, np.ndarray]
    warnings: tuple[str, ...]


def smooth_and_differentiate(
    coordinates: dict[str, np.ndarray],
    time_sec: np.ndarray,
    method: str,
    window_sec: float,
    polyorder: int,
) -> SmoothingResult:
    warnings: list[str] = []
    if time_sec.size < 3:
        return SmoothingResult(dict(coordinates), {}, ("Too few samples for smoothing.",))

    dt = float(np.median(np.diff(time_sec)))
    if dt <= 0:
        positive_diffs = np.diff(time_sec)
        positive_diffs = positive_diffs[np.isfinite(positive_diffs) & (positive_diffs > 0)]
        dt = float(np.mean(positive_diffs)) if positive_diffs.size else 1.0
    window_length = max(int(round(window_sec / dt)), 3)
    if window_length % 2 == 0:
        window_length += 1
    if window_length > len(time_sec):
        window_length = len(time_sec) if len(time_sec) % 2 == 1 else len(time_sec) - 1
    if window_length < 3:
        window_length = 3
    if window_length <= polyorder:
        window_length = polyorder + 2 if (polyorder + 2) % 2 == 1 else polyorder + 3

    smoothed: dict[str, np.ndarray] = {}
    velocities: dict[str, np.ndarray] = {}

    for joint_id, points in coordinates.items():
        joint_smoothed = np.full_like(points, np.nan, dtype=float)
        joint_velocity = np.full_like(points, np.nan, dtype=float)
        for axis_index in range(points.shape[1]):
            series = points[:, axis_index]
            if not np.isfinite(series).any():
                continue
            filled = fill_nans(series)
            if method == "savitzky_golay" and savgol_filter is not None and window_length > polyorder:
                filtered = savgol_filter(filled, window_length=window_length, polyorder=polyorder, mode="interp")
            else:
                if method == "savitzky_golay" and savgol_filter is None:
                    warnings.append("SciPy is unavailable; used moving-average smoothing instead of Savitzky-Golay.")
                filtered = moving_average(filled, window_length)
            joint_smoothed[:, axis_index] = filtered
            joint_velocity[:, axis_index] = np.gradient(filtered, time_sec)
        smoothed[joint_id] = joint_smoothed
        velocities[joint_id] = joint_velocity

    return SmoothingResult(
        smoothed=smoothed,
        velocities=velocities,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def fill_nans(series: np.ndarray) -> np.ndarray:
    filled = np.array(series, dtype=float, copy=True)
    valid = np.isfinite(filled)
    if not np.any(valid):
        return np.zeros_like(filled)
    indices = np.arange(len(filled))
    filled[~valid] = np.interp(indices[~valid], indices[valid], filled[valid])
    return filled


def moving_average(series: np.ndarray, window_length: int) -> np.ndarray:
    if window_length <= 1:
        return series.copy()
    kernel = np.ones(window_length, dtype=float) / window_length
    padded = np.pad(series, (window_length // 2,), mode="edge")
    return np.convolve(padded, kernel, mode="valid")
