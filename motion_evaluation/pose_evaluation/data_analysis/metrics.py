from __future__ import annotations

import math

import numpy as np


def valid_pair_mask(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    return np.isfinite(reference).all(axis=1) & np.isfinite(candidate).all(axis=1)


def euclidean_distance_series(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    distances = np.full(len(reference), np.nan, dtype=float)
    mask = valid_pair_mask(reference, candidate)
    if np.any(mask):
        distances[mask] = np.linalg.norm(reference[mask] - candidate[mask], axis=1)
    return distances


def rmse(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(values**2)))


def trajectory_rmse_2d(reference: np.ndarray, candidate: np.ndarray) -> float:
    return rmse(euclidean_distance_series(reference, candidate))


def mpjpe_2d(reference_joints: dict[str, np.ndarray], candidate_joints: dict[str, np.ndarray]) -> float:
    distances: list[np.ndarray] = []
    for joint_id, reference in reference_joints.items():
        candidate = candidate_joints.get(joint_id)
        if candidate is None:
            continue
        distances.append(euclidean_distance_series(reference, candidate))
    if not distances:
        return float("nan")
    stacked = np.vstack(distances)
    return float(np.nanmean(stacked))


def pck_2d(reference_joints: dict[str, np.ndarray], candidate_joints: dict[str, np.ndarray], threshold: float) -> float:
    hits: list[np.ndarray] = []
    for joint_id, reference in reference_joints.items():
        candidate = candidate_joints.get(joint_id)
        if candidate is None:
            continue
        distances = euclidean_distance_series(reference, candidate)
        joint_hits = np.full(len(distances), np.nan, dtype=float)
        finite = np.isfinite(distances)
        joint_hits[finite] = distances[finite] <= threshold
        hits.append(joint_hits)
    if not hits:
        return float("nan")
    stacked = np.vstack(hits)
    finite = np.isfinite(stacked)
    if not np.any(finite):
        return float("nan")
    return float(np.mean(stacked[finite]))


def pearson_correlation(
    reference: np.ndarray,
    candidate: np.ndarray,
    min_valid_samples: int = 2,
    min_std: float = 1e-12,
) -> float:
    mask = np.isfinite(reference) & np.isfinite(candidate)
    if np.sum(mask) < min_valid_samples:
        return float("nan")
    reference_valid = reference[mask]
    candidate_valid = candidate[mask]
    if np.std(reference_valid) <= min_std or np.std(candidate_valid) <= min_std:
        return float("nan")
    ref = reference_valid - np.mean(reference_valid)
    cand = candidate_valid - np.mean(candidate_valid)
    denom = math.sqrt(float(np.sum(ref**2) * np.sum(cand**2)))
    if denom == 0:
        return float("nan")
    return float(np.sum(ref * cand) / denom)


def speed(series: np.ndarray) -> np.ndarray:
    if series.ndim != 2:
        raise ValueError("Expected 2D velocity array")
    return np.linalg.norm(series, axis=1)


def bland_altman_stats(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(reference) & np.isfinite(candidate)
    if np.sum(mask) == 0:
        return {
            "bias": float("nan"),
            "sd_diff": float("nan"),
            "loa_lower": float("nan"),
            "loa_upper": float("nan"),
        }
    differences = candidate[mask] - reference[mask]
    bias = float(np.mean(differences))
    sd_diff = float(np.std(differences, ddof=1)) if differences.size > 1 else float("nan")
    if np.isfinite(sd_diff):
        loa_lower = bias - 1.96 * sd_diff
        loa_upper = bias + 1.96 * sd_diff
    else:
        loa_lower = float("nan")
        loa_upper = float("nan")
    return {
        "bias": bias,
        "sd_diff": sd_diff,
        "loa_lower": float(loa_lower),
        "loa_upper": float(loa_upper),
    }


def bland_altman_paired_values(reference: np.ndarray, candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    mask = np.isfinite(reference) & np.isfinite(candidate)
    reference = reference[mask]
    candidate = candidate[mask]
    return (reference + candidate) / 2.0, candidate - reference


def residual_jitter_stats(raw_signal: np.ndarray, smoothed_signal: np.ndarray) -> dict[str, float]:
    raw_signal = np.asarray(raw_signal, dtype=float)
    smoothed_signal = np.asarray(smoothed_signal, dtype=float)
    mask = np.isfinite(raw_signal) & np.isfinite(smoothed_signal)
    if not np.any(mask):
        return {
            "jitter_rms": float("nan"),
        }
    residual = raw_signal[mask] - smoothed_signal[mask]
    return {
        "jitter_rms": float(np.sqrt(np.mean(residual**2))),
    }


def principal_axis(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    mask = np.isfinite(points).all(axis=1)
    valid_points = points[mask]
    if len(valid_points) < 2:
        return np.array([1.0, 0.0]), np.array([0.0, 1.0]), float("nan")
    centered = valid_points - np.mean(valid_points, axis=0)
    covariance = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    primary = eigenvectors[:, order[0]]
    orthogonal = eigenvectors[:, order[1]]
    explained_ratio = float(eigenvalues[order[0]] / np.sum(eigenvalues)) if np.sum(eigenvalues) > 0 else float("nan")
    return primary, orthogonal, explained_ratio


def project_onto_axis(points: np.ndarray, axis: np.ndarray) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    values = np.full(len(points), np.nan, dtype=float)
    mask = np.isfinite(points).all(axis=1)
    if np.any(mask):
        values[mask] = points[mask] @ axis
    return values


def inter_event_interval_rmse(reference_event_times: np.ndarray, candidate_event_times: np.ndarray) -> float:
    if len(reference_event_times) < 2 or len(candidate_event_times) < 2:
        return float("nan")
    reference_intervals = np.diff(reference_event_times)
    candidate_intervals = np.diff(candidate_event_times)
    length = min(len(reference_intervals), len(candidate_intervals))
    if length == 0:
        return float("nan")
    return rmse(candidate_intervals[:length] - reference_intervals[:length])
