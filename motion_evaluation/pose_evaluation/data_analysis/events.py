from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EventDetectionResult:
    maxima_indices: np.ndarray
    minima_indices: np.ndarray
    reversal_indices: np.ndarray
    velocity_zero_crossings: np.ndarray


@dataclass(frozen=True)
class EventMatchResult:
    matched_reference_indices: np.ndarray
    matched_candidate_indices: np.ndarray
    matched_time_errors_sec: np.ndarray
    missed_count: int
    extra_count: int


def detect_events(
    signal: np.ndarray,
    velocity: np.ndarray,
    time_sec: np.ndarray,
    min_distance_sec: float,
    prominence_ratio: float,
) -> EventDetectionResult:
    min_distance_samples = max(int(round(min_distance_sec / median_dt(time_sec))), 1)
    threshold = prominence_ratio * float(np.nanstd(signal))
    maxima = filter_extrema(local_extrema_indices(signal, "max"), signal, min_distance_samples, "max", threshold)
    minima = filter_extrema(local_extrema_indices(signal, "min"), signal, min_distance_samples, "min", threshold)
    reversal = np.sort(np.unique(np.concatenate([maxima, minima])))
    zero_crossings = zero_crossing_indices(velocity)
    return EventDetectionResult(maxima, minima, reversal, zero_crossings)


def match_event_times(reference_times: np.ndarray, candidate_times: np.ndarray, tolerance_sec: float) -> EventMatchResult:
    matched_reference: list[int] = []
    matched_candidate: list[int] = []
    matched_errors: list[float] = []
    used_candidate: set[int] = set()

    for ref_index, ref_time in enumerate(reference_times):
        best_index = None
        best_delta = None
        for cand_index, cand_time in enumerate(candidate_times):
            if cand_index in used_candidate:
                continue
            delta = abs(float(cand_time) - float(ref_time))
            if delta > tolerance_sec:
                continue
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_index = cand_index
        if best_index is None:
            continue
        used_candidate.add(best_index)
        matched_reference.append(ref_index)
        matched_candidate.append(best_index)
        matched_errors.append(float(candidate_times[best_index] - ref_time))

    return EventMatchResult(
        matched_reference_indices=np.asarray(matched_reference, dtype=int),
        matched_candidate_indices=np.asarray(matched_candidate, dtype=int),
        matched_time_errors_sec=np.asarray(matched_errors, dtype=float),
        missed_count=max(len(reference_times) - len(matched_reference), 0),
        extra_count=max(len(candidate_times) - len(matched_candidate), 0),
    )


def local_extrema_indices(signal: np.ndarray, mode: str) -> np.ndarray:
    indices: list[int] = []
    for index in range(1, len(signal) - 1):
        current = signal[index]
        prev_value = signal[index - 1]
        next_value = signal[index + 1]
        if not np.isfinite([prev_value, current, next_value]).all():
            continue
        if mode == "max" and current > prev_value and current >= next_value:
            indices.append(index)
        elif mode == "min" and current < prev_value and current <= next_value:
            indices.append(index)
    return np.asarray(indices, dtype=int)


def filter_extrema(
    candidate_indices: np.ndarray,
    signal: np.ndarray,
    min_distance_samples: int,
    mode: str,
    threshold: float,
) -> np.ndarray:
    if candidate_indices.size == 0:
        return candidate_indices
    center = float(np.nanmedian(signal))
    if threshold > 0:
        if mode == "max":
            candidate_indices = np.asarray([index for index in candidate_indices if signal[index] >= center + threshold], dtype=int)
        else:
            candidate_indices = np.asarray([index for index in candidate_indices if signal[index] <= center - threshold], dtype=int)
    if candidate_indices.size == 0:
        return candidate_indices

    kept: list[int] = [int(candidate_indices[0])]
    for index in candidate_indices[1:]:
        if index - kept[-1] >= min_distance_samples:
            kept.append(int(index))
            continue
        previous = kept[-1]
        if mode == "max" and signal[index] > signal[previous]:
            kept[-1] = int(index)
        elif mode == "min" and signal[index] < signal[previous]:
            kept[-1] = int(index)
    return np.asarray(kept, dtype=int)


def zero_crossing_indices(signal: np.ndarray) -> np.ndarray:
    finite = np.isfinite(signal)
    indices: list[int] = []
    for index in range(1, len(signal)):
        if not finite[index - 1] or not finite[index]:
            continue
        if signal[index - 1] == 0:
            indices.append(index - 1)
        elif signal[index] == 0:
            indices.append(index)
        elif np.sign(signal[index - 1]) != np.sign(signal[index]):
            indices.append(index)
    return np.asarray(sorted(set(indices)), dtype=int)


def median_dt(time_sec: np.ndarray) -> float:
    if len(time_sec) < 2:
        return 1.0
    diffs = np.diff(time_sec)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    return float(np.median(diffs)) if diffs.size else 1.0
