"""Landmark smoothing filters for drum hit-detection input."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .schema import PoseFrame, PosePoint


@dataclass(frozen=True, slots=True)
class SmoothingConfig:
    min_cutoff: float = 1.0
    beta: float = 0.04
    derivative_cutoff: float = 1.0
    max_missing_frames: int = 5


@dataclass(slots=True)
class _PointFilterState:
    x_filter: "_OneEuroFilter"
    y_filter: "_OneEuroFilter"
    x_raw_filter: "_OneEuroFilter"
    y_raw_filter: "_OneEuroFilter"
    missing_frames: int = 0


class LandmarkSmoother:
    def __init__(self, config: SmoothingConfig | None = None) -> None:
        self.config = config or SmoothingConfig()
        self._states: dict[str, _PointFilterState] = {}

    def update(self, frame: PoseFrame) -> PoseFrame:
        smoothed_points: dict[str, PosePoint] = {}
        updated_ids: set[str] = set()

        for point_id, point in frame.points.items():
            if not _valid_point(point):
                continue

            state = self._states.get(point_id)
            if state is None:
                state = _PointFilterState(
                    x_filter=_OneEuroFilter(self.config),
                    y_filter=_OneEuroFilter(self.config),
                    x_raw_filter=_OneEuroFilter(self.config),
                    y_raw_filter=_OneEuroFilter(self.config),
                )
                self._states[point_id] = state

            state.missing_frames = 0
            updated_ids.add(point_id)
            x_raw = (
                state.x_raw_filter.filter(float(point.x_raw), frame.timestamp_seconds)
                if point.x_raw is not None
                else None
            )
            y_raw = (
                state.y_raw_filter.filter(float(point.y_raw), frame.timestamp_seconds)
                if point.y_raw is not None
                else None
            )
            smoothed_points[point_id] = PosePoint(
                id=point.id,
                x=state.x_filter.filter(float(point.x), frame.timestamp_seconds),
                y=state.y_filter.filter(float(point.y), frame.timestamp_seconds),
                confidence=point.confidence,
                tracking_present=point.tracking_present,
                x_raw=x_raw,
                y_raw=y_raw,
            )

        self._age_missing_states(updated_ids)
        return PoseFrame(timestamp_seconds=frame.timestamp_seconds, points=smoothed_points)

    def reset(self) -> None:
        self._states.clear()

    def _age_missing_states(self, updated_ids: set[str]) -> None:
        expired_ids: list[str] = []
        for point_id, state in self._states.items():
            if point_id in updated_ids:
                continue
            state.missing_frames += 1
            if state.missing_frames > self.config.max_missing_frames:
                expired_ids.append(point_id)

        for point_id in expired_ids:
            del self._states[point_id]


@dataclass(slots=True)
class _LowPassFilter:
    value: float | None = None

    def filter(self, value: float, alpha: float) -> float:
        if self.value is None:
            self.value = value
            return value
        self.value = (alpha * value) + ((1.0 - alpha) * self.value)
        return self.value


class _OneEuroFilter:
    def __init__(self, config: SmoothingConfig) -> None:
        self.config = config
        self._value_filter = _LowPassFilter()
        self._derivative_filter = _LowPassFilter()
        self._last_timestamp: float | None = None

    def filter(self, value: float, timestamp_seconds: float) -> float:
        dt = self._time_delta(timestamp_seconds)
        previous_value = self._value_filter.value
        derivative = 0.0 if previous_value is None else (value - previous_value) / dt
        derivative_hat = self._derivative_filter.filter(
            derivative,
            _alpha(dt, self.config.derivative_cutoff),
        )
        cutoff = self.config.min_cutoff + (self.config.beta * abs(derivative_hat))
        value_hat = self._value_filter.filter(value, _alpha(dt, cutoff))
        self._last_timestamp = timestamp_seconds
        return value_hat

    def _time_delta(self, timestamp_seconds: float) -> float:
        if self._last_timestamp is None:
            return 1.0 / 60.0
        dt = timestamp_seconds - self._last_timestamp
        if dt <= 0.0:
            return 1.0 / 60.0
        return dt


def _alpha(dt: float, cutoff: float) -> float:
    cutoff = max(cutoff, 1e-6)
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + (tau / dt))


def _valid_point(point: PosePoint) -> bool:
    return point.tracking_present and point.x is not None and point.y is not None
