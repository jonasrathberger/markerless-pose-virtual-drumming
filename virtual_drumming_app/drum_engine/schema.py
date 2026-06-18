"""Generic pose data structures for drum-engine processing."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PosePoint:
    id: str
    x: float | None
    y: float | None
    confidence: float = 1.0
    tracking_present: bool = True
    x_raw: float | None = None
    y_raw: float | None = None


@dataclass(frozen=True, slots=True)
class PoseFrame:
    timestamp_seconds: float
    points: dict[str, PosePoint]
