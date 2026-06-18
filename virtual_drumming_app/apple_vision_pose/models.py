"""Shared data models and rolling performance statistics."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class Landmark:
    group: str
    joint: str
    x_norm: float
    y_norm: float
    x_rel: float | None
    y_rel: float | None
    x_px: float
    y_px: float
    confidence: float
    tracking_present: bool
    chirality: str | None = None
    z_rel: float | None = None


@dataclass(slots=True)
class ProcessedFrame:
    frame_id: int
    frame_bgr: Any
    landmarks: dict[str, Landmark]
    detect_seconds: float
    convert_seconds: float = 0.0
    raw_frame_bgr: Any | None = None


class TimingStats:
    def __init__(self, window_size: int = 120) -> None:
        self._values = defaultdict(lambda: deque(maxlen=window_size))
        self._last_frame_at: float | None = None

    def add(self, name: str, seconds: float) -> None:
        self._values[name].append(seconds)

    def mark_frame(self) -> None:
        now = time.perf_counter()
        if self._last_frame_at is not None:
            self.add("frame_interval", now - self._last_frame_at)
        self._last_frame_at = now

    def avg_ms(self, name: str) -> float:
        values = self._values[name]
        return 1000.0 * sum(values) / max(1, len(values))

    def serial_stage_ms(self) -> float:
        return sum(self.avg_ms(name) for name in ("read", "detect", "convert", "draw", "display"))

    def fps(self) -> float:
        frame_interval_ms = self.avg_ms("frame_interval")
        if frame_interval_ms <= 0:
            return 0.0
        return 1000.0 / frame_interval_ms

    def summary(self) -> str:
        return (
            f"fps={self.fps():.1f} "
            f"read={self.avg_ms('read'):.1f}ms "
            f"detect={self.avg_ms('detect'):.1f}ms "
            f"convert={self.avg_ms('convert'):.1f}ms "
            f"draw={self.avg_ms('draw'):.1f}ms "
            f"display={self.avg_ms('display'):.1f}ms "
            f"serial={self.serial_stage_ms():.1f}ms"
        )

    def hud_line(self) -> str:
        return (
            f"fps {self.fps():.1f} | "
            f"read {self.avg_ms('read'):.1f} | "
            f"detect {self.avg_ms('detect'):.1f} | "
            f"convert {self.avg_ms('convert'):.1f} | "
            f"draw {self.avg_ms('draw'):.1f} | "
            f"display {self.avg_ms('display'):.1f} ms"
        )
