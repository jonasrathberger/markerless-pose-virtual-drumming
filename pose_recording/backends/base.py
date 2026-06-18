"""Abstract backend interface used by the recorder loop."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from models.schema import FrameResult
from overlays.renderer import OverlayRenderer


class BackendUnavailableError(RuntimeError):
    """Raised when a backend dependency stack is unavailable."""


class PoseBackend(ABC):
    """Backend contract for webcam pose-estimation pipelines."""

    backend_name: str = "unknown"

    def __init__(self) -> None:
        self.overlay_renderer = OverlayRenderer()

    @abstractmethod
    def initialize(self) -> None:
        """Allocate model objects and other runtime resources."""

    @abstractmethod
    def process_frame(self, frame_bgr, timestamp_monotonic_sec: float) -> FrameResult:
        """Run inference for one BGR frame and return normalized landmarks."""

    def draw_overlay(
        self,
        frame_bgr,
        result: FrameResult,
        *,
        overlay_enabled: bool = True,
        hud_lines: list[str] | None = None,
    ):
        return self.overlay_renderer.draw(
            frame_bgr,
            result,
            overlay_enabled=overlay_enabled,
            hud_lines=hud_lines,
        )

    def get_configuration(self) -> dict[str, Any]:
        return {}

    @abstractmethod
    def shutdown(self) -> None:
        """Release model resources cleanly."""

