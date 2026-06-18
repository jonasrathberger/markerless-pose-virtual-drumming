"""Drawing routines for normalized skeleton overlays and HUD information."""

from __future__ import annotations

from dataclasses import dataclass

import cv2

from models.canonical import OVERLAY_CONNECTIONS
from models.schema import FrameResult, LandmarkObservation


@dataclass(slots=True)
class OverlayRenderer:
    line_thickness: int = 2
    point_radius: int = 3
    font_scale: float = 0.55

    def draw(
        self,
        frame_bgr,
        result: FrameResult,
        *,
        overlay_enabled: bool = True,
        hud_lines: list[str] | None = None,
    ):
        annotated = frame_bgr.copy()
        if overlay_enabled:
            self._draw_connections(annotated, result)
            self._draw_points(annotated, result)
            self._draw_hand_labels(annotated, result)
        if hud_lines:
            self._draw_hud(annotated, hud_lines)
        return annotated

    def _draw_connections(self, frame_bgr, result: FrameResult) -> None:
        for start_id, end_id in OVERLAY_CONNECTIONS:
            start = result.landmarks.get(start_id)
            end = result.landmarks.get(end_id)
            if not self._is_drawable(start) or not self._is_drawable(end):
                continue
            color = self._color_for_landmark(start_id)
            cv2.line(
                frame_bgr,
                (int(start.x_px), int(start.y_px)),
                (int(end.x_px), int(end.y_px)),
                color,
                self.line_thickness,
                cv2.LINE_AA,
            )

    def _draw_points(self, frame_bgr, result: FrameResult) -> None:
        for landmark_id, observation in result.landmarks.items():
            if not self._is_drawable(observation):
                continue
            cv2.circle(
                frame_bgr,
                (int(observation.x_px), int(observation.y_px)),
                self.point_radius,
                self._color_for_landmark(landmark_id),
                thickness=-1,
                lineType=cv2.LINE_AA,
            )

    def _draw_hud(self, frame_bgr, hud_lines: list[str]) -> None:
        x_origin = 14
        y_origin = 22
        line_height = 22
        for index, line in enumerate(hud_lines):
            y_coord = y_origin + (index * line_height)
            cv2.putText(
                frame_bgr,
                line,
                (x_origin, y_coord),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame_bgr,
                line,
                (x_origin, y_coord),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                (30, 30, 30),
                1,
                cv2.LINE_AA,
            )

    def _draw_hand_labels(self, frame_bgr, result: FrameResult) -> None:
        wrist_ids = {
            "left_hand:left:wrist": "L hand",
            "right_hand:right:wrist": "R hand",
        }
        for landmark_id, label in wrist_ids.items():
            observation = result.landmarks.get(landmark_id)
            if not self._is_drawable(observation):
                continue
            cv2.putText(
                frame_bgr,
                label,
                (int(observation.x_px) + 8, int(observation.y_px) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    @staticmethod
    def _is_drawable(observation: LandmarkObservation | None) -> bool:
        if not observation or not observation.present:
            return False
        return observation.x_px is not None and observation.y_px is not None

    @staticmethod
    def _color_for_landmark(landmark_id: str) -> tuple[int, int, int]:
        if landmark_id.startswith("body:"):
            return (0, 220, 255)
        if landmark_id.startswith("left_hand:"):
            return (90, 220, 90)
        if landmark_id.startswith("right_hand:"):
            return (255, 140, 60)
        return (255, 255, 255)

