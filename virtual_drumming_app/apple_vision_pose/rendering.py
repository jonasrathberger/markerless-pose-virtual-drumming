"""OpenCV landmark overlay rendering."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .constants import BODY_CONNECTIONS, HAND_CONNECTIONS, TRACKED_CONNECTIONS, TRACKED_LANDMARK_IDS
from .dependencies import load_opencv
from .image_utils import black_frame
from .models import Landmark



def draw_landmarks(
    frame_bgr: Any,
    landmarks: dict[str, Landmark],
    performance_line: str | None = None,
    *,
    draw_all: bool = False,
    mirror_preview: bool = True,
    hand_drum_labels: dict[str, str] | None = None,
    text_scale: float = 1.0,
) -> Any:
    cv2 = load_opencv()
    if frame_bgr is None:
        frame_bgr = black_frame(width=640, height=480)
    annotated = cv2.flip(frame_bgr, 1) if mirror_preview else frame_bgr.copy()
    display_landmarks = mirrored_landmarks(landmarks, width=annotated.shape[1]) if mirror_preview else landmarks

    for start, end in overlay_connections(draw_all=draw_all):
        draw_connection(annotated, display_landmarks, start, end, color_for_key(start))

    for key, landmark in display_landmarks.items():
        if not draw_all and key not in TRACKED_LANDMARK_IDS:
            continue
        if not landmark.tracking_present:
            continue
        color = color_for_key(key)
        cv2.circle(
            annotated,
            (int(landmark.x_px), int(landmark.y_px)),
            4,
            color,
            thickness=-1,
            lineType=cv2.LINE_AA,
        )
        draw_coordinate_label(annotated, landmark, text_scale=text_scale)

    if hand_drum_labels:
        draw_hand_drum_labels(
            annotated,
            display_landmarks,
            hand_drum_labels,
            text_scale=text_scale,
        )

    height, width = annotated.shape[:2]
    hud_y = scaled_int(26, text_scale, minimum=16)
    draw_hud(
        annotated,
        f"Apple Vision {width}x{height} | landmarks: {len(landmarks)} | q/Esc quits",
        y_origin=hud_y,
        text_scale=text_scale,
    )
    if performance_line:
        draw_wrapped_hud(
            annotated,
            performance_line,
            y_origin=hud_y + scaled_int(24, text_scale, minimum=18),
            text_scale=text_scale,
        )
    return annotated


def mirrored_landmarks(landmarks: dict[str, Landmark], *, width: int) -> dict[str, Landmark]:
    return {
        key: replace(
            landmark,
            x_norm=1.0 - landmark.x_norm,
            x_rel=-landmark.x_rel if landmark.x_rel is not None else None,
            x_px=float(width) - landmark.x_px,
        )
        for key, landmark in landmarks.items()
    }


def overlay_connections(*, draw_all: bool) -> list[tuple[str, str]]:
    if not draw_all:
        return TRACKED_CONNECTIONS

    connections = [(f"body:{start}", f"body:{end}") for start, end in BODY_CONNECTIONS]
    for hand_group in ("hand_left", "hand_right"):
        connections.extend(
            (f"{hand_group}:{start}", f"{hand_group}:{end}")
            for start, end in HAND_CONNECTIONS
        )
    return connections


def draw_connection(
    frame_bgr: Any,
    landmarks: dict[str, Landmark],
    start_key: str,
    end_key: str,
    color: tuple[int, int, int],
) -> None:
    cv2 = load_opencv()
    start = landmarks.get(start_key)
    end = landmarks.get(end_key)
    if not start or not end or not start.tracking_present or not end.tracking_present:
        return
    cv2.line(
        frame_bgr,
        (int(start.x_px), int(start.y_px)),
        (int(end.x_px), int(end.y_px)),
        color,
        2,
        cv2.LINE_AA,
    )


def draw_coordinate_label(frame_bgr: Any, landmark: Landmark, *, text_scale: float = 1.0) -> None:
    if landmark.x_rel is None or landmark.y_rel is None:
        return

    cv2 = load_opencv()
    label = f"{landmark.x_rel:+.2f},{landmark.y_rel:+.2f}"
    font_scale = scaled_float(0.35, text_scale)
    text_thickness = scaled_int(1, text_scale)
    outline_thickness = scaled_int(2, text_scale)
    offset = scaled_int(7, text_scale)
    margin = scaled_int(2, text_scale)
    text_width, text_height = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)[0]
    x = min(max(margin, int(landmark.x_px) + offset), max(margin, frame_bgr.shape[1] - text_width - margin))
    y = min(
        max(text_height + margin, int(landmark.y_px) - offset),
        max(text_height + margin, frame_bgr.shape[0] - margin),
    )
    origin = (x, y)
    cv2.putText(
        frame_bgr,
        label,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        outline_thickness,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame_bgr,
        label,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (20, 20, 20),
        text_thickness,
        cv2.LINE_AA,
    )


def draw_hand_drum_labels(
    frame_bgr: Any,
    landmarks: dict[str, Landmark],
    hand_drum_labels: dict[str, str],
    *,
    text_scale: float = 1.0,
) -> None:
    cv2 = load_opencv()
    for side, drum in hand_drum_labels.items():
        wrist = landmarks.get(f"hand_{side}:VNHumanHandPoseObservationJointNameWrist")
        if wrist is None or not wrist.tracking_present:
            continue
        label = f"{side}: {drum}"
        font_scale = scaled_float(0.62, text_scale)
        text_thickness = scaled_int(2, text_scale)
        outline_thickness = scaled_int(4, text_scale)
        x_margin = scaled_int(10, text_scale)
        y_margin = scaled_int(8, text_scale)
        text_width, text_height = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)[0]
        x = min(
            max(2, int(wrist.x_px) + scaled_int(12, text_scale)),
            max(2, frame_bgr.shape[1] - text_width - x_margin),
        )
        y = min(
            max(text_height + x_margin, int(wrist.y_px) - scaled_int(18, text_scale)),
            max(text_height + x_margin, frame_bgr.shape[0] - y_margin),
        )
        origin = (x, y)
        cv2.putText(
            frame_bgr,
            label,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (20, 20, 20),
            outline_thickness,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame_bgr,
            label,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color_for_key(f"hand_{side}:"),
            text_thickness,
            cv2.LINE_AA,
        )


def draw_hud(frame_bgr: Any, text: str, y_origin: int = 26, *, text_scale: float = 1.0) -> None:
    cv2 = load_opencv()
    font_scale = scaled_float(0.55, text_scale)
    text_thickness = scaled_int(1, text_scale)
    outline_thickness = scaled_int(3, text_scale)
    x_origin = scaled_int(14, text_scale, minimum=14)
    cv2.putText(
        frame_bgr,
        text,
        (x_origin, y_origin),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        outline_thickness,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame_bgr,
        text,
        (x_origin, y_origin),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (20, 20, 20),
        text_thickness,
        cv2.LINE_AA,
    )


def draw_wrapped_hud(frame_bgr: Any, text: str, y_origin: int = 50, *, text_scale: float = 1.0) -> None:
    max_width = max(120, frame_bgr.shape[1] - 28)
    line_height = scaled_int(22, text_scale, minimum=18)
    for index, line in enumerate(wrap_hud_text(text, max_width=max_width, text_scale=text_scale)):
        draw_hud(frame_bgr, line, y_origin=y_origin + (index * line_height), text_scale=text_scale)


def wrap_hud_text(text: str, *, max_width: int, text_scale: float = 1.0) -> list[str]:
    cv2 = load_opencv()
    font_scale = scaled_float(0.55, text_scale)
    text_thickness = scaled_int(1, text_scale)
    parts = [part.strip() for part in text.split("|")]
    lines: list[str] = []
    current = ""

    for part in parts:
        candidate = part if not current else f"{current} | {part}"
        text_width = cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)[0][0]
        if current and text_width > max_width:
            lines.append(current)
            current = part
        else:
            current = candidate

    if current:
        lines.append(current)
    return lines


def scaled_float(value: float, text_scale: float) -> float:
    return max(0.1, value * max(0.1, text_scale))


def scaled_int(value: int, text_scale: float, *, minimum: int = 1) -> int:
    return max(minimum, round(value * max(0.1, text_scale)))


def color_for_key(key: str) -> tuple[int, int, int]:
    if key.startswith("body:"):
        return (0, 220, 255)
    if key.startswith("hand_left:"):
        return (90, 220, 90)
    if key.startswith("hand_right:"):
        return (255, 140, 60)
    return (255, 255, 255)
