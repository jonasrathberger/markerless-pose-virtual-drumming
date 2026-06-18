"""Helpers for current MediaPipe Tasks-based Python integrations."""

from __future__ import annotations

import logging
import urllib.error
import urllib.request
from pathlib import Path

import cv2

from utils.io import ensure_directory


LOGGER = logging.getLogger(__name__)

MEDIAPIPE_MODELS_DIR = Path("model_assets") / "mediapipe"

# Current MediaPipe Tasks model bundle layout used by the official examples.
POSE_LANDMARKER_LITE_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


def ensure_model_asset(
    *,
    output_path: str | Path | None,
    default_filename: str,
    download_url: str,
    allow_download: bool = True,
) -> Path:
    """Return a local model path, downloading it when allowed and needed."""

    if output_path is None:
        output_path = MEDIAPIPE_MODELS_DIR / default_filename
    path = Path(output_path)
    if path.exists():
        return path

    ensure_directory(path.parent)
    if not allow_download:
        raise FileNotFoundError(
            f"Required MediaPipe model asset is missing: {path}. "
            f"Download it from {download_url} and retry."
        )

    LOGGER.info("Downloading MediaPipe model asset to %s", path)
    try:
        urllib.request.urlretrieve(download_url, path)
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Unable to download required MediaPipe model asset from {download_url}. "
            f"Download it manually to {path} and retry."
        ) from exc
    return path


def mediapipe_image_from_bgr(mp, frame_bgr):
    """Convert an OpenCV BGR frame into a MediaPipe SRGB image."""

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)


def monotonic_sec_to_timestamp_ms(timestamp_monotonic_sec: float) -> int:
    """Convert floating-point monotonic seconds to MediaPipe video timestamps."""

    return max(0, int(round(timestamp_monotonic_sec * 1000.0)))
