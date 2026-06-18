"""Lazy imports for optional native and scientific dependencies."""

from __future__ import annotations

from typing import Any

cv2: Any | None = None
np: Any | None = None


def load_opencv() -> Any:
    global cv2
    if cv2 is not None:
        return cv2
    try:
        import cv2 as cv2_module
    except ImportError as exc:
        raise RuntimeError(
            "Missing OpenCV. From the repository root, run:\n"
            "  python -m pip install -r requirements.txt"
        ) from exc
    cv2 = cv2_module
    return cv2


def load_numpy() -> Any:
    global np
    if np is not None:
        return np
    try:
        import numpy as numpy_module
    except ImportError as exc:
        raise RuntimeError(
            "Missing NumPy. From the repository root, run:\n"
            "  python -m pip install -r requirements.txt"
        ) from exc
    np = numpy_module
    return np


def load_avfoundation_stack() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import AVFoundation
        import CoreMedia
        import dispatch
        import Foundation
    except ImportError as exc:
        raise RuntimeError(
            "Missing AVFoundation PyObjC bindings. From the repository root, run:\n"
            "  python -m pip install -r requirements.txt"
        ) from exc

    try:
        import CoreVideo
    except ImportError:
        import Quartz as CoreVideo

    return AVFoundation, CoreMedia, CoreVideo, Foundation, dispatch
