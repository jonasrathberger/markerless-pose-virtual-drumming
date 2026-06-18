"""Image conversion and resizing helpers."""

from __future__ import annotations

import ctypes
from typing import Any

from .dependencies import load_numpy, load_opencv


def pixel_buffer_to_bgr(pixel_buffer: Any, CoreVideo: Any) -> Any:
    numpy_module = load_numpy()
    lock_flags = getattr(CoreVideo, "kCVPixelBufferLock_ReadOnly", 1)
    CoreVideo.CVPixelBufferLockBaseAddress(pixel_buffer, lock_flags)
    try:
        width = int(CoreVideo.CVPixelBufferGetWidth(pixel_buffer))
        height = int(CoreVideo.CVPixelBufferGetHeight(pixel_buffer))
        bytes_per_row = int(CoreVideo.CVPixelBufferGetBytesPerRow(pixel_buffer))
        base_address = CoreVideo.CVPixelBufferGetBaseAddress(pixel_buffer)
        if base_address is None:
            return black_frame(width=width, height=height)

        buffer_size = bytes_per_row * height
        if hasattr(base_address, "as_buffer"):
            pixel_data = base_address.as_buffer(buffer_size)
        else:
            try:
                address = int(base_address)
            except TypeError:
                address = ctypes.cast(base_address, ctypes.c_void_p).value
            if address is None:
                return black_frame(width=width, height=height)
            pixel_data = (ctypes.c_ubyte * buffer_size).from_address(address)

        bgra = numpy_module.frombuffer(pixel_data, dtype=numpy_module.uint8).reshape(height, bytes_per_row // 4, 4)
        return bgra[:, :width, :3].copy()
    finally:
        CoreVideo.CVPixelBufferUnlockBaseAddress(pixel_buffer, lock_flags)


def resize_to_request(frame_bgr: Any, *, width: int, height: int) -> Any:
    cv2 = load_opencv()
    if width <= 0 or height <= 0:
        return frame_bgr
    current_height, current_width = frame_bgr.shape[:2]
    if (current_width, current_height) == (width, height):
        return frame_bgr
    interpolation = cv2.INTER_AREA if width < current_width or height < current_height else cv2.INTER_LINEAR
    return cv2.resize(frame_bgr, (width, height), interpolation=interpolation)


def black_frame(*, width: int, height: int) -> Any:
    numpy_module = load_numpy()
    return numpy_module.zeros((height, width, 3), dtype=numpy_module.uint8)

