"""OpenCV camera setup helpers with macOS built-in camera preference."""

from __future__ import annotations

import logging
import platform
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np


LOGGER = logging.getLogger(__name__)

BUILTIN_CAMERA_PATTERNS = (
    "facetime",
    "face time",
    "built-in",
    "builtin",
    "integrated",
    "macbook",
)
EXTERNAL_CAMERA_PATTERNS = (
    "iphone",
    "continuity",
    "desk view",
    "ipad",
    "external",
    "usb",
)


@dataclass(frozen=True, slots=True)
class CameraDeviceInfo:
    index: int
    name: str


class FFmpegAVFoundationCapture:
    """Read raw frames from a specific macOS AVFoundation device via FFmpeg."""

    def __init__(self, camera_index: int, width: int, height: int, fps: int) -> None:
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_bytes = width * height * 3
        self._lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._latest_frame_id = 0
        self._last_read_frame_id = 0
        self._stop_event = threading.Event()
        self.process = self._start_process()
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()

    def _start_process(self):
        command = [
            "ffmpeg",
            "-f",
            "avfoundation",
            "-framerate",
            str(self.fps),
            "-video_size",
            f"{self.width}x{self.height}",
            "-i",
            f"{self.camera_index}:none",
            "-an",
            "-sn",
            "-pix_fmt",
            "bgr24",
            "-vcodec",
            "rawvideo",
            "-f",
            "rawvideo",
            "pipe:1",
            "-loglevel",
            "error",
            "-nostdin",
        ]
        try:
            return subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=self.frame_bytes * 2,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "FFmpeg is required for deterministic macOS camera capture but was not found in PATH."
            ) from exc

    def isOpened(self) -> bool:
        return self.process.poll() is None and self.process.stdout is not None

    def read(self) -> tuple[bool, np.ndarray | None]:
        deadline = time.time() + 2.0
        while time.time() < deadline:
            with self._lock:
                if self._latest_frame is not None and self._latest_frame_id != self._last_read_frame_id:
                    self._last_read_frame_id = self._latest_frame_id
                    return True, self._latest_frame.copy()
            if not self.isOpened():
                break
            time.sleep(0.001)
        return False, None

    def release(self) -> None:
        self._stop_event.set()
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=1.0)
        if self._reader_thread.is_alive():
            self._reader_thread.join(timeout=1.0)

    def _reader_loop(self) -> None:
        if self.process.stdout is None:
            return
        while not self._stop_event.is_set() and self.isOpened():
            raw_frame = self._read_exact(self.frame_bytes)
            if raw_frame is None:
                break
            frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((self.height, self.width, 3)).copy()
            with self._lock:
                self._latest_frame = frame
                self._latest_frame_id += 1

    def _read_exact(self, size: int) -> bytes | None:
        if self.process.stdout is None:
            return None
        chunks = bytearray()
        while len(chunks) < size and not self._stop_event.is_set():
            chunk = self.process.stdout.read(size - len(chunks))
            if not chunk:
                return None
            chunks.extend(chunk)
        return bytes(chunks) if len(chunks) == size else None


def open_camera(camera_index: int, width: int, height: int, fps: int) -> cv2.VideoCapture:
    if platform.system() == "Darwin" and shutil.which("ffmpeg"):
        capture = FFmpegAVFoundationCapture(camera_index, width, height, fps)
        if capture.isOpened():
            LOGGER.info("Using FFmpeg AVFoundation capture for camera index %s", camera_index)
            return capture

    capture = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
    if not capture.isOpened():
        capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open webcam at index {camera_index}.")

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, fps)
    return capture


def get_capture_resolution(capture: cv2.VideoCapture) -> tuple[int, int]:
    if hasattr(capture, "width") and hasattr(capture, "height"):
        return int(capture.width), int(capture.height)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return width, height


def select_camera(
    *,
    requested_index: int | None,
    prefer_builtin_macbook_camera: bool = True,
    allow_external_camera: bool = False,
) -> CameraDeviceInfo:
    """Resolve which camera index to open on the current machine."""

    if platform.system() != "Darwin":
        if requested_index is None:
            return CameraDeviceInfo(index=0, name="Default camera")
        return CameraDeviceInfo(index=requested_index, name=f"Camera {requested_index}")

    discovered_devices = discover_macos_video_devices()
    if requested_index is not None:
        requested_name = next((device.name for device in discovered_devices if device.index == requested_index), None)
        if requested_name is None and prefer_builtin_macbook_camera and not allow_external_camera:
            raise RuntimeError(
                f"Camera index {requested_index} was requested, but macOS device enumeration could not verify which "
                "physical camera it maps to. The app refuses to guess because Continuity Camera may be selected. "
                "Install `pyobjc-framework-AVFoundation` for reliable device identification, or pass "
                "`--allow-external-camera` if you want to override this safety check."
            )
        if requested_name and prefer_builtin_macbook_camera and not allow_external_camera and _looks_external(requested_name):
            raise RuntimeError(
                f"Camera index {requested_index} maps to '{requested_name}', which is not the built-in MacBook camera. "
                "Use a built-in device index or pass --allow-external-camera."
            )
        return CameraDeviceInfo(index=requested_index, name=requested_name or f"Camera {requested_index}")

    if not prefer_builtin_macbook_camera:
        if discovered_devices:
            return discovered_devices[0]
        return CameraDeviceInfo(index=0, name="Default camera")

    builtin_candidates = [device for device in discovered_devices if _looks_builtin(device.name)]
    if builtin_candidates:
        return builtin_candidates[0]

    external_candidates = [device for device in discovered_devices if _looks_external(device.name)]
    if external_candidates and not allow_external_camera:
        names = ", ".join(f"{device.index}:{device.name}" for device in discovered_devices)
        raise RuntimeError(
            "Unable to identify a built-in MacBook camera, and macOS appears to expose only external/Continuity "
            f"devices: {names}. Disable Continuity Camera, connect the built-in camera, or pass "
            "`--allow-external-camera --camera-index <index>` explicitly."
        )

    if discovered_devices:
        names = ", ".join(f"{device.index}:{device.name}" for device in discovered_devices)
        raise RuntimeError(
            "Unable to confidently identify the built-in MacBook camera from the available macOS devices: "
            f"{names}. Install `pyobjc-framework-AVFoundation` for more reliable enumeration or pass "
            "`--allow-external-camera --camera-index <index>` if you want to override this safety check."
        )

    raise RuntimeError(
        "Unable to enumerate macOS video devices, so the app will not guess and risk opening Continuity Camera. "
        "Install `pyobjc-framework-AVFoundation`, or pass `--allow-external-camera --camera-index <index>` "
        "to bypass built-in camera enforcement."
    )


def discover_macos_video_devices() -> list[CameraDeviceInfo]:
    """Best-effort enumeration of macOS video devices."""

    devices = _discover_devices_via_avfoundation()
    if devices:
        return devices
    devices = _discover_devices_via_ffmpeg()
    if devices:
        return devices
    return []


def _discover_devices_via_avfoundation() -> list[CameraDeviceInfo]:
    try:
        import AVFoundation
    except ImportError:
        return []

    try:
        devices = AVFoundation.AVCaptureDevice.devicesWithMediaType_(AVFoundation.AVMediaTypeVideo)
    except Exception as exc:  # pragma: no cover - best effort only
        LOGGER.debug("AVFoundation device enumeration failed: %s", exc)
        return []

    results: list[CameraDeviceInfo] = []
    for index, device in enumerate(devices):
        try:
            name = str(device.localizedName())
        except Exception:
            name = f"Camera {index}"
        results.append(CameraDeviceInfo(index=index, name=name))
    return results


def _discover_devices_via_ffmpeg() -> list[CameraDeviceInfo]:
    try:
        result = subprocess.run(
            ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return []

    output = result.stderr or ""
    device_pattern = re.compile(r"\[AVFoundation indev @ .*?\]\s+\[(\d+)\]\s+(.*)")
    devices: list[CameraDeviceInfo] = []
    in_video_section = False
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if "AVFoundation video devices:" in line:
            in_video_section = True
            continue
        if "AVFoundation audio devices:" in line:
            break
        if not in_video_section:
            continue
        match = device_pattern.search(line)
        if not match:
            continue
        devices.append(CameraDeviceInfo(index=int(match.group(1)), name=match.group(2).strip()))
    return devices


def try_get_camera_name(camera_index: int | None = None) -> str | None:
    if platform.system() != "Darwin":
        return None

    devices = discover_macos_video_devices()
    if camera_index is not None:
        match = next((device.name for device in devices if device.index == camera_index), None)
        if match:
            return match
    return devices[0].name if devices else None


def _looks_builtin(name: str) -> bool:
    name_lower = name.lower()
    return any(pattern in name_lower for pattern in BUILTIN_CAMERA_PATTERNS) and not _looks_external(name)


def _looks_external(name: str) -> bool:
    name_lower = name.lower()
    return any(pattern in name_lower for pattern in EXTERNAL_CAMERA_PATTERNS)
