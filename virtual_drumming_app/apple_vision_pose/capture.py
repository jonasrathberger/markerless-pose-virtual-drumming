"""Camera capture backends and camera-selection helpers."""

from __future__ import annotations

import contextlib
import os
import sys
import threading
import time
from typing import Any

from .constants import IPHONE_CAMERA_HINTS, MAC_WEBCAM_HINTS
from .dependencies import load_avfoundation_stack, load_opencv
from .image_utils import black_frame, pixel_buffer_to_bgr, resize_to_request
from .models import ProcessedFrame


class LatestFrameCapture:
    def __init__(self, capture: Any) -> None:
        self.capture = capture
        self.condition = threading.Condition()
        self.frame: Any | None = None
        self.frame_id = 0
        self.stopped = False
        self.failed = False
        self.thread = threading.Thread(target=self._run, name="latest-frame-capture", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def read(self, last_seen_frame_id: int) -> tuple[bool, int, Any | None]:
        with self.condition:
            while not self.stopped and not self.failed and self.frame_id <= last_seen_frame_id:
                self.condition.wait(timeout=0.25)
            if self.failed or self.frame is None:
                return False, last_seen_frame_id, None
            return True, self.frame_id, self.frame

    def stop(self) -> None:
        with self.condition:
            self.stopped = True
            self.condition.notify_all()
        self.thread.join(timeout=1.0)

    def _run(self) -> None:
        while True:
            with self.condition:
                if self.stopped:
                    return

            ok, frame = self.capture.read()
            with self.condition:
                if not ok:
                    self.failed = True
                    self.condition.notify_all()
                    return
                self.frame = frame
                self.frame_id += 1
                self.condition.notify_all()


class AVFoundationPoseCapture:
    def __init__(
        self,
        *,
        camera_index: int,
        detector: Any,
        width: int,
        height: int,
        detect_width: int,
        display_camera_image: bool,
        retain_raw_frame: bool = False,
    ) -> None:
        self.AVFoundation, self.CoreMedia, self.CoreVideo, self.Foundation, self.dispatch = load_avfoundation_stack()
        self.detector = detector
        self.width = width
        self.height = height
        self.detect_width = detect_width
        self.display_camera_image = display_camera_image
        self.retain_raw_frame = retain_raw_frame
        self.condition = threading.Condition()
        self.latest_frame: ProcessedFrame | None = None
        self.last_error: BaseException | None = None
        self.frame_id = 0
        self.stopped = False
        self.preview_conversion_failed = False
        self.software_resize_reported = False

        devices = self.AVFoundation.AVCaptureDevice.devicesWithMediaType_(self.AVFoundation.AVMediaTypeVideo)
        if camera_index < 0 or camera_index >= len(devices):
            raise RuntimeError(f"AVFoundation camera index {camera_index} is out of range.")

        self.session = self.AVFoundation.AVCaptureSession.alloc().init()
        self.session.beginConfiguration()
        self._set_session_preset(width=width, height=height)

        device = devices[camera_index]
        camera_input, error = self.AVFoundation.AVCaptureDeviceInput.deviceInputWithDevice_error_(device, None)
        if camera_input is None or error is not None:
            raise RuntimeError(f"Could not create AVFoundation camera input: {error}")
        if not self.session.canAddInput_(camera_input):
            raise RuntimeError("Could not add AVFoundation camera input to the capture session.")
        self.session.addInput_(camera_input)
        self._select_device_format(device, width=width, height=height)

        self.video_output = self.AVFoundation.AVCaptureVideoDataOutput.alloc().init()
        self.video_output.setAlwaysDiscardsLateVideoFrames_(True)
        self.video_output.setVideoSettings_(self._video_settings(width=width, height=height))
        if not self.session.canAddOutput_(self.video_output):
            raise RuntimeError("Could not add AVFoundation video output to the capture session.")
        self.session.addOutput_(self.video_output)

        global AVFoundationFrameDelegate
        if AVFoundationFrameDelegate is None:
            AVFoundationFrameDelegate = make_avfoundation_frame_delegate_class()
        self.delegate = AVFoundationFrameDelegate.alloc().initWithOwner_(self)
        self.queue = self.dispatch.dispatch_queue_create(b"apple-vision-live.capture", None)
        self.video_output.setSampleBufferDelegate_queue_(self.delegate, self.queue)
        self.session.commitConfiguration()

    def start(self) -> None:
        self.session.startRunning()

    def stop(self) -> None:
        with self.condition:
            self.stopped = True
            self.condition.notify_all()
        self.video_output.setSampleBufferDelegate_queue_(None, None)
        self.session.stopRunning()

    def read(self, last_seen_frame_id: int) -> tuple[bool, int, ProcessedFrame | None]:
        with self.condition:
            while self.latest_frame is None or self.latest_frame.frame_id <= last_seen_frame_id:
                if self.stopped or self.last_error is not None:
                    return False, last_seen_frame_id, None
                self.condition.wait(timeout=0.25)
            return True, self.latest_frame.frame_id, self.latest_frame

    def process_sample_buffer(self, sample_buffer: Any) -> None:
        try:
            pixel_buffer = self.CoreMedia.CMSampleBufferGetImageBuffer(sample_buffer)
            if pixel_buffer is None:
                return

            source_width = int(self.CoreVideo.CVPixelBufferGetWidth(pixel_buffer))
            source_height = int(self.CoreVideo.CVPixelBufferGetHeight(pixel_buffer))
            target_width, target_height = self._processing_size(
                source_width=source_width,
                source_height=source_height,
            )

            if (target_width, target_height) != (source_width, source_height):
                self._report_software_resize(
                    source_width=source_width,
                    source_height=source_height,
                    target_width=target_width,
                    target_height=target_height,
                )
                t0 = time.perf_counter()
                raw_frame_bgr = pixel_buffer_to_bgr(pixel_buffer, self.CoreVideo)
                frame_bgr = resize_to_request(raw_frame_bgr, width=target_width, height=target_height)
                t1 = time.perf_counter()
                landmarks = self.detector.detect(frame_bgr)
                t2 = time.perf_counter()
                detect_seconds = t2 - t1
                convert_seconds = t1 - t0
            else:
                t0 = time.perf_counter()
                landmarks = self.detector.detect_pixel_buffer(pixel_buffer, source_width, source_height)
                t1 = time.perf_counter()
                raw_frame_bgr = self._raw_frame(pixel_buffer, source_width, source_height)
                frame_bgr = self._preview_frame(raw_frame_bgr, source_width, source_height)
                t2 = time.perf_counter()
                detect_seconds = t1 - t0
                convert_seconds = t2 - t1

            with self.condition:
                self.frame_id += 1
                self.latest_frame = ProcessedFrame(
                    frame_id=self.frame_id,
                    frame_bgr=frame_bgr,
                    raw_frame_bgr=raw_frame_bgr,
                    landmarks=landmarks,
                    detect_seconds=detect_seconds,
                    convert_seconds=convert_seconds,
                )
                self.condition.notify_all()
        except BaseException as exc:
            with self.condition:
                self.last_error = exc
                self.condition.notify_all()

    def _raw_frame(self, pixel_buffer: Any, width: int, height: int) -> Any | None:
        if not self.display_camera_image and not self.retain_raw_frame:
            return None
        try:
            return pixel_buffer_to_bgr(pixel_buffer, self.CoreVideo)
        except Exception as exc:
            if not self.preview_conversion_failed:
                print(
                    f"Could not copy CVPixelBuffer preview image ({exc}). Drawing on a black canvas instead.",
                    file=sys.stderr,
                    flush=True,
                )
                self.preview_conversion_failed = True
            return None

    def _preview_frame(self, raw_frame_bgr: Any | None, width: int, height: int) -> Any:
        if not self.display_camera_image or raw_frame_bgr is None:
            return black_frame(width=width, height=height)
        return raw_frame_bgr

    def _processing_size(self, *, source_width: int, source_height: int) -> tuple[int, int]:
        if self.detect_width > 0 and source_width > self.detect_width:
            target_height = max(1, round(source_height * (self.detect_width / source_width)))
            return self.detect_width, target_height
        return source_width, source_height

    def _report_software_resize(
        self,
        *,
        source_width: int,
        source_height: int,
        target_width: int,
        target_height: int,
    ) -> None:
        if self.software_resize_reported:
            return
        print(
            f"AVFoundation is delivering {source_width}x{source_height}; resizing to "
            f"{target_width}x{target_height} before Vision and preview.",
            file=sys.stderr,
            flush=True,
        )
        self.software_resize_reported = True

    def _video_settings(self, *, width: int, height: int) -> dict[Any, Any]:
        settings = {self.CoreVideo.kCVPixelBufferPixelFormatTypeKey: self.CoreVideo.kCVPixelFormatType_32BGRA}
        width_key = getattr(self.CoreVideo, "kCVPixelBufferWidthKey", None)
        height_key = getattr(self.CoreVideo, "kCVPixelBufferHeightKey", None)
        if width_key is not None and height_key is not None:
            settings[width_key] = width
            settings[height_key] = height
        return settings

    def _set_session_preset(self, *, width: int, height: int) -> None:
        candidates = []
        if width <= 640 and height <= 480 and hasattr(self.AVFoundation, "AVCaptureSessionPreset640x480"):
            candidates.append(self.AVFoundation.AVCaptureSessionPreset640x480)
        if width <= 1280 and height <= 720 and hasattr(self.AVFoundation, "AVCaptureSessionPreset1280x720"):
            candidates.append(self.AVFoundation.AVCaptureSessionPreset1280x720)
        if width <= 1920 and height <= 1080 and hasattr(self.AVFoundation, "AVCaptureSessionPreset1920x1080"):
            candidates.append(self.AVFoundation.AVCaptureSessionPreset1920x1080)
        if not candidates and hasattr(self.AVFoundation, "AVCaptureSessionPresetHigh"):
            candidates.append(self.AVFoundation.AVCaptureSessionPresetHigh)

        for preset in candidates:
            if self.session.canSetSessionPreset_(preset):
                self.session.setSessionPreset_(preset)
                return

    def _select_device_format(self, device: Any, *, width: int, height: int) -> None:
        best_format = None
        best_dimensions: tuple[int, int] | None = None
        best_score: tuple[int, int, int] | None = None

        for camera_format in device.formats():
            format_width, format_height = self._format_dimensions(camera_format)
            fits_request = format_width >= width and format_height >= height
            score = (
                0 if fits_request else 1,
                abs(format_width - width) + abs(format_height - height),
                format_width * format_height,
            )
            if best_score is None or score < best_score:
                best_score = score
                best_format = camera_format
                best_dimensions = (format_width, format_height)

        if best_format is None:
            return

        locked, error = device.lockForConfiguration_(None)
        if not locked:
            print(f"Could not lock camera for format selection: {error}", file=sys.stderr, flush=True)
            return
        try:
            device.setActiveFormat_(best_format)
            if hasattr(self.AVFoundation, "AVCaptureSessionPresetInputPriority"):
                input_priority = self.AVFoundation.AVCaptureSessionPresetInputPriority
                if self.session.canSetSessionPreset_(input_priority):
                    self.session.setSessionPreset_(input_priority)
            if best_dimensions is not None:
                print(
                    f"Selected AVFoundation camera format {best_dimensions[0]}x{best_dimensions[1]} "
                    f"for request {width}x{height}.",
                    file=sys.stderr,
                    flush=True,
                )
        finally:
            device.unlockForConfiguration()

    def _format_dimensions(self, camera_format: Any) -> tuple[int, int]:
        description = camera_format.formatDescription()
        try:
            dimensions = self.CoreMedia.CMVideoFormatDescriptionGetDimensions(description)
        except AttributeError:
            dimensions = self.CoreMedia.CMVideoFormatDescriptionGetDimensions_(description)
        return int(dimensions.width), int(dimensions.height)


def make_avfoundation_frame_delegate_class() -> Any:
    _avfoundation, _core_media, _core_video, foundation, _dispatch = load_avfoundation_stack()
    import objc

    class Delegate(foundation.NSObject):
        def initWithOwner_(self, owner: AVFoundationPoseCapture) -> Any:
            self = objc.super(Delegate, self).init()
            if self is None:
                return None
            self.owner = owner
            return self

        def captureOutput_didOutputSampleBuffer_fromConnection_(
            self,
            _output: Any,
            sample_buffer: Any,
            _connection: Any,
        ) -> None:
            self.owner.process_sample_buffer(sample_buffer)

    return Delegate


AVFoundationFrameDelegate: Any | None = None


def list_avfoundation_cameras() -> list[dict[str, str]]:
    try:
        import AVFoundation
    except ImportError as exc:
        raise RuntimeError("Missing AVFoundation bindings.") from exc

    devices = AVFoundation.AVCaptureDevice.devicesWithMediaType_(AVFoundation.AVMediaTypeVideo)
    cameras: list[dict[str, str]] = []
    for index, device in enumerate(devices):
        name = str(device.localizedName())
        unique_id = str(device.uniqueID()) if hasattr(device, "uniqueID") else ""
        device_type = str(device.deviceType()) if hasattr(device, "deviceType") else ""
        cameras.append({"index": str(index), "name": name, "unique_id": unique_id, "device_type": device_type})
    return cameras


def select_camera_index(explicit_index: int | None) -> int:
    if explicit_index is not None:
        return explicit_index

    try:
        cameras = list_avfoundation_cameras()
    except RuntimeError:
        return first_open_cv_camera_index(preferred_indices=[1, 2, 3, 4, 5, 0], quiet=True)

    if not cameras:
        return 0

    for camera in cameras:
        searchable = f"{camera['name']} {camera['unique_id']} {camera['device_type']}".lower()
        if any(hint in searchable for hint in MAC_WEBCAM_HINTS) and not any(
            hint in searchable for hint in IPHONE_CAMERA_HINTS
        ):
            return int(camera["index"])

    for camera in cameras:
        searchable = f"{camera['name']} {camera['unique_id']} {camera['device_type']}".lower()
        if not any(hint in searchable for hint in IPHONE_CAMERA_HINTS):
            return int(camera["index"])

    return 0


def first_open_cv_camera_index(preferred_indices: list[int], *, quiet: bool = False) -> int:
    cv2 = load_opencv()
    for index in preferred_indices:
        with suppress_native_stderr(enabled=quiet):
            capture = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        opened = capture.isOpened()
        capture.release()
        if opened:
            return index
    return 0


def list_open_cv_camera_indices(max_index: int = 5, *, quiet: bool = False) -> list[int]:
    cv2 = load_opencv()
    indices: list[int] = []
    for index in range(max_index + 1):
        with suppress_native_stderr(enabled=quiet):
            capture = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        opened = capture.isOpened()
        capture.release()
        if opened:
            indices.append(index)
    return indices


@contextlib.contextmanager
def suppress_native_stderr(*, enabled: bool):
    if not enabled:
        yield
        return

    stderr_fd = sys.stderr.fileno()
    saved_fd = os.dup(stderr_fd)
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            os.dup2(devnull.fileno(), stderr_fd)
            yield
    finally:
        os.dup2(saved_fd, stderr_fd)
        os.close(saved_fd)


def print_camera_list() -> None:
    try:
        cameras = list_avfoundation_cameras()
    except RuntimeError:
        print("AVFoundation PyObjC bindings are not installed, so camera names are unavailable.")
        print("From the repository root, install/update dependencies with: python -m pip install -r requirements.txt")
        try:
            indices = list_open_cv_camera_indices(quiet=True)
        except RuntimeError as exc:
            print(exc)
            return
        if not indices:
            print("No OpenCV AVFoundation camera indices opened.")
            return
        selected = first_open_cv_camera_index(preferred_indices=[1, 2, 3, 4, 5, 0], quiet=True)
        for index in indices:
            marker = "default candidate" if index == selected else ""
            suffix = f" ({marker})" if marker else ""
            print(f"{index}: OpenCV camera index {index}{suffix}")
        if indices == [0]:
            print(
                "Only index 0 is visible to OpenCV. If that is your iPhone, install "
                "pyobjc-framework-AVFoundation so this script can select cameras by name."
            )
        return

    if not cameras:
        print("No AVFoundation video devices found.")
        return

    for camera in cameras:
        marker = "mac webcam candidate" if int(camera["index"]) == select_camera_index(None) else ""
        suffix = f" ({marker})" if marker else ""
        print(
            f"{camera['index']}: {camera['name']}"
            f" | type={camera['device_type']}"
            f" | unique_id={camera['unique_id']}{suffix}"
        )


def avfoundation_bindings_available() -> bool:
    try:
        import AVFoundation  # noqa: F401
    except ImportError:
        return False
    return True
