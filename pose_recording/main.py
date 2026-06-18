"""CLI entry point for the macOS webcam pose recorder."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2

from backends import BackendUnavailableError, create_backend, list_backends
from config.defaults import (
    DEFAULT_CAMERA_INDEX,
    DEFAULT_FPS,
    DEFAULT_HEIGHT,
    DEFAULT_RECORDINGS_DIR,
    DEFAULT_WIDTH,
    DEFAULT_WINDOW_NAME,
)
from exporters import SessionExporter
from utils.camera import (
    CameraDeviceInfo,
    discover_macos_video_devices,
    get_capture_resolution,
    open_camera,
    select_camera,
    try_get_camera_name,
)
from utils.logging_utils import configure_logging
from utils.runtime_env import configure_local_cache_environment
from utils.time_utils import make_session_id, monotonic_time_sec, wallclock_iso


LOGGER = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record webcam pose estimates with pluggable backends for research use."
    )
    parser.add_argument("--backend", help=f"Backend name. Options: {', '.join(list_backends())}")
    parser.add_argument("--camera-index", type=int, default=None)
    parser.add_argument(
        "--allow-external-camera",
        action="store_true",
        help="Allow external or Continuity Camera devices instead of enforcing the built-in MacBook camera.",
    )
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--mediapipe-pose-model", default=None, help="Path to a local MediaPipe pose .task model bundle.")
    parser.add_argument("--mediapipe-hand-model", default=None, help="Path to a local MediaPipe hand .task model bundle.")
    parser.add_argument(
        "--apple-vision-3d-stride",
        type=int,
        default=3,
        help="Run Apple Vision 3D body depth every Nth frame and reuse the latest z values in between.",
    )
    parser.add_argument(
        "--no-mediapipe-auto-download",
        action="store_true",
        help="Require local MediaPipe .task model files instead of downloading them automatically.",
    )
    parser.add_argument("--recordings-dir", default=DEFAULT_RECORDINGS_DIR)
    parser.add_argument("--session-name", default=None)
    parser.add_argument("--notes", default="")
    parser.add_argument("--save-raw-video", action="store_true")
    parser.add_argument("--save-annotated-video", action="store_true")
    parser.add_argument(
        "--no-preview-hud",
        action="store_true",
        help="Hide preview HUD text while keeping the pose overlay and hand labels visible.",
    )
    parser.add_argument("--no-parquet", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--list-backends", action="store_true")
    parser.add_argument("--list-cameras", action="store_true", help="List cameras that macOS device enumeration can identify.")
    return parser


def make_backend_from_args(args) -> object:
    backend_kwargs = {}
    if args.backend == "mediapipe_pose_hands":
        backend_kwargs["pose_model_path"] = args.mediapipe_pose_model
        backend_kwargs["hand_model_path"] = args.mediapipe_hand_model
        backend_kwargs["allow_model_download"] = not args.no_mediapipe_auto_download
    if args.backend == "apple_vision_3d":
        backend_kwargs["body_3d_stride"] = args.apple_vision_3d_stride
    return create_backend(args.backend, **backend_kwargs)


def hud_lines(
    *,
    backend_name: str,
    session_id: str,
    recording: bool,
    overlay_enabled: bool,
    preview_annotated: bool,
    fps_value: float | None,
    landmark_count: int | None,
    status_message: str | None,
) -> list[str]:
    lines = [
        f"Backend: {backend_name}",
        f"Session: {session_id}",
        f"Recording: {'ON' if recording else 'OFF'}",
        f"Overlay: {'ON' if overlay_enabled else 'OFF'}",
        f"Preview: {'annotated' if preview_annotated else 'raw'}",
        "Controls: q quit | r record | o overlay | h HUD | s snapshot | i info | v preview | p pause",
    ]
    if fps_value is not None:
        lines.insert(2, f"FPS: {fps_value:.2f}")
    if landmark_count is not None:
        lines.insert(3 if fps_value is not None else 2, f"Landmarks: {landmark_count}")
    if status_message:
        lines.append(status_message)
    return lines


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.list_backends:
        print("\n".join(list_backends()))
        return 0
    if args.list_cameras:
        devices = discover_macos_video_devices()
        if not devices:
            print("No cameras could be identified by macOS device enumeration.")
            return 1
        for device in devices:
            print(f"{device.index}: {device.name}")
        return 0
    if not args.backend:
        parser.error("--backend is required unless you use --list-backends or --list-cameras")

    configure_local_cache_environment()
    configure_logging(verbose=args.verbose)
    logging.getLogger("matplotlib").setLevel(logging.ERROR)

    try:
        backend = make_backend_from_args(args)
        backend.initialize()
    except (ValueError, BackendUnavailableError, RuntimeError, FileNotFoundError) as exc:
        if args.verbose:
            LOGGER.exception("Backend initialization failed")
        else:
            LOGGER.error("%s", exc)
            LOGGER.info("Re-run with --verbose to see the full traceback.")
        return 1

    session_id = make_session_id(args.session_name)
    recordings_dir = Path(args.recordings_dir)
    session_dir = recordings_dir / session_id

    try:
        selected_camera: CameraDeviceInfo = select_camera(
            requested_index=args.camera_index,
            prefer_builtin_macbook_camera=True,
            allow_external_camera=args.allow_external_camera,
        )
        capture = open_camera(selected_camera.index, args.width, args.height, args.fps)
    except RuntimeError as exc:
        LOGGER.error("%s", exc)
        backend.shutdown()
        return 1

    capture_width, capture_height = get_capture_resolution(capture)
    camera_name = selected_camera.name or try_get_camera_name(selected_camera.index) or f"OpenCV camera {selected_camera.index}"
    exporter = SessionExporter(
        session_id=session_id,
        model_name=args.backend,
        session_dir=session_dir,
        command_line_settings=vars(args),
        model_configuration=backend.get_configuration(),
        notes=args.notes,
        target_fps=args.fps,
        camera_name=camera_name,
        save_raw_video=args.save_raw_video,
        save_annotated_video=args.save_annotated_video,
        save_parquet=not args.no_parquet,
    )

    LOGGER.info("Selected backend: %s", args.backend)
    LOGGER.info("Selected camera: %s (index %s)", camera_name, selected_camera.index)
    LOGGER.info("Session directory: %s", session_dir)
    LOGGER.info("Controls: q quit | r record | o overlay | h HUD | s snapshot | i info | v preview | p pause")

    recording = False
    overlay_enabled = True
    preview_annotated = True
    preview_hud_enabled = not args.no_preview_hud
    paused = False
    frame_index = 0
    fps_value = None
    status_message = "Ready. Press r to start recording."
    status_message_expires = 0.0
    last_loop_time = None
    last_result = None
    last_raw_frame = None
    last_annotated_frame = None

    cv2.namedWindow(DEFAULT_WINDOW_NAME, cv2.WINDOW_NORMAL)

    def visible_hud_lines_for(result) -> list[str] | None:
        if not preview_hud_enabled:
            return None
        return hud_lines(
            backend_name=args.backend,
            session_id=session_id,
            recording=recording,
            overlay_enabled=overlay_enabled,
            preview_annotated=preview_annotated,
            fps_value=fps_value,
            landmark_count=result.present_landmark_count(),
            status_message=status_message,
        )

    try:
        while True:
            now_monotonic = monotonic_time_sec()
            if not paused:
                success, frame_bgr = capture.read()
                if not success:
                    exporter.note_capture_failure()
                    status_message = "Camera frame read failed."
                    status_message_expires = now_monotonic + 1.5
                    key_code = cv2.waitKey(1) & 0xFF
                    if key_code == ord("q"):
                        break
                    continue

                frame_index += 1
                timestamp_monotonic_sec = monotonic_time_sec()
                timestamp_wallclock_iso = wallclock_iso()
                result = backend.process_frame(frame_bgr, timestamp_monotonic_sec)

                if last_loop_time is not None:
                    elapsed = timestamp_monotonic_sec - last_loop_time
                    if elapsed > 0:
                        fps_value = 1.0 / elapsed
                last_loop_time = timestamp_monotonic_sec

                if monotonic_time_sec() > status_message_expires:
                    status_message = None

                last_raw_frame = frame_bgr.copy()
                last_result = result
                last_annotated_frame = backend.draw_overlay(
                    frame_bgr,
                    result,
                    overlay_enabled=overlay_enabled,
                    hud_lines=visible_hud_lines_for(result),
                )

                if recording:
                    exporter.append_frame(
                        result=result,
                        frame_index=frame_index,
                        timestamp_monotonic_sec=timestamp_monotonic_sec,
                        timestamp_wallclock_iso=timestamp_wallclock_iso,
                        image_width=capture_width,
                        image_height=capture_height,
                        raw_frame_bgr=last_raw_frame if args.save_raw_video else None,
                        annotated_frame_bgr=last_annotated_frame if args.save_annotated_video else None,
                    )
            else:
                if last_raw_frame is None or last_result is None or last_annotated_frame is None:
                    key_code = cv2.waitKey(1) & 0xFF
                    if key_code == ord("q"):
                        break
                    continue

            preview_frame = last_annotated_frame if preview_annotated else last_raw_frame
            cv2.imshow(DEFAULT_WINDOW_NAME, preview_frame)

            key_code = cv2.waitKey(1) & 0xFF
            if key_code == ord("q"):
                break
            if key_code == ord("r"):
                recording = not recording
                if recording:
                    if exporter.recording_started_at is None:
                        exporter.recording_started_at = wallclock_iso()
                    exporter.open_video_writers(capture_width, capture_height)
                    exporter.log_event(f"[recording-start] frame={frame_index} ts={exporter.recording_started_at}")
                    status_message = "Recording started."
                else:
                    exporter.log_event(f"[recording-stop] frame={frame_index} ts={wallclock_iso()}")
                    exporter.flush_rows()
                    status_message = "Recording stopped."
                status_message_expires = monotonic_time_sec() + 2.0
            if key_code == ord("o"):
                overlay_enabled = not overlay_enabled
                status_message = f"Overlay {'enabled' if overlay_enabled else 'disabled'}."
                status_message_expires = monotonic_time_sec() + 2.0
            if key_code == ord("h"):
                preview_hud_enabled = not preview_hud_enabled
                status_message = f"HUD {'enabled' if preview_hud_enabled else 'hidden'}."
                status_message_expires = monotonic_time_sec() + 2.0
                if last_raw_frame is not None and last_result is not None:
                    last_annotated_frame = backend.draw_overlay(
                        last_raw_frame,
                        last_result,
                        overlay_enabled=overlay_enabled,
                        hud_lines=visible_hud_lines_for(last_result),
                    )
            if key_code == ord("v"):
                preview_annotated = not preview_annotated
                status_message = f"Preview set to {'annotated' if preview_annotated else 'raw'}."
                status_message_expires = monotonic_time_sec() + 2.0
            if key_code == ord("p"):
                paused = not paused
                exporter.log_event(f"[pause-toggle] paused={paused} frame={frame_index} ts={wallclock_iso()}")
                status_message = f"Preview {'paused' if paused else 'resumed'}."
                status_message_expires = monotonic_time_sec() + 2.0
            if key_code == ord("s") and preview_frame is not None:
                snapshot_name = f"snapshot_{frame_index:06d}.png"
                snapshot_path = exporter.save_snapshot(preview_frame, snapshot_name)
                exporter.log_event(f"[snapshot] path={snapshot_path.name} frame={frame_index}")
                status_message = f"Saved {snapshot_path.name}"
                status_message_expires = monotonic_time_sec() + 2.0
            if key_code == ord("i"):
                info_message = (
                    f"session={session_id} backend={args.backend} frame={frame_index} "
                    f"recording={recording} fps={fps_value:.2f}" if fps_value else
                    f"session={session_id} backend={args.backend} frame={frame_index} recording={recording}"
                )
                LOGGER.info(info_message)
                exporter.log_event(f"[info] {info_message}")
                status_message = info_message
                status_message_expires = monotonic_time_sec() + 3.0

    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user.")
    finally:
        capture.release()
        cv2.destroyAllWindows()
        exporter.close(resolution=(capture_width, capture_height))
        backend.shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())
