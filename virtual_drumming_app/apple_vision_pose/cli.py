"""Command-line entrypoint and runtime loops."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from drum_engine import (
    HandHitDetector,
    HandHitDetectorConfig,
    LandmarkSmoother,
    MidiOutput,
    MidiOutputConfig,
    PedalHitDetector,
    apple_vision_landmarks_to_pose_frame,
    available_midi_outputs,
)
from drum_engine.hand_hit_detection import HandHitEvent
from drum_engine.pedal_hit_detection import HitEvent
from drum_engine.target_classification import (
    DrumTargetObservation,
    DrumTargetPrediction,
    KNN_DISTANCE_METRICS,
    KNN_DISTANCE_SQUARED_EUCLIDEAN,
    KnnDrumTargetClassifier,
    TARGET_DRUMS,
    TARGET_HAND_SIDES,
    load_or_create_target_sample_file,
    target_observation_to_sample_record,
    write_target_sample_file,
)

from .capture import (
    AVFoundationPoseCapture,
    LatestFrameCapture,
    avfoundation_bindings_available,
    list_open_cv_camera_indices,
    print_camera_list,
    select_camera_index,
)
from .dependencies import load_opencv
from .models import TimingStats
from .rendering import draw_landmarks
from .vision import AppleVisionPose

PREVIEW_WIDTH = 1920
PREVIEW_HEIGHT = 1080
BACKSPACE_KEYS = (8, 127)
DEFAULT_COLLECTION_RAW_VIDEO_DIR = Path("target_sample_videos")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live Apple Vision pose landmarks through OpenCV.")
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="Camera index. Defaults to the built-in Mac webcam when detectable.",
    )
    parser.add_argument(
        "--capture-backend",
        choices=("avfoundation", "opencv"),
        default="avfoundation",
        help="Use native AVFoundation CVPixelBuffer capture or the older OpenCV capture path.",
    )
    parser.add_argument("--list-cameras", action="store_true", help="List AVFoundation video devices and exit.")
    parser.add_argument("--list-midi-outputs", action="store_true", help="List MIDI output ports and exit.")
    parser.add_argument(
        "--midi-out",
        nargs="?",
        const="",
        default="",
        metavar="PORT",
        help="Send drum hits to a MIDI output port. Omit PORT to use the default output.",
    )
    parser.add_argument(
        "--no-midi-out",
        action="store_true",
        help="Disable MIDI output and only print detected hits.",
    )
    parser.add_argument(
        "--midi-channel",
        type=int,
        default=10,
        help="MIDI channel number from 1 to 16. Channel 10 is the General MIDI drum channel.",
    )
    parser.add_argument("--width", type=int, default=1920, help="Requested camera capture width.")
    parser.add_argument("--height", type=int, default=1080, help="Requested camera capture height.")
    parser.add_argument(
        "--detect-width",
        type=int,
        default=0,
        help="Resize frames to this width before Vision inference. Use 0 for capture resolution.",
    )
    parser.add_argument(
        "--no-latest-frame",
        action="store_true",
        help="OpenCV backend only: disable the capture thread that drops stale frames.",
    )
    parser.add_argument(
        "--no-camera-image",
        action="store_true",
        help="AVFoundation backend only: draw landmarks on a black canvas instead of copying the camera image.",
    )
    parser.add_argument(
        "--stats-every",
        type=int,
        default=0,
        help="Print average timing every N processed frames. Use 0 to disable.",
    )
    parser.add_argument(
        "--preview-text-scale",
        type=float,
        default=1.0,
        help="Scale live preview HUD and landmark label text. Use values like 1.5 or 2.0 for screenshots.",
    )
    parser.add_argument(
        "--print-all",
        action="store_true",
        help="Extract and draw every detected body/hand landmark for visual debugging.",
    )
    parser.add_argument(
        "--hand-target-model",
        default=None,
        metavar="PATH",
        help="Optional KNN hand target model JSON path. Defaults to knn_100.json.",
    )
    parser.add_argument(
        "--knn-distance",
        choices=KNN_DISTANCE_METRICS,
        default=KNN_DISTANCE_SQUARED_EUCLIDEAN,
        help="Distance metric for KNN hand-target classification.",
    )
    parser.add_argument(
        "--hand-up-threshold",
        type=float,
        default=None,
        help="Override upward prep distance required to arm hand hits. Lower values catch smaller strokes.",
    )
    parser.add_argument(
        "--hand-down-velocity-threshold",
        type=float,
        default=None,
        help="Override downward hand velocity required to trigger a hit.",
    )
    parser.add_argument(
        "--collect-target-samples",
        default=None,
        metavar="OUT.json",
        help="Collect labeled hand-target hit samples for KNN training.",
    )
    parser.add_argument(
        "--samples-per-target",
        type=int,
        default=50,
        help="Clean hits to collect for each hand/drum prompt.",
    )
    parser.add_argument(
        "--collection-countdown",
        type=float,
        default=3.0,
        help="Countdown seconds after pressing Space before recording starts.",
    )
    parser.add_argument(
        "--collection-output-mode",
        choices=("append", "overwrite"),
        default="append",
        help="Append to an existing target sample file or overwrite it.",
    )
    parser.add_argument(
        "--collection-raw-video",
        default=None,
        metavar="OUT.mp4",
        help="Record unannotated camera video while target sample collection is actively recording.",
    )
    parser.add_argument(
        "--collection-raw-video-fps",
        type=float,
        default=30.0,
        help="Frame rate written to --collection-raw-video.",
    )
    parser.add_argument(
        "--record-evaluation-video",
        default=None,
        metavar="OUT.mp4",
        help="Record an unannotated full-run video for later classifier evaluation.",
    )
    parser.add_argument(
        "--evaluation-annotations",
        default=None,
        metavar="OUT.json",
        help="Sidecar JSON for --record-evaluation-video events. Defaults to the video path with .json suffix.",
    )
    parser.add_argument(
        "--evaluation-video-fps",
        type=float,
        default=30.0,
        help="Frame rate written to --record-evaluation-video.",
    )
    parser.add_argument(
        "--evaluation-countdown",
        type=float,
        default=10.0,
        help="Countdown seconds before evaluation video and hit recording starts.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.list_cameras:
        print_camera_list()
        return 0
    if args.list_midi_outputs:
        print_midi_outputs()
        return 0

    try:
        if args.capture_backend == "avfoundation":
            return run_avfoundation_loop(args)
        return run_opencv_loop(args)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


def run_avfoundation_loop(args: argparse.Namespace) -> int:
    hand_hit_detector, collection = create_hand_hit_detector(args)
    evaluation = create_evaluation_recording_session(args)
    cv2 = load_opencv()
    detector = AppleVisionPose(max_hands=2, detect_width=0, extract_all_landmarks=args.print_all)
    camera_index = select_camera_index(args.camera)
    print(f"Using AVFoundation camera index {camera_index}.", file=sys.stderr)

    capture = AVFoundationPoseCapture(
        camera_index=camera_index,
        detector=detector,
        width=args.width,
        height=args.height,
        detect_width=args.detect_width,
        display_camera_image=not args.no_camera_image,
        retain_raw_frame=(collection is not None and collection.records_raw_video) or evaluation is not None,
    )

    frame_index = 0
    last_capture_frame_id = 0
    window_name = "Apple Vision Pose"
    setup_preview_window(cv2, window_name)
    stats = TimingStats()
    landmark_smoother = LandmarkSmoother()
    pedal_hit_detector = PedalHitDetector()
    midi_output = None if collection is not None else create_midi_output(args)
    if collection is not None:
        collection.print_prompt()

    capture.start()
    try:
        while True:
            t0 = time.perf_counter()
            ok, last_capture_frame_id, processed = capture.read(last_capture_frame_id)
            t1 = time.perf_counter()
            if not ok or processed is None:
                if capture.last_error is not None:
                    print(f"Apple Vision pose detection failed: {capture.last_error}", file=sys.stderr, flush=True)
                    return 1
                print("Camera frame read failed.", file=sys.stderr)
                break

            stats.mark_frame()
            frame_index += 1
            stats.add("read", t1 - t0)
            stats.add("detect", processed.detect_seconds)
            stats.add("convert", processed.convert_seconds)
            timestamp_seconds = time.perf_counter()
            raw_frame_bgr = processed.raw_frame_bgr if processed.raw_frame_bgr is not None else processed.frame_bgr
            if collection is not None:
                collection.record_video_frame(raw_frame_bgr, timestamp_seconds)
            emit_hits = evaluation is None or evaluation.is_recording(timestamp_seconds)
            if evaluation is not None:
                evaluation.record_video_frame(raw_frame_bgr, timestamp_seconds)
            smoothed_frame, hand_events, pedal_events = process_drum_engine_hits(
                processed.landmarks,
                landmark_smoother=landmark_smoother,
                hand_hit_detector=hand_hit_detector,
                pedal_hit_detector=pedal_hit_detector,
                midi_output=midi_output if emit_hits else None,
                timestamp_seconds=timestamp_seconds,
                emit_hits=emit_hits,
            )
            if evaluation is not None:
                evaluation.record_events(hand_events=hand_events, pedal_events=pedal_events, timestamp_seconds=timestamp_seconds)
            hand_drum_labels = current_hand_drum_labels(hand_hit_detector, smoothed_frame, collection=collection)
            if collection is not None and collection.completed:
                break

            t2 = time.perf_counter()
            performance_line = stats.hud_line()
            if collection is not None:
                performance_line = collection.preview_line(time.perf_counter())
            elif evaluation is not None:
                performance_line = evaluation.preview_line(time.perf_counter())
            annotated = draw_landmarks(
                processed.frame_bgr,
                processed.landmarks,
                performance_line,
                draw_all=args.print_all,
                hand_drum_labels=hand_drum_labels,
                text_scale=args.preview_text_scale,
            )
            t3 = time.perf_counter()
            stats.add("draw", t3 - t2)

            cv2.imshow(window_name, resize_preview(cv2, annotated))
            key = cv2.waitKey(1) & 0xFF
            if collection is not None:
                collection.handle_key(key, time.perf_counter())
            t4 = time.perf_counter()
            stats.add("display", t4 - t3)

            if args.stats_every > 0 and frame_index % args.stats_every == 0:
                print(stats.summary(), file=sys.stderr, flush=True)

            if key in (ord("q"), 27):
                break
    finally:
        if collection is not None:
            collection.write_outputs()
        if evaluation is not None:
            evaluation.write_outputs()
        if midi_output is not None:
            midi_output.close()
        capture.stop()
        cv2.destroyAllWindows()

    return 0


def run_opencv_loop(args: argparse.Namespace) -> int:
    hand_hit_detector, collection = create_hand_hit_detector(args)
    evaluation = create_evaluation_recording_session(args)
    cv2 = load_opencv()
    if args.camera is None and not avfoundation_bindings_available():
        indices = list_open_cv_camera_indices(quiet=True)
        if indices == [0]:
            print(
                "Only OpenCV camera index 0 is available, and AVFoundation camera names are unavailable.\n"
                "Because index 0 is your iPhone/Continuity Camera, install the shared requirements from the repository root first:\n"
                "  python -m pip install -r requirements.txt\n"
                "Then run:\n"
                "  python apple_vision_live.py --list-cameras\n"
                "If you still want to force index 0, run: python apple_vision_live.py --camera 0",
                file=sys.stderr,
            )
            return 1

    detector = AppleVisionPose(max_hands=2, detect_width=args.detect_width, extract_all_landmarks=args.print_all)

    camera_index = select_camera_index(args.camera)
    print(f"Using camera index {camera_index}.", file=sys.stderr)
    capture = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not capture.isOpened():
        print(f"Could not open camera index {camera_index}.", file=sys.stderr)
        return 1

    frame_index = 0
    last_capture_frame_id = 0
    window_name = "Apple Vision Pose"
    setup_preview_window(cv2, window_name)
    stats = TimingStats()
    landmark_smoother = LandmarkSmoother()
    pedal_hit_detector = PedalHitDetector()
    midi_output = None if collection is not None else create_midi_output(args)
    if collection is not None:
        collection.print_prompt()
    latest_capture: LatestFrameCapture | None = None
    if not args.no_latest_frame:
        latest_capture = LatestFrameCapture(capture)
        latest_capture.start()

    try:
        while True:
            t0 = time.perf_counter()
            if latest_capture is None:
                ok, frame_bgr = capture.read()
            else:
                ok, last_capture_frame_id, frame_bgr = latest_capture.read(last_capture_frame_id)
            t1 = time.perf_counter()
            if not ok:
                print("Camera frame read failed.", file=sys.stderr)
                break

            stats.mark_frame()
            frame_index += 1
            landmarks = detector.detect(frame_bgr)
            t2 = time.perf_counter()
            stats.add("read", t1 - t0)
            stats.add("detect", t2 - t1)
            if collection is not None:
                collection.record_video_frame(frame_bgr, t2)
            emit_hits = evaluation is None or evaluation.is_recording(t2)
            if evaluation is not None:
                evaluation.record_video_frame(frame_bgr, t2)
            smoothed_frame, hand_events, pedal_events = process_drum_engine_hits(
                landmarks,
                landmark_smoother=landmark_smoother,
                hand_hit_detector=hand_hit_detector,
                pedal_hit_detector=pedal_hit_detector,
                midi_output=midi_output if emit_hits else None,
                timestamp_seconds=t2,
                emit_hits=emit_hits,
            )
            if evaluation is not None:
                evaluation.record_events(hand_events=hand_events, pedal_events=pedal_events, timestamp_seconds=t2)
            hand_drum_labels = current_hand_drum_labels(hand_hit_detector, smoothed_frame, collection=collection)
            if collection is not None and collection.completed:
                break

            t2 = time.perf_counter()
            performance_line = stats.hud_line()
            if collection is not None:
                performance_line = collection.preview_line(time.perf_counter())
            elif evaluation is not None:
                performance_line = evaluation.preview_line(time.perf_counter())
            annotated = draw_landmarks(
                frame_bgr,
                landmarks,
                performance_line,
                draw_all=args.print_all,
                hand_drum_labels=hand_drum_labels,
                text_scale=args.preview_text_scale,
            )
            t3 = time.perf_counter()
            stats.add("draw", t3 - t2)

            cv2.imshow(window_name, resize_preview(cv2, annotated))
            key = cv2.waitKey(1) & 0xFF
            if collection is not None:
                collection.handle_key(key, time.perf_counter())
            t4 = time.perf_counter()
            stats.add("display", t4 - t3)

            if args.stats_every > 0 and frame_index % args.stats_every == 0:
                print(stats.summary(), file=sys.stderr, flush=True)

            if key in (ord("q"), 27):
                break
    finally:
        if collection is not None:
            collection.write_outputs()
        if evaluation is not None:
            evaluation.write_outputs()
        if midi_output is not None:
            midi_output.close()
        if latest_capture is not None:
            latest_capture.stop()
        capture.release()
        cv2.destroyAllWindows()

    return 0


def setup_preview_window(cv2, window_name: str) -> None:
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, PREVIEW_WIDTH, PREVIEW_HEIGHT)


def resize_preview(cv2, frame_bgr):
    height, width = frame_bgr.shape[:2]
    if (width, height) == (PREVIEW_WIDTH, PREVIEW_HEIGHT):
        return frame_bgr
    return cv2.resize(frame_bgr, (PREVIEW_WIDTH, PREVIEW_HEIGHT), interpolation=cv2.INTER_LINEAR)


def print_midi_outputs() -> None:
    outputs = available_midi_outputs()
    if not outputs:
        print("No MIDI output ports found.")
        return
    for index, output in enumerate(outputs):
        print(f"{index}: {output}")


def create_midi_output(args: argparse.Namespace) -> MidiOutput | None:
    if args.no_midi_out:
        return None
    channel = max(1, min(16, args.midi_channel)) - 1
    port_name = args.midi_out or None
    output = MidiOutput(MidiOutputConfig(port_name=port_name, channel=channel))
    target = port_name if port_name is not None else "default MIDI output"
    print(f"Sending MIDI drum hits to {target}.", file=sys.stderr, flush=True)
    return output


def create_hand_hit_detector(args: argparse.Namespace) -> tuple[HandHitDetector, "TargetSampleCollectionSession | None"]:
    if args.collect_target_samples is None and args.collection_raw_video is not None:
        raise ValueError("--collection-raw-video requires --collect-target-samples.")
    if args.collect_target_samples is not None:
        collection = TargetSampleCollectionSession(
            output_path=Path(args.collect_target_samples),
            samples_per_target=max(1, args.samples_per_target),
            countdown_seconds=max(0.0, args.collection_countdown),
            append=args.collection_output_mode == "append",
            raw_video_path=collection_raw_video_path(args.collection_raw_video),
            raw_video_fps=max(1.0, args.collection_raw_video_fps),
        )
        return create_hand_detector_instance(args, target_classifier=collection), collection
    return create_hand_detector_instance(args, target_classifier=create_target_classifier(args)), None


def create_hand_detector_instance(args, *, target_classifier) -> HandHitDetector:
    config = hand_detector_config_from_args(args)
    return HandHitDetector(config=config, target_classifier=target_classifier)


def hand_detector_config_from_args(args) -> HandHitDetectorConfig:
    defaults = HandHitDetectorConfig()
    return HandHitDetectorConfig(
        up_threshold=defaults.up_threshold if args.hand_up_threshold is None else args.hand_up_threshold,
        down_velocity_threshold=(
            defaults.down_velocity_threshold
            if args.hand_down_velocity_threshold is None
            else args.hand_down_velocity_threshold
        ),
        min_hit_interval_seconds=defaults.min_hit_interval_seconds,
        max_missing_frames=defaults.max_missing_frames,
        target_history_size=defaults.target_history_size,
        hand_motion_weight=defaults.hand_motion_weight,
        forearm_motion_weight=defaults.forearm_motion_weight,
    )


def create_evaluation_recording_session(args: argparse.Namespace) -> "EvaluationRecordingSession | None":
    if args.record_evaluation_video is None:
        if args.evaluation_annotations is not None:
            raise ValueError("--evaluation-annotations requires --record-evaluation-video.")
        return None
    video_path = recorded_video_path(args.record_evaluation_video)
    annotations_path = (
        recorded_video_path(args.evaluation_annotations)
        if args.evaluation_annotations is not None
        else video_path.with_suffix(".json")
    )
    return EvaluationRecordingSession(
        video_path=video_path,
        annotations_path=annotations_path,
        fps=max(1.0, args.evaluation_video_fps),
        countdown_seconds=max(0.0, args.evaluation_countdown),
    )


def collection_raw_video_path(raw_video_path: str | None) -> Path | None:
    if raw_video_path is None:
        return None
    return recorded_video_path(raw_video_path)


def recorded_video_path(raw_video_path: str) -> Path:
    path = Path(raw_video_path)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == str(DEFAULT_COLLECTION_RAW_VIDEO_DIR):
        return path
    return DEFAULT_COLLECTION_RAW_VIDEO_DIR / path


def create_target_classifier(args: argparse.Namespace) -> KnnDrumTargetClassifier:
    if args.hand_target_model is not None:
        return KnnDrumTargetClassifier.from_path(args.hand_target_model, distance_metric=args.knn_distance)
    return KnnDrumTargetClassifier.from_default_model(distance_metric=args.knn_distance)


class TargetSampleCollectionSession:
    def __init__(
        self,
        *,
        output_path: Path,
        samples_per_target: int,
        countdown_seconds: float,
        append: bool,
        raw_video_path: Path | None = None,
        raw_video_fps: float = 30.0,
        raw_video_recorder: "RawVideoRecorder | None" = None,
    ) -> None:
        self.output_path = output_path
        self.data = load_or_create_target_sample_file(output_path, append=append)
        self.samples_per_target = samples_per_target
        self.countdown_seconds = countdown_seconds
        self.prompts = [(side, drum) for side in TARGET_HAND_SIDES for drum in TARGET_DRUMS]
        self.prompt_index = 0
        self.current_prompt_count = 0
        self.state = "waiting"
        self.countdown_started_at: float | None = None
        self.records: list[dict] = []
        self.completed = False
        self._written = False
        if raw_video_path is not None and raw_video_recorder is not None:
            raise ValueError("Pass either raw_video_path or raw_video_recorder, not both.")
        self.raw_video_recorder = raw_video_recorder
        if raw_video_path is not None:
            self.raw_video_recorder = RawVideoRecorder(output_path=raw_video_path, fps=raw_video_fps)

    @property
    def records_raw_video(self) -> bool:
        return self.raw_video_recorder is not None

    @property
    def current_side(self) -> str:
        return self.prompts[self.prompt_index][0]

    @property
    def current_drum(self) -> str:
        return self.prompts[self.prompt_index][1]

    def print_prompt(self) -> None:
        print(
            f"Collecting {self.current_side} {self.current_drum}: press Space, wait "
            f"{self.countdown_seconds:.0f}s, then play {self.samples_per_target} clean hits.",
            file=sys.stderr,
            flush=True,
        )

    def handle_key(self, key: int, now: float) -> None:
        if key == ord("p") and not self.completed:
            self._toggle_pause(now)
            return
        if key == ord(" ") and self.state == "waiting" and not self.completed:
            self.state = "countdown"
            self.countdown_started_at = now
            print("Collection countdown started.", file=sys.stderr, flush=True)
            return
        if key in BACKSPACE_KEYS:
            self.undo_latest()

    def preview_line(self, now: float) -> str:
        self._update_countdown(now)
        if self.completed:
            return f"Collection complete | wrote {len(self.records)} samples"
        target = f"{self.current_side} {self.current_drum} {self.current_prompt_count}/{self.samples_per_target}"
        if self.state == "paused":
            return f"Paused {target} | Press p to resume | Backspace undo | q/Esc saves and quits"
        if self.state == "waiting":
            return f"Collect {target} | Press Space to start | p pause | Backspace undo | q/Esc saves and quits"
        if self.state == "countdown":
            remaining = self._countdown_remaining(now)
            return f"Collect {target} | Recording starts in {remaining:.1f}s | p pause"
        return f"Recording {target} | Play clean hits | p pause | Backspace undo | q/Esc saves and quits"

    def classify(self, observation: DrumTargetObservation) -> DrumTargetPrediction | None:
        if self.completed:
            return None
        self._update_countdown(time.perf_counter())
        if self.state != "recording":
            return None
        if observation.side != self.current_side:
            return None
        record = target_observation_to_sample_record(
            drum=self.current_drum,
            observation=observation,
            timestamp_seconds=observation.timestamp_seconds,
            velocity=observation.strike_velocity,
        )
        if record is None:
            return None
        raw_video_reference = self._raw_video_sample_reference(observation.timestamp_seconds)
        if raw_video_reference is not None:
            record["raw_video"] = raw_video_reference
        self.records.append(record)
        self.current_prompt_count += 1
        print(
            f"sample drum={self.current_drum} side={self.current_side} "
            f"{self.current_prompt_count}/{self.samples_per_target}",
            file=sys.stderr,
            flush=True,
        )
        if self.current_prompt_count >= self.samples_per_target:
            self.prompt_index += 1
            self.current_prompt_count = 0
            self.state = "waiting"
            self.countdown_started_at = None
            if self.prompt_index >= len(self.prompts):
                self.completed = True
                self.write_outputs()
            else:
                self.print_prompt()
        return DrumTargetPrediction(drum=record["drum"], context_name="target_sample_collection", confidence=1.0)

    def undo_latest(self) -> None:
        if not self.records:
            return
        latest = self.records[-1]
        if latest["drum"] != self.current_drum or latest["side"] != self.current_side:
            return
        self.records.pop()
        self.current_prompt_count = max(0, self.current_prompt_count - 1)
        print(
            f"removed sample drum={self.current_drum} side={self.current_side} "
            f"{self.current_prompt_count}/{self.samples_per_target}",
            file=sys.stderr,
            flush=True,
        )

    def write_outputs(self) -> None:
        if self._written:
            return
        self.close_video()
        if self.raw_video_recorder is not None and self.raw_video_recorder.frames_written > 0:
            self.data.setdefault("raw_videos", []).append(self.raw_video_recorder.metadata())
        self.data["samples"].extend(self.records)
        write_target_sample_file(self.output_path, self.data)
        print(f"Wrote {len(self.records)} target samples to {self.output_path}.", file=sys.stderr, flush=True)
        self._written = True

    def record_video_frame(self, frame_bgr, timestamp_seconds: float) -> None:
        self._update_countdown(timestamp_seconds)
        if self.raw_video_recorder is None or self.state != "recording" or self.completed:
            return
        self.raw_video_recorder.write_frame(frame_bgr, timestamp_seconds=timestamp_seconds)

    def close_video(self) -> None:
        if self.raw_video_recorder is not None:
            self.raw_video_recorder.close()

    def _raw_video_sample_reference(self, timestamp_seconds: float) -> dict | None:
        if self.raw_video_recorder is None:
            return None
        return self.raw_video_recorder.sample_reference(timestamp_seconds=timestamp_seconds)

    def _countdown_remaining(self, now: float) -> float:
        if self.countdown_started_at is None:
            return self.countdown_seconds
        return max(0.0, self.countdown_seconds - (now - self.countdown_started_at))

    def _update_countdown(self, now: float) -> None:
        if self.state == "countdown" and self._countdown_remaining(now) <= 0.0:
            self.state = "recording"
            print(
                f"Recording {self.current_side} {self.current_drum}.",
                file=sys.stderr,
                flush=True,
            )

    def _toggle_pause(self, now: float) -> None:
        if self.state == "paused":
            self.state = "waiting"
            self.countdown_started_at = None
            print("Collection resumed. Press Space to start.", file=sys.stderr, flush=True)
            return
        self.state = "paused"
        self.countdown_started_at = None
        print("Collection paused. Press p to resume.", file=sys.stderr, flush=True)


class RawVideoRecorder:
    def __init__(self, *, output_path: Path, fps: float) -> None:
        self.output_path = output_path
        self.fps = float(fps)
        self.writer = None
        self.frames_written = 0
        self.width: int | None = None
        self.height: int | None = None
        self.started_at_seconds: float | None = None
        self.last_frame_timestamp_seconds: float | None = None
        self._closed = False

    def write_frame(self, frame_bgr, *, timestamp_seconds: float) -> None:
        if self._closed:
            return
        if self.writer is None:
            self._open(frame_bgr)
            self.started_at_seconds = timestamp_seconds
            print(f"Recording raw collection video to {self.output_path}.", file=sys.stderr, flush=True)
        self.writer.write(frame_bgr)
        self.frames_written += 1
        self.last_frame_timestamp_seconds = timestamp_seconds

    def sample_reference(self, *, timestamp_seconds: float) -> dict | None:
        if self.frames_written <= 0:
            return None
        frame_index = self.frames_written - 1
        return {
            "path": str(self.output_path),
            "frame_index": frame_index,
            "video_timestamp_seconds": frame_index / self.fps,
            "sample_timestamp_seconds": float(timestamp_seconds),
        }

    def metadata(self) -> dict:
        return {
            "path": str(self.output_path),
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "frames": self.frames_written,
            "started_at_seconds": self.started_at_seconds,
            "last_frame_timestamp_seconds": self.last_frame_timestamp_seconds,
        }

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        if not self._closed and self.frames_written > 0:
            print(
                f"Wrote {self.frames_written} raw video frames to {self.output_path}.",
                file=sys.stderr,
                flush=True,
            )
        self._closed = True

    def _open(self, frame_bgr) -> None:
        cv2 = load_opencv()
        height, width = frame_bgr.shape[:2]
        self.width = int(width)
        self.height = int(height)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(self.output_path), fourcc, self.fps, (self.width, self.height))
        if not self.writer.isOpened():
            self.writer.release()
            self.writer = None
            raise RuntimeError(f"Could not open raw video writer at {self.output_path}.")


class EvaluationRecordingSession:
    def __init__(
        self,
        *,
        video_path: Path,
        annotations_path: Path,
        fps: float,
        countdown_seconds: float = 10.0,
        raw_video_recorder: RawVideoRecorder | None = None,
    ) -> None:
        self.video_path = video_path
        self.annotations_path = annotations_path
        self.countdown_seconds = countdown_seconds
        self.raw_video_recorder = raw_video_recorder or RawVideoRecorder(output_path=video_path, fps=fps)
        self.events: list[dict] = []
        self.started_at_seconds: float | None = None
        self.recording_started = False
        self._written = False

    def is_recording(self, now: float) -> bool:
        if self.started_at_seconds is None:
            self.started_at_seconds = now
            if self.countdown_seconds > 0.0:
                print(
                    f"Evaluation recording starts in {self.countdown_seconds:.1f}s.",
                    file=sys.stderr,
                    flush=True,
                )
        recording = now - self.started_at_seconds >= self.countdown_seconds
        if recording and not self.recording_started:
            self.recording_started = True
            print(f"Recording evaluation video to {self.video_path}.", file=sys.stderr, flush=True)
        return recording

    def preview_line(self, now: float) -> str:
        if not self.is_recording(now):
            return f"Evaluation recording starts in {self.countdown_remaining(now):.1f}s | q/Esc quits"
        return f"Recording evaluation | events {len(self.events)} | q/Esc saves and quits"

    def countdown_remaining(self, now: float) -> float:
        if self.started_at_seconds is None:
            return self.countdown_seconds
        return max(0.0, self.countdown_seconds - (now - self.started_at_seconds))

    def record_video_frame(self, frame_bgr, timestamp_seconds: float) -> None:
        if not self.is_recording(timestamp_seconds):
            return
        self.raw_video_recorder.write_frame(frame_bgr, timestamp_seconds=timestamp_seconds)

    def record_events(
        self,
        *,
        hand_events: list[HandHitEvent],
        pedal_events: list[HitEvent],
        timestamp_seconds: float,
    ) -> None:
        if not self.is_recording(timestamp_seconds):
            return
        frame_reference = self.raw_video_recorder.sample_reference(timestamp_seconds=timestamp_seconds)
        for event in hand_events:
            self.events.append(self._hand_event_record(event, frame_reference=frame_reference))
        for event in pedal_events:
            self.events.append(self._pedal_event_record(event, frame_reference=frame_reference))

    def write_outputs(self) -> None:
        if self._written:
            return
        self.raw_video_recorder.close()
        self.annotations_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "recording_type": "virtual_drumming_evaluation_recording_v1",
            "video": self.raw_video_recorder.metadata(),
            "events": self.events,
        }
        with self.annotations_path.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, sort_keys=True)
            file.write("\n")
        print(
            f"Wrote {len(self.events)} evaluation events to {self.annotations_path}.",
            file=sys.stderr,
            flush=True,
        )
        self._written = True

    def _hand_event_record(self, event: HandHitEvent, *, frame_reference: dict | None) -> dict:
        record = {
            "type": "hand",
            "side": event.side,
            "drum": event.drum,
            "timestamp_seconds": event.timestamp_seconds,
            "velocity": event.velocity,
            "wrist_x": event.wrist_x,
            "wrist_y": event.wrist_y,
            "strike_motion_y": event.strike_motion_y,
            "hand_motion_y": event.hand_motion_y,
            "forearm_motion_y": event.forearm_motion_y,
            "context_name": event.context_name,
            "confidence": event.confidence,
        }
        if frame_reference is not None:
            record["video"] = frame_reference
        return record

    def _pedal_event_record(self, event: HitEvent, *, frame_reference: dict | None) -> dict:
        record = {
            "type": "pedal",
            "side": event.side,
            "pedal": event.pedal_id,
            "point": event.point_id,
            "timestamp_seconds": event.timestamp_seconds,
            "velocity": event.velocity,
            "y": event.y,
            "previous_y": event.previous_y,
        }
        if frame_reference is not None:
            record["video"] = frame_reference
        return record


def process_drum_engine_hits(
    landmarks: dict,
    *,
    landmark_smoother: LandmarkSmoother,
    hand_hit_detector: HandHitDetector,
    pedal_hit_detector: PedalHitDetector,
    midi_output: MidiOutput | None,
    timestamp_seconds: float,
    emit_hits: bool = True,
):
    pose_frame = apple_vision_landmarks_to_pose_frame(landmarks, timestamp_seconds=timestamp_seconds)
    smoothed_frame = landmark_smoother.update(pose_frame)
    if not emit_hits:
        return smoothed_frame, [], []
    hand_events = hand_hit_detector.update(smoothed_frame)
    for event in hand_events:
        print(format_hand_hit_event(event), flush=True)
        if midi_output is not None:
            midi_output.send_hand_hit(event)
    pedal_events = pedal_hit_detector.update(smoothed_frame)
    for event in pedal_events:
        print(format_hit_event(event), flush=True)
        if midi_output is not None:
            midi_output.send_pedal_hit(event)
    return smoothed_frame, hand_events, pedal_events


def current_hand_drum_labels(
    hand_hit_detector: HandHitDetector,
    smoothed_frame,
    *,
    collection: "TargetSampleCollectionSession | None",
) -> dict[str, str]:
    if collection is not None:
        return {}
    return hand_hit_detector.classify_current_targets(smoothed_frame)


def format_hit_event(event: HitEvent) -> str:
    return (
        f"hit side={event.side} "
        f"pedal={event.pedal_id} "
        f"point={event.point_id} "
        f"time={event.timestamp_seconds:.3f} "
        f"velocity={event.velocity:.3f} "
        f"y={event.y:.3f} "
        f"previous_y={event.previous_y:.3f}"
    )


def format_hand_hit_event(event: HandHitEvent) -> str:
    base = (
        f"hand_hit side={event.side} "
        f"drum={event.drum} "
        f"time={event.timestamp_seconds:.3f} "
        f"velocity={event.velocity:.3f} "
        f"wrist_x={event.wrist_x:.3f} "
        f"wrist_y={event.wrist_y:.3f} "
        f"motion_y={event.strike_motion_y:.3f} "
        f"hand_motion_y={event.hand_motion_y:.3f} "
        f"forearm_motion_y={event.forearm_motion_y:.3f}"
    )
    return (
        f"{base} "
        f"context={event.context_name} "
        f"confidence={event.confidence:.2f}"
    )
