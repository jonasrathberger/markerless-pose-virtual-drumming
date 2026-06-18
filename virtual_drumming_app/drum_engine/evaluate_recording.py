"""Evaluate recorded drumming video against a reference MIDI file."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .apple_vision import apple_vision_landmarks_to_pose_frame
from .hand_hit_detection import (
    HAND_SIDES,
    HandHitDetector,
    HandHitDetectorConfig,
    _hand_sample,
)
from .midi import MidiOutput, MidiOutputConfig
from .pedal_hit_detection import PedalHitDetector, _pedal_signal_y
from .schema import PoseFrame
from .smoothing import LandmarkSmoother
from .target_classification import (
    DrumTargetClassifier,
)

MatchMode = Literal["type", "drum"]

MIDI_NOTE_TO_DRUM = {
    36: "right_pedal",
    37: "snare",
    38: "snare",
    42: "hi_hat",
    43: "floor_tom",
    44: "left_pedal",
    45: "tom_2",
    46: "hi_hat",
    47: "tom_2",
    48: "tom_1",
    49: "crash",
    50: "tom_1",
    51: "ride",
}


@dataclass(frozen=True, slots=True)
class EvaluationEvent:
    time_seconds: float
    event_type: str
    drum: str
    side: str | None = None
    note: int | None = None


@dataclass(frozen=True, slots=True)
class MatchStats:
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    false_negative_drums: dict[str, int]
    false_positive_drums: dict[str, int]


@dataclass(frozen=True, slots=True)
class AlignmentResult:
    offset_seconds: float
    scale: float
    stats: MatchStats


@dataclass(frozen=True, slots=True)
class SignalSample:
    time_seconds: float
    event_type: str
    side: str
    signal_y: float
    velocity: float | None
    state_active: bool
    down_displacement: float | None = None


@dataclass(frozen=True, slots=True)
class ReplayData:
    detected_events: list[EvaluationEvent]
    signal_samples: list[SignalSample]


@dataclass(frozen=True, slots=True)
class _MatchDetails:
    true_positive_count: int
    false_negatives: list[tuple[EvaluationEvent, float]]
    false_positives: list[EvaluationEvent]


def load_midi_reference_events(path: Path) -> list[EvaluationEvent]:
    mido = _load_mido()
    midi = mido.MidiFile(path)
    events: list[EvaluationEvent] = []
    timestamp_seconds = 0.0
    for message in midi:
        timestamp_seconds += float(message.time)
        if message.type != "note_on" or message.velocity <= 0:
            continue
        drum = MIDI_NOTE_TO_DRUM.get(message.note)
        if drum is None:
            continue
        events.append(
            EvaluationEvent(
                time_seconds=timestamp_seconds,
                event_type="pedal" if drum in ("left_pedal", "right_pedal") else "hand",
                drum=drum,
                side=_side_for_drum(drum),
                note=message.note,
            )
        )
    return events


def replay_video_events(video_path: Path, *, progress_every: int = 300) -> list[EvaluationEvent]:
    return replay_video_analysis(video_path, progress_every=progress_every, collect_signals=False).detected_events


def replay_video_analysis(
    video_path: Path,
    *,
    progress_every: int = 300,
    collect_signals: bool = True,
    hand_hit_detector_config: HandHitDetectorConfig | None = None,
    preview: bool = False,
    preview_speed: float = 1.0,
    analysis_speed: float = 1.0,
    preview_mirror: bool = True,
    midi_output: MidiOutput | None = None,
    target_classifier: DrumTargetClassifier | None = None,
) -> ReplayData:
    cv2 = _load_opencv()
    from apple_vision_pose.vision import AppleVisionPose

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video {video_path}.")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    detector = AppleVisionPose(max_hands=2, detect_width=0, extract_all_landmarks=False)
    smoother = LandmarkSmoother()
    hand_hit_detector = HandHitDetector(config=hand_hit_detector_config, target_classifier=target_classifier)
    pedal_hit_detector = PedalHitDetector()
    detected: list[EvaluationEvent] = []
    recent_hits: list[EvaluationEvent] = []
    signal_samples: list[SignalSample] = []
    frame_index = 0
    started_at = time.perf_counter()
    window_name = "Evaluation Replay"
    if preview:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
    try:
        while True:
            frame_started_at = time.perf_counter()
            ok, frame = capture.read()
            if not ok:
                break
            timestamp_seconds = _frame_timestamp(frame_index, fps=fps, analysis_speed=analysis_speed)
            landmarks = detector.detect(frame)
            pose_frame = apple_vision_landmarks_to_pose_frame(landmarks, timestamp_seconds=timestamp_seconds)
            smoothed_frame = smoother.update(pose_frame)
            if collect_signals:
                signal_samples.extend(_signal_samples_from_frame(smoothed_frame, hand_hit_detector, pedal_hit_detector))
            hand_hit_events = hand_hit_detector.update(smoothed_frame)
            pedal_hit_events = pedal_hit_detector.update(smoothed_frame)
            if midi_output is not None:
                for event in hand_hit_events:
                    midi_output.send_hand_hit(event)
                for event in pedal_hit_events:
                    midi_output.send_pedal_hit(event)
            frame_events = [
                *_hand_hit_events_to_evaluation_events(hand_hit_events),
                *_pedal_hit_events_to_evaluation_events(pedal_hit_events),
            ]
            detected.extend(frame_events)
            if preview:
                recent_hits.extend(frame_events)
                recent_hits = [
                    event for event in recent_hits if timestamp_seconds - event.time_seconds <= 1.25
                ]
                hand_drum_labels = hand_hit_detector.classify_current_targets(smoothed_frame)
                annotated = _preview_frame(
                    frame,
                    landmarks,
                    timestamp_seconds=timestamp_seconds,
                    frame_index=frame_index,
                    total_detected=len(detected),
                    recent_hits=recent_hits,
                    hand_drum_labels=hand_drum_labels,
                    draw_all=False,
                    mirror_preview=preview_mirror,
                )
                cv2.imshow(window_name, annotated)
                _sleep_for_preview_timing(frame_started_at, fps=fps, speed=preview_speed)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
            frame_index += 1
            if progress_every > 0 and frame_index % progress_every == 0:
                elapsed = time.perf_counter() - started_at
                print(f"Processed {frame_index} frames in {elapsed:.1f}s.", file=sys.stderr, flush=True)
    finally:
        capture.release()
        if preview:
            cv2.destroyWindow(window_name)
    return ReplayData(detected_events=detected, signal_samples=signal_samples)


def match_events(
    reference_events: list[EvaluationEvent],
    detected_events: list[EvaluationEvent],
    *,
    offset_seconds: float,
    scale: float,
    tolerance_seconds: float,
    mode: MatchMode,
) -> MatchStats:
    details = _match_event_details(
        reference_events,
        detected_events,
        offset_seconds=offset_seconds,
        scale=scale,
        tolerance_seconds=tolerance_seconds,
        mode=mode,
    )
    true_positives = details.true_positive_count
    false_positives = details.false_positives
    false_negatives = details.false_negatives
    precision = _ratio(true_positives, true_positives + len(false_positives))
    recall = _ratio(true_positives, true_positives + len(false_negatives))
    f1 = _ratio(2 * precision * recall, precision + recall)
    return MatchStats(
        true_positives=true_positives,
        false_positives=len(false_positives),
        false_negatives=len(false_negatives),
        precision=precision,
        recall=recall,
        f1=f1,
        false_negative_drums=dict(Counter(event.drum for event, _target in false_negatives)),
        false_positive_drums=dict(Counter(event.drum for event in false_positives)),
    )


def _match_event_details(
    reference_events: list[EvaluationEvent],
    detected_events: list[EvaluationEvent],
    *,
    offset_seconds: float,
    scale: float,
    tolerance_seconds: float,
    mode: MatchMode,
) -> _MatchDetails:
    detected_by_key: dict[tuple[str, ...], list[tuple[float, int]]] = defaultdict(list)
    for index, event in enumerate(detected_events):
        detected_by_key[_match_key(event, mode)].append((event.time_seconds, index))
    for events in detected_by_key.values():
        events.sort()

    used_detected: set[int] = set()
    false_negatives: list[tuple[EvaluationEvent, float]] = []
    for reference in reference_events:
        target_time = offset_seconds + (reference.time_seconds * scale)
        candidates = detected_by_key.get(_match_key(reference, mode), [])
        candidate_index = bisect.bisect_left(candidates, (target_time - tolerance_seconds, -1))
        best: tuple[float, int] | None = None
        while candidate_index < len(candidates) and candidates[candidate_index][0] <= target_time + tolerance_seconds:
            detected_time, detected_index = candidates[candidate_index]
            if detected_index not in used_detected:
                delta = abs(detected_time - target_time)
                if best is None or delta < best[0]:
                    best = (delta, detected_index)
            candidate_index += 1
        if best is None:
            false_negatives.append((reference, target_time))
            continue
        used_detected.add(best[1])

    false_positives = [
        event for index, event in enumerate(detected_events) if index not in used_detected
    ]
    return _MatchDetails(
        true_positive_count=len(used_detected),
        false_negatives=false_negatives,
        false_positives=false_positives,
    )


def find_best_alignment(
    reference_events: list[EvaluationEvent],
    detected_events: list[EvaluationEvent],
    *,
    tolerance_seconds: float,
    mode: MatchMode,
    offset_min: float = 0.0,
    offset_max: float = 5.0,
    scale_min: float = 1.8,
    scale_max: float = 2.2,
) -> AlignmentResult:
    best = _alignment_grid_search(
        reference_events,
        detected_events,
        tolerance_seconds=tolerance_seconds,
        mode=mode,
        offset_min=offset_min,
        offset_max=offset_max,
        offset_step=0.05,
        scale_min=scale_min,
        scale_max=scale_max,
        scale_step=0.005,
    )
    return _alignment_grid_search(
        reference_events,
        detected_events,
        tolerance_seconds=tolerance_seconds,
        mode=mode,
        offset_min=max(offset_min, best.offset_seconds - 0.15),
        offset_max=min(offset_max, best.offset_seconds + 0.15),
        offset_step=0.01,
        scale_min=max(scale_min, best.scale - 0.02),
        scale_max=min(scale_max, best.scale + 0.02),
        scale_step=0.001,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    reference_events = load_midi_reference_events(args.midi)
    midi_output = create_midi_output(args)
    try:
        replay_data = replay_video_analysis(
            args.video,
            progress_every=args.progress_every,
            collect_signals=args.miss_analysis_csv is not None,
            hand_hit_detector_config=hand_detector_config_from_args(args),
            preview=args.preview,
            preview_speed=args.preview_speed,
            analysis_speed=args.analysis_speed,
            preview_mirror=not args.no_preview_mirror,
            midi_output=midi_output,
        )
    finally:
        if midi_output is not None:
            midi_output.close()
    detected_events = replay_data.detected_events
    report = build_report(
        reference_events,
        detected_events,
        tolerance_seconds=args.tolerance,
        offset_min=args.offset_min,
        offset_max=args.offset_max,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
    )
    if args.miss_analysis_csv is not None:
        mode_report = report["modes"][args.miss_analysis_mode]
        write_miss_analysis_csv(
            args.miss_analysis_csv,
            reference_events,
            detected_events,
            replay_data.signal_samples,
            offset_seconds=mode_report["offset_seconds"],
            scale=mode_report["scale"],
            tolerance_seconds=args.tolerance,
            mode=args.miss_analysis_mode,
            window_seconds=args.miss_analysis_window,
        )
        print(f"Wrote miss analysis CSV to {args.miss_analysis_csv}.", file=sys.stderr, flush=True)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_text_report(args.video, args.midi, report)
    return 0


def build_report(
    reference_events: list[EvaluationEvent],
    detected_events: list[EvaluationEvent],
    *,
    tolerance_seconds: float,
    offset_min: float | None = None,
    offset_max: float | None = None,
    scale_min: float | None = None,
    scale_max: float | None = None,
) -> dict:
    resolved_offset_min, resolved_offset_max, resolved_scale_min, resolved_scale_max = _resolve_alignment_bounds(
        reference_events,
        detected_events,
        offset_min=offset_min,
        offset_max=offset_max,
        scale_min=scale_min,
        scale_max=scale_max,
    )
    modes: dict[str, dict] = {}
    for mode in ("type", "drum"):
        alignment = find_best_alignment(
            reference_events,
            detected_events,
            tolerance_seconds=tolerance_seconds,
            mode=mode,
            offset_min=resolved_offset_min,
            offset_max=resolved_offset_max,
            scale_min=resolved_scale_min,
            scale_max=resolved_scale_max,
        )
        modes[mode] = _alignment_to_dict(alignment)
    return {
        "reference_events": len(reference_events),
        "detected_events": len(detected_events),
        "reference_drums": dict(Counter(event.drum for event in reference_events)),
        "detected_drums": dict(Counter(event.drum for event in detected_events)),
        "tolerance_seconds": tolerance_seconds,
        "alignment_search": {
            "offset_min": resolved_offset_min,
            "offset_max": resolved_offset_max,
            "scale_min": resolved_scale_min,
            "scale_max": resolved_scale_max,
        },
        "modes": modes,
    }


def write_miss_analysis_csv(
    path: Path,
    reference_events: list[EvaluationEvent],
    detected_events: list[EvaluationEvent],
    signal_samples: list[SignalSample],
    *,
    offset_seconds: float,
    scale: float,
    tolerance_seconds: float,
    mode: MatchMode,
    window_seconds: float = 0.250,
) -> None:
    details = _match_event_details(
        reference_events,
        detected_events,
        offset_seconds=offset_seconds,
        scale=scale,
        tolerance_seconds=tolerance_seconds,
        mode=mode,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "mode",
        "midi_time_seconds",
        "expected_video_time_seconds",
        "event_type",
        "drum",
        "note",
        "side",
        "simultaneous_drums",
        "nearest_same_type_time_seconds",
        "nearest_same_type_delta_ms",
        "nearest_same_type_drum",
        "nearest_same_drum_time_seconds",
        "nearest_same_drum_delta_ms",
        "nearest_same_drum_drum",
        "nearest_any_time_seconds",
        "nearest_any_delta_ms",
        "nearest_any_type",
        "nearest_any_drum",
        "local_sample_count",
        "local_peak_velocity",
        "local_peak_velocity_time_seconds",
        "local_peak_velocity_delta_ms",
        "local_peak_velocity_side",
        "local_peak_signal_y",
        "local_peak_state_active",
        "local_peak_down_displacement",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for reference, target_time in details.false_negatives:
            peak = _local_peak_signal(reference, target_time, signal_samples, window_seconds)
            row = {
                "mode": mode,
                "midi_time_seconds": _format_float(reference.time_seconds),
                "expected_video_time_seconds": _format_float(target_time),
                "event_type": reference.event_type,
                "drum": reference.drum,
                "note": "" if reference.note is None else reference.note,
                "side": reference.side or "",
                "simultaneous_drums": "+".join(_simultaneous_drums(reference_events, reference)),
                "local_sample_count": _local_signal_count(reference, target_time, signal_samples, window_seconds),
            }
            row.update(_nearest_event_fields("nearest_same_type", target_time, detected_events, lambda event: event.event_type == reference.event_type))
            row.update(_nearest_event_fields("nearest_same_drum", target_time, detected_events, lambda event: event.drum == reference.drum))
            row.update(_nearest_event_fields("nearest_any", target_time, detected_events, lambda _event: True))
            row.update(_peak_signal_fields(peak, target_time))
            writer.writerow(row)


def print_text_report(video_path: Path, midi_path: Path, report: dict) -> None:
    print(f"Video: {video_path}")
    print(f"MIDI: {midi_path}")
    print(f"Reference events: {report['reference_events']}")
    print(f"Detected events: {report['detected_events']}")
    print(f"Tolerance: {report['tolerance_seconds']:.3f}s")
    search = report["alignment_search"]
    print(
        "Alignment search: offset={offset_min:.3f}..{offset_max:.3f}s scale={scale_min:.3f}..{scale_max:.3f}".format(
            **search
        )
    )
    for mode_name, mode_report in report["modes"].items():
        print()
        print(f"{mode_name} match")
        print(f"  offset_seconds: {mode_report['offset_seconds']:.3f}")
        print(f"  scale: {mode_report['scale']:.3f}")
        print(
            "  tp={tp} fp={fp} fn={fn} precision={precision:.3f} recall={recall:.3f} f1={f1:.3f}".format(
                **mode_report["stats"]
            )
        )
        print(f"  missed_by_drum: {mode_report['stats']['false_negative_drums']}")
        print(f"  extra_by_drum: {mode_report['stats']['false_positive_drums']}")


def hand_detector_config_from_args(args: argparse.Namespace) -> HandHitDetectorConfig:
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


def create_midi_output(args: argparse.Namespace) -> MidiOutput | None:
    if args.midi_out is None:
        return None
    channel = max(1, min(16, args.midi_channel)) - 1
    port_name = args.midi_out or None
    target = port_name if port_name is not None else "default MIDI output"
    print(f"Sending replay hits to {target}.", file=sys.stderr, flush=True)
    return MidiOutput(MidiOutputConfig(port_name=port_name, channel=channel))


def _preview_frame(
    frame,
    landmarks: dict,
    *,
    timestamp_seconds: float,
    frame_index: int,
    total_detected: int,
    recent_hits: list[EvaluationEvent],
    hand_drum_labels: dict[str, str],
    draw_all: bool,
    mirror_preview: bool,
):
    from apple_vision_pose.rendering import draw_landmarks, draw_wrapped_hud

    performance_line = (
        f"replay {timestamp_seconds:.2f}s | frame {frame_index} | hits {total_detected} | "
        f"recent {_format_recent_hits(recent_hits)}"
    )
    annotated = draw_landmarks(
        frame,
        landmarks,
        performance_line,
        draw_all=draw_all,
        mirror_preview=mirror_preview,
        hand_drum_labels=hand_drum_labels,
    )
    if recent_hits:
        draw_wrapped_hud(
            annotated,
            "detected: " + " | ".join(_format_event(event) for event in recent_hits[-5:]),
            y_origin=95,
        )
    return annotated


def _format_recent_hits(events: list[EvaluationEvent]) -> str:
    if not events:
        return "-"
    return ", ".join(_format_event(event) for event in events[-4:])


def _format_event(event: EvaluationEvent) -> str:
    if event.event_type == "pedal":
        return event.drum
    return f"{event.side or '?'}:{event.drum}"


def _sleep_for_preview_timing(frame_started_at: float, *, fps: float, speed: float) -> None:
    if speed <= 0.0:
        return
    target_seconds = (1.0 / max(fps, 1.0)) / speed
    remaining = target_seconds - (time.perf_counter() - frame_started_at)
    if remaining > 0:
        time.sleep(remaining)


def _frame_timestamp(frame_index: int, *, fps: float, analysis_speed: float) -> float:
    return frame_index / (fps * max(analysis_speed, 1e-9))


def _alignment_grid_search(
    reference_events: list[EvaluationEvent],
    detected_events: list[EvaluationEvent],
    *,
    tolerance_seconds: float,
    mode: MatchMode,
    offset_min: float,
    offset_max: float,
    offset_step: float,
    scale_min: float,
    scale_max: float,
    scale_step: float,
) -> AlignmentResult:
    best: AlignmentResult | None = None
    scale = _first_grid_value(scale_min, scale_step)
    while scale <= scale_max + 1e-9:
        offset = _first_grid_value(offset_min, offset_step)
        while offset <= offset_max + 1e-9:
            stats = match_events(
                reference_events,
                detected_events,
                offset_seconds=offset,
                scale=scale,
                tolerance_seconds=tolerance_seconds,
                mode=mode,
            )
            candidate = AlignmentResult(offset_seconds=offset, scale=scale, stats=stats)
            if best is None or _alignment_rank(candidate) > _alignment_rank(best):
                best = candidate
            offset += offset_step
        scale += scale_step
    assert best is not None
    return best


def _first_grid_value(minimum: float, step: float) -> float:
    return math.ceil((minimum - 1e-9) / step) * step


def _resolve_alignment_bounds(
    reference_events: list[EvaluationEvent],
    detected_events: list[EvaluationEvent],
    *,
    offset_min: float | None,
    offset_max: float | None,
    scale_min: float | None,
    scale_max: float | None,
) -> tuple[float, float, float, float]:
    auto_offset_min, auto_offset_max, auto_scale_min, auto_scale_max = _auto_alignment_bounds(
        reference_events,
        detected_events,
    )
    return (
        auto_offset_min if offset_min is None else offset_min,
        auto_offset_max if offset_max is None else offset_max,
        auto_scale_min if scale_min is None else scale_min,
        auto_scale_max if scale_max is None else scale_max,
    )


def _auto_alignment_bounds(
    reference_events: list[EvaluationEvent],
    detected_events: list[EvaluationEvent],
) -> tuple[float, float, float, float]:
    if len(reference_events) < 2 or len(detected_events) < 2:
        return 0.0, 5.0, 1.8, 2.2
    reference_span = max(reference_events[-1].time_seconds - reference_events[0].time_seconds, 1e-9)
    detected_span = max(detected_events[-1].time_seconds - detected_events[0].time_seconds, 1e-9)
    estimated_scale = detected_span / reference_span
    scale_min = max(0.1, estimated_scale * 0.85)
    scale_max = estimated_scale * 1.15
    estimated_offset = detected_events[0].time_seconds - (reference_events[0].time_seconds * estimated_scale)
    return estimated_offset - 5.0, estimated_offset + 5.0, scale_min, scale_max


def _alignment_rank(alignment: AlignmentResult) -> tuple[float, int, int, int]:
    stats = alignment.stats
    return (
        stats.f1,
        stats.true_positives,
        -stats.false_positives,
        -stats.false_negatives,
    )


def _hand_events_from_frame(frame: PoseFrame, detector: HandHitDetector) -> list[EvaluationEvent]:
    return _hand_hit_events_to_evaluation_events(detector.update(frame))


def _hand_hit_events_to_evaluation_events(events) -> list[EvaluationEvent]:
    return [
        EvaluationEvent(
            time_seconds=event.timestamp_seconds,
            event_type="hand",
            drum=event.drum,
            side=event.side,
        )
        for event in events
    ]


def _pedal_events_from_frame(frame: PoseFrame, detector: PedalHitDetector) -> list[EvaluationEvent]:
    return _pedal_hit_events_to_evaluation_events(detector.update(frame))


def _pedal_hit_events_to_evaluation_events(events) -> list[EvaluationEvent]:
    return [
        EvaluationEvent(
            time_seconds=event.timestamp_seconds,
            event_type="pedal",
            drum=event.pedal_id,
            side=event.side,
        )
        for event in events
    ]


def _signal_samples_from_frame(
    frame: PoseFrame,
    hand_detector: HandHitDetector,
    pedal_detector: PedalHitDetector,
) -> list[SignalSample]:
    return [
        *_hand_signal_samples_from_frame(frame, hand_detector),
        *_pedal_signal_samples_from_frame(frame, pedal_detector),
    ]


def _hand_signal_samples_from_frame(frame: PoseFrame, detector: HandHitDetector) -> list[SignalSample]:
    samples: list[SignalSample] = []
    for side in HAND_SIDES:
        hand_sample = _hand_sample(frame, side)
        if hand_sample is None:
            continue
        state = detector._states[side]
        signal_y = hand_sample.strike_motion_y(
            hand_weight=detector.config.hand_motion_weight,
            forearm_weight=detector.config.forearm_motion_weight,
        )
        velocity = None
        if state.last_motion_y is not None and state.last_timestamp_seconds is not None:
            dt = frame.timestamp_seconds - state.last_timestamp_seconds
            if dt <= 0.0:
                dt = 1.0 / 60.0
            velocity = (signal_y - state.last_motion_y) / dt
        samples.append(
            SignalSample(
                time_seconds=frame.timestamp_seconds,
                event_type="hand",
                side=side,
                signal_y=signal_y,
                velocity=velocity,
                state_active=state.armed,
            )
        )
    return samples


def _pedal_signal_samples_from_frame(frame: PoseFrame, detector: PedalHitDetector) -> list[SignalSample]:
    samples: list[SignalSample] = []
    for side, point_id in detector.pedal_points.items():
        point = frame.points.get(point_id)
        if point is None or not point.tracking_present:
            continue
        anchor_id = detector.anchor_points.get(side)
        anchor = frame.points.get(anchor_id) if anchor_id is not None else None
        signal_y = _pedal_signal_y(point, anchor)
        if signal_y is None:
            continue
        state = detector._states[side]
        velocity = None
        down_displacement = None
        if state.last_y is not None and state.last_timestamp_seconds is not None:
            dt = frame.timestamp_seconds - state.last_timestamp_seconds
            if dt <= 0.0:
                dt = 1.0 / 60.0
            velocity = (signal_y - state.last_y) / dt
            if state.downstroke_start_y is not None:
                down_displacement = signal_y - state.downstroke_start_y
            elif velocity > 0:
                down_displacement = signal_y - state.last_y
        samples.append(
            SignalSample(
                time_seconds=frame.timestamp_seconds,
                event_type="pedal",
                side=side,
                signal_y=signal_y,
                velocity=velocity,
                state_active=state.pressed,
                down_displacement=down_displacement,
            )
        )
    return samples


def _nearest_event_fields(
    prefix: str,
    target_time: float,
    detected_events: list[EvaluationEvent],
    predicate,
) -> dict:
    nearest = _nearest_event(target_time, detected_events, predicate)
    if nearest is None:
        fields = {
            f"{prefix}_time_seconds": "",
            f"{prefix}_delta_ms": "",
            f"{prefix}_drum": "",
        }
        if prefix == "nearest_any":
            fields[f"{prefix}_type"] = ""
        return fields
    delta_seconds, event = nearest
    fields = {
        f"{prefix}_time_seconds": _format_float(event.time_seconds),
        f"{prefix}_delta_ms": _format_float(delta_seconds * 1000.0),
        f"{prefix}_drum": event.drum,
    }
    if prefix == "nearest_any":
        fields[f"{prefix}_type"] = event.event_type
    return fields


def _nearest_event(
    target_time: float,
    detected_events: list[EvaluationEvent],
    predicate,
) -> tuple[float, EvaluationEvent] | None:
    nearest: tuple[float, EvaluationEvent] | None = None
    for event in detected_events:
        if not predicate(event):
            continue
        delta_seconds = event.time_seconds - target_time
        if nearest is None or abs(delta_seconds) < abs(nearest[0]):
            nearest = (delta_seconds, event)
    return nearest


def _local_peak_signal(
    reference: EvaluationEvent,
    target_time: float,
    signal_samples: list[SignalSample],
    window_seconds: float,
) -> SignalSample | None:
    relevant = [
        sample
        for sample in signal_samples
        if sample.event_type == reference.event_type
        and (reference.side is None or sample.side == reference.side)
        and abs(sample.time_seconds - target_time) <= window_seconds
        and sample.velocity is not None
    ]
    if not relevant:
        return None
    return max(relevant, key=lambda sample: sample.velocity if sample.velocity is not None else float("-inf"))


def _local_signal_count(
    reference: EvaluationEvent,
    target_time: float,
    signal_samples: list[SignalSample],
    window_seconds: float,
) -> int:
    return sum(
        1
        for sample in signal_samples
        if sample.event_type == reference.event_type
        and (reference.side is None or sample.side == reference.side)
        and abs(sample.time_seconds - target_time) <= window_seconds
    )


def _peak_signal_fields(peak: SignalSample | None, target_time: float) -> dict:
    if peak is None:
        return {
            "local_peak_velocity": "",
            "local_peak_velocity_time_seconds": "",
            "local_peak_velocity_delta_ms": "",
            "local_peak_velocity_side": "",
            "local_peak_signal_y": "",
            "local_peak_state_active": "",
            "local_peak_down_displacement": "",
        }
    return {
        "local_peak_velocity": _format_float(peak.velocity),
        "local_peak_velocity_time_seconds": _format_float(peak.time_seconds),
        "local_peak_velocity_delta_ms": _format_float((peak.time_seconds - target_time) * 1000.0),
        "local_peak_velocity_side": peak.side,
        "local_peak_signal_y": _format_float(peak.signal_y),
        "local_peak_state_active": int(peak.state_active),
        "local_peak_down_displacement": _format_float(peak.down_displacement),
    }


def _simultaneous_drums(reference_events: list[EvaluationEvent], reference: EvaluationEvent) -> list[str]:
    return sorted(
        event.drum
        for event in reference_events
        if abs(event.time_seconds - reference.time_seconds) < 1e-9
    )


def _match_key(event: EvaluationEvent, mode: MatchMode) -> tuple[str, ...]:
    if mode == "type":
        return (event.event_type,)
    return (event.event_type, event.drum)


def _side_for_drum(drum: str) -> str | None:
    if drum == "right_pedal":
        return "right"
    if drum == "left_pedal":
        return "left"
    return None


def _ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _format_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def _alignment_to_dict(alignment: AlignmentResult) -> dict:
    return {
        "offset_seconds": alignment.offset_seconds,
        "scale": alignment.scale,
        "stats": {
            "tp": alignment.stats.true_positives,
            "fp": alignment.stats.false_positives,
            "fn": alignment.stats.false_negatives,
            "precision": alignment.stats.precision,
            "recall": alignment.stats.recall,
            "f1": alignment.stats.f1,
            "false_negative_drums": alignment.stats.false_negative_drums,
            "false_positive_drums": alignment.stats.false_positive_drums,
        },
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a drumming video against reference MIDI.")
    parser.add_argument("video", type=Path, help="Recorded evaluation video.")
    parser.add_argument("--midi", type=Path, required=True, help="Reference MIDI file.")
    parser.add_argument("--tolerance", type=float, default=0.120, help="Match tolerance in seconds.")
    parser.add_argument("--offset-min", type=float, default=None, help="Minimum MIDI-to-video offset to search; omitted means automatic.")
    parser.add_argument("--offset-max", type=float, default=None, help="Maximum MIDI-to-video offset to search; omitted means automatic.")
    parser.add_argument("--scale-min", type=float, default=None, help="Minimum MIDI-to-video time scale to search; omitted means automatic.")
    parser.add_argument("--scale-max", type=float, default=None, help="Maximum MIDI-to-video time scale to search; omitted means automatic.")
    parser.add_argument("--hand-up-threshold", type=float, default=None, help="Override upward prep distance required to arm hand hits.")
    parser.add_argument("--hand-down-velocity-threshold", type=float, default=None, help="Override downward hand velocity required to trigger hand hits.")
    parser.add_argument("--progress-every", type=int, default=300, help="Print progress every N frames; 0 disables.")
    parser.add_argument("--preview", action="store_true", help="Display an annotated live replay while evaluating the video.")
    parser.add_argument("--preview-speed", type=float, default=1.0, help="Preview playback speed multiplier.")
    parser.add_argument(
        "--analysis-speed",
        type=float,
        default=1.0,
        help="Speed multiplier for detector timestamps. Use 2.0 for half-speed recordings so hand velocities match live motion.",
    )
    parser.add_argument("--no-preview-mirror", action="store_true", help="Do not mirror the recorded video preview.")
    parser.add_argument(
        "--midi-out",
        nargs="?",
        const="",
        default=None,
        metavar="PORT",
        help="Send detected replay hits to a MIDI output port. Omit PORT to use the default output.",
    )
    parser.add_argument(
        "--midi-channel",
        type=int,
        default=10,
        help="MIDI output channel number from 1 to 16. Channel 10 is the General MIDI drum channel.",
    )
    parser.add_argument("--miss-analysis-csv", type=Path, default=None, help="Write false-negative signal diagnostics to CSV.")
    parser.add_argument(
        "--miss-analysis-mode",
        choices=("type", "drum"),
        default="type",
        help="Match mode whose false negatives should be analyzed.",
    )
    parser.add_argument(
        "--miss-analysis-window",
        type=float,
        default=0.250,
        help="Seconds around each miss to search for local peak motion signals.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args(argv)


def _load_mido():
    try:
        import mido
    except ImportError as exc:
        raise RuntimeError(
            "Missing MIDI dependencies. From the repository root, run:\n"
            "  python -m pip install -r requirements.txt"
        ) from exc
    return mido


def _load_opencv():
    try:
        from apple_vision_pose.dependencies import load_opencv
    except ImportError as exc:
        raise RuntimeError(
            "Missing video dependencies. From the repository root, run:\n"
            "  python -m pip install -r requirements.txt"
        ) from exc
    return load_opencv()


if __name__ == "__main__":
    raise SystemExit(main())
