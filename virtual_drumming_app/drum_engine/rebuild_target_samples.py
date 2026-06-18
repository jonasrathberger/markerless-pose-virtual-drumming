"""Rebuild hand-target samples from raw video hit-frame references."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

from .apple_vision import apple_vision_landmarks_to_pose_frame
from .hand_hit_detection import target_observation_from_pose_frame
from .schema import PoseFrame
from .smoothing import LandmarkSmoother
from .target_classification import (
    DrumTargetHandSample,
    DrumTargetObservation,
    TARGET_FEATURE_SCHEMA,
    TARGET_TEMPORAL_FEATURE_SCHEMA,
    TARGET_DRUMS,
    TARGET_HAND_SIDES,
    new_target_sample_file,
    target_observation_to_sample_record,
    write_target_sample_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild target sample features from raw collection videos.")
    parser.add_argument("samples_path", help="Input target sample JSON with raw_video sample references.")
    parser.add_argument("output_path", help="Output target sample JSON using the current feature schema.")
    parser.add_argument(
        "--lookback-frames",
        type=int,
        default=9,
        help="Frames before each hit to load before rebuilding the hit-frame features.",
    )
    parser.add_argument(
        "--feature-schema",
        choices=("static", "temporal", TARGET_FEATURE_SCHEMA, TARGET_TEMPORAL_FEATURE_SCHEMA),
        default=TARGET_FEATURE_SCHEMA,
        help="Feature schema to write. Use 'temporal' for the temporal-window KNN feature set.",
    )
    parser.add_argument(
        "--detect-width",
        type=int,
        default=0,
        help="Resize video frames to this width before Vision inference. Use 0 for video resolution.",
    )
    parser.add_argument(
        "--allow-stored-observation",
        action="store_true",
        help="Fallback to each sample's stored observation when raw video is missing or unreadable.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    samples_path = Path(args.samples_path)
    try:
        source = read_target_sample_file_for_rebuild(samples_path)
        rebuilt = rebuild_target_sample_file(
            source,
            samples_path=samples_path,
            lookback_frames=max(0, args.lookback_frames),
            detect_width=args.detect_width,
            feature_schema=_normalize_feature_schema(args.feature_schema),
            allow_stored_observation=args.allow_stored_observation,
        )
        write_target_sample_file(args.output_path, rebuilt)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Wrote {len(rebuilt['samples'])} rebuilt target samples to {args.output_path}.")
    return 0


def read_target_sample_file_for_rebuild(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("Target sample file must contain a JSON object.")
    if not isinstance(data.get("samples"), list):
        raise ValueError("Target sample file must contain a samples list.")
    return data


def rebuild_target_sample_file(
    data: dict,
    *,
    samples_path: Path,
    lookback_frames: int,
    detect_width: int,
    feature_schema: str,
    allow_stored_observation: bool = False,
) -> dict:
    records_by_index: dict[int, dict] = {}
    skipped = 0
    detector = None
    raw_video_samples: dict[Path, list[tuple[int, dict]]] = defaultdict(list)

    for sample_index, sample in enumerate(data["samples"]):
        raw_video = sample.get("raw_video")
        if isinstance(raw_video, dict):
            _validate_sample_label(sample)
            video_path = _resolve_video_path(str(raw_video["path"]), samples_path=samples_path)
            raw_video_samples[video_path].append((sample_index, sample))
            continue
        if allow_stored_observation:
            record = rebuild_sample_record_from_stored_observation(sample, feature_schema=feature_schema)
            if record is not None:
                records_by_index[sample_index] = record
                continue
        skipped += 1

    for video_path, indexed_samples in raw_video_samples.items():
        if detector is None:
            detector = _create_detector(detect_width=detect_width)
        try:
            pose_frames = raw_video_pose_frames_for_samples(
                video_path,
                [sample for _index, sample in indexed_samples],
                detector=detector,
                lookback_frames=lookback_frames,
            )
        except (KeyError, RuntimeError, ValueError):
            if not allow_stored_observation:
                raise
            pose_frames = {}

        for sample_index, sample in indexed_samples:
            record = None
            if pose_frames:
                pose_window = pose_window_for_sample(sample, pose_frames, lookback_frames=lookback_frames)
                record = rebuild_sample_record_from_pose_window(
                    sample,
                    pose_window,
                    lookback_frames=lookback_frames,
                    feature_schema=feature_schema,
                )
            if record is None and allow_stored_observation:
                record = rebuild_sample_record_from_stored_observation(sample, feature_schema=feature_schema)
            if record is None:
                skipped += 1
                continue
            records_by_index[sample_index] = record

    records = [
        records_by_index[index]
        for index in range(len(data["samples"]))
        if index in records_by_index
    ]

    rebuilt = new_target_sample_file(records, feature_schema=feature_schema)
    if "raw_videos" in data:
        rebuilt["raw_videos"] = data["raw_videos"]
    rebuilt["rebuilt_from"] = {
        "samples_path": str(samples_path),
        "lookback_frames": lookback_frames,
        "skipped_samples": skipped,
    }
    return rebuilt


def raw_video_pose_frames_for_samples(
    video_path: Path,
    samples: list[dict],
    *,
    detector: Any,
    lookback_frames: int,
) -> dict[int, PoseFrame]:
    frame_indices = needed_frame_indices(samples, lookback_frames=lookback_frames)
    if not frame_indices:
        return {}
    fps = _video_fps(video_path)
    detector.reset_tracking()
    raw_pose_frames = _read_selected_video_pose_frames(
        video_path,
        frame_indices=frame_indices,
        detector=detector,
        fps=fps,
    )
    print(
        f"Processed {len(raw_pose_frames)} unique raw-video frames from {video_path}.",
        file=sys.stderr,
        flush=True,
    )
    return raw_pose_frames


def needed_frame_indices(samples: list[dict], *, lookback_frames: int) -> list[int]:
    indices: set[int] = set()
    for sample in samples:
        raw_video = sample.get("raw_video")
        if not isinstance(raw_video, dict):
            continue
        hit_frame = int(raw_video["frame_index"])
        start_frame = max(0, hit_frame - lookback_frames)
        indices.update(range(start_frame, hit_frame + 1))
    return sorted(indices)


def pose_window_for_sample(
    sample: dict,
    pose_frames: dict[int, PoseFrame],
    *,
    lookback_frames: int,
) -> list[PoseFrame]:
    raw_video = sample.get("raw_video")
    if not isinstance(raw_video, dict):
        return []
    hit_frame = int(raw_video["frame_index"])
    start_frame = max(0, hit_frame - lookback_frames)
    return [
        pose_frames[frame_index]
        for frame_index in range(start_frame, hit_frame + 1)
        if frame_index in pose_frames
    ]


def _read_selected_video_pose_frames(
    video_path: Path,
    *,
    frame_indices: list[int],
    detector: Any,
    fps: float,
) -> dict[int, PoseFrame]:
    cv2 = _load_opencv()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open raw video {video_path}.")
    pose_frames: dict[int, PoseFrame] = {}
    try:
        current_position: int | None = None
        for processed_count, frame_index in enumerate(frame_indices, start=1):
            if current_position != frame_index:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                current_position = frame_index
            ok, frame_bgr = capture.read()
            if not ok:
                raise RuntimeError(f"Could not read frame {frame_index} from raw video {video_path}.")
            current_position = frame_index + 1
            timestamp_seconds = frame_index / fps
            landmarks = _detect_frame_landmarks(detector, frame_bgr)
            pose_frames[frame_index] = apple_vision_landmarks_to_pose_frame(
                landmarks,
                timestamp_seconds=timestamp_seconds,
            )
            if processed_count % 500 == 0:
                print(
                    f"Processed {processed_count}/{len(frame_indices)} frames from {video_path}.",
                    file=sys.stderr,
                    flush=True,
                )
    finally:
        capture.release()
    return pose_frames


def _detect_frame_landmarks(detector: Any, frame_bgr: Any) -> dict:
    foundation = getattr(detector, "Foundation", None)
    pool = None
    if foundation is not None and hasattr(foundation, "NSAutoreleasePool"):
        pool = foundation.NSAutoreleasePool.alloc().init()
    try:
        return detector.detect(frame_bgr)
    finally:
        if pool is not None and hasattr(pool, "drain"):
            pool.drain()


def smoothed_pose_window(pose_window: list[PoseFrame]) -> list[PoseFrame]:
    smoother = LandmarkSmoother()
    return [smoother.update(frame) for frame in pose_window]


def rebuild_sample_record(
    sample: dict,
    *,
    samples_path: Path,
    detector: Any,
    lookback_frames: int,
    feature_schema: str,
    allow_stored_observation: bool = False,
) -> dict | None:
    _validate_sample_label(sample)
    raw_video = sample.get("raw_video")
    if isinstance(raw_video, dict):
        try:
            pose_window = raw_video_pose_window(
                raw_video,
                samples_path=samples_path,
                detector=detector,
                lookback_frames=lookback_frames,
            )
            return rebuild_sample_record_from_pose_window(
                sample,
                pose_window,
                lookback_frames=lookback_frames,
                feature_schema=feature_schema,
            )
        except (KeyError, RuntimeError, ValueError):
            if not allow_stored_observation:
                raise

    if allow_stored_observation:
        return rebuild_sample_record_from_stored_observation(sample, feature_schema=feature_schema)
    raise ValueError("Sample is missing raw_video metadata. Pass --allow-stored-observation to use stored observations.")


def raw_video_pose_window(
    raw_video: dict,
    *,
    samples_path: Path,
    detector: Any,
    lookback_frames: int,
) -> list[PoseFrame]:
    frame_index = int(raw_video["frame_index"])
    video_path = _resolve_video_path(raw_video["path"], samples_path=samples_path)
    start_frame = max(0, frame_index - lookback_frames)
    frames = _read_video_frames(video_path, start_frame=start_frame, end_frame=frame_index)
    fps = _video_fps(video_path)
    detector.reset_tracking()
    pose_frames: list[PoseFrame] = []
    for offset, frame_bgr in enumerate(frames):
        absolute_frame_index = start_frame + offset
        timestamp_seconds = absolute_frame_index / fps
        landmarks = _detect_frame_landmarks(detector, frame_bgr)
        pose_frame = apple_vision_landmarks_to_pose_frame(landmarks, timestamp_seconds=timestamp_seconds)
        pose_frames.append(pose_frame)
    return pose_frames


def rebuild_sample_record_from_pose_window(
    sample: dict,
    pose_window: list[PoseFrame],
    *,
    lookback_frames: int,
    feature_schema: str = TARGET_FEATURE_SCHEMA,
) -> dict | None:
    pose_window = smoothed_pose_window(pose_window)
    if not pose_window:
        return None
    velocity = float(sample.get("velocity", 0.0))
    observation = target_observation_from_pose_frame(
        pose_window[-1],
        side=str(sample["side"]),
        strike_velocity=velocity,
        history_frames=pose_window[:-1],
    )
    if observation is None:
        return None
    observation = replace(observation, timestamp_seconds=float(sample.get("timestamp_seconds", observation.timestamp_seconds)))
    record = target_observation_to_sample_record(
        drum=str(sample["drum"]),
        observation=observation,
        timestamp_seconds=observation.timestamp_seconds,
        velocity=velocity,
        feature_schema=feature_schema,
    )
    if record is None:
        return None
    _copy_rebuild_metadata(record, sample, lookback_frames=lookback_frames, window_start_frame=None)
    return record


def rebuild_sample_record_from_stored_observation(
    sample: dict,
    *,
    feature_schema: str = TARGET_FEATURE_SCHEMA,
) -> dict | None:
    existing_observation = sample.get("observation")
    if not isinstance(existing_observation, dict):
        return None
    active = _stored_hand_sample(existing_observation.get("active"))
    if active is None:
        return None
    velocity = float(sample.get("velocity", existing_observation.get("strike_velocity", 0.0)))
    observation = DrumTargetObservation(
        side=str(sample["side"]),
        active=active,
        other=None,
        strike_motion_y=float(existing_observation.get("strike_motion_y", active.hand_motion_y)),
        timestamp_seconds=float(sample.get("timestamp_seconds", 0.0)),
        strike_velocity=velocity,
    )
    record = target_observation_to_sample_record(
        drum=str(sample["drum"]),
        observation=observation,
        timestamp_seconds=observation.timestamp_seconds,
        velocity=velocity,
        feature_schema=feature_schema,
    )
    if record is None:
        return None
    if "raw_video" in sample:
        record["raw_video"] = sample["raw_video"]
    return record


def _stored_hand_sample(data: Any) -> DrumTargetHandSample | None:
    if not isinstance(data, dict):
        return None
    required = (
        "wrist_x",
        "wrist_y",
        "thumb_mcp_x",
        "thumb_mcp_y",
        "middle_mcp_x",
        "middle_mcp_y",
        "little_mcp_x",
        "little_mcp_y",
        "elbow_x",
        "elbow_y",
    )
    if any(name not in data for name in required):
        return None
    return DrumTargetHandSample(
        wrist_x=float(data["wrist_x"]),
        wrist_y=float(data["wrist_y"]),
        thumb_mcp_x=float(data["thumb_mcp_x"]),
        thumb_mcp_y=float(data["thumb_mcp_y"]),
        middle_mcp_x=float(data["middle_mcp_x"]),
        middle_mcp_y=float(data["middle_mcp_y"]),
        little_mcp_x=float(data["little_mcp_x"]),
        little_mcp_y=float(data["little_mcp_y"]),
        elbow_x=float(data["elbow_x"]),
        elbow_y=float(data["elbow_y"]),
        shoulder_x=float(data["shoulder_x"]) if data.get("shoulder_x") is not None else None,
        shoulder_y=float(data["shoulder_y"]) if data.get("shoulder_y") is not None else None,
    )


def _copy_rebuild_metadata(
    record: dict,
    sample: dict,
    *,
    lookback_frames: int,
    window_start_frame: int | None,
) -> None:
    raw_video = sample.get("raw_video")
    if isinstance(raw_video, dict):
        record["raw_video"] = raw_video
        frame_index = int(raw_video.get("frame_index", 0))
        start_frame = max(0, frame_index - lookback_frames) if window_start_frame is None else window_start_frame
        record["raw_video_window"] = {
            "path": raw_video.get("path"),
            "start_frame": start_frame,
            "hit_frame": frame_index,
            "lookback_frames": lookback_frames,
        }


def _read_video_frames(video_path: Path, *, start_frame: int, end_frame: int) -> list[Any]:
    if end_frame < start_frame:
        return []
    cv2 = _load_opencv()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open raw video {video_path}.")
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames = []
        for frame_index in range(start_frame, end_frame + 1):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Could not read frame {frame_index} from raw video {video_path}.")
            frames.append(frame)
        return frames
    finally:
        capture.release()


def _video_fps(video_path: Path) -> float:
    cv2 = _load_opencv()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open raw video {video_path}.")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    return fps if fps > 0.0 else 30.0


def _resolve_video_path(path: str, *, samples_path: Path) -> Path:
    video_path = Path(path)
    if video_path.is_absolute():
        return video_path
    return samples_path.parent / video_path


def _validate_sample_label(sample: dict) -> None:
    if sample.get("side") not in TARGET_HAND_SIDES:
        raise ValueError("Target sample has an unsupported hand side.")
    if sample.get("drum") not in TARGET_DRUMS:
        raise ValueError("Target sample has an unsupported drum.")


def _normalize_feature_schema(value: str) -> str:
    if value == "static":
        return TARGET_FEATURE_SCHEMA
    if value == "temporal":
        return TARGET_TEMPORAL_FEATURE_SCHEMA
    return value


def _create_detector(*, detect_width: int) -> Any:
    from apple_vision_pose.vision import AppleVisionPose

    return AppleVisionPose(max_hands=2, detect_width=detect_width)


def _load_opencv() -> Any:
    from apple_vision_pose.dependencies import load_opencv

    return load_opencv()


if __name__ == "__main__":
    raise SystemExit(main())
