#!/usr/bin/env python3

from __future__ import annotations

import csv
import argparse
import json
import re
import shutil
import sys
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
THESIS_DATA_DIR = REPO_ROOT / "thesis_data" / "pose_evaluation"
MOCAP_INPUT_DIR = REPO_ROOT / "thesis_data" / "raw" / "mocap"
POSE_INPUT_DIR = REPO_ROOT / "thesis_data" / "raw" / "pose"
RESULT_DIR = THESIS_DATA_DIR / "prepared_result"

OUTPUT_FIELDS = [
    "recording_id",
    "source",
    "model_name",
    "target",
    "target_variant",
    "side",
    "frame_index",
    "time_sec",
    "x",
    "y",
    "z",
    "coord_space",
    "unit",
    "x_px",
    "y_px",
    "confidence",
    "visibility",
    "tracking_present",
    "valid",
    "source_label",
]

MANIFEST_FIELDS = [
    "recording_id",
    "source",
    "model_name",
    "target",
    "target_variant",
    "side",
    "result_path",
    "row_count",
    "coord_space",
    "source_label",
]

@dataclass(frozen=True)
class MocapSpec:
    target: str
    target_variant: str
    side: str
    source_label: str
    mode: str
    names: tuple[str, ...]


@dataclass(frozen=True)
class PoseSpec:
    target: str
    target_variant: str
    side: str
    landmark_group: str
    landmark_name: str
    source_label: str


MOCAP_SPECS = (
    MocapSpec(
        target="shoulder",
        target_variant="default",
        side="left",
        source_label="Skeleton 002:LShoulderTop+Skeleton 002:LShoulderBack",
        mode="mean_marker",
        names=("Skeleton 002:LShoulderTop", "Skeleton 002:LShoulderBack"),
    ),
    MocapSpec(
        target="shoulder",
        target_variant="default",
        side="right",
        source_label="Skeleton 002:RShoulderTop+Skeleton 002:RShoulderBack",
        mode="mean_marker",
        names=("Skeleton 002:RShoulderTop", "Skeleton 002:RShoulderBack"),
    ),
    MocapSpec(
        target="elbow",
        target_variant="default",
        side="left",
        source_label="Skeleton 002:LElbowOut",
        mode="direct_marker",
        names=("Skeleton 002:LElbowOut",),
    ),
    MocapSpec(
        target="elbow",
        target_variant="default",
        side="right",
        source_label="Skeleton 002:RElbowOut",
        mode="direct_marker",
        names=("Skeleton 002:RElbowOut",),
    ),
    MocapSpec(
        target="hip",
        target_variant="default",
        side="left",
        source_label="Skeleton 002:WaistLFront+Skeleton 002:WaistLBack",
        mode="mean_marker",
        names=("Skeleton 002:WaistLFront", "Skeleton 002:WaistLBack"),
    ),
    MocapSpec(
        target="hip",
        target_variant="default",
        side="right",
        source_label="Skeleton 002:WaistRFront+Skeleton 002:WaistRBack",
        mode="mean_marker",
        names=("Skeleton 002:WaistRFront", "Skeleton 002:WaistRBack"),
    ),
    MocapSpec(
        target="pinky_knuckle",
        target_variant="proxy_pinky1",
        side="left",
        source_label="Skeleton 002:LPinky1",
        mode="direct_bone",
        names=("Skeleton 002:LPinky1",),
    ),
    MocapSpec(
        target="pinky_knuckle",
        target_variant="proxy_pinky1",
        side="right",
        source_label="Skeleton 002:RPinky1",
        mode="direct_bone",
        names=("Skeleton 002:RPinky1",),
    ),
    MocapSpec(
        target="knee",
        target_variant="default",
        side="left",
        source_label="Skeleton 002:LKneeOut",
        mode="direct_marker",
        names=("Skeleton 002:LKneeOut",),
    ),
    MocapSpec(
        target="knee",
        target_variant="default",
        side="right",
        source_label="Skeleton 002:RKneeOut",
        mode="direct_marker",
        names=("Skeleton 002:RKneeOut",),
    ),
    MocapSpec(
        target="ankle",
        target_variant="default",
        side="left",
        source_label="Skeleton 002:LAnkleOut",
        mode="direct_marker",
        names=("Skeleton 002:LAnkleOut",),
    ),
    MocapSpec(
        target="ankle",
        target_variant="default",
        side="right",
        source_label="Skeleton 002:RAnkleOut",
        mode="direct_marker",
        names=("Skeleton 002:RAnkleOut",),
    ),
    MocapSpec(
        target="wrist",
        target_variant="center",
        side="left",
        source_label="Skeleton 002:LWristIn+Skeleton 002:LWristOut",
        mode="mean_marker",
        names=("Skeleton 002:LWristIn", "Skeleton 002:LWristOut"),
    ),
    MocapSpec(
        target="wrist",
        target_variant="center",
        side="right",
        source_label="Skeleton 002:RWristIn+Skeleton 002:RWristOut",
        mode="mean_marker",
        names=("Skeleton 002:RWristIn", "Skeleton 002:RWristOut"),
    ),
)

POSE_SPECS = (
    PoseSpec(
        target="shoulder",
        target_variant="default",
        side="left",
        landmark_group="body",
        landmark_name="shoulder",
        source_label="body:shoulder",
    ),
    PoseSpec(
        target="shoulder",
        target_variant="default",
        side="right",
        landmark_group="body",
        landmark_name="shoulder",
        source_label="body:shoulder",
    ),
    PoseSpec(
        target="elbow",
        target_variant="default",
        side="left",
        landmark_group="body",
        landmark_name="elbow",
        source_label="body:elbow",
    ),
    PoseSpec(
        target="elbow",
        target_variant="default",
        side="right",
        landmark_group="body",
        landmark_name="elbow",
        source_label="body:elbow",
    ),
    PoseSpec(
        target="hip",
        target_variant="default",
        side="left",
        landmark_group="body",
        landmark_name="hip",
        source_label="body:hip",
    ),
    PoseSpec(
        target="hip",
        target_variant="default",
        side="right",
        landmark_group="body",
        landmark_name="hip",
        source_label="body:hip",
    ),
    PoseSpec(
        target="pinky_knuckle",
        target_variant="default",
        side="left",
        landmark_group="left_hand",
        landmark_name="pinky_mcp",
        source_label="left_hand:pinky_mcp",
    ),
    PoseSpec(
        target="pinky_knuckle",
        target_variant="default",
        side="right",
        landmark_group="right_hand",
        landmark_name="pinky_mcp",
        source_label="right_hand:pinky_mcp",
    ),
    PoseSpec(
        target="knee",
        target_variant="default",
        side="left",
        landmark_group="body",
        landmark_name="knee",
        source_label="body:knee",
    ),
    PoseSpec(
        target="knee",
        target_variant="default",
        side="right",
        landmark_group="body",
        landmark_name="knee",
        source_label="body:knee",
    ),
    PoseSpec(
        target="ankle",
        target_variant="default",
        side="left",
        landmark_group="body",
        landmark_name="ankle",
        source_label="body:ankle",
    ),
    PoseSpec(
        target="ankle",
        target_variant="default",
        side="right",
        landmark_group="body",
        landmark_name="ankle",
        source_label="body:ankle",
    ),
    PoseSpec(
        target="wrist",
        target_variant="hand",
        side="left",
        landmark_group="left_hand",
        landmark_name="wrist",
        source_label="left_hand:wrist",
    ),
    PoseSpec(
        target="wrist",
        target_variant="hand",
        side="right",
        landmark_group="right_hand",
        landmark_name="wrist",
        source_label="right_hand:wrist",
    ),
    PoseSpec(
        target="wrist",
        target_variant="body",
        side="left",
        landmark_group="body",
        landmark_name="wrist",
        source_label="body:wrist",
    ),
    PoseSpec(
        target="wrist",
        target_variant="body",
        side="right",
        landmark_group="body",
        landmark_name="wrist",
        source_label="body:wrist",
    ),
)


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    return float(text)


def format_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.15g}"


def stringify_bool(value: bool | None) -> str:
    if value is None:
        return ""
    return "True" if value else "False"


def parse_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    text = value.strip().lower()
    if text == "":
        return None
    if text == "true":
        return True
    if text == "false":
        return False
    raise ValueError(f"Unsupported boolean value: {value!r}")


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def ensure_clean_result_dir(result_dir: Path) -> None:
    if result_dir.exists():
        shutil.rmtree(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)


def parse_metadata_row(row: list[str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for index in range(0, len(row) - 1, 2):
        key = row[index].strip()
        value = row[index + 1].strip()
        if key:
            metadata[key] = value
    return metadata


def build_position_lookup(
    type_row: list[str],
    name_row: list[str],
    measure_row: list[str],
    axis_row: list[str],
) -> dict[tuple[str, str], dict[str, int]]:
    lookup: dict[tuple[str, str], dict[str, int]] = {}
    for index in range(len(type_row)):
        if index >= len(name_row) or index >= len(measure_row) or index >= len(axis_row):
            continue
        source_type = type_row[index].strip()
        source_name = name_row[index].strip()
        measure = measure_row[index].strip()
        axis = axis_row[index].strip()
        if not source_type or not source_name or measure != "Position" or axis not in {"X", "Y", "Z"}:
            continue
        lookup.setdefault((source_type, source_name), {})[axis] = index
    return lookup


def find_position_columns(
    lookup: dict[tuple[str, str], dict[str, int]],
    source_name: str,
    source_types: Iterable[str],
) -> tuple[int, int, int]:
    for source_type in source_types:
        axes = lookup.get((source_type, source_name))
        if axes and {"X", "Y", "Z"} <= set(axes):
            return axes["X"], axes["Y"], axes["Z"]
    raise KeyError(f"Could not find position columns for {source_name!r} with types {tuple(source_types)!r}")


def get_row_values(row: list[str], columns: tuple[int, int, int]) -> tuple[float | None, float | None, float | None]:
    values: list[float | None] = []
    for column in columns:
        if column >= len(row):
            return None, None, None
        values.append(parse_float(row[column]))
    return values[0], values[1], values[2]


def build_mocap_output_row(
    row: list[str],
    recording_id: str,
    spec: MocapSpec,
    column_map: dict[tuple[str, str, str], tuple[int, int, int]],
) -> dict[str, str]:
    if spec.mode == "direct_bone":
        x, y, z = get_row_values(row, column_map[(spec.target, spec.target_variant, spec.side)])
        valid = all(value is not None for value in (x, y, z))
    elif spec.mode == "direct_marker":
        x, y, z = get_row_values(row, column_map[(spec.target, spec.target_variant, spec.side)])
        valid = all(value is not None for value in (x, y, z))
    elif spec.mode == "mean_marker":
        inner_columns, outer_columns = (
            column_map[(spec.target, spec.target_variant, spec.side, "inner")],
            column_map[(spec.target, spec.target_variant, spec.side, "outer")],
        )
        inner_values = get_row_values(row, inner_columns)
        outer_values = get_row_values(row, outer_columns)
        if all(value is not None for value in inner_values + outer_values):
            x = (inner_values[0] + outer_values[0]) / 2.0
            y = (inner_values[1] + outer_values[1]) / 2.0
            z = (inner_values[2] + outer_values[2]) / 2.0
            valid = True
        else:
            x = y = z = None
            valid = False
    else:
        raise ValueError(f"Unsupported mocap mode: {spec.mode}")

    return {
        "recording_id": recording_id,
        "source": "mocap",
        "model_name": "optitrack",
        "target": spec.target,
        "target_variant": spec.target_variant,
        "side": spec.side,
        "frame_index": row[0].strip(),
        "time_sec": row[1].strip(),
        "x": format_float(x),
        "y": format_float(y),
        "z": format_float(z),
        "coord_space": "global",
        "unit": "meters",
        "x_px": "",
        "y_px": "",
        "confidence": "",
        "visibility": "",
        "tracking_present": "",
        "valid": stringify_bool(valid),
        "source_label": spec.source_label,
    }


def normalize_mocap_recording_id(recording_id: str) -> str:
    return re.sub(r"_\d+$", "", recording_id)


def prepare_mocap_outputs(manifest_rows: list[dict[str, str]], result_dir: Path) -> None:
    for csv_path in sorted(MOCAP_INPUT_DIR.glob("*.csv")):
        with csv_path.open(newline="") as handle:
            reader = csv.reader(handle)
            header_rows = [next(reader) for _ in range(7)]
            metadata = parse_metadata_row(header_rows[0])
            type_row = header_rows[2]
            name_row = header_rows[3]
            measure_row = header_rows[5]
            axis_row = header_rows[6]
            raw_recording_id = metadata.get("Take Name", csv_path.stem) or csv_path.stem
            recording_id = normalize_mocap_recording_id(raw_recording_id)
            lookup = build_position_lookup(type_row, name_row, measure_row, axis_row)

            column_map: dict[tuple[str, str, str] | tuple[str, str, str, str], tuple[int, int, int]] = {}
            for spec in MOCAP_SPECS:
                if spec.mode == "direct_bone":
                    column_map[(spec.target, spec.target_variant, spec.side)] = find_position_columns(
                        lookup,
                        spec.names[0],
                        ("Bone",),
                    )
                elif spec.mode == "direct_marker":
                    column_map[(spec.target, spec.target_variant, spec.side)] = find_position_columns(
                        lookup,
                        spec.names[0],
                        ("Marker", "Bone Marker"),
                    )
                elif spec.mode == "mean_marker":
                    column_map[(spec.target, spec.target_variant, spec.side, "inner")] = find_position_columns(
                        lookup,
                        spec.names[0],
                        ("Marker", "Bone Marker"),
                    )
                    column_map[(spec.target, spec.target_variant, spec.side, "outer")] = find_position_columns(
                        lookup,
                        spec.names[1],
                        ("Marker", "Bone Marker"),
                    )
                else:
                    raise ValueError(f"Unsupported mocap mode: {spec.mode}")

            output_dir = result_dir / "mocap" / recording_id
            output_dir.mkdir(parents=True, exist_ok=True)

            with ExitStack() as stack:
                writers: dict[tuple[str, str, str], csv.DictWriter] = {}
                counts: dict[tuple[str, str, str], int] = {}
                output_paths: dict[tuple[str, str, str], Path] = {}

                for spec in MOCAP_SPECS:
                    key = (spec.target, spec.target_variant, spec.side)
                    output_path = output_dir / f"{spec.target}_{spec.target_variant}_{spec.side}.csv"
                    output_paths[key] = output_path
                    writer_handle = stack.enter_context(output_path.open("w", newline=""))
                    writer = csv.DictWriter(writer_handle, fieldnames=OUTPUT_FIELDS)
                    writer.writeheader()
                    writers[key] = writer
                    counts[key] = 0

                for row in reader:
                    for spec in MOCAP_SPECS:
                        key = (spec.target, spec.target_variant, spec.side)
                        output_row = build_mocap_output_row(row, recording_id, spec, column_map)
                        writers[key].writerow(output_row)
                        counts[key] += 1

            for spec in MOCAP_SPECS:
                key = (spec.target, spec.target_variant, spec.side)
                manifest_rows.append(
                    {
                        "recording_id": recording_id,
                        "source": "mocap",
                        "model_name": "optitrack",
                        "target": spec.target,
                        "target_variant": spec.target_variant,
                        "side": spec.side,
                        "result_path": display_path(output_paths[key]),
                        "row_count": str(counts[key]),
                        "coord_space": "global",
                        "source_label": spec.source_label,
                    }
                )


def build_pose_output_row(
    row: dict[str, str],
    recording_id: str,
    model_name: str,
    spec: PoseSpec,
    time_offset: float,
) -> dict[str, str]:
    x = parse_float(row.get("x_norm"))
    y = parse_float(row.get("y_norm"))
    z = parse_float(row.get("z_rel"))
    x_px = row.get("x_px", "").strip()
    y_px = row.get("y_px", "").strip()
    confidence = row.get("confidence", "").strip()
    visibility = row.get("visibility", "").strip()
    tracking_present = parse_bool(row.get("tracking_present"))
    raw_time = parse_float(row.get("timestamp_monotonic_sec"))
    if raw_time is None:
        raise ValueError(f"Missing timestamp_monotonic_sec for pose row in {recording_id}")
    # Apple Vision often has no z value, so validity only requires tracked x/y coordinates.
    valid = bool(tracking_present) and x is not None and y is not None

    return {
        "recording_id": recording_id,
        "source": "pose",
        "model_name": model_name,
        "target": spec.target,
        "target_variant": spec.target_variant,
        "side": spec.side,
        "frame_index": row.get("frame_index", "").strip(),
        "time_sec": format_float(raw_time - time_offset),
        "x": format_float(x),
        "y": format_float(y),
        "z": format_float(z),
        "coord_space": "image_normalized_relz",
        "unit": "normalized",
        "x_px": x_px,
        "y_px": y_px,
        "confidence": confidence,
        "visibility": visibility,
        "tracking_present": stringify_bool(tracking_present),
        "valid": stringify_bool(valid),
        "source_label": spec.source_label,
    }


def prepare_pose_outputs(manifest_rows: list[dict[str, str]], result_dir: Path) -> None:
    if not POSE_INPUT_DIR.exists():
        return

    session_dirs = sorted(path.parent for path in POSE_INPUT_DIR.rglob("landmarks.csv"))
    for session_dir in session_dirs:
        landmarks_path = session_dir / "landmarks.csv"
        metadata_path = session_dir / "metadata.json"
        if not landmarks_path.exists() or not metadata_path.exists():
            continue

        metadata = json.loads(metadata_path.read_text())
        recording_id = str(metadata.get("session_id") or session_dir.name)
        model_name = str(metadata.get("model_name") or "unknown_pose_model")
        output_dir = result_dir / "pose" / recording_id
        output_dir.mkdir(parents=True, exist_ok=True)

        with ExitStack() as stack:
            writers: dict[tuple[str, str, str], csv.DictWriter] = {}
            counts: dict[tuple[str, str, str], int] = {}
            output_paths: dict[tuple[str, str, str], Path] = {}

            for spec in POSE_SPECS:
                key = (spec.target, spec.target_variant, spec.side)
                output_path = output_dir / f"{spec.target}_{spec.target_variant}_{spec.side}.csv"
                output_paths[key] = output_path
                writer_handle = stack.enter_context(output_path.open("w", newline=""))
                writer = csv.DictWriter(writer_handle, fieldnames=OUTPUT_FIELDS)
                writer.writeheader()
                writers[key] = writer
                counts[key] = 0

            with landmarks_path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                time_offset: float | None = None
                for row in reader:
                    raw_time = parse_float(row.get("timestamp_monotonic_sec"))
                    if raw_time is not None and time_offset is None:
                        time_offset = raw_time
                    if time_offset is None:
                        continue

                    for spec in POSE_SPECS:
                        if (
                            row.get("landmark_group", "").strip() == spec.landmark_group
                            and row.get("landmark_name", "").strip() == spec.landmark_name
                            and row.get("side", "").strip() == spec.side
                        ):
                            key = (spec.target, spec.target_variant, spec.side)
                            output_row = build_pose_output_row(row, recording_id, model_name, spec, time_offset)
                            writers[key].writerow(output_row)
                            counts[key] += 1

        for spec in POSE_SPECS:
            key = (spec.target, spec.target_variant, spec.side)
            manifest_rows.append(
                {
                    "recording_id": recording_id,
                    "source": "pose",
                    "model_name": model_name,
                    "target": spec.target,
                    "target_variant": spec.target_variant,
                    "side": spec.side,
                    "result_path": display_path(output_paths[key]),
                    "row_count": str(counts[key]),
                    "coord_space": "image_normalized_relz",
                    "source_label": spec.source_label,
                }
            )

def write_manifest(manifest_rows: list[dict[str, str]], result_dir: Path) -> None:
    manifest_path = result_dir / "manifest.csv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in sorted(
            manifest_rows,
            key=lambda item: (
                item["source"],
                item["recording_id"],
                item["target"],
                item["target_variant"],
                item["side"],
            ),
        ):
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare canonical pose-evaluation landmark CSVs.")
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR, help="Directory for prepared CSV outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_dir = args.output_dir.resolve()
    ensure_clean_result_dir(result_dir)
    manifest_rows: list[dict[str, str]] = []
    prepare_mocap_outputs(manifest_rows, result_dir)
    prepare_pose_outputs(manifest_rows, result_dir)
    write_manifest(manifest_rows, result_dir)
    print(f"Wrote {len(manifest_rows)} baseline result files under {result_dir}")


if __name__ == "__main__":
    main()
