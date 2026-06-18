from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from motion_types import MotionRecording


def parse_metadata_row(row: list[str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for idx in range(0, len(row) - 1, 2):
        key = row[idx].strip()
        value = row[idx + 1].strip()
        if key:
            metadata[key] = value
    return metadata


def _deduplicate(columns: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for name in columns:
        count = seen.get(name, 0)
        if count == 0:
            out.append(name)
        else:
            out.append(f"{name}__{count}")
        seen[name] = count + 1
    return out


def build_column_names(
    row_type: list[str],
    row_name: list[str],
    row_transform: list[str],
    row_axis: list[str],
) -> list[str]:
    n_cols = len(row_axis)
    columns: list[str] = []

    for i in range(n_cols):
        if i == 0:
            columns.append("frame")
            continue
        if i == 1:
            columns.append("time_s")
            continue

        name = row_name[i].strip() if i < len(row_name) else ""
        transform = row_transform[i].strip() if i < len(row_transform) else ""
        axis = row_axis[i].strip() if i < len(row_axis) else ""
        node_type = row_type[i].strip() if i < len(row_type) else ""

        parts = [p for p in [name, transform, axis] if p]
        if not parts:
            parts = [node_type or f"col_{i}"]
        columns.append("|".join(parts))

    return _deduplicate(columns)


def parse_motion_csv(path: str | Path, source: str) -> MotionRecording:
    csv_path = Path(path)

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = [next(reader, []) for _ in range(7)]

    metadata = parse_metadata_row(rows[0])
    colnames = build_column_names(rows[2], rows[3], rows[5], rows[6])

    df = pd.read_csv(csv_path, skiprows=7, header=None, names=colnames)

    # Force all columns numeric where possible and drop trailing empty rows.
    df = df.apply(pd.to_numeric, errors="coerce")
    if "frame" in df:
        df = df[df["frame"].notna()].copy()
        df["frame"] = df["frame"].astype("int64")
    if "time_s" in df:
        df = df[df["time_s"].notna()].copy()

    return MotionRecording(source=source, path=csv_path, metadata=metadata, data=df)


def list_landmarks(recording: MotionRecording) -> list[str]:
    landmarks: set[str] = set()
    for col in recording.columns:
        if col in {"frame", "time_s"}:
            continue
        pieces = col.split("|")
        if pieces:
            landmarks.add(pieces[0])
    return sorted(landmarks)


def get_xyz_columns(recording: MotionRecording, landmark: str) -> dict[str, str]:
    out: dict[str, str] = {}
    prefix = f"{landmark}|"
    for col in recording.columns:
        if not col.startswith(prefix):
            continue
        if col.endswith("|X"):
            out["x"] = col
        elif col.endswith("|Y"):
            out["y"] = col
        elif col.endswith("|Z"):
            out["z"] = col
    return out
