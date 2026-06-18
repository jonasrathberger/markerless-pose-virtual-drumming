"""Validate a recording folder and print summary statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_landmarks(recording_dir: Path) -> pd.DataFrame:
    parquet_path = recording_dir / "landmarks.parquet"
    csv_path = recording_dir / "landmarks.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError("No landmarks.parquet or landmarks.csv found in the recording directory.")


def summarize(recording_dir: Path) -> None:
    metadata_path = recording_dir / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    dataframe = load_landmarks(recording_dir)
    if dataframe.empty:
        print("No landmark rows found.")
        return

    frames = dataframe["frame_index"].nunique()
    rows = len(dataframe)
    landmarks_per_frame = rows / frames if frames else 0.0
    missing_rows = (~dataframe["tracking_present"].fillna(False)).sum()
    missing_rate = missing_rows / rows if rows else 0.0

    monotonic_by_frame = (
        dataframe[["frame_index", "timestamp_monotonic_sec"]]
        .drop_duplicates(subset=["frame_index"])
        .sort_values("frame_index")
    )
    average_fps = None
    if len(monotonic_by_frame) >= 2:
        elapsed = monotonic_by_frame["timestamp_monotonic_sec"].iloc[-1] - monotonic_by_frame["timestamp_monotonic_sec"].iloc[0]
        if elapsed > 0:
            average_fps = (len(monotonic_by_frame) - 1) / elapsed

    print(f"Recording folder: {recording_dir}")
    print(f"Session ID: {metadata.get('session_id', 'unknown')}")
    print(f"Model name: {metadata.get('model_name', 'unknown')}")
    print(f"Total frames: {frames}")
    print(f"Total landmark rows: {rows}")
    print(f"Average landmarks per frame: {landmarks_per_frame:.2f}")
    print(f"Missing-data rate: {missing_rate:.2%}")
    if average_fps is not None:
        print(f"Average FPS: {average_fps:.2f}")
    if metadata:
        print(f"Metadata target FPS: {metadata.get('target_fps')}")
        print(f"Metadata dropped frames: {metadata.get('dropped_frame_count')}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a recording folder and print summary stats.")
    parser.add_argument("recording_dir", help="Path to a session directory under recordings/.")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    summarize(Path(args.recording_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

