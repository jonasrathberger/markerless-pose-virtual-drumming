from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


ROOT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
THESIS_DATA_DIR = REPO_ROOT / "thesis_data" / "pose_evaluation"
INPUT_RESULT_DIR = THESIS_DATA_DIR / "prepared_result"
OUTPUT_RESULT_DIR = THESIS_DATA_DIR / "aligned_result"
CONFIG_PATH = Path(__file__).resolve().with_name("left_knee_alignment_config.json")

MANIFEST_COLUMNS = [
    "source",
    "recording_id",
    "csv_name",
    "result_path",
    "row_count",
    "trim_window_start_sec",
    "trim_window_end_sec",
    "alignment_peak_source_time_sec",
    "alignment_peak_offset_sec",
]


def load_config() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text())


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def prepare_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def trim_csv(csv_path: Path, trim_start_sec: float, trim_end_sec: float, peak_time_sec: float, nominal_peak_offset_sec: float) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    if "time_sec" not in frame.columns:
        raise ValueError(f"{csv_path} does not contain a time_sec column")

    source_time = pd.to_numeric(frame["time_sec"], errors="coerce")
    mask = source_time.notna() & (source_time >= trim_start_sec) & (source_time <= trim_end_sec)
    trimmed = frame.loc[mask].copy()
    trimmed_time = source_time.loc[mask]
    if trimmed.empty:
        return trimmed

    first_kept_time = float(trimmed_time.iloc[0])
    actual_peak_offset_sec = peak_time_sec - first_kept_time

    insert_at = trimmed.columns.get_loc("time_sec") + 1
    trimmed.insert(insert_at, "source_time_sec", trimmed_time.to_numpy())
    trimmed["time_sec"] = trimmed_time.to_numpy() - first_kept_time
    trimmed["trim_window_start_sec"] = trim_start_sec
    trimmed["trim_window_end_sec"] = trim_end_sec
    trimmed["alignment_peak_source_time_sec"] = peak_time_sec
    trimmed["alignment_peak_offset_sec"] = actual_peak_offset_sec
    trimmed["alignment_peak_nominal_offset_sec"] = nominal_peak_offset_sec
    return trimmed


def write_manifest(output_dir: Path, manifest_rows: list[dict[str, object]]) -> None:
    manifest_path = output_dir / "manifest.csv"
    manifest_frame = pd.DataFrame(manifest_rows, columns=MANIFEST_COLUMNS)
    manifest_frame.to_csv(manifest_path, index=False)


def trim_tree(input_root: Path, output_root: Path, config: dict[str, object]) -> None:
    window = config["trim_window"]
    before_sec = float(window["seconds_before_peak"])
    after_sec = float(window["seconds_after_peak"])
    nominal_peak_offset_sec = float(window["aligned_peak_offset_sec"])
    manifest_rows: list[dict[str, object]] = []

    for recording in config["recordings"]:
        source = str(recording["source"])
        recording_id = str(recording["recording_id"])
        peak_time_sec = float(recording["peak_time_sec"])
        trim_start_sec = peak_time_sec - before_sec
        trim_end_sec = peak_time_sec + after_sec

        input_dir = input_root / source / recording_id
        if not input_dir.exists():
            continue
        output_dir = output_root / source / recording_id
        output_dir.mkdir(parents=True, exist_ok=True)

        for csv_path in sorted(input_dir.glob("*.csv")):
            trimmed = trim_csv(
                csv_path=csv_path,
                trim_start_sec=trim_start_sec,
                trim_end_sec=trim_end_sec,
                peak_time_sec=peak_time_sec,
                nominal_peak_offset_sec=nominal_peak_offset_sec,
            )
            output_path = output_dir / csv_path.name
            trimmed.to_csv(output_path, index=False)
            manifest_rows.append(
                {
                    "source": source,
                    "recording_id": recording_id,
                    "csv_name": csv_path.name,
                    "result_path": display_path(output_path),
                    "row_count": int(len(trimmed)),
                    "trim_window_start_sec": trim_start_sec,
                    "trim_window_end_sec": trim_end_sec,
                    "alignment_peak_source_time_sec": peak_time_sec,
                    "alignment_peak_offset_sec": float(trimmed["alignment_peak_offset_sec"].iloc[0]) if not trimmed.empty else nominal_peak_offset_sec,
                }
            )

    write_manifest(output_root, manifest_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trim prepared landmark CSVs to the configured alignment window.")
    parser.add_argument("--input-dir", type=Path, default=INPUT_RESULT_DIR, help="Prepared CSV input directory.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_RESULT_DIR, help="Aligned CSV output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    config = load_config()
    prepare_output_dir(output_dir)
    trim_tree(input_dir, output_dir, config)
    print(f"Wrote trimmed aligned CSVs under {output_dir}")


if __name__ == "__main__":
    main()
