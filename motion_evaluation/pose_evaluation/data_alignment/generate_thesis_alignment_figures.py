from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_figures import THESIS_DATA_ROOT, THESIS_OUTPUT_ROOT, THESIS_STYLE, apply_thesis_style, figure_size, save_single_figure, style_axis

os.environ.setdefault("MPLCONFIGDIR", str((REPO_ROOT / ".cache" / "matplotlib").resolve()))
apply_thesis_style()

import matplotlib.pyplot as plt


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PREPARED_DIR = THESIS_DATA_ROOT / "pose_evaluation" / "prepared_result"
DEFAULT_ALIGNED_DIR = THESIS_DATA_ROOT / "pose_evaluation" / "aligned_result"
DEFAULT_CONFIG_PATH = ROOT_DIR / "data_alignment" / "left_knee_alignment_config.json"

RECORDING_ORDER = ["air_knees", "drums", "apple_vision", "mediapipe"]
RECORDING_LABELS = {
    "air_knees": "OptiTrack Motion Capture: Air Knees",
    "drums": "OptiTrack Motion Capture: Drums",
    "apple_vision": "Apple Vision",
    "mediapipe": "MediaPipe",
}
TRAJECTORY_FIGURE_SIZE = (8.2, 4.1)
PALETTE = {
    "air_knees": "#1d4ed8",
    "drums": "#dc2626",
    "apple_vision": "#059669",
    "mediapipe": "#d97706",
}


def _load_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column in ("time_sec", "source_time_sec", "x", "y", "z", "confidence", "visibility", "alignment_peak_offset_sec", "alignment_peak_nominal_offset_sec"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "valid" in frame.columns:
        frame["valid"] = frame["valid"].fillna("").astype(str).str.lower().eq("true")
    return frame


def _signal(
    path: Path,
    *,
    axis_name: str,
    source_name: str,
    valid_only: bool = True,
    normalize: bool = False,
    invert_pose_vertical: bool = True,
) -> pd.DataFrame:
    frame = _load_csv(path)
    if valid_only and "valid" in frame.columns:
        frame = frame[frame["valid"]]
    signal = frame[["time_sec", axis_name]].dropna().copy()
    if signal.empty:
        return pd.DataFrame(columns=["plot_time_sec", "plot_y"])
    if invert_pose_vertical and source_name == "pose" and axis_name in {"y", "y_px"}:
        signal[axis_name] = -signal[axis_name]
    if normalize:
        centered = signal[axis_name] - signal[axis_name].mean()
        std = float(signal[axis_name].std(ddof=0))
        signal[axis_name] = centered if std == 0 else centered / std
    signal["plot_time_sec"] = signal["time_sec"]
    signal["plot_y"] = signal[axis_name]
    return signal[["plot_time_sec", "plot_y"]]


def _prepared_path(prepared_dir: Path, recording_name: str, csv_name: str) -> tuple[str, Path]:
    source_name = "pose" if recording_name in {"apple_vision", "mediapipe"} else "mocap"
    return source_name, prepared_dir / source_name / recording_name / csv_name


def _aligned_path(aligned_dir: Path, recording_name: str, csv_name: str) -> tuple[str, Path]:
    source_name = "pose" if recording_name in {"apple_vision", "mediapipe"} else "mocap"
    return source_name, aligned_dir / source_name / recording_name / csv_name


def _load_peak_times(config_path: Path) -> dict[str, float]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return {str(row["recording_id"]): float(row["peak_time_sec"]) for row in config["recordings"]}


def _offset_series(series_map: dict[str, pd.DataFrame], offset_step: float) -> dict[str, pd.DataFrame]:
    available = [name for name in RECORDING_ORDER if name in series_map and not series_map[name].empty]
    centered_index = (len(available) - 1) / 2.0
    shifted: dict[str, pd.DataFrame] = {}
    for index, recording_name in enumerate(available):
        frame = series_map[recording_name].copy()
        frame["plot_y"] = frame["plot_y"] + (centered_index - index) * offset_step
        shifted[recording_name] = frame
    return shifted


def _plot_overlay(
    series_map: dict[str, pd.DataFrame],
    output_path: Path,
    *,
    title: str,
    x_axis_label: str,
    y_axis_label: str = "Offset visualization",
    x_window: tuple[float, float] | None = None,
    reference_line: tuple[float, str] | None = None,
) -> Path:
    figure, axis = plt.subplots(figsize=TRAJECTORY_FIGURE_SIZE, dpi=THESIS_STYLE.dpi)
    for recording_name in RECORDING_ORDER:
        frame = series_map.get(recording_name, pd.DataFrame())
        if frame.empty:
            continue
        local = frame
        if x_window is not None:
            local = local[(local["plot_time_sec"] >= x_window[0]) & (local["plot_time_sec"] <= x_window[1])]
        if local.empty:
            continue
        if len(local) > 2600:
            local = local.iloc[:: max(1, len(local) // 2600)]
        axis.plot(
            local["plot_time_sec"],
            local["plot_y"],
            color=PALETTE.get(recording_name, THESIS_STYLE.blue),
            linewidth=1.0,
            label=RECORDING_LABELS.get(recording_name, recording_name),
        )
    if reference_line is not None:
        x_value, label = reference_line
        axis.axvline(x_value, color=THESIS_STYLE.spine_color, linewidth=0.9, linestyle=(0, (4, 3)))
        axis.text(
            x_value + 0.6,
            0.94,
            label,
            transform=axis.get_xaxis_transform(),
            fontsize=THESIS_STYLE.small_font_size,
            color=THESIS_STYLE.muted_text_color,
            va="top",
        )
    axis.set_xlabel(x_axis_label)
    axis.set_ylabel(y_axis_label)
    if x_window is not None:
        axis.set_xlim(*x_window)
    style_axis(axis, y_grid=True, x_grid=True, hide_y_tick_labels=True)
    axis.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18), frameon=False, columnspacing=1.2, handlelength=2.2)
    figure.suptitle(title, fontsize=THESIS_STYLE.figure_title_size, y=0.98)
    figure.subplots_adjust(left=0.08, right=0.985, bottom=0.17, top=0.78)
    return save_single_figure(figure, output_path)


def plot_air_knees_left_knee(prepared_dir: Path, output_path: Path) -> Path | None:
    _source_name, path = _prepared_path(prepared_dir, "air_knees", "knee_default_left.csv")
    if not path.exists():
        return None
    frame = _signal(path, axis_name="y", source_name="mocap", valid_only=True, normalize=False, invert_pose_vertical=False)
    figure, axis = plt.subplots(figsize=figure_size("wide", aspect=0.52), dpi=THESIS_STYLE.dpi)
    axis.plot(frame["plot_time_sec"], frame["plot_y"], color=THESIS_STYLE.blue, linewidth=0.9, label="y")
    axis.set_title("LK trajectory in air_knees")
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Value")
    axis.legend(loc="upper right")
    style_axis(axis, y_grid=True, x_grid=True)
    figure.subplots_adjust(left=0.10, right=0.985, bottom=0.17, top=0.88)
    return save_single_figure(figure, output_path)


def plot_aligned_full_left_knee(prepared_dir: Path, config_path: Path, output_path: Path) -> Path | None:
    peak_times = _load_peak_times(config_path)
    series_map: dict[str, pd.DataFrame] = {}
    for recording_name in RECORDING_ORDER:
        source_name, path = _prepared_path(prepared_dir, recording_name, "knee_default_left.csv")
        if not path.exists() or recording_name not in peak_times:
            continue
        frame = _signal(path, axis_name="y", source_name=source_name, normalize=True, invert_pose_vertical=True)
        if frame.empty:
            continue
        frame["plot_time_sec"] = frame["plot_time_sec"] - peak_times[recording_name]
        series_map[recording_name] = frame
    if not series_map:
        return None
    return _plot_overlay(
        _offset_series(series_map, offset_step=2.4),
        output_path,
        title="Aligned Full Recordings: LK",
        x_axis_label="Aligned time relative to saved anchor (s)",
        x_window=(-16.0, 112.0),
        reference_line=(0.0, "saved anchor"),
    )


def plot_trimmed_right_wrist(aligned_dir: Path, output_path: Path) -> Path | None:
    series_map: dict[str, pd.DataFrame] = {}
    for recording_name in RECORDING_ORDER:
        csv_name = "wrist_body_right.csv" if recording_name in {"apple_vision", "mediapipe"} else "wrist_center_right.csv"
        source_name, path = _aligned_path(aligned_dir, recording_name, csv_name)
        if not path.exists():
            continue
        frame = _signal(path, axis_name="y", source_name=source_name, normalize=True, invert_pose_vertical=True)
        if not frame.empty:
            series_map[recording_name] = frame
    if not series_map:
        return None
    return _plot_overlay(
        _offset_series(series_map, offset_step=2.4),
        output_path,
        title="Trimmed Recordings: body-based RW",
        x_axis_label="Trimmed clip time (s)",
        x_window=(0.0, 118.0),
        reference_line=(16.0, "saved anchor"),
    )


def generate_figures(
    *,
    prepared_dir: Path = DEFAULT_PREPARED_DIR,
    aligned_dir: Path = DEFAULT_ALIGNED_DIR,
    output_dir: Path,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        plot_air_knees_left_knee(prepared_dir, output_dir / "air_knees_left_knee.png"),
        plot_aligned_full_left_knee(prepared_dir, config_path, output_dir / "aligned_full_left_knee.png"),
        plot_trimmed_right_wrist(aligned_dir, output_dir / "trimmed_right_wrist_overlay.png"),
    ]
    return [path for path in outputs if path is not None]


if __name__ == "__main__":
    for generated_path in generate_figures(output_dir=THESIS_OUTPUT_ROOT / "pose_evaluation"):
        print(generated_path)
