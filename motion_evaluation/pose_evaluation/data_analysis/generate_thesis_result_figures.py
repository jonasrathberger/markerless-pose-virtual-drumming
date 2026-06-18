from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_figures import THESIS_OUTPUT_ROOT, THESIS_STYLE, apply_thesis_style, figure_size, resolve_input_dir
from thesis_figures import finish_figure as shared_finish_figure
from thesis_figures import save_single_figure, style_axis as shared_style_axis

os.environ.setdefault("MPLCONFIGDIR", str((REPO_ROOT / ".cache" / "matplotlib").resolve()))
apply_thesis_style()

import matplotlib.pyplot as plt


ROOT_DIR = Path(__file__).resolve().parents[1]
BASELINE_OUTPUT_DIR = resolve_input_dir(
    "pose_evaluation",
    ROOT_DIR / "data_analysis" / "output",
    required=("summary_by_domain.csv", "summary_by_module.csv", "bland_altman.csv"),
)
FIGURES_DIR = THESIS_OUTPUT_ROOT / "pose_evaluation"
ALIGNMENT_RESULT_DIR = resolve_input_dir(
    "pose_evaluation/aligned_result",
    ROOT_DIR / "data_alignment" / "aligned_result",
    required=(
        "mocap/air_knees/wrist_center_right.csv",
        "mocap/drums/wrist_center_right.csv",
        "pose/apple_vision/wrist_body_right.csv",
        "pose/mediapipe/wrist_body_right.csv",
    ),
)

PALETTE = {
    "air_knees": "#1d4ed8",
    "drums": "#dc2626",
    "apple_vision": "#059669",
    "mediapipe": "#d97706",
    "optitrack": "#1d4ed8",
}

LABELS = {
    "air_knees": "OptiTrack Motion Capture: Air Knees",
    "drums": "OptiTrack Motion Capture: Drums",
    "apple_vision": "Apple Vision",
    "mediapipe": "MediaPipe",
    "optitrack": "OptiTrack",
}

ALIGNMENT_LABELS = {
    "body_only": "Body only",
    "body_plus_similarity": "Body +\nsimilarity",
    "body_plus_torso_similarity": "Body + torso\nsimilarity",
}

CONDITION_LABELS = {
    "air_knees": "Air Knees",
    "drums": "Drums",
}

TRIAL_SHORT_LABELS = {
    "air_knees": "AK",
    "drums": "DR",
}

SIDE_MARKERS = {
    "left": "o",
    "right": "s",
}

RECORDING_ORDER = ["air_knees", "drums", "apple_vision", "mediapipe"]
PANEL_TITLE_SIZE = 8.2
COMPACT_TITLE_SIZE = 7.7
COMPACT_LABEL_SIZE = 7.4
COMPACT_LEGEND_SIZE = 7.7
TRAJECTORY_FIGURE_SIZE = (8.2, 4.1)


def style_axis(axis: plt.Axes, *, y_grid: bool = True, x_grid: bool = True, hide_y_tick_labels: bool = False) -> None:
    shared_style_axis(axis, y_grid=y_grid, x_grid=x_grid, hide_y_tick_labels=hide_y_tick_labels)


def finalize_figure(figure: plt.Figure, title: str, *, top: float = 0.86, left: float = 0.07, right: float = 0.985, bottom: float = 0.12, hspace: float = 0.35, wspace: float = 0.25) -> None:
    shared_finish_figure(figure, title, top=top, left=left, right=right, bottom=bottom, hspace=hspace, wspace=wspace)


def save_figure(figure: plt.Figure, output_path: Path) -> None:
    save_single_figure(figure, output_path)


def load_csv(name: str, input_dir: Path = BASELINE_OUTPUT_DIR) -> pd.DataFrame:
    path = input_dir / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def metric_value(
    summary_by_domain: pd.DataFrame,
    *,
    comparison_name: str,
    wrist_variant: str,
    alignment_mode: str,
    metric_domain: str,
    metric_name: str,
    trial_id: str | None = None,
    system_name: str = "pair",
    side_mode: str = "mean",
) -> float:
    frame = summary_by_domain[
        (summary_by_domain["comparison_name"] == comparison_name)
        & (summary_by_domain["wrist_variant"] == wrist_variant)
        & (summary_by_domain["alignment_mode"] == alignment_mode)
        & (summary_by_domain["metric_domain"] == metric_domain)
        & (summary_by_domain["metric_name"] == metric_name)
        & (summary_by_domain["system_name"] == system_name)
    ].copy()
    if trial_id is not None:
        frame = frame[frame["trial_id"] == trial_id]
    if frame.empty:
        return float("nan")
    if side_mode == "both":
        frame = frame[frame["side"] == "both"]
        if frame.empty:
            return float("nan")
        return float(frame.iloc[0]["mean_metric_value"])
    if side_mode == "mean":
        frame = frame[frame["side"].isin(["left", "right"])]
        return float(frame["mean_metric_value"].mean()) if not frame.empty else float("nan")
    if side_mode in {"left", "right"}:
        frame = frame[frame["side"] == side_mode]
        if frame.empty:
            return float("nan")
        return float(frame.iloc[0]["mean_metric_value"])
    raise ValueError(f"Unsupported side_mode: {side_mode}")


def jitter_value(
    summary_by_module: pd.DataFrame,
    *,
    comparison_name: str,
    wrist_variant: str,
    alignment_mode: str,
    system_name: str,
    metric_name: str,
) -> float:
    frame = summary_by_module[
        (summary_by_module["comparison_name"] == comparison_name)
        & (summary_by_module["wrist_variant"] == wrist_variant)
        & (summary_by_module["alignment_mode"] == alignment_mode)
        & (summary_by_module["system_name"] == system_name)
        & (summary_by_module["metric_family"] == "jitter")
        & (summary_by_module["metric_name"] == metric_name)
    ]
    if frame.empty:
        return float("nan")
    return float(frame["mean_metric_value"].mean())


def value_labels(axis: plt.Axes, bars, *, fmt: str = "{:.3f}", fontsize: float = COMPACT_LABEL_SIZE) -> None:
    y_min, y_max = axis.get_ylim()
    offset = 0.02 * (y_max - y_min if y_max > y_min else 1.0)
    for bar in bars:
        height = float(bar.get_height())
        axis.text(
            bar.get_x() + (bar.get_width() / 2.0),
            height + offset,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color=THESIS_STYLE.muted_text_color,
        )


def plot_alignment_comparison(summary_by_domain: pd.DataFrame, output_path: Path) -> None:
    alignments = ["body_only", "body_plus_similarity", "body_plus_torso_similarity"]
    comparisons = [
        ("applevision_vs_optitrack", "apple_vision"),
        ("mediapipe_vs_optitrack", "mediapipe"),
    ]
    metric_specs = [
        ("core_spatial", "mpjpe_2d", "both", "MPJPE", "Lower is better"),
        ("core_spatial", "pck_2d@0.5", "both", "PCK@0.5", "Higher is better"),
        ("knee_pedal", "excursion_mean_abs_error", "mean", "RK/LK Excursion MAE", "Lower is better"),
    ]

    figure, axes = plt.subplots(1, 3, figsize=figure_size("wide", aspect=0.56), dpi=THESIS_STYLE.dpi)
    width = 0.34
    x = np.arange(len(alignments), dtype=float)

    for axis, (metric_domain, metric_name, side_mode, metric_title, metric_note) in zip(axes, metric_specs):
        for index, (comparison_name, system_key) in enumerate(comparisons):
            values = [
                metric_value(
                    summary_by_domain,
                    comparison_name=comparison_name,
                    wrist_variant="body",
                    alignment_mode=alignment_mode,
                    metric_domain=metric_domain,
                    metric_name=metric_name,
                    side_mode=side_mode,
                )
                for alignment_mode in alignments
            ]
            bars = axis.bar(
                x + ((index - 0.5) * width),
                values,
                width=width,
                color=PALETTE[system_key],
                label=LABELS[system_key],
                alpha=0.95,
            )
            value_labels(axis, bars)

        axis.set_xticks(x)
        axis.set_xticklabels([ALIGNMENT_LABELS[item] for item in alignments])
        axis.set_title(f"{metric_title}\n{metric_note}", fontsize=PANEL_TITLE_SIZE, color=THESIS_STYLE.text_color, pad=7)
        style_axis(axis)
        axis.tick_params(axis="x", labelsize=COMPACT_LABEL_SIZE)

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.865),
        ncol=2,
        frameon=False,
        fontsize=COMPACT_LEGEND_SIZE,
        columnspacing=1.8,
        handlelength=2.8,
    )
    finalize_figure(figure, "Baseline Alignment Comparison", top=0.72, bottom=0.18, wspace=0.32)
    save_figure(figure, output_path)


def plot_system_comparison(summary_by_domain: pd.DataFrame, output_path: Path) -> None:
    comparison_to_system = {
        "applevision_vs_optitrack": "apple_vision",
        "mediapipe_vs_optitrack": "mediapipe",
    }
    metric_specs = [
        ("core_spatial", "mpjpe_2d", "both", "MPJPE", "Lower is better"),
        ("core_spatial", "pck_2d@0.5", "both", "PCK@0.5", "Higher is better"),
        ("pearson", "pearson_velocity", "mean", "RW/LW Velocity Pearson", "Higher is better"),
        ("pearson", "pearson_angle", "mean", "Angle Pearson", "Higher is better"),
        ("pearson", "pearson_along_axis_velocity", "mean", "RK/LK Along-Axis Pearson", "Higher is better"),
        ("knee_pedal", "excursion_mean_abs_error", "mean", "RK/LK Excursion MAE", "Lower is better"),
    ]

    figure, axes = plt.subplots(2, 3, figsize=figure_size("wide", aspect=0.98), dpi=THESIS_STYLE.dpi)
    axes = axes.flatten()

    for axis, (metric_domain, metric_name, side_mode, metric_title, metric_note) in zip(axes, metric_specs):
        systems = []
        values = []
        colors = []
        for comparison_name, system_key in comparison_to_system.items():
            systems.append(LABELS[system_key])
            values.append(
                metric_value(
                    summary_by_domain,
                    comparison_name=comparison_name,
                    wrist_variant="body",
                    alignment_mode="body_plus_similarity",
                    metric_domain=metric_domain,
                    metric_name=metric_name,
                    side_mode=side_mode,
                )
            )
            colors.append(PALETTE[system_key])
        bars = axis.bar(systems, values, color=colors, width=0.62, alpha=0.95)
        axis.set_title(f"{metric_title}\n{metric_note}", fontsize=PANEL_TITLE_SIZE, color=THESIS_STYLE.text_color, pad=7)
        style_axis(axis)
        axis.tick_params(axis="x", labelsize=COMPACT_LABEL_SIZE)
        value_labels(axis, bars)

    finalize_figure(figure, "Baseline System Comparison Against OptiTrack", top=0.89, bottom=0.11, hspace=0.68, wspace=0.34)
    save_figure(figure, output_path)


def plot_condition_comparison(summary_by_domain: pd.DataFrame, output_path: Path) -> None:
    system_rows = [
        ("applevision_vs_optitrack", "Apple Vision"),
        ("mediapipe_vs_optitrack", "MediaPipe"),
    ]
    metric_specs = [
        ("pearson", "pearson_velocity", "mean", "RW/LW Velocity Pearson", "Higher is better"),
        ("pearson", "pearson_angle", "mean", "Angle Pearson", "Higher is better"),
        ("pearson", "pearson_along_axis_velocity", "mean", "RK/LK Along-Axis Pearson", "Higher is better"),
        ("hand_arm", "reversal_mean_abs_timing_error_sec", "mean", "RW/LW Reversal MAE (s)", "Lower is better"),
    ]
    trials = ["air_knees", "drums"]
    colors = [PALETTE["air_knees"], PALETTE["drums"]]

    figure, axes = plt.subplots(2, 4, figsize=figure_size("wide", aspect=1.02), dpi=THESIS_STYLE.dpi)

    for row_index, (comparison_name, system_label) in enumerate(system_rows):
        for col_index, (metric_domain, metric_name, side_mode, metric_title, metric_note) in enumerate(metric_specs):
            axis = axes[row_index, col_index]
            values = [
                metric_value(
                    summary_by_domain,
                    comparison_name=comparison_name,
                    wrist_variant="body",
                    alignment_mode="body_plus_similarity",
                    metric_domain=metric_domain,
                    metric_name=metric_name,
                    side_mode=side_mode,
                    trial_id=trial_id,
                )
                for trial_id in trials
            ]
            bars = axis.bar(
                [CONDITION_LABELS[item] for item in trials],
                values,
                color=colors,
                width=0.62,
                alpha=0.95,
            )
            if row_index == 0:
                axis.set_title(f"{metric_title}\n{metric_note}", fontsize=COMPACT_TITLE_SIZE, color=THESIS_STYLE.text_color, pad=7)
            if col_index == 0:
                axis.set_ylabel(system_label, fontsize=COMPACT_LABEL_SIZE, color=THESIS_STYLE.text_color)
            style_axis(axis)
            axis.tick_params(axis="x", labelsize=COMPACT_LABEL_SIZE)
            value_labels(axis, bars)

    finalize_figure(figure, "Baseline Condition Comparison", top=0.88, bottom=0.10, hspace=0.58, wspace=0.42)
    save_figure(figure, output_path)


def prepare_overlay_series(csv_path: Path, *, invert_pose_vertical: bool, offset_step: float) -> tuple[pd.DataFrame, float]:
    frame = pd.read_csv(csv_path)
    if "valid" in frame.columns:
        frame["valid"] = frame["valid"].fillna("").astype(str).str.lower().eq("true")
        frame = frame[frame["valid"]]
    signal = frame[["time_sec", "y"]].dropna().copy()
    if invert_pose_vertical:
        signal["y"] = -signal["y"]
    signal["y"] = signal["y"] - signal["y"].mean()
    std = float(signal["y"].std(ddof=0))
    if np.isfinite(std) and std > 0:
        signal["y"] = signal["y"] / std
    nominal_offset = 16.0
    if "alignment_peak_nominal_offset_sec" in frame.columns:
        series = pd.to_numeric(frame["alignment_peak_nominal_offset_sec"], errors="coerce").dropna()
        if not series.empty:
            nominal_offset = float(series.iloc[0])
    signal["plot_time_sec"] = signal["time_sec"] - nominal_offset
    signal["plot_y"] = signal["y"]
    return signal[["plot_time_sec", "plot_y"]], nominal_offset


def apply_vertical_offsets(series_map: dict[str, pd.DataFrame], offset_step: float) -> dict[str, pd.DataFrame]:
    shifted_map: dict[str, pd.DataFrame] = {}
    centered_index = (len(RECORDING_ORDER) - 1) / 2.0
    for index, recording_name in enumerate(RECORDING_ORDER):
        frame = series_map.get(recording_name, pd.DataFrame(columns=["plot_time_sec", "plot_y"])).copy()
        if frame.empty:
            shifted_map[recording_name] = frame
            continue
        offset = (centered_index - index) * offset_step
        frame["plot_y"] = frame["plot_y"] + offset
        shifted_map[recording_name] = frame
    return shifted_map


def plot_representative_overlay(
    output_path: Path,
    *,
    alignment_result_dir: Path = ALIGNMENT_RESULT_DIR,
) -> bool:
    file_map = {
        "air_knees": alignment_result_dir / "mocap" / "air_knees" / "wrist_center_right.csv",
        "drums": alignment_result_dir / "mocap" / "drums" / "wrist_center_right.csv",
        "apple_vision": alignment_result_dir / "pose" / "apple_vision" / "wrist_body_right.csv",
        "mediapipe": alignment_result_dir / "pose" / "mediapipe" / "wrist_body_right.csv",
    }
    missing_paths = [path for path in file_map.values() if not path.exists()]
    if missing_paths:
        print(
            "Skipping representative trajectory overlay; missing source CSV(s): "
            + ", ".join(str(path) for path in missing_paths),
            file=sys.stderr,
        )
        return False

    series_map = {
        recording_name: prepare_overlay_series(path, invert_pose_vertical=("pose" in path.parts), offset_step=2.4)[0]
        for recording_name, path in file_map.items()
    }
    shifted_map = apply_vertical_offsets(series_map, offset_step=2.4)

    figure, axis = plt.subplots(figsize=TRAJECTORY_FIGURE_SIZE, dpi=THESIS_STYLE.dpi)
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")

    for recording_name in RECORDING_ORDER:
        frame = shifted_map[recording_name]
        axis.plot(
            frame["plot_time_sec"],
            frame["plot_y"],
            color=PALETTE[recording_name],
            linewidth=1.15,
            label=LABELS[recording_name],
        )

    y_min, y_max = axis.get_ylim()
    axis.axvline(0.0, color="#0f172a", linewidth=0.9, linestyle=(0, (6, 4)))
    axis.text(
        0.4,
        y_max - 0.06 * (y_max - y_min),
        "saved anchor",
        fontsize=THESIS_STYLE.small_font_size,
        color=THESIS_STYLE.text_color,
        ha="left",
        va="top",
    )

    axis.set_xlabel("Aligned time relative to saved anchor (s)", fontsize=THESIS_STYLE.label_size)
    axis.set_ylabel("Offset visualization", fontsize=THESIS_STYLE.label_size)
    style_axis(axis, hide_y_tick_labels=True)
    finalize_figure(figure, "Representative Trajectory Overlay: RW", top=0.76, bottom=0.14, left=0.08)
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.895),
        ncol=4,
        frameon=False,
        fontsize=COMPACT_LEGEND_SIZE,
        columnspacing=1.2,
        handlelength=2.2,
    )
    save_figure(figure, output_path)
    return True


def plot_bland_altman_summaries(bland_altman: pd.DataFrame, output_path: Path) -> None:
    systems = [
        ("applevision_vs_optitrack", "apple_vision", "Apple Vision"),
        ("mediapipe_vs_optitrack", "mediapipe", "MediaPipe"),
    ]
    variables = [
        ("knee_along_axis_excursion", "RK/LK Along-Axis Excursion"),
        ("wrist_peak_speed", "RW/LW Peak Speed"),
    ]

    figure, axes = plt.subplots(2, 2, figsize=figure_size("wide", aspect=0.72), dpi=THESIS_STYLE.dpi)

    for row_index, (comparison_name, system_key, system_label) in enumerate(systems):
        for col_index, (variable_name, variable_label) in enumerate(variables):
            axis = axes[row_index, col_index]
            panel = bland_altman[
                (bland_altman["comparison_name"] == comparison_name)
                & (bland_altman["wrist_variant"] == "body")
                & (bland_altman["alignment_mode"] == "body_plus_similarity")
                & (bland_altman["variable_name"] == variable_name)
            ].copy()
            panel = panel.sort_values(["trial_id", "side"])
            for _, row in panel.iterrows():
                marker = SIDE_MARKERS.get(str(row["side"]), "o")
                label = f"{TRIAL_SHORT_LABELS.get(str(row['trial_id']), str(row['trial_id']))}-{str(row['side'])[0].upper()}"
                axis.scatter(
                    float(row["mean_of_methods"]),
                    float(row["mean_difference"]),
                    color=PALETTE[system_key],
                    marker=marker,
                    s=50,
                    alpha=0.95,
                )
                axis.text(
                    float(row["mean_of_methods"]),
                    float(row["mean_difference"]),
                    f" {label}",
                    fontsize=THESIS_STYLE.annotation_font_size,
                    color=THESIS_STYLE.muted_text_color,
                    ha="left",
                    va="center",
                )

            if not panel.empty:
                bias = float(panel["bias"].mean())
                loa_lower = float(panel["loa_lower"].mean())
                loa_upper = float(panel["loa_upper"].mean())
                axis.axhline(bias, color="#0f172a", linewidth=1.2, linestyle="-")
                axis.axhline(loa_lower, color="#475569", linewidth=1.0, linestyle=(0, (4, 3)))
                axis.axhline(loa_upper, color="#475569", linewidth=1.0, linestyle=(0, (4, 3)))

            axis.set_title(f"{system_label}: {variable_label}", fontsize=THESIS_STYLE.axes_title_size, color=THESIS_STYLE.text_color)
            axis.set_xlabel("Mean of methods", fontsize=THESIS_STYLE.label_size)
            axis.set_ylabel("Difference", fontsize=THESIS_STYLE.label_size)
            style_axis(axis)

    finalize_figure(figure, "Baseline Bland-Altman Summaries", top=0.9, bottom=0.1, hspace=0.35, wspace=0.25)
    save_figure(figure, output_path)


def plot_jitter_comparison(summary_by_module: pd.DataFrame, output_path: Path) -> None:
    metric_specs = [
        ("jitter_x", "RW/LW Jitter X", "Lower is better"),
        ("jitter_y", "RW/LW Jitter Y", "Lower is better"),
        ("jitter_along_axis", "RK/LK Jitter Along Axis", "Lower is better"),
    ]
    systems = ["optitrack", "apple_vision", "mediapipe"]
    colors = [PALETTE["optitrack"], PALETTE["apple_vision"], PALETTE["mediapipe"]]

    figure, axes = plt.subplots(1, 3, figsize=figure_size("wide", aspect=0.58), dpi=THESIS_STYLE.dpi)

    for axis, (metric_name, metric_title, metric_note) in zip(axes, metric_specs):
        values = [
            jitter_value(
                summary_by_module,
                comparison_name="applevision_vs_optitrack" if system_name in {"optitrack", "apple_vision"} else "mediapipe_vs_optitrack",
                wrist_variant="body",
                alignment_mode="body_plus_similarity",
                system_name=system_name,
                metric_name=metric_name,
            )
            for system_name in systems
        ]
        bars = axis.bar(
            ["OptiTrack", "Apple\nVision", "MediaPipe"],
            values,
            color=colors,
            width=0.62,
            alpha=0.95,
        )
        axis.set_title(f"{metric_title}\n{metric_note}", fontsize=PANEL_TITLE_SIZE, color=THESIS_STYLE.text_color, pad=7)
        style_axis(axis)
        axis.tick_params(axis="x", labelsize=COMPACT_LABEL_SIZE)
        value_labels(axis, bars)

    finalize_figure(figure, "Baseline Jitter Comparison", top=0.82, bottom=0.20, wspace=0.34)
    save_figure(figure, output_path)


def write_manifest(rows: list[tuple[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["filename", "description"])
        writer.writerows(rows)


def generate_figures(
    *,
    input_dir: Path = BASELINE_OUTPUT_DIR,
    output_dir: Path = FIGURES_DIR,
    alignment_result_dir: Path = ALIGNMENT_RESULT_DIR,
) -> list[Path]:
    summary_by_domain = load_csv("summary_by_domain.csv", input_dir=input_dir)
    summary_by_module = load_csv("summary_by_module.csv", input_dir=input_dir)
    bland_altman = load_csv("bland_altman.csv", input_dir=input_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []

    alignment_path = output_dir / "baseline_alignment_comparison.png"
    plot_alignment_comparison(summary_by_domain, alignment_path)
    manifest_rows.append((alignment_path.name, "Grouped comparison of baseline alignment modes for MPJPE, PCK@0.5, and knee excursion MAE."))

    system_path = output_dir / "baseline_system_comparison.png"
    plot_system_comparison(summary_by_domain, system_path)
    manifest_rows.append((system_path.name, "Apple Vision versus MediaPipe against OptiTrack under body-plus-similarity alignment and the body-wrist variant."))

    condition_path = output_dir / "baseline_condition_comparison.png"
    plot_condition_comparison(summary_by_domain, condition_path)
    manifest_rows.append((condition_path.name, "Air-knees versus drums condition comparison for the main temporal metrics."))

    overlay_path = output_dir / "baseline_representative_trajectory_overlay.png"
    if plot_representative_overlay(overlay_path, alignment_result_dir=alignment_result_dir):
        manifest_rows.append((overlay_path.name, "Representative full-recording overlay for RW using the baseline aligned recordings."))

    bland_altman_path = output_dir / "baseline_bland_altman_summaries.png"
    plot_bland_altman_summaries(bland_altman, bland_altman_path)
    manifest_rows.append((bland_altman_path.name, "Selected Bland-Altman summary plots for RK/LK excursion and RW/LW peak speed."))

    jitter_path = output_dir / "baseline_jitter_comparison.png"
    plot_jitter_comparison(summary_by_module, jitter_path)
    manifest_rows.append((jitter_path.name, "Baseline jitter comparison across OptiTrack, Apple Vision, and MediaPipe."))

    manifest_path = output_dir / "figure_manifest.csv"
    write_manifest(manifest_rows, manifest_path)
    return [output_dir / filename for filename, _description in manifest_rows] + [manifest_path]


def main() -> None:
    outputs = generate_figures()

    print(f"Wrote {len(outputs) - 1} figures to {FIGURES_DIR}")
    for output in outputs:
        print(f"- {output.name}")


if __name__ == "__main__":
    main()
