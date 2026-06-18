#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Avoid matplotlib cache warnings in restricted environments.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str((PROJECT_ROOT / ".cache" / "matplotlib").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str((PROJECT_ROOT / ".cache").resolve()))

from thesis_figures import (
    THESIS_DATA_ROOT,
    THESIS_OUTPUT_ROOT,
    THESIS_STYLE,
    apply_thesis_style,
    figure_size,
    resolve_input_dir,
    save_single_figure,
    style_axis,
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch


DEFAULT_RESULTS_DIR = resolve_input_dir(
    "motion_analysis",
    THESIS_DATA_ROOT / "motion_analysis",
    required=(
        "proxy_dynamic_metrics.csv",
        "knee_condition_vs_foot_condition_segments.csv",
        "knee_condition_vs_foot_condition_tests.csv",
        "knee_condition_vs_foot_condition_summary.csv",
    ),
)
DEFAULT_OUTPUT_DIR = THESIS_OUTPUT_ROOT / "motion_analysis"
DEFAULT_OVERLAY_DIR = resolve_input_dir(
    "motion_analysis/proxy_overlays",
    THESIS_DATA_ROOT / "motion_analysis" / "proxy_overlays",
    required=(
        "proxy_dynamic_overlay_top_drums_L.png",
        "proxy_dynamic_overlay_top_drums_R.png",
    ),
)

FAMILY_ORDER = [
    "fixed_k",
    "dynamic_k_model",
    "direct_tip_model",
    "advanced_supervised_model",
]

FAMILY_FIGURE_LABELS = {
    "fixed_k": "Fixed-k\nRW/LW-relative",
    "dynamic_k_model": "Dynamic\nk(t)",
    "direct_tip_model": "Direct RST/LST\nprediction",
    "advanced_supervised_model": "Advanced\nsupervised",
}

FAMILY_COLORS = {
    "advanced_supervised_model": "#2F6BDE",
    "direct_tip_model": "#E94B3C",
    "dynamic_k_model": "#10A37F",
    "fixed_k": "#E88C00",
}

PROXY_OVERLAY_EXAMPLES_SIZE = (9.8, 6.6)

CONDITION_ORDER = [
    ("air_feet", "L"),
    ("air_feet", "R"),
    ("air_knees", "L"),
    ("air_knees", "R"),
    ("drums", "L"),
    ("drums", "R"),
]

CONDITION_LABELS = {
    ("air_feet", "L"): "Air feet (L)",
    ("air_feet", "R"): "Air feet (R)",
    ("air_knees", "L"): "Air knees (L)",
    ("air_knees", "R"): "Air knees (R)",
    ("drums", "L"): "Drums (L)",
    ("drums", "R"): "Drums (R)",
}


def _apply_reference_style() -> None:
    apply_thesis_style()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _condition_key(df: pd.DataFrame) -> pd.Series:
    return df["recording"].astype(str) + "|" + df["side"].astype(str)


def _condition_sort_columns(df: pd.DataFrame) -> pd.DataFrame:
    order = {f"{rec}|{side}": i for i, (rec, side) in enumerate(CONDITION_ORDER)}
    out = df.copy()
    out["_condition_order"] = _condition_key(out).map(order)
    return out.sort_values("_condition_order")


def _load_proxy_metrics(results_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(results_dir / "proxy_dynamic_metrics.csv")
    df = df[df["split"] == "test"].copy()
    df["condition_label"] = [CONDITION_LABELS[(rec, side)] for rec, side in zip(df["recording"], df["side"])]
    return _condition_sort_columns(df)


def plot_proxy_winner_heatmap(proxy_metrics: pd.DataFrame, out_path: Path) -> None:
    grouped = (
        proxy_metrics.groupby(["recording", "side", "model_family"], as_index=False)
        .agg(
            composite_score=("composite_score", "min"),
        )
    )

    pivot = (
        grouped.pivot_table(
            index=["recording", "side"],
            columns="model_family",
            values="composite_score",
            aggfunc="min",
        )
        .reindex(index=CONDITION_ORDER, columns=FAMILY_ORDER)
    )

    values = pivot.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    norm = Normalize(vmin=float(np.min(finite)), vmax=float(np.max(finite)))
    cmap = LinearSegmentedColormap.from_list("thesis_blues", ["#D9E7FB", "#83A7EA", "#2F6BDE"])

    fig, ax = plt.subplots(figsize=figure_size("wide", aspect=0.72))
    im = ax.imshow(values, aspect="auto", cmap=cmap, norm=norm)

    ax.set_xticks(np.arange(len(FAMILY_ORDER)))
    ax.set_xticklabels([FAMILY_FIGURE_LABELS[f] for f in FAMILY_ORDER])
    ax.set_yticks(np.arange(len(CONDITION_ORDER)))
    ax.set_yticklabels([CONDITION_LABELS[idx] for idx in CONDITION_ORDER])
    ax.set_title("Proxy Model Winners by Condition")
    ax.set_xlabel("Model family")
    ax.set_ylabel("Recording / side")

    winner_positions: set[tuple[int, int]] = set()
    for row_idx, condition in enumerate(CONDITION_ORDER):
        row = pivot.loc[condition]
        if row.notna().any():
            col_idx = int(np.nanargmin(row.to_numpy(dtype=float)))
            winner_positions.add((row_idx, col_idx))

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if not np.isfinite(value):
                continue
            score_text = f"{value:.3f}"
            weight = "bold" if (i, j) in winner_positions else "regular"
            winner_tag = "\nWinner" if (i, j) in winner_positions else ""
            ax.text(
                j,
                i,
                f"{score_text}{winner_tag}",
                ha="center",
                va="center",
                color="white" if norm(value) > 0.45 else "#1F2A44",
                fontsize=THESIS_STYLE.annotation_font_size,
                fontweight=weight,
            )

    ax.set_xticks(np.arange(-0.5, len(FAMILY_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(CONDITION_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
    cbar.set_label("Best composite score within family\n(lower is better)")

    fig.tight_layout()
    save_single_figure(fig, out_path)


def plot_proxy_delta_vs_baseline(proxy_metrics: pd.DataFrame, out_path: Path) -> None:
    best = (
        proxy_metrics.sort_values(["recording", "side", "composite_score", "rmse"], ascending=[True, True, True, True])
        .groupby(["recording", "side"], as_index=False)
        .first()
    )
    baseline = proxy_metrics[proxy_metrics["model_name"] == "fixed_k_best_rmse"].copy()

    merged = best.merge(
        baseline[["recording", "side", "rmse", "f1", "onset_mae_ms"]],
        on=["recording", "side"],
        suffixes=("", "_baseline"),
        how="inner",
    )
    merged["condition_label"] = [CONDITION_LABELS[(rec, side)] for rec, side in zip(merged["recording"], merged["side"])]
    merged = _condition_sort_columns(merged)

    metrics = [
        ("rmse", "RMSE delta vs fixed-k RMSE baseline", "winner - baseline", True),
        ("f1", "F1 delta vs fixed-k RMSE baseline", "winner - baseline", False),
        ("onset_mae_ms", "Onset MAE delta vs fixed-k RMSE baseline", "winner - baseline (ms)", True),
    ]

    fig, axes = plt.subplots(1, 3, figsize=figure_size("wide", aspect=0.52), sharey=True)
    fig.suptitle(
        "Per-Condition Improvement Relative to Fixed-k RMSE Baseline",
        y=0.98,
        fontsize=THESIS_STYLE.figure_title_size,
    )

    y = np.arange(len(merged))
    for ax, (metric, title, xlabel, lower_is_better) in zip(axes, metrics):
        delta = merged[metric] - merged[f"{metric}_baseline"]
        colors = []
        for value in delta:
            improved = value < 0 if lower_is_better else value > 0
            colors.append("#10A37F" if improved else "#E94B3C")
        bars = ax.barh(y, delta.to_numpy(dtype=float), color=colors, edgecolor="none", height=0.72)
        ax.axvline(0.0, color="#1F2A44", linewidth=1.5, alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_yticks(y)
        ax.set_yticklabels(merged["condition_label"])
        ax.invert_yaxis()
        for bar, value in zip(bars, delta):
            x_text = value + (0.002 if metric != "onset_mae_ms" else 0.35) * (1 if value >= 0 else -1)
            ax.text(
                x_text,
                bar.get_y() + bar.get_height() / 2,
                f"{value:+.3f}" if metric != "onset_mae_ms" else f"{value:+.1f}",
                va="center",
                ha="left" if value >= 0 else "right",
                fontsize=THESIS_STYLE.annotation_font_size,
                color=THESIS_STYLE.text_color,
            )

        mean_delta = float(delta.mean())
        ax.text(
            0.02,
            0.03,
            f"Mean delta: {mean_delta:+.4f}" if metric != "onset_mae_ms" else f"Mean delta: {mean_delta:+.2f} ms",
            transform=ax.transAxes,
            fontsize=THESIS_STYLE.small_font_size,
            color=THESIS_STYLE.text_color,
            bbox={"facecolor": "#F6F8FC", "edgecolor": "#DDE5F0", "boxstyle": "round,pad=0.35"},
        )

        style_axis(ax, y_grid=False, x_grid=True)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_single_figure(fig, out_path)


def plot_proxy_overlay_examples(overlay_dir: Path, out_path: Path) -> None:
    selected_examples = [
        ("drums", "L", "Left-side example: direct LST prediction winner"),
        ("drums", "R", "Right-side example: dynamic k(t) winner"),
    ]
    overlay_paths = [
        overlay_dir / f"proxy_dynamic_overlay_top_{recording}_{side}.png"
        for recording, side, _ in selected_examples
    ]
    if not any(path.exists() for path in overlay_paths):
        return

    fig, axes = plt.subplots(2, 1, figsize=PROXY_OVERLAY_EXAMPLES_SIZE)
    fig.suptitle(
        "Representative Trajectory Overlays for Proxy Prediction",
        y=0.975,
        fontsize=THESIS_STYLE.figure_title_size,
    )

    for ax, (recording, side, title_suffix), path in zip(np.atleast_1d(axes), selected_examples, overlay_paths):
        ax.set_axis_off()
        if not path.exists():
            ax.text(
                0.5,
                0.5,
                "Overlay not found",
                ha="center",
                va="center",
                fontsize=THESIS_STYLE.label_size,
                color=THESIS_STYLE.text_color,
            )
            ax.set_title(title_suffix)
            continue
        img = plt.imread(path)
        ax.imshow(img)
        ax.set_title(title_suffix, pad=4)

    fig.subplots_adjust(left=0.04, right=0.96, bottom=0.04, top=0.91, hspace=0.16)
    save_single_figure(fig, out_path)


def _p_value_label(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "p = n/a"
    if p_value < 0.001:
        return "p < 0.001"
    return f"p = {p_value:.3f}"


def plot_rom_side_comparison(
    segments: pd.DataFrame,
    tests: pd.DataFrame,
    summary: pd.DataFrame,
    out_path: Path,
) -> None:
    filtered = segments[segments["keep_segment"]].copy()

    metrics = [
        ("peak_to_peak", "Peak-to-peak range"),
        ("p95_p05", "p95-p05 range"),
    ]
    sides = ["L", "R"]

    fig, axes = plt.subplots(2, 2, figsize=figure_size("wide", aspect=0.78))
    fig.suptitle(
        "Range-of-Motion Comparison Between RK/LK and RT/LT",
        y=0.98,
        fontsize=THESIS_STYLE.figure_title_size,
    )

    for row_idx, (metric_key, metric_title) in enumerate(metrics):
        for col_idx, side in enumerate(sides):
            ax = axes[row_idx, col_idx]
            side_data = filtered[filtered["side"] == side]
            knee_col = f"knee_{metric_key}"
            toe_col = f"toe_{metric_key}"

            if side_data.empty:
                ax.text(
                    0.5,
                    0.5,
                    "No filtered segments",
                    ha="center",
                    va="center",
                    fontsize=THESIS_STYLE.label_size,
                    color=THESIS_STYLE.text_color,
                )
                ax.set_axis_off()
                continue

            bp = ax.boxplot(
                [side_data[knee_col].to_numpy(dtype=float), side_data[toe_col].to_numpy(dtype=float)],
                widths=0.55,
                patch_artist=True,
                showmeans=False,
                medianprops={"color": "#1F2A44", "linewidth": 2.2},
                whiskerprops={"color": "#7A8AA6", "linewidth": 1.4},
                capprops={"color": "#7A8AA6", "linewidth": 1.4},
                boxprops={"linewidth": 1.4, "color": "#7A8AA6"},
            )

            for patch, color in zip(bp["boxes"], ["#2F6BDE", "#E88C00"]):
                patch.set_facecolor(color)
                patch.set_alpha(0.85)

            x_jitter = np.array([-0.06, 0.06])
            for x_base, col_name, color in [(1.0, knee_col, "#2F6BDE"), (2.0, toe_col, "#E88C00")]:
                values = side_data[col_name].to_numpy(dtype=float)
                jitter = np.linspace(x_base + x_jitter[0], x_base + x_jitter[1], num=len(values))
                ax.scatter(jitter, values, s=28, alpha=0.45, color=color, edgecolors="none")

            test_row = tests[(tests["side"] == side) & (tests["metric"] == metric_key)]
            summary_row = summary[summary["side"] == side]

            title = f"{'Left' if side == 'L' else 'Right'} side: {metric_title}"
            if not test_row.empty:
                p_label = _p_value_label(float(test_row.iloc[0]["p_value"]))
                title = f"{title}\n{p_label}"
            ax.set_title(title)
            ax.set_xticks([1, 2])
            if side == "L":
                marker_labels = ("LK", "LT")
            else:
                marker_labels = ("RK", "RT")
            ax.set_xticklabels(marker_labels)
            ax.set_ylabel("Filtered segment range")

            if not summary_row.empty:
                median_knee = float(summary_row.iloc[0][f"knee_filtered_median_{metric_key}"])
                median_toe = float(summary_row.iloc[0][f"toe_filtered_median_{metric_key}"])
                ax.text(
                    0.03,
                    0.96,
                    f"Median {marker_labels[0]}: {median_knee:.6f}\nMedian {marker_labels[1]}: {median_toe:.6f}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=THESIS_STYLE.annotation_font_size,
                    color=THESIS_STYLE.text_color,
                    bbox={"facecolor": "#F6F8FC", "edgecolor": "#DDE5F0", "boxstyle": "round,pad=0.35"},
                )
            style_axis(ax, y_grid=True, x_grid=False)

    handles = [
        Patch(facecolor="#2F6BDE", alpha=0.85, label="RK/LK"),
        Patch(facecolor="#E88C00", alpha=0.85, label="RT/LT"),
    ]

    fig.legend(handles=handles, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.93), frameon=False, handlelength=2.6)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_single_figure(fig, out_path)


def generate_results_section_figures(
    results_dir: Path,
    output_dir: Path,
    overlay_dir: Path,
) -> list[Path]:
    _apply_reference_style()
    _ensure_dir(output_dir)

    proxy_metrics = _load_proxy_metrics(results_dir)
    rom_segments = pd.read_csv(results_dir / "knee_condition_vs_foot_condition_segments.csv")
    rom_tests = pd.read_csv(results_dir / "knee_condition_vs_foot_condition_tests.csv")
    rom_summary = pd.read_csv(results_dir / "knee_condition_vs_foot_condition_summary.csv")

    outputs = [
        output_dir / "proxy_winner_heatmap.png",
        output_dir / "proxy_delta_vs_fixed_k_baseline.png",
        output_dir / "proxy_overlay_examples.png",
        output_dir / "knee_toe_rom_side_comparison.png",
    ]

    plot_proxy_winner_heatmap(proxy_metrics, outputs[0])
    plot_proxy_delta_vs_baseline(proxy_metrics, outputs[1])
    plot_proxy_overlay_examples(overlay_dir, outputs[2])
    plot_rom_side_comparison(rom_segments, rom_tests, rom_summary, outputs[3])
    return [path for path in outputs if path.exists()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate thesis figures for the results section.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR, help="Directory with result CSV files.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to write the generated figures.")
    parser.add_argument("--overlay-dir", type=Path, default=DEFAULT_OVERLAY_DIR, help="Directory containing saved proxy overlay PNGs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = generate_results_section_figures(
        results_dir=args.results_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        overlay_dir=args.overlay_dir.resolve(),
    )
    print("Generated figures:")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
