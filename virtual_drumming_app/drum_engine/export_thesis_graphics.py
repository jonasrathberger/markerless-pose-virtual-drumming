"""Create thesis graphics from exported evaluation metrics."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_figures import (
    THESIS_DATA_ROOT,
    THESIS_OUTPUT_ROOT,
    THESIS_STYLE,
    apply_thesis_style,
    figure_size,
    resolve_input_dir,
    save_multi_format_figure,
    style_axis,
)

DEFAULT_METRICS_DIR = resolve_input_dir(
    "virtual_drumming",
    THESIS_DATA_ROOT / "virtual_drumming",
    required=("summary_metrics.csv",),
)
DEFAULT_OUTPUT_DIR = THESIS_OUTPUT_ROOT / "virtual_drumming"
REFERENCE_LABELS = {
    "computer": "computer.mid",
    "human": "human.mid",
}
MATCH_LABELS = {
    "type": "Type",
    "drum": "Drum",
}
STATUS_COLORS = {
    "matched_correct": "#2e7d32",
    "matched_wrong": "#f9a825",
    "missed": "#c62828",
    "extra": "#6a1b9a",
}


def export_graphics(
    *,
    metrics_dir: Path = DEFAULT_METRICS_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    formats: tuple[str, ...] = ("png",),
) -> list[Path]:
    _configure_matplotlib_cache()
    apply_thesis_style()
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = read_summary_metrics(metrics_dir / "summary_metrics.csv")
    outputs: list[Path] = []

    outputs.extend(
        save_figure(
            plot_metric_bars(summary_rows),
            output_dir / "metrics_precision_recall_f1",
            formats=formats,
        )
    )
    outputs.extend(
        save_figure(
            plot_count_bars(summary_rows),
            output_dir / "counts_tp_fp_fn",
            formats=formats,
        )
    )

    for reference_name in ("computer", "human"):
        outputs.extend(
            save_figure(
                plot_confusion_heatmap(
                    read_matrix_csv(metrics_dir / f"confusion_full_{reference_name}.csv"),
                    title=f"Full-system confusion matrix: {REFERENCE_LABELS[reference_name]}",
                ),
                output_dir / f"confusion_full_{reference_name}",
                formats=formats,
            )
        )
        outputs.extend(
            save_figure(
                plot_confusion_heatmap(
                    read_matrix_csv(metrics_dir / f"confusion_classification_{reference_name}.csv"),
                    title=f"Classification-only confusion matrix: {REFERENCE_LABELS[reference_name]}",
                ),
                output_dir / f"confusion_classification_{reference_name}",
                formats=formats,
            )
        )

    outputs.extend(
        save_figure(
            plot_error_by_drum(metrics_dir, error_kind="missed"),
            output_dir / "missed_hits_by_drum",
            formats=formats,
        )
    )
    outputs.extend(
        save_figure(
            plot_error_by_drum(metrics_dir, error_kind="extra"),
            output_dir / "extra_detections_by_drum",
            formats=formats,
        )
    )

    for reference_name in ("computer", "human"):
        outputs.extend(
            save_figure(
                plot_event_timeline(
                    read_matched_events(metrics_dir / f"matched_events_{reference_name}.csv"),
                    title=f"Event timeline: {REFERENCE_LABELS[reference_name]}",
                ),
                output_dir / f"event_timeline_{reference_name}",
                formats=formats,
            )
        )

    plt.close("all")
    return outputs


def read_summary_metrics(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        for field in ("precision", "recall", "f1", "offset_seconds", "scale", "tolerance_seconds"):
            row[field] = float(row[field])
        for field in ("reference_events", "detected_events", "tp", "fp", "fn"):
            row[field] = int(row[field])
    return rows


def read_matrix_csv(path: Path) -> tuple[list[str], list[str], list[list[int]]]:
    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.reader(file))
    columns = rows[0][1:]
    labels = []
    values = []
    for row in rows[1:]:
        labels.append(row[0])
        values.append([int(value) for value in row[1:]])
    return labels, columns, values


def read_matched_events(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        row["plot_time"] = _event_plot_time(row)
        row["timeline_status"] = _timeline_status(row)
    return rows


def plot_metric_bars(rows: list[dict[str, object]]):
    import matplotlib.pyplot as plt

    labels = [_row_label(row) for row in rows]
    metrics = ("precision", "recall", "f1")
    colors = ("#2f6f9f", "#d0812c", "#5f8f3f")
    x_positions = list(range(len(labels)))
    width = 0.24
    fig, ax = plt.subplots(figsize=figure_size("wide", aspect=0.62), constrained_layout=True)
    for metric_index, metric in enumerate(metrics):
        offsets = [position + ((metric_index - 1) * width) for position in x_positions]
        values = [float(row[metric]) for row in rows]
        bars = ax.bar(offsets, values, width=width, label=metric.upper(), color=colors[metric_index])
        ax.bar_label(bars, labels=[f"{value:.3f}" for value in values], fontsize=THESIS_STYLE.annotation_font_size, padding=2)
    ax.set_title("Evaluation metrics by reference and match mode")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.08)
    ax.set_xticks(x_positions, labels, rotation=20, ha="right")
    ax.legend(ncols=3, loc="upper center")
    style_axis(ax, y_grid=True, x_grid=False)
    return fig


def plot_count_bars(rows: list[dict[str, object]]):
    import matplotlib.pyplot as plt

    labels = [_row_label(row) for row in rows]
    counts = ("tp", "fp", "fn")
    colors = ("#2e7d32", "#ef6c00", "#c62828")
    x_positions = list(range(len(labels)))
    width = 0.24
    fig, ax = plt.subplots(figsize=figure_size("wide", aspect=0.62), constrained_layout=True)
    for count_index, count_name in enumerate(counts):
        offsets = [position + ((count_index - 1) * width) for position in x_positions]
        values = [int(row[count_name]) for row in rows]
        bars = ax.bar(offsets, values, width=width, label=count_name.upper(), color=colors[count_index])
        ax.bar_label(bars, labels=[str(value) for value in values], fontsize=THESIS_STYLE.annotation_font_size, padding=2)
    ax.set_title("True positives, false positives, and false negatives")
    ax.set_ylabel("Event count")
    ax.set_xticks(x_positions, labels, rotation=20, ha="right")
    ax.legend(ncols=3, loc="upper center")
    style_axis(ax, y_grid=True, x_grid=False)
    return fig


def plot_confusion_heatmap(matrix_data: tuple[list[str], list[str], list[list[int]]], *, title: str):
    import matplotlib.pyplot as plt
    import numpy as np

    row_labels, column_labels, values = matrix_data
    array = np.array(values, dtype=float)
    fig_width = 6.3
    fig_height = max(4.2, 0.32 * len(row_labels) + 1.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
    image = ax.imshow(array, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Detected drum")
    ax.set_ylabel("Reference drum")
    ax.set_xticks(range(len(column_labels)), [_human_label(label) for label in column_labels], rotation=35, ha="right")
    ax.set_yticks(range(len(row_labels)), [_human_label(label) for label in row_labels])
    threshold = array.max() * 0.55 if array.size else 0
    for row_index, row in enumerate(values):
        for column_index, value in enumerate(row):
            if value == 0:
                continue
            text_color = "white" if value >= threshold else "#202020"
            ax.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color=text_color,
                fontsize=THESIS_STYLE.annotation_font_size,
            )
    fig.colorbar(image, ax=ax, shrink=0.82, label="Event count")
    return fig


def plot_error_by_drum(metrics_dir: Path, *, error_kind: str):
    import matplotlib.pyplot as plt

    if error_kind not in ("missed", "extra"):
        raise ValueError("error_kind must be 'missed' or 'extra'.")

    reference_names = ("computer", "human")
    series = []
    all_drums: list[str] = []
    for reference_name in reference_names:
        rows, columns, values = read_matrix_csv(metrics_dir / f"confusion_full_{reference_name}.csv")
        counts = missed_counts(rows, columns, values) if error_kind == "missed" else extra_counts(rows, columns, values)
        series.append((reference_name, counts))
        for drum in counts:
            if drum not in all_drums:
                all_drums.append(drum)

    x_positions = list(range(len(all_drums)))
    width = 0.36
    fig, ax = plt.subplots(figsize=figure_size("wide", aspect=0.62), constrained_layout=True)
    colors = ("#2f6f9f", "#d0812c")
    for index, (reference_name, counts) in enumerate(series):
        offsets = [position + ((index - 0.5) * width) for position in x_positions]
        values_for_ref = [counts.get(drum, 0) for drum in all_drums]
        bars = ax.bar(offsets, values_for_ref, width=width, label=REFERENCE_LABELS[reference_name], color=colors[index])
        ax.bar_label(
            bars,
            labels=[str(value) if value > 0 else "" for value in values_for_ref],
            fontsize=THESIS_STYLE.annotation_font_size,
            padding=2,
        )
    title = "Missed hits by drum family" if error_kind == "missed" else "Extra detections by drum family"
    ax.set_title(title)
    ax.set_ylabel("Event count")
    ax.set_xticks(x_positions, [_human_label(drum) for drum in all_drums], rotation=25, ha="right")
    ax.legend()
    style_axis(ax, y_grid=True, x_grid=False)
    return fig


def plot_event_timeline(rows: list[dict[str, object]], *, title: str):
    import matplotlib.pyplot as plt

    lanes = [
        ("matched_correct", "Matched, correct drum"),
        ("matched_wrong", "Matched, wrong drum"),
        ("missed", "Missed"),
        ("extra", "Extra"),
    ]
    y_by_status = {status: index for index, (status, _label) in enumerate(lanes)}
    fig, ax = plt.subplots(figsize=figure_size("wide", aspect=0.48), constrained_layout=True)
    for status, label in lanes:
        times = [float(row["plot_time"]) for row in rows if row["timeline_status"] == status]
        y_values = [y_by_status[status]] * len(times)
        ax.scatter(times, y_values, s=18, color=STATUS_COLORS[status], label=label, alpha=0.85, edgecolors="none")
    ax.set_title(title)
    ax.set_xlabel("Video time (s)")
    ax.set_yticks([y_by_status[status] for status, _label in lanes], [label for _status, label in lanes])
    ax.set_ylim(-0.6, len(lanes) - 0.4)
    style_axis(ax, y_grid=False, x_grid=True)
    ax.legend(ncols=2, loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False)
    return fig


def missed_counts(rows: list[str], columns: list[str], values: list[list[int]]) -> dict[str, int]:
    if "missed" not in columns:
        return {}
    missed_index = columns.index("missed")
    return {
        row: values[row_index][missed_index]
        for row_index, row in enumerate(rows)
        if row != "extra" and values[row_index][missed_index] > 0
    }


def extra_counts(rows: list[str], columns: list[str], values: list[list[int]]) -> dict[str, int]:
    if "extra" not in rows:
        return {}
    extra_index = rows.index("extra")
    return {
        column: values[extra_index][column_index]
        for column_index, column in enumerate(columns)
        if column != "missed" and values[extra_index][column_index] > 0
    }


def save_figure(fig, base_path: Path, *, formats: tuple[str, ...]) -> list[Path]:
    return save_multi_format_figure(fig, base_path, formats=formats)


def _row_label(row: dict[str, object]) -> str:
    return f"{REFERENCE_LABELS[str(row['reference_name'])]}\n{MATCH_LABELS[str(row['match_mode'])]}"


def _event_plot_time(row: dict[str, str]) -> float:
    expected = row.get("expected_video_time_seconds", "")
    if expected:
        return float(expected)
    detected = row.get("detected_time_seconds", "")
    if detected:
        return float(detected)
    return 0.0


def _timeline_status(row: dict[str, str]) -> str:
    status = row["status"]
    if status == "missed":
        return "missed"
    if status == "extra":
        return "extra"
    if row["reference_drum"] == row["detected_drum"]:
        return "matched_correct"
    return "matched_wrong"


def _human_label(label: str) -> str:
    return label.replace("_", " ")


def _configure_matplotlib_cache() -> None:
    cache_dir = Path(".cache/matplotlib").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    xdg_cache_dir = Path(".cache").resolve()
    xdg_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache_dir))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create thesis graphics from exported evaluation metrics.")
    parser.add_argument("--metrics-dir", type=Path, default=DEFAULT_METRICS_DIR, help="Directory containing thesis metrics CSV files.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for generated figures.")
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png"],
        choices=("png", "pdf", "svg"),
        help="Figure formats to write.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    outputs = export_graphics(
        metrics_dir=args.metrics_dir,
        output_dir=args.output_dir,
        formats=tuple(args.formats),
    )
    print(f"Wrote {len(outputs)} figure files to {args.output_dir}.")
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
