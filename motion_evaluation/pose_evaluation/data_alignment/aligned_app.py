from __future__ import annotations

import json
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    st.set_option("global.dataFrameSerialization", "legacy")
except Exception:
    pass


ROOT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
ALIGNED_RESULT_DIR = REPO_ROOT / "thesis_data" / "pose_evaluation" / "aligned_result"
MOCAP_DIR = ALIGNED_RESULT_DIR / "mocap"
POSE_DIR = ALIGNED_RESULT_DIR / "pose"
CONFIG_PATH = Path(__file__).resolve().with_name("left_knee_alignment_config.json")
NUMERIC_COLUMNS = [
    "frame_index",
    "time_sec",
    "source_time_sec",
    "x",
    "y",
    "z",
    "x_px",
    "y_px",
    "confidence",
    "visibility",
    "alignment_peak_source_time_sec",
    "alignment_peak_offset_sec",
    "alignment_peak_nominal_offset_sec",
]
BOOLEAN_COLUMNS = ["tracking_present", "valid"]
PLOTTABLE_AXES = ["x", "y", "z", "x_px", "y_px", "confidence", "visibility"]


def load_alignment_config() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text())


def list_recordings(base_dir: Path, target_file: str) -> dict[str, Path]:
    if not base_dir.exists():
        return {}
    return {path.parent.name: path for path in sorted(base_dir.glob(f"*/{target_file}"))}


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def parse_target_stem(target_stem: str) -> tuple[str, str, str]:
    parts = target_stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected target stem: {target_stem}")
    side = parts[-1]
    target_variant = parts[-2]
    target = "_".join(parts[:-2])
    return target, target_variant, side


def format_target_label(target_stem: str) -> str:
    target, target_variant, side = parse_target_stem(target_stem)
    return f"{side.title()} {target.replace('_', ' ')} ({target_variant.replace('_', ' ')})"


def list_target_options() -> list[str]:
    stems: set[str] = set()
    for base_dir in (MOCAP_DIR, POSE_DIR):
        if not base_dir.exists():
            continue
        for csv_path in base_dir.glob("*/*.csv"):
            stems.add(csv_path.stem)
    return sorted(stems, key=lambda stem: (parse_target_stem(stem)[2], parse_target_stem(stem)[0], parse_target_stem(stem)[1]))


@st.cache_data(show_spinner=False)
def load_recording(csv_path: str) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    for column in NUMERIC_COLUMNS:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in BOOLEAN_COLUMNS:
        if column in frame.columns:
            frame[column] = frame[column].fillna("").astype(str).str.lower().eq("true")
    return frame


def build_summary(recording_name: str, source_name: str, csv_path: Path, frame: pd.DataFrame) -> dict[str, object]:
    model_name = frame["model_name"].dropna().iloc[0] if "model_name" in frame.columns and not frame.empty else ""
    coord_space = frame["coord_space"].dropna().iloc[0] if "coord_space" in frame.columns and not frame.empty else ""
    unit = frame["unit"].dropna().iloc[0] if "unit" in frame.columns and not frame.empty else ""
    return {
        "recording": recording_name,
        "source": source_name,
        "model_name": model_name,
        "rows": len(frame),
        "time_start_sec": frame["time_sec"].min() if "time_sec" in frame.columns else None,
        "time_end_sec": frame["time_sec"].max() if "time_sec" in frame.columns else None,
        "source_time_start_sec": frame["source_time_sec"].min() if "source_time_sec" in frame.columns else None,
        "source_time_end_sec": frame["source_time_sec"].max() if "source_time_sec" in frame.columns else None,
        "coord_space": coord_space,
        "unit": unit,
        "file": display_path(csv_path),
    }


def prepare_table_frame(frame: pd.DataFrame) -> pd.DataFrame:
    table_frame = frame.copy()
    table_frame.columns = [str(column) for column in table_frame.columns]
    for column in table_frame.columns:
        if pd.api.types.is_numeric_dtype(table_frame[column]) or pd.api.types.is_bool_dtype(table_frame[column]):
            continue
        table_frame[column] = table_frame[column].where(table_frame[column].notna(), "")
        table_frame[column] = table_frame[column].map(lambda value: value if isinstance(value, str) else str(value))
        table_frame[column] = table_frame[column].astype(object)
    return table_frame


def prepare_plot_frame(frame: pd.DataFrame, source_name: str, axis_name: str, valid_only: bool, normalize: bool, invert_pose_vertical: bool) -> pd.DataFrame:
    plot_frame = frame.copy()
    if valid_only and "valid" in plot_frame.columns:
        plot_frame = plot_frame[plot_frame["valid"]]
    plot_frame = plot_frame[["time_sec", axis_name]].dropna()
    if plot_frame.empty:
        return pd.DataFrame()
    if invert_pose_vertical and source_name == "pose" and axis_name in {"y", "y_px"}:
        plot_frame[axis_name] = -plot_frame[axis_name]
    if normalize:
        centered = plot_frame[axis_name] - plot_frame[axis_name].mean()
        std = plot_frame[axis_name].std(ddof=0)
        plot_frame[axis_name] = centered if pd.isna(std) or std == 0 else centered / std
    return plot_frame.rename(columns={axis_name: axis_name}).set_index("time_sec")


def downsample_series(series: pd.Series, max_points: int = 1200) -> pd.Series:
    if len(series) <= max_points:
        return series
    step = max(1, len(series) // max_points)
    return series.iloc[::step]


def build_svg_chart(combined: pd.DataFrame, title: str, y_axis_label: str, reference_time_sec: float | None) -> str:
    plot_width = 980
    plot_height = 320
    margin_left = 72
    margin_right = 24
    margin_top = 24
    margin_bottom = 52
    inner_width = plot_width - margin_left - margin_right
    inner_height = plot_height - margin_top - margin_bottom
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#17becf"]

    chart_frame = combined.sort_index()
    x_values = pd.to_numeric(chart_frame.index.to_series(), errors="coerce").dropna()
    y_values = pd.to_numeric(chart_frame.stack(), errors="coerce").dropna()
    if x_values.empty or y_values.empty:
        return "<p>No plottable values.</p>"

    x_min = float(x_values.min())
    x_max = float(x_values.max())
    y_min = float(y_values.min())
    y_max = float(y_values.max())
    if x_min == x_max:
        x_min -= 1.0
        x_max += 1.0
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0

    def scale_x(value: float) -> float:
        return margin_left + ((value - x_min) / (x_max - x_min)) * inner_width

    def scale_y(value: float) -> float:
        return margin_top + inner_height - ((value - y_min) / (y_max - y_min)) * inner_height

    y_ticks = []
    for tick_index in range(5):
        ratio = tick_index / 4
        value = y_min + ratio * (y_max - y_min)
        y = scale_y(value)
        y_ticks.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{plot_width - margin_right}" y2="{y:.2f}" stroke="#e5e7eb" stroke-width="1" />'
            f'<text x="{margin_left - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="12" fill="#475569">{value:.3f}</text>'
        )

    x_ticks = []
    for tick_index in range(5):
        ratio = tick_index / 4
        value = x_min + ratio * (x_max - x_min)
        x = scale_x(value)
        x_ticks.append(
            f'<line x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" y2="{margin_top + inner_height}" stroke="#f1f5f9" stroke-width="1" />'
            f'<text x="{x:.2f}" y="{plot_height - 16}" text-anchor="middle" font-size="12" fill="#475569">{value:.2f}</text>'
        )

    reference_line = ""
    if reference_time_sec is not None and x_min <= reference_time_sec <= x_max:
        reference_x = scale_x(reference_time_sec)
        reference_line = (
            f'<line x1="{reference_x:.2f}" y1="{margin_top}" x2="{reference_x:.2f}" y2="{margin_top + inner_height}" '
            f'stroke="#dc2626" stroke-width="2" stroke-dasharray="6 4" />'
        )

    series_paths = []
    legend_items = []
    for index, column in enumerate(chart_frame.columns):
        color = palette[index % len(palette)]
        series = pd.to_numeric(chart_frame[column], errors="coerce").dropna()
        if series.empty:
            continue
        series = downsample_series(series)
        points = " ".join(f"{scale_x(float(time_sec)):.2f},{scale_y(float(value)):.2f}" for time_sec, value in series.items())
        if len(points) < 2:
            continue
        series_paths.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{points}" />')
        legend_y = margin_top + 8 + index * 18
        legend_items.append(
            f'<line x1="{plot_width - 220}" y1="{legend_y:.2f}" x2="{plot_width - 198}" y2="{legend_y:.2f}" stroke="{color}" stroke-width="3" />'
            f'<text x="{plot_width - 192}" y="{legend_y + 4:.2f}" font-size="12" fill="#0f172a">{escape(str(column))}</text>'
        )

    return f"""
    <svg viewBox="0 0 {plot_width} {plot_height}" width="100%" height="auto" xmlns="http://www.w3.org/2000/svg">
      <rect x="0" y="0" width="{plot_width}" height="{plot_height}" fill="#ffffff" rx="12" />
      <text x="{plot_width / 2}" y="18" text-anchor="middle" font-size="16" fill="#0f172a">{escape(title)}</text>
      {''.join(y_ticks)}
      {''.join(x_ticks)}
      <line x1="{margin_left}" y1="{margin_top + inner_height}" x2="{plot_width - margin_right}" y2="{margin_top + inner_height}" stroke="#94a3b8" stroke-width="1.5" />
      <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + inner_height}" stroke="#94a3b8" stroke-width="1.5" />
      {reference_line}
      {''.join(series_paths)}
      {''.join(legend_items)}
      <text x="{plot_width / 2}" y="{plot_height - 4}" text-anchor="middle" font-size="12" fill="#475569">Trimmed time (s)</text>
      <text x="16" y="{plot_height / 2}" text-anchor="middle" font-size="12" fill="#475569" transform="rotate(-90 16 {plot_height / 2})">{escape(y_axis_label)}</text>
    </svg>
    """


config = load_alignment_config()
nominal_peak_offset_sec = float(config["trim_window"]["aligned_peak_offset_sec"])
window_before_sec = float(config["trim_window"]["seconds_before_peak"])
window_after_sec = float(config["trim_window"]["seconds_after_peak"])

st.set_page_config(page_title="Aligned Trimmed Dashboard", layout="wide")
st.title("Aligned Trimmed Dashboard")
st.caption(
    f"Trimmed clips are loaded from {display_path(ALIGNED_RESULT_DIR)}. Each clip starts at t = 0, and the saved alignment peak should appear at about t = {nominal_peak_offset_sec:.1f}s."
)

available_targets = list_target_options()
if not available_targets:
    st.error(f"No trimmed aligned CSVs found under {ALIGNED_RESULT_DIR}. Run `python3 data_alignment/trim_to_alignment_window.py` first.")
    st.stop()

with st.sidebar:
    st.header("Options")
    default_target = "knee_default_left" if "knee_default_left" in available_targets else available_targets[0]
    selected_target = st.selectbox("Marker / landmark", available_targets, index=available_targets.index(default_target), format_func=format_target_label)
    valid_only = st.checkbox("Only plot valid rows", value=True)
    value_mode = st.radio("Value mode", ["Raw values", "Normalized values"], index=0)
    invert_pose_vertical = st.checkbox("Invert pose vertical axes (y, y_px)", value=True)
    axes = st.multiselect("Axes", PLOTTABLE_AXES, default=["y"])

normalize = value_mode == "Normalized values"
target_file = f"{selected_target}.csv"
mocap_recordings = list_recordings(MOCAP_DIR, target_file)
pose_recordings = list_recordings(POSE_DIR, target_file)
recording_specs: list[tuple[str, str, Path]] = [("mocap", name, mocap_recordings[name]) for name in mocap_recordings]
recording_specs.extend(("pose", name, pose_recordings[name]) for name in pose_recordings)

if not recording_specs:
    st.error(f"No trimmed recordings found for {format_target_label(selected_target)}")
    st.stop()

loaded_frames: dict[tuple[str, str], pd.DataFrame] = {}
summary_rows: list[dict[str, object]] = []
for source_name, recording_name, csv_path in recording_specs:
    frame = load_recording(str(csv_path))
    loaded_frames[(source_name, recording_name)] = frame
    summary_rows.append(build_summary(recording_name, source_name, csv_path, frame))

st.subheader("Summary")
st.dataframe(prepare_table_frame(pd.DataFrame(summary_rows)), use_container_width=True)

if not axes:
    st.info("Select at least one axis to plot.")
    st.stop()

st.subheader(f"{format_target_label(selected_target)} Plots")
st.caption(f"The red dashed line marks the saved alignment peak in each trimmed clip. The trimmed window spans 0s to {window_before_sec + window_after_sec:.1f}s.")
if invert_pose_vertical:
    st.caption("Pose vertical axes are inverted before plotting so image-space down maps to physical up.")

for source_name, recording_name, csv_path in recording_specs:
    combined: pd.DataFrame | None = None
    for axis_name in axes:
        plot_frame = prepare_plot_frame(
            frame=loaded_frames[(source_name, recording_name)],
            source_name=source_name,
            axis_name=axis_name,
            valid_only=valid_only,
            normalize=normalize,
            invert_pose_vertical=invert_pose_vertical,
        )
        if plot_frame.empty:
            continue
        combined = plot_frame if combined is None else combined.join(plot_frame, how="outer")

    model_name = summary_rows[[row["recording"] for row in summary_rows].index(recording_name)]["model_name"]
    st.markdown(f"**{recording_name}**  `source={source_name}`  `model={model_name}`  `file={display_path(csv_path)}`")
    if combined is None or combined.empty:
        st.warning(f"No plottable values found for `{recording_name}` with the current filters.")
        continue
    reference_time_sec = None
    peak_series = loaded_frames[(source_name, recording_name)]["alignment_peak_offset_sec"] if "alignment_peak_offset_sec" in loaded_frames[(source_name, recording_name)] else None
    if peak_series is not None and not peak_series.dropna().empty:
        reference_time_sec = float(peak_series.dropna().iloc[0])
    st.markdown(
        build_svg_chart(
            combined=combined,
            title=f"{recording_name} {format_target_label(selected_target).lower()}",
            y_axis_label="normalized value" if normalize else "value",
            reference_time_sec=reference_time_sec,
        ),
        unsafe_allow_html=True,
    )

st.subheader("Raw Data")
preview_source = st.selectbox("Preview recording", [f"{source_name}:{recording_name}" for source_name, recording_name, _ in recording_specs])
preview_source_name, preview_recording_name = preview_source.split(":", maxsplit=1)
st.dataframe(
    prepare_table_frame(loaded_frames[(preview_source_name, preview_recording_name)]),
    use_container_width=True,
    height=420,
)
