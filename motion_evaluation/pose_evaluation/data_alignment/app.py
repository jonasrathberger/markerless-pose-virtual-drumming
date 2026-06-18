from __future__ import annotations

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
PREPARED_RESULT_DIR = REPO_ROOT / "thesis_data" / "pose_evaluation" / "prepared_result"
MOCAP_DIR = PREPARED_RESULT_DIR / "mocap"
POSE_DIR = PREPARED_RESULT_DIR / "pose"
NUMERIC_COLUMNS = ["frame_index", "time_sec", "x", "y", "z", "x_px", "y_px", "confidence", "visibility"]
BOOLEAN_COLUMNS = ["tracking_present", "valid"]
PLOTTABLE_AXES = ["x", "y", "z", "x_px", "y_px", "confidence", "visibility"]


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
    target_text = target.replace("_", " ")
    variant_text = target_variant.replace("_", " ")
    return f"{side.title()} {target_text} ({variant_text})"


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
    time_min = frame["time_sec"].min() if "time_sec" in frame.columns else None
    time_max = frame["time_sec"].max() if "time_sec" in frame.columns else None
    valid_ratio = frame["valid"].mean() if "valid" in frame.columns and not frame.empty else None
    model_name = frame["model_name"].dropna().iloc[0] if "model_name" in frame.columns and not frame.empty else ""
    coord_space = frame["coord_space"].dropna().iloc[0] if "coord_space" in frame.columns and not frame.empty else ""
    unit = frame["unit"].dropna().iloc[0] if "unit" in frame.columns and not frame.empty else ""
    return {
        "recording": recording_name,
        "source": source_name,
        "model_name": model_name,
        "rows": len(frame),
        "time_start_sec": time_min,
        "time_end_sec": time_max,
        "valid_ratio": valid_ratio,
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


def prepare_plot_frame(
    frame: pd.DataFrame,
    source_name: str,
    axis_name: str,
    valid_only: bool,
    normalize: bool,
    invert_pose_vertical: bool,
) -> pd.DataFrame:
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


def align_time_axis(
    combined: pd.DataFrame,
    alignment_axis: str,
    window_sec: float | int | None,
) -> tuple[pd.DataFrame, float | None, str | None]:
    if combined.empty:
        return combined, None, None

    reference_axis = alignment_axis if alignment_axis in combined.columns else next(iter(combined.columns), None)
    if reference_axis is None:
        return combined, None, None

    reference_series = pd.to_numeric(combined[reference_axis], errors="coerce").dropna()
    if reference_series.empty:
        return combined, None, reference_axis

    baseline = float(reference_series.median())
    reference_time = float((reference_series - baseline).abs().idxmax())

    aligned = combined.copy()
    aligned.index = aligned.index.astype(float) - reference_time

    if window_sec is not None:
        aligned = aligned[(aligned.index >= -float(window_sec)) & (aligned.index <= float(window_sec))]

    return aligned, reference_time, reference_axis


def downsample_series(series: pd.Series, max_points: int = 1200) -> pd.Series:
    if len(series) <= max_points:
        return series
    step = max(1, len(series) // max_points)
    return series.iloc[::step]


def build_svg_chart(combined: pd.DataFrame, title: str, y_axis_label: str) -> str:
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
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{plot_width - margin_right}" y2="{y:.2f}" '
            f'stroke="#e5e7eb" stroke-width="1" />'
            f'<text x="{margin_left - 10}" y="{y + 4:.2f}" text-anchor="end" '
            f'font-size="12" fill="#475569">{value:.3f}</text>'
        )

    x_ticks = []
    for tick_index in range(5):
        ratio = tick_index / 4
        value = x_min + ratio * (x_max - x_min)
        x = scale_x(value)
        x_ticks.append(
            f'<line x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" y2="{margin_top + inner_height}" '
            f'stroke="#f1f5f9" stroke-width="1" />'
            f'<text x="{x:.2f}" y="{plot_height - 16}" text-anchor="middle" '
            f'font-size="12" fill="#475569">{value:.2f}</text>'
        )

    series_paths: list[str] = []
    legend_items: list[str] = []
    for index, column in enumerate(chart_frame.columns):
        color = palette[index % len(palette)]
        series = pd.to_numeric(chart_frame[column], errors="coerce").dropna()
        if series.empty:
            continue
        series = downsample_series(series)
        points = [f"{scale_x(float(time_sec)):.2f},{scale_y(float(value)):.2f}" for time_sec, value in series.items()]
        if len(points) < 2:
            continue
        series_paths.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2" '
            f'points="{" ".join(points)}" />'
        )
        legend_y = margin_top + 8 + index * 18
        legend_items.append(
            f'<line x1="{plot_width - 220}" y1="{legend_y:.2f}" x2="{plot_width - 198}" y2="{legend_y:.2f}" '
            f'stroke="{color}" stroke-width="3" />'
            f'<text x="{plot_width - 192}" y="{legend_y + 4:.2f}" font-size="12" fill="#0f172a">{escape(str(column))}</text>'
        )

    svg = f"""
    <svg viewBox="0 0 {plot_width} {plot_height}" width="100%" height="auto" xmlns="http://www.w3.org/2000/svg">
      <rect x="0" y="0" width="{plot_width}" height="{plot_height}" fill="#ffffff" rx="12" />
      <text x="{plot_width / 2}" y="18" text-anchor="middle" font-size="16" fill="#0f172a">{escape(title)}</text>
      {''.join(y_ticks)}
      {''.join(x_ticks)}
      <line x1="{margin_left}" y1="{margin_top + inner_height}" x2="{plot_width - margin_right}" y2="{margin_top + inner_height}" stroke="#94a3b8" stroke-width="1.5" />
      <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + inner_height}" stroke="#94a3b8" stroke-width="1.5" />
      {''.join(series_paths)}
      {''.join(legend_items)}
      <text x="{plot_width / 2}" y="{plot_height - 4}" text-anchor="middle" font-size="12" fill="#475569">Time (s)</text>
      <text x="16" y="{plot_height / 2}" text-anchor="middle" font-size="12" fill="#475569" transform="rotate(-90 16 {plot_height / 2})">{escape(y_axis_label)}</text>
    </svg>
    """
    return svg


st.set_page_config(page_title="Landmark Comparison Dashboard", layout="wide")
st.title("Landmark Comparison Dashboard")
st.caption("Temporary Streamlit view for stacked mocap and pose marker/landmark outputs.")

available_targets = list_target_options()
if not available_targets:
    st.error(f"No prepared result CSVs found under {MOCAP_DIR} or {POSE_DIR}")
    st.stop()

with st.sidebar:
    st.header("Options")
    selected_target = st.selectbox(
        "Marker / landmark",
        available_targets,
        index=available_targets.index("knee_default_right") if "knee_default_right" in available_targets else 0,
        format_func=format_target_label,
    )
    valid_only = st.checkbox("Only plot valid rows", value=True)
    value_mode = st.radio("Value mode", ["Raw values", "Normalized values"], index=0)
    invert_pose_vertical = st.checkbox("Invert pose vertical axes (y, y_px)", value=True)
    align_time = st.checkbox("Align recordings around main event", value=True)
    alignment_axis = st.selectbox("Alignment axis", PLOTTABLE_AXES, index=PLOTTABLE_AXES.index("y"))
    window_sec = st.slider("Alignment window (seconds)", min_value=2, max_value=120, value=8, step=1)
    axes = st.multiselect("Axes", PLOTTABLE_AXES, default=["y"])

normalize = value_mode == "Normalized values"

target_file = f"{selected_target}.csv"
mocap_recordings = list_recordings(MOCAP_DIR, target_file)
pose_recordings = list_recordings(POSE_DIR, target_file)

recording_specs: list[tuple[str, str, Path]] = [
    ("mocap", recording_name, mocap_recordings[recording_name]) for recording_name in mocap_recordings
]
recording_specs.extend(
    ("pose", recording_name, pose_recordings[recording_name]) for recording_name in pose_recordings
)

if not recording_specs:
    st.error(f"No recordings found for {format_target_label(selected_target)}")
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

selected_target_label = format_target_label(selected_target)
st.subheader(f"{selected_target_label} Plots")
if invert_pose_vertical:
    st.caption("Pose vertical axes are inverted before plotting so image-space down maps to physical up.")
if align_time:
    st.caption(f"Each recording is centered on its largest deviation in `{alignment_axis}` and cropped to +/- {window_sec} s.")

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

    reference_time: float | None = None
    reference_axis: str | None = None
    if align_time:
        combined, reference_time, reference_axis = align_time_axis(combined, alignment_axis, window_sec)
        if combined.empty:
            st.warning(f"No values remain for `{recording_name}` after alignment windowing.")
            continue

    detail_parts = [f"mode={'normalized' if normalize else 'raw'}"]
    if reference_time is not None and reference_axis is not None:
        detail_parts.append(f"aligned_on={reference_axis}@{reference_time:.2f}s")
    elif align_time:
        detail_parts.append("aligned_on=unavailable")
    st.caption("  ".join(detail_parts))

    st.markdown(
        build_svg_chart(
            combined,
            title=f"{recording_name} {selected_target_label.lower()}",
            y_axis_label="normalized value" if normalize else "value",
        ),
        unsafe_allow_html=True,
    )

st.subheader("Raw Data")
preview_source = st.selectbox(
    "Preview recording",
    [f"{source_name}:{recording_name}" for source_name, recording_name, _ in recording_specs],
)
preview_source_name, preview_recording_name = preview_source.split(":", maxsplit=1)
st.dataframe(
    prepare_table_frame(loaded_frames[(preview_source_name, preview_recording_name)]),
    use_container_width=True,
    height=420,
)
