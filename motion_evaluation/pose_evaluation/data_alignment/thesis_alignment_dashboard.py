from __future__ import annotations

import io
import json
import os
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
THESIS_POSE_DIR = REPO_ROOT / "thesis_data" / "pose_evaluation"
ORIGINAL_RESULT_DIR = THESIS_POSE_DIR / "prepared_result"
TRIMMED_RESULT_DIR = THESIS_POSE_DIR / "aligned_result"
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
RECORDING_ID_ALIASES = {
    "air_knees": "air_knees",
    "drums": "drums",
}
RECORDING_ORDER = ["air_knees", "drums", "apple_vision", "mediapipe"]
RECORDING_LABELS = {
    "air_knees": "OptiTrack Motion Capture: Air-Knees",
    "drums": "OptiTrack Motion Capture: Physical Drumming",
    "apple_vision": "Apple Vision",
    "mediapipe": "MediaPipe",
}
PALETTE = {
    "air_knees": "#1d4ed8",
    "drums": "#dc2626",
    "apple_vision": "#059669",
    "mediapipe": "#d97706",
}
PNG_EXPORT_ERROR: str | None = None


def load_alignment_config() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text())


def canonical_recording_id(recording_id: str) -> str:
    return RECORDING_ID_ALIASES.get(recording_id, recording_id)


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
    base_label = f"{side.title()} {target.replace('_', ' ')}"
    if target_variant == "default":
        return base_label
    return f"{base_label} ({target_variant.replace('_', ' ')})"


def list_target_options(base_dir: Path) -> list[str]:
    stems: set[str] = set()
    for source_name in ("mocap", "pose"):
        source_dir = base_dir / source_name
        if not source_dir.exists():
            continue
        for csv_path in source_dir.glob("*/*.csv"):
            stems.add(csv_path.stem)
    return sorted(stems, key=lambda stem: (parse_target_stem(stem)[2], parse_target_stem(stem)[0], parse_target_stem(stem)[1]))


def list_recording_paths(base_dir: Path, target_file: str) -> list[tuple[str, str, Path]]:
    specs: list[tuple[str, str, Path]] = []
    for source_name in ("mocap", "pose"):
        source_dir = base_dir / source_name
        if not source_dir.exists():
            continue
        for csv_path in sorted(source_dir.glob(f"*/{target_file}")):
            specs.append((source_name, canonical_recording_id(csv_path.parent.name), csv_path))
    return sorted(specs, key=lambda item: RECORDING_ORDER.index(item[1]) if item[1] in RECORDING_ORDER else len(RECORDING_ORDER))


def list_comparison_recording_paths(
    base_dir: Path,
    mocap_target_file: str,
    pose_target_file: str,
) -> list[tuple[str, str, Path]]:
    specs: list[tuple[str, str, Path]] = []
    target_files = {
        "mocap": mocap_target_file,
        "pose": pose_target_file,
    }
    for source_name in ("mocap", "pose"):
        source_dir = base_dir / source_name
        if not source_dir.exists():
            continue
        for csv_path in sorted(source_dir.glob(f"*/{target_files[source_name]}")):
            specs.append((source_name, canonical_recording_id(csv_path.parent.name), csv_path))
    return sorted(specs, key=lambda item: RECORDING_ORDER.index(item[1]) if item[1] in RECORDING_ORDER else len(RECORDING_ORDER))


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


def prepare_signal(
    frame: pd.DataFrame,
    source_name: str,
    axis_name: str,
    valid_only: bool,
    normalize: bool,
    invert_pose_vertical: bool,
) -> pd.DataFrame:
    signal = frame.copy()
    if valid_only and "valid" in signal.columns:
        signal = signal[signal["valid"]]
    signal = signal[["time_sec", axis_name]].dropna()
    if signal.empty:
        return pd.DataFrame(columns=["time_sec", axis_name])

    if invert_pose_vertical and source_name == "pose" and axis_name in {"y", "y_px"}:
        signal[axis_name] = -signal[axis_name]

    if normalize:
        centered = signal[axis_name] - signal[axis_name].mean()
        std = signal[axis_name].std(ddof=0)
        signal[axis_name] = centered if pd.isna(std) or std == 0 else centered / std

    return signal


def build_aligned_original_series(
    recording_specs: list[tuple[str, str, Path]],
    peak_times: dict[str, float],
    axis_name: str,
    valid_only: bool,
    normalize: bool,
    invert_pose_vertical: bool,
) -> dict[str, pd.DataFrame]:
    series_map: dict[str, pd.DataFrame] = {}
    for source_name, recording_name, csv_path in recording_specs:
        frame = load_recording(str(csv_path))
        signal = prepare_signal(frame, source_name, axis_name, valid_only, normalize, invert_pose_vertical)
        if signal.empty or recording_name not in peak_times:
            series_map[recording_name] = pd.DataFrame(columns=["plot_time_sec", "plot_y"])
            continue
        signal["plot_time_sec"] = signal["time_sec"] - peak_times[recording_name]
        signal["plot_y"] = signal[axis_name]
        series_map[recording_name] = signal[["plot_time_sec", "plot_y"]]
    return series_map


def build_trimmed_series(
    recording_specs: list[tuple[str, str, Path]],
    axis_name: str,
    valid_only: bool,
    normalize: bool,
    invert_pose_vertical: bool,
) -> tuple[dict[str, pd.DataFrame], dict[str, float]]:
    series_map: dict[str, pd.DataFrame] = {}
    peak_offsets: dict[str, float] = {}
    for source_name, recording_name, csv_path in recording_specs:
        frame = load_recording(str(csv_path))
        signal = prepare_signal(frame, source_name, axis_name, valid_only, normalize, invert_pose_vertical)
        if signal.empty:
            series_map[recording_name] = pd.DataFrame(columns=["plot_time_sec", "plot_y"])
            continue
        signal["plot_time_sec"] = signal["time_sec"]
        signal["plot_y"] = signal[axis_name]
        series_map[recording_name] = signal[["plot_time_sec", "plot_y"]]
        if "alignment_peak_offset_sec" in frame.columns and not frame["alignment_peak_offset_sec"].dropna().empty:
            peak_offsets[recording_name] = float(frame["alignment_peak_offset_sec"].dropna().iloc[0])
    return series_map, peak_offsets


def downsample_series(frame: pd.DataFrame, max_points: int = 2400) -> pd.DataFrame:
    if len(frame) <= max_points:
        return frame
    step = max(1, len(frame) // max_points)
    return frame.iloc[::step]


def apply_vertical_offsets(series_map: dict[str, pd.DataFrame], offset_step: float) -> dict[str, pd.DataFrame]:
    available_recordings = [name for name in RECORDING_ORDER if name in series_map and not series_map[name].empty]
    if not available_recordings:
        return series_map

    centered_index = (len(available_recordings) - 1) / 2.0
    shifted_map: dict[str, pd.DataFrame] = {}
    for index, recording_name in enumerate(available_recordings):
        offset = (centered_index - index) * offset_step
        frame = series_map[recording_name].copy()
        frame["plot_y"] = frame["plot_y"] + offset
        shifted_map[recording_name] = frame

    for recording_name, frame in series_map.items():
        if recording_name not in shifted_map:
            shifted_map[recording_name] = frame
    return shifted_map


def build_overlay_svg(
    series_map: dict[str, pd.DataFrame],
    title: str,
    x_axis_label: str,
    y_axis_label: str,
    x_window: tuple[float, float] | None,
    reference_lines: list[tuple[float, str]],
    show_y_tick_labels: bool,
) -> str:
    plot_width = 1280
    plot_height = 560
    margin_left = 86
    margin_right = 32
    margin_top = 104
    margin_bottom = 64
    inner_width = plot_width - margin_left - margin_right
    inner_height = plot_height - margin_top - margin_bottom

    combined_rows: list[pd.DataFrame] = []
    for recording_name in RECORDING_ORDER:
        frame = series_map.get(recording_name)
        if frame is None or frame.empty:
            continue
        local = frame.copy()
        local["series"] = recording_name
        combined_rows.append(local)

    if not combined_rows:
        return "<p>No plottable values.</p>"

    combined = pd.concat(combined_rows, ignore_index=True)
    if x_window is not None:
        x_start, x_end = x_window
        combined = combined[(combined["plot_time_sec"] >= x_start) & (combined["plot_time_sec"] <= x_end)]
    if combined.empty:
        return "<p>No plottable values in the selected window.</p>"

    x_values = pd.to_numeric(combined["plot_time_sec"], errors="coerce").dropna()
    y_values = pd.to_numeric(combined["plot_y"], errors="coerce").dropna()
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
    for tick_index in range(6):
        ratio = tick_index / 5
        value = y_min + ratio * (y_max - y_min)
        y = scale_y(value)
        label_svg = ""
        if show_y_tick_labels:
            label_svg = f'<text x="{margin_left - 12}" y="{y + 4:.2f}" text-anchor="end" font-size="14" fill="#334155">{value:.3f}</text>'
        y_ticks.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{plot_width - margin_right}" y2="{y:.2f}" stroke="#e5e7eb" stroke-width="1" />'
            f"{label_svg}"
        )

    x_ticks = []
    for tick_index in range(6):
        ratio = tick_index / 5
        value = x_min + ratio * (x_max - x_min)
        x = scale_x(value)
        x_ticks.append(
            f'<line x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" y2="{margin_top + inner_height}" stroke="#f1f5f9" stroke-width="1" />'
            f'<text x="{x:.2f}" y="{plot_height - 18}" text-anchor="middle" font-size="14" fill="#334155">{value:.1f}</text>'
        )

    reference_svg = []
    for value, label in reference_lines:
        if x_min <= value <= x_max:
            x = scale_x(value)
            reference_svg.append(
                f'<line x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" y2="{margin_top + inner_height}" stroke="#0f172a" stroke-width="1.5" stroke-dasharray="6 4" />'
                f'<text x="{x + 6:.2f}" y="{margin_top + 18:.2f}" font-size="13" fill="#0f172a">{escape(label)}</text>'
            )

    series_paths = []
    legend_items = []
    legend_x = margin_left
    legend_y = 64
    legend_step = 275
    for index, recording_name in enumerate(RECORDING_ORDER):
        frame = series_map.get(recording_name)
        if frame is None or frame.empty:
            continue
        local = frame.copy()
        if x_window is not None:
            x_start, x_end = x_window
            local = local[(local["plot_time_sec"] >= x_start) & (local["plot_time_sec"] <= x_end)]
        if local.empty:
            continue
        local = downsample_series(local)
        points = " ".join(
            f"{scale_x(float(time_sec)):.2f},{scale_y(float(value)):.2f}"
            for time_sec, value in zip(local["plot_time_sec"], local["plot_y"])
        )
        color = PALETTE.get(recording_name, "#475569")
        label = RECORDING_LABELS.get(recording_name, recording_name)
        series_paths.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{points}" />')
        item_x = legend_x + index * legend_step
        legend_items.append(
            f'<line x1="{item_x:.2f}" y1="{legend_y:.2f}" x2="{item_x + 30:.2f}" y2="{legend_y:.2f}" stroke="{color}" stroke-width="4" />'
            f'<text x="{item_x + 40:.2f}" y="{legend_y + 5:.2f}" font-size="14" fill="#0f172a">{escape(label)}</text>'
        )

    return f"""
    <svg viewBox="0 0 {plot_width} {plot_height}" width="100%" height="auto" xmlns="http://www.w3.org/2000/svg">
      <rect x="0" y="0" width="{plot_width}" height="{plot_height}" fill="#ffffff" rx="14" />
      <text x="{plot_width / 2}" y="30" text-anchor="middle" font-size="24" fill="#0f172a">{escape(title)}</text>
      {''.join(y_ticks)}
      {''.join(x_ticks)}
      <line x1="{margin_left}" y1="{margin_top + inner_height}" x2="{plot_width - margin_right}" y2="{margin_top + inner_height}" stroke="#94a3b8" stroke-width="1.5" />
      <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + inner_height}" stroke="#94a3b8" stroke-width="1.5" />
      {''.join(reference_svg)}
      {''.join(series_paths)}
      {''.join(legend_items)}
      <text x="{plot_width / 2}" y="{plot_height - 6}" text-anchor="middle" font-size="15" fill="#334155">{escape(x_axis_label)}</text>
      <text x="22" y="{plot_height / 2}" text-anchor="middle" font-size="15" fill="#334155" transform="rotate(-90 22 {plot_height / 2})">{escape(y_axis_label)}</text>
    </svg>
    """


def build_overlay_png(
    series_map: dict[str, pd.DataFrame],
    title: str,
    x_axis_label: str,
    y_axis_label: str,
    x_window: tuple[float, float] | None,
    reference_lines: list[tuple[float, str]],
    show_y_tick_labels: bool,
) -> bytes:
    global PNG_EXPORT_ERROR
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    try:
        import matplotlib
    except ModuleNotFoundError:
        PNG_EXPORT_ERROR = "PNG export requires matplotlib."
        return b""

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(12.8, 5.0), dpi=220)
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")
    figure.subplots_adjust(left=0.08, right=0.985, bottom=0.14, top=0.79)

    plotted_any = False
    for recording_name in RECORDING_ORDER:
        frame = series_map.get(recording_name)
        if frame is None or frame.empty:
            continue
        local = frame.copy()
        if x_window is not None:
            x_start, x_end = x_window
            local = local[(local["plot_time_sec"] >= x_start) & (local["plot_time_sec"] <= x_end)]
        if local.empty:
            continue
        local = downsample_series(local)
        axis.plot(
            local["plot_time_sec"],
            local["plot_y"],
            color=PALETTE.get(recording_name, "#475569"),
            linewidth=2.4,
            label=RECORDING_LABELS.get(recording_name, recording_name),
        )
        plotted_any = True

    if not plotted_any:
        plt.close(figure)
        return b""

    if x_window is not None:
        axis.set_xlim(*x_window)

    y_min, y_max = axis.get_ylim()
    for value, label in reference_lines:
        axis.axvline(value, color="#0f172a", linewidth=1.2, linestyle=(0, (6, 4)))
        if y_max > y_min:
            axis.text(
                value + 0.4,
                y_max - 0.06 * (y_max - y_min),
                label,
                fontsize=10,
                color="#0f172a",
                ha="left",
                va="top",
            )

    figure.suptitle(title, fontsize=18, y=0.975)
    axis.set_xlabel(x_axis_label, fontsize=12)
    axis.set_ylabel(y_axis_label, fontsize=12)
    axis.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    axis.grid(axis="x", color="#f1f5f9", linewidth=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#94a3b8")
    axis.spines["bottom"].set_color("#94a3b8")
    axis.tick_params(labelsize=11, colors="#334155")
    if not show_y_tick_labels:
        axis.set_yticklabels([])
        axis.tick_params(axis="y", length=0)

    handles, labels = axis.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.895),
        ncol=4,
        frameon=False,
        fontsize=11,
        columnspacing=1.6,
        handlelength=2.6,
    )

    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", facecolor="white", bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)
    return buffer.getvalue()


config = load_alignment_config()
peak_times = {
    canonical_recording_id(str(item["recording_id"])): float(item["peak_time_sec"])
    for item in config["recordings"]
}
nominal_peak_offset_sec = float(config["trim_window"]["aligned_peak_offset_sec"])
trim_length_sec = float(config["trim_window"]["seconds_before_peak"]) + float(config["trim_window"]["seconds_after_peak"])

st.set_page_config(page_title="Thesis Alignment Dashboard", layout="wide")
st.title("Thesis Alignment Dashboard")
st.caption("Two presentation-style overlays: full recordings aligned by the saved left-knee anchors, and trimmed recordings after applying the saved window.")

available_targets = list_target_options(ORIGINAL_RESULT_DIR)
if not available_targets:
    st.error(f"No prepared result CSVs found under {ORIGINAL_RESULT_DIR}.")
    st.stop()

with st.sidebar:
    st.header("Plot")
    default_target = "knee_default_left" if "knee_default_left" in available_targets else available_targets[0]
    selected_target = st.selectbox("Marker / landmark", available_targets, index=available_targets.index(default_target), format_func=format_target_label)
    selected_target_name, selected_target_variant, selected_side = parse_target_stem(selected_target)
    pose_wrist_variant = None
    if selected_target_name == "wrist" and selected_target_variant == "center":
        pose_wrist_variant = st.selectbox("Pose wrist source", ["body", "hand"], index=0)
    axis_name = st.selectbox("Axis", PLOTTABLE_AXES, index=PLOTTABLE_AXES.index("y"))
    value_mode = st.radio("Value mode", ["Raw values", "Normalized values"], index=1)
    valid_only = st.checkbox("Only plot valid rows", value=True)
    invert_pose_vertical = st.checkbox("Invert pose vertical axes (y, y_px)", value=True)
    apply_offsets = st.checkbox("Apply vertical offsets for readability", value=True)
    offset_step = st.slider("Vertical offset step", min_value=1.0, max_value=8.0, value=3.5, step=0.5, disabled=not apply_offsets)
    st.caption("Normalized values are the better default for cross-system thesis figures because mocap and pose use different coordinate scales.")
    st.header("Windows")
    full_aligned_window = st.slider("Aligned full view (s)", min_value=20, max_value=180, value=140, step=5)
    trimmed_x_max = st.slider("Trimmed view max x (s)", min_value=30, max_value=140, value=118, step=2)

normalize = value_mode == "Normalized values"
target_file = f"{selected_target}.csv"
comparison_label_suffix = ""
if pose_wrist_variant is not None:
    pose_target_stem = f"wrist_{pose_wrist_variant}_{selected_side}"
    pose_target_file = f"{pose_target_stem}.csv"
    original_specs = list_comparison_recording_paths(ORIGINAL_RESULT_DIR, target_file, pose_target_file)
    trimmed_specs = list_comparison_recording_paths(TRIMMED_RESULT_DIR, target_file, pose_target_file)
    comparison_label_suffix = f" | pose {pose_wrist_variant} wrist"
else:
    original_specs = list_recording_paths(ORIGINAL_RESULT_DIR, target_file)
    trimmed_specs = list_recording_paths(TRIMMED_RESULT_DIR, target_file)

if not original_specs:
    st.error(f"No original recordings found for {format_target_label(selected_target)}")
    st.stop()
if not trimmed_specs:
    st.error(f"No trimmed recordings found for {format_target_label(selected_target)}. Run `python3 data_alignment/trim_to_alignment_window.py` first.")
    st.stop()

original_series = build_aligned_original_series(
    recording_specs=original_specs,
    peak_times=peak_times,
    axis_name=axis_name,
    valid_only=valid_only,
    normalize=normalize,
    invert_pose_vertical=invert_pose_vertical,
)
trimmed_series, _trimmed_peak_offsets = build_trimmed_series(
    recording_specs=trimmed_specs,
    axis_name=axis_name,
    valid_only=valid_only,
    normalize=normalize,
    invert_pose_vertical=invert_pose_vertical,
)

if apply_offsets:
    original_series = apply_vertical_offsets(original_series, float(offset_step))
    trimmed_series = apply_vertical_offsets(trimmed_series, float(offset_step))

selected_target_label = format_target_label(selected_target)
display_target_label = selected_target_label + comparison_label_suffix
y_axis_label = "Offset visualization" if apply_offsets else f"{axis_name} ({'normalized' if normalize else 'raw'})"

st.subheader("Full Recordings Aligned By Saved Left-Knee Anchor")
st.caption("These plots use the original prepared recordings. Time is shifted so the saved anchor is at t = 0, but no trimming is applied.")
full_svg = build_overlay_svg(
    series_map=original_series,
    title=f"Aligned Full Recordings: {display_target_label}",
    x_axis_label="Aligned time relative to saved anchor (s)",
    y_axis_label=y_axis_label,
    x_window=(-16.0, float(full_aligned_window) - 16.0),
    reference_lines=[(0.0, "saved anchor")],
    show_y_tick_labels=not apply_offsets,
)
st.markdown(full_svg, unsafe_allow_html=True)
full_png = build_overlay_png(
    series_map=original_series,
    title=f"Aligned Full Recordings: {display_target_label}",
    x_axis_label="Aligned time relative to saved anchor (s)",
    y_axis_label=y_axis_label,
    x_window=(-16.0, float(full_aligned_window) - 16.0),
    reference_lines=[(0.0, "saved anchor")],
    show_y_tick_labels=not apply_offsets,
)
st.download_button(
    "Download Full Plot SVG",
    data=full_svg,
    file_name=f"aligned_full_{selected_target}_{axis_name}.svg",
    mime="image/svg+xml",
)
if full_png:
    st.download_button(
        "Download Full Plot PNG",
        data=full_png,
        file_name=f"aligned_full_{selected_target}_{axis_name}.png",
        mime="image/png",
    )
elif PNG_EXPORT_ERROR:
    st.info(PNG_EXPORT_ERROR + " Install it to enable the PNG download buttons.")

st.subheader("Trimmed Recordings")
st.caption("These plots use the trimmed aligned recordings. Each clip starts at t = 0. The dashed line marks the nominal saved anchor position inside the trimmed window.")
trimmed_svg = build_overlay_svg(
    series_map=trimmed_series,
    title=f"Trimmed Recordings: {display_target_label}",
    x_axis_label="Trimmed clip time (s)",
    y_axis_label=y_axis_label,
    x_window=(0.0, float(trimmed_x_max)),
    reference_lines=[(nominal_peak_offset_sec, "saved anchor")],
    show_y_tick_labels=not apply_offsets,
)
st.markdown(trimmed_svg, unsafe_allow_html=True)
trimmed_png = build_overlay_png(
    series_map=trimmed_series,
    title=f"Trimmed Recordings: {display_target_label}",
    x_axis_label="Trimmed clip time (s)",
    y_axis_label=y_axis_label,
    x_window=(0.0, float(trimmed_x_max)),
    reference_lines=[(nominal_peak_offset_sec, "saved anchor")],
    show_y_tick_labels=not apply_offsets,
)
st.download_button(
    "Download Trimmed Plot SVG",
    data=trimmed_svg,
    file_name=f"trimmed_{selected_target}_{axis_name}.svg",
    mime="image/svg+xml",
)
if trimmed_png:
    st.download_button(
        "Download Trimmed Plot PNG",
        data=trimmed_png,
        file_name=f"trimmed_{selected_target}_{axis_name}.png",
        mime="image/png",
    )
