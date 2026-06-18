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
RESULT_DIR = REPO_ROOT / "thesis_data" / "pose_evaluation" / "prepared_result"
ALIGNMENT_FILE = "knee_default_left.csv"
NUMERIC_COLUMNS = ["frame_index", "time_sec", "x", "y", "z", "x_px", "y_px", "confidence", "visibility"]
BOOLEAN_COLUMNS = ["tracking_present", "valid"]
SEARCH_PRESETS = {
    "air_knees": (20.0, 40.0),
    "drums": (34.0, 45.0),
    "apple_vision": (24.0, 32.0),
    "mediapipe": (24.0, 36.0),
}
PEAK_DIRECTION_OVERRIDES = {
    "drums": "valley",
}
PEAK_SIGMA_OVERRIDES = {
    "drums": 1.5,
}
RECORDING_ORDER = ["air_knees", "drums", "apple_vision", "mediapipe"]
PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#17becf"]


def list_recording_specs() -> list[tuple[str, str, Path]]:
    specs: list[tuple[str, str, Path]] = []
    for source_name in ("mocap", "pose"):
        base_dir = RESULT_DIR / source_name
        if not base_dir.exists():
            continue
        for csv_path in sorted(base_dir.glob(f"*/{ALIGNMENT_FILE}")):
            specs.append((source_name, csv_path.parent.name, csv_path))
    return sorted(specs, key=lambda item: RECORDING_ORDER.index(item[1]) if item[1] in RECORDING_ORDER else len(RECORDING_ORDER))


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


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


def prepare_signal_frame(frame: pd.DataFrame, source_name: str, invert_pose_y: bool) -> pd.DataFrame:
    signal_frame = frame.copy()
    if "valid" in signal_frame.columns:
        signal_frame = signal_frame[signal_frame["valid"]]
    signal_frame = signal_frame[["frame_index", "time_sec", "y"]].dropna().copy()
    if invert_pose_y and source_name == "pose":
        signal_frame["y"] = -signal_frame["y"]
    return signal_frame.sort_values("time_sec").reset_index(drop=True)


def zscore_series(series: pd.Series) -> pd.Series:
    std = float(series.std(ddof=0))
    centered = series - float(series.mean())
    return centered if std == 0 or pd.isna(std) else centered / std


def cluster_candidates(candidates: pd.DataFrame, min_peak_distance_sec: float, direction: str) -> pd.DataFrame:
    if candidates.empty:
        return candidates

    clustered_rows: list[pd.Series] = []
    current_group: list[pd.Series] = []
    last_time: float | None = None

    for row in candidates.sort_values("time_sec").iterrows():
        series = row[1]
        current_time = float(series["time_sec"])
        if last_time is None or current_time - last_time <= min_peak_distance_sec:
            current_group.append(series)
        else:
            if direction == "valley":
                clustered_rows.append(min(current_group, key=lambda item: (float(item["smooth_y"]), float(item["time_sec"]))))
            else:
                clustered_rows.append(max(current_group, key=lambda item: (float(item["smooth_y"]), -float(item["time_sec"]))))
            current_group = [series]
        last_time = current_time

    if current_group:
        if direction == "valley":
            clustered_rows.append(min(current_group, key=lambda item: (float(item["smooth_y"]), float(item["time_sec"]))))
        else:
            clustered_rows.append(max(current_group, key=lambda item: (float(item["smooth_y"]), -float(item["time_sec"]))))

    return pd.DataFrame(clustered_rows).sort_values("time_sec").reset_index(drop=True)


def detect_first_peak(
    signal_frame: pd.DataFrame,
    search_start_sec: float,
    search_end_sec: float,
    smoothing_points: int,
    min_peak_sigma: float,
    min_peak_distance_sec: float,
    direction: str,
) -> dict[str, object]:
    search_frame = signal_frame[
        (signal_frame["time_sec"] >= search_start_sec) & (signal_frame["time_sec"] <= search_end_sec)
    ].copy()
    if search_frame.empty:
        return {
            "status": "no_data_in_window",
            "search_frame": search_frame,
            "candidates": pd.DataFrame(),
            "selected_peak": None,
            "threshold": None,
            "robust_scale": None,
        }

    search_frame["smooth_y"] = (
        search_frame["y"].rolling(int(max(1, smoothing_points)), center=True, min_periods=1).median()
    )
    baseline = float(search_frame["smooth_y"].median())
    mad = float((search_frame["smooth_y"] - baseline).abs().median())
    robust_scale = mad * 1.4826 if mad > 0 else float(search_frame["smooth_y"].std(ddof=0) or 1.0)
    previous_values = search_frame["smooth_y"].shift(1)
    next_values = search_frame["smooth_y"].shift(-1)
    if direction == "valley":
        threshold = baseline - min_peak_sigma * robust_scale
        raw_candidates = search_frame[
            (search_frame["smooth_y"] < previous_values)
            & (search_frame["smooth_y"] <= next_values)
            & (search_frame["smooth_y"] <= threshold)
        ].copy()
        raw_candidates["peak_score"] = baseline - raw_candidates["smooth_y"]
    else:
        threshold = baseline + min_peak_sigma * robust_scale
        raw_candidates = search_frame[
            (search_frame["smooth_y"] > previous_values)
            & (search_frame["smooth_y"] >= next_values)
            & (search_frame["smooth_y"] >= threshold)
        ].copy()
        raw_candidates["peak_score"] = raw_candidates["smooth_y"] - baseline
    candidates = cluster_candidates(raw_candidates, min_peak_distance_sec, direction)

    if not candidates.empty:
        selected_peak = candidates.iloc[0]
        status = "detected"
    else:
        fallback_peak = search_frame.nsmallest(1, "smooth_y") if direction == "valley" else search_frame.nlargest(1, "smooth_y")
        selected_peak = fallback_peak.iloc[0] if not fallback_peak.empty else None
        status = "fallback_max" if selected_peak is not None else "no_peak"

    return {
        "status": status,
        "direction": direction,
        "search_frame": search_frame,
        "candidates": candidates,
        "selected_peak": selected_peak,
        "threshold": threshold,
        "robust_scale": robust_scale,
    }


def build_alignment_summary(
    recording_name: str,
    source_name: str,
    csv_path: Path,
    detection: dict[str, object],
    before_sec: float,
    after_sec: float,
) -> dict[str, object]:
    selected_peak = detection["selected_peak"]
    if selected_peak is None:
        return {
            "recording": recording_name,
            "source": source_name,
            "status": detection["status"],
            "direction": detection["direction"],
            "search_start_sec": None,
            "search_end_sec": None,
            "peak_time_sec": None,
            "peak_frame_index": None,
            "peak_y": None,
            "clip_start_sec": None,
            "clip_end_sec": None,
            "candidate_count": len(detection["candidates"]),
            "file": display_path(csv_path),
        }

    peak_time = float(selected_peak["time_sec"])
    return {
        "recording": recording_name,
        "source": source_name,
        "status": detection["status"],
        "direction": detection["direction"],
        "search_start_sec": float(detection["search_frame"]["time_sec"].min()),
        "search_end_sec": float(detection["search_frame"]["time_sec"].max()),
        "peak_time_sec": peak_time,
        "peak_frame_index": int(selected_peak["frame_index"]),
        "peak_y": float(selected_peak["y"]),
        "clip_start_sec": peak_time - before_sec,
        "clip_end_sec": peak_time + after_sec,
        "candidate_count": len(detection["candidates"]),
        "file": display_path(csv_path),
    }


def prepare_trimmed_clip(
    signal_frame: pd.DataFrame,
    peak_time_sec: float,
    before_sec: float,
    after_sec: float,
    normalize_clip: bool,
) -> pd.DataFrame:
    clip_frame = signal_frame[
        (signal_frame["time_sec"] >= peak_time_sec - before_sec) & (signal_frame["time_sec"] <= peak_time_sec + after_sec)
    ].copy()
    clip_frame["aligned_time_sec"] = clip_frame["time_sec"] - peak_time_sec
    clip_frame["plot_y"] = zscore_series(clip_frame["y"]) if normalize_clip else clip_frame["y"]
    return clip_frame[["aligned_time_sec", "plot_y"]].dropna()


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


def svg_axes(x_min: float, x_max: float, y_min: float, y_max: float, width: int, height: int) -> tuple[float, float, float, float]:
    if x_min == x_max:
        x_min -= 1.0
        x_max += 1.0
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0
    return x_min, x_max, y_min, y_max


def build_detection_svg(
    signal_frame: pd.DataFrame,
    title: str,
    search_start_sec: float,
    search_end_sec: float,
    candidate_times: list[float],
    selected_peak_time: float | None,
    clip_start_sec: float | None,
    clip_end_sec: float | None,
    normalize_display: bool,
) -> str:
    plot_width = 980
    plot_height = 280
    margin_left = 72
    margin_right = 24
    margin_top = 20
    margin_bottom = 44
    inner_width = plot_width - margin_left - margin_right
    inner_height = plot_height - margin_top - margin_bottom

    plot_frame = signal_frame.copy()
    plot_frame["plot_y"] = zscore_series(plot_frame["y"]) if normalize_display else plot_frame["y"]
    x_values = plot_frame["time_sec"].dropna()
    y_values = plot_frame["plot_y"].dropna()
    if x_values.empty or y_values.empty:
        return "<p>No plottable values.</p>"

    x_min, x_max, y_min, y_max = svg_axes(float(x_values.min()), float(x_values.max()), float(y_values.min()), float(y_values.max()), plot_width, plot_height)

    def scale_x(value: float) -> float:
        return margin_left + ((value - x_min) / (x_max - x_min)) * inner_width

    def scale_y(value: float) -> float:
        return margin_top + inner_height - ((value - y_min) / (y_max - y_min)) * inner_height

    search_left = scale_x(max(search_start_sec, x_min))
    search_right = scale_x(min(search_end_sec, x_max))
    search_width = max(0.0, search_right - search_left)

    points = " ".join(
        f"{scale_x(float(time_sec)):.2f},{scale_y(float(value)):.2f}"
        for time_sec, value in zip(plot_frame["time_sec"], plot_frame["plot_y"])
    )

    candidate_marks = []
    for peak_time in candidate_times:
        peak_row = plot_frame.iloc[(plot_frame["time_sec"] - peak_time).abs().argsort()[:1]]
        if peak_row.empty:
            continue
        x = scale_x(float(peak_row["time_sec"].iloc[0]))
        y = scale_y(float(peak_row["plot_y"].iloc[0]))
        candidate_marks.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="#f59e0b" />')

    selected_marker = ""
    selected_line = ""
    clip_lines = ""
    if selected_peak_time is not None:
        peak_row = plot_frame.iloc[(plot_frame["time_sec"] - selected_peak_time).abs().argsort()[:1]]
        if not peak_row.empty:
            x = scale_x(float(peak_row["time_sec"].iloc[0]))
            y = scale_y(float(peak_row["plot_y"].iloc[0]))
            selected_marker = f'<circle cx="{x:.2f}" cy="{y:.2f}" r="6" fill="#dc2626" />'
            selected_line = (
                f'<line x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" y2="{margin_top + inner_height}" '
                f'stroke="#dc2626" stroke-width="2" stroke-dasharray="6 4" />'
            )
        if clip_start_sec is not None and clip_end_sec is not None:
            for clip_time in (clip_start_sec, clip_end_sec):
                clip_x = scale_x(min(max(clip_time, x_min), x_max))
                clip_lines += (
                    f'<line x1="{clip_x:.2f}" y1="{margin_top}" x2="{clip_x:.2f}" y2="{margin_top + inner_height}" '
                    f'stroke="#64748b" stroke-width="1.5" stroke-dasharray="4 4" />'
                )

    y_ticks = []
    for tick_index in range(5):
        ratio = tick_index / 4
        value = y_min + ratio * (y_max - y_min)
        y = scale_y(value)
        y_ticks.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{plot_width - margin_right}" y2="{y:.2f}" stroke="#e5e7eb" stroke-width="1" />'
            f'<text x="{margin_left - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="12" fill="#475569">{value:.2f}</text>'
        )

    svg = f"""
    <svg viewBox="0 0 {plot_width} {plot_height}" width="100%" height="auto" xmlns="http://www.w3.org/2000/svg">
      <rect x="0" y="0" width="{plot_width}" height="{plot_height}" fill="#ffffff" rx="12" />
      <text x="{plot_width / 2}" y="18" text-anchor="middle" font-size="16" fill="#0f172a">{escape(title)}</text>
      {''.join(y_ticks)}
      <rect x="{search_left:.2f}" y="{margin_top}" width="{search_width:.2f}" height="{inner_height}" fill="#dbeafe" opacity="0.45" />
      <line x1="{margin_left}" y1="{margin_top + inner_height}" x2="{plot_width - margin_right}" y2="{margin_top + inner_height}" stroke="#94a3b8" stroke-width="1.5" />
      <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + inner_height}" stroke="#94a3b8" stroke-width="1.5" />
      <polyline fill="none" stroke="#2563eb" stroke-width="2" points="{points}" />
      {''.join(candidate_marks)}
      {selected_line}
      {clip_lines}
      {selected_marker}
      <text x="{plot_width / 2}" y="{plot_height - 4}" text-anchor="middle" font-size="12" fill="#475569">Time (s)</text>
      <text x="16" y="{plot_height / 2}" text-anchor="middle" font-size="12" fill="#475569" transform="rotate(-90 16 {plot_height / 2})">{'normalized y' if normalize_display else 'processed y'}</text>
    </svg>
    """
    return svg


def build_overlay_svg(series_map: dict[str, pd.DataFrame], title: str, normalize_display: bool) -> str:
    plot_width = 980
    plot_height = 320
    margin_left = 72
    margin_right = 24
    margin_top = 24
    margin_bottom = 48
    inner_width = plot_width - margin_left - margin_right
    inner_height = plot_height - margin_top - margin_bottom

    combined_rows = []
    for label, frame in series_map.items():
        if frame.empty:
            continue
        local_frame = frame.copy()
        local_frame["series"] = label
        combined_rows.append(local_frame)
    if not combined_rows:
        return "<p>No aligned clips available.</p>"

    combined = pd.concat(combined_rows, ignore_index=True)
    x_values = combined["aligned_time_sec"].dropna()
    y_values = combined["plot_y"].dropna()
    if x_values.empty or y_values.empty:
        return "<p>No aligned clips available.</p>"

    x_min, x_max, y_min, y_max = svg_axes(float(x_values.min()), float(x_values.max()), float(y_values.min()), float(y_values.max()), plot_width, plot_height)

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
            f'<text x="{margin_left - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="12" fill="#475569">{value:.2f}</text>'
        )

    series_paths = []
    legend_items = []
    for index, label in enumerate(RECORDING_ORDER):
        if label not in series_map or series_map[label].empty:
            continue
        frame = series_map[label].copy()
        points = " ".join(
            f"{scale_x(float(time_sec)):.2f},{scale_y(float(value)):.2f}"
            for time_sec, value in zip(frame["aligned_time_sec"], frame["plot_y"])
        )
        color = PALETTE[index % len(PALETTE)]
        series_paths.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{points}" />')
        legend_y = margin_top + 10 + len(legend_items) * 18
        legend_items.append(
            f'<line x1="{plot_width - 210}" y1="{legend_y:.2f}" x2="{plot_width - 188}" y2="{legend_y:.2f}" stroke="{color}" stroke-width="3" />'
            f'<text x="{plot_width - 182}" y="{legend_y + 4:.2f}" font-size="12" fill="#0f172a">{escape(label)}</text>'
        )

    zero_x = scale_x(0.0) if x_min <= 0 <= x_max else None
    zero_line = (
        f'<line x1="{zero_x:.2f}" y1="{margin_top}" x2="{zero_x:.2f}" y2="{margin_top + inner_height}" stroke="#dc2626" stroke-width="2" stroke-dasharray="6 4" />'
        if zero_x is not None
        else ""
    )

    svg = f"""
    <svg viewBox="0 0 {plot_width} {plot_height}" width="100%" height="auto" xmlns="http://www.w3.org/2000/svg">
      <rect x="0" y="0" width="{plot_width}" height="{plot_height}" fill="#ffffff" rx="12" />
      <text x="{plot_width / 2}" y="18" text-anchor="middle" font-size="16" fill="#0f172a">{escape(title)}</text>
      {''.join(y_ticks)}
      <line x1="{margin_left}" y1="{margin_top + inner_height}" x2="{plot_width - margin_right}" y2="{margin_top + inner_height}" stroke="#94a3b8" stroke-width="1.5" />
      <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + inner_height}" stroke="#94a3b8" stroke-width="1.5" />
      {zero_line}
      {''.join(series_paths)}
      {''.join(legend_items)}
      <text x="{plot_width / 2}" y="{plot_height - 4}" text-anchor="middle" font-size="12" fill="#475569">Aligned time (s)</text>
      <text x="16" y="{plot_height / 2}" text-anchor="middle" font-size="12" fill="#475569" transform="rotate(-90 16 {plot_height / 2})">{'normalized y' if normalize_display else 'processed y'}</text>
    </svg>
    """
    return svg


st.set_page_config(page_title="Left Knee Peak Alignment", layout="wide")
st.title("Left Knee Peak Alignment")
st.caption("Detect the first isolated left-knee y peak per recording, align it to t = 0, and trim before/after windows for comparison.")

recording_specs = list_recording_specs()
if not recording_specs:
    st.error(f"No `{ALIGNMENT_FILE}` recordings were found under {RESULT_DIR}.")
    st.stop()

with st.sidebar:
    st.header("Detection")
    invert_pose_y = st.checkbox("Invert pose y", value=True)
    smoothing_points = st.slider("Smoothing points", min_value=1, max_value=31, value=9, step=2)
    min_peak_sigma = st.slider("Peak threshold (sigma)", min_value=1.0, max_value=8.0, value=3.0, step=0.25)
    min_peak_distance_sec = st.slider("Min peak distance (s)", min_value=0.25, max_value=10.0, value=1.5, step=0.25)
    normalize_display = st.checkbox("Normalize plots for display", value=True)
    st.header("Trim Window")
    before_sec = st.slider("Seconds before peak", min_value=0.0, max_value=30.0, value=5.0, step=0.5)
    after_sec = st.slider("Seconds after peak", min_value=1.0, max_value=120.0, value=90.0, step=1.0)

st.subheader("Search Windows")
st.caption("The default mocap windows start after 20 seconds so the detector ignores setup outliers and looks for the later performance peaks.")
st.caption("`drums` is configured to detect a valley instead of a positive peak, because that hi-hat knee motion is inverted.")
search_windows: dict[str, tuple[float, float]] = {}
search_cols = st.columns(2)
for index, (_, recording_name, csv_path) in enumerate(recording_specs):
    signal_preview = load_recording(str(csv_path))
    time_max = float(signal_preview["time_sec"].max())
    default_start, default_end = SEARCH_PRESETS.get(recording_name, (0.0, min(20.0, time_max)))
    default_start = max(0.0, min(default_start, time_max))
    default_end = max(default_start, min(default_end, time_max))
    column = search_cols[index % 2]
    with column:
        st.markdown(f"**{recording_name}**")
        start_value = st.number_input(
            f"{recording_name} start (s)",
            min_value=0.0,
            max_value=time_max,
            value=default_start,
            step=0.5,
            key=f"{recording_name}_start",
        )
        end_value = st.number_input(
            f"{recording_name} end (s)",
            min_value=start_value,
            max_value=time_max,
            value=max(start_value, default_end),
            step=0.5,
            key=f"{recording_name}_end",
        )
        search_windows[recording_name] = (float(start_value), float(end_value))

processed_frames: dict[str, pd.DataFrame] = {}
detections: dict[str, dict[str, object]] = {}
summary_rows: list[dict[str, object]] = []
aligned_clips: dict[str, pd.DataFrame] = {}
recording_meta: dict[str, tuple[str, Path]] = {}

for source_name, recording_name, csv_path in recording_specs:
    recording_meta[recording_name] = (source_name, csv_path)
    raw_frame = load_recording(str(csv_path))
    signal_frame = prepare_signal_frame(raw_frame, source_name, invert_pose_y=invert_pose_y)
    processed_frames[recording_name] = signal_frame
    search_start_sec, search_end_sec = search_windows[recording_name]
    direction = PEAK_DIRECTION_OVERRIDES.get(recording_name, "peak")
    min_peak_sigma_for_recording = PEAK_SIGMA_OVERRIDES.get(recording_name, min_peak_sigma)
    detection = detect_first_peak(
        signal_frame=signal_frame,
        search_start_sec=search_start_sec,
        search_end_sec=search_end_sec,
        smoothing_points=smoothing_points,
        min_peak_sigma=min_peak_sigma_for_recording,
        min_peak_distance_sec=min_peak_distance_sec,
        direction=direction,
    )
    detections[recording_name] = detection
    summary_rows.append(build_alignment_summary(recording_name, source_name, csv_path, detection, before_sec, after_sec))

    selected_peak = detection["selected_peak"]
    if selected_peak is not None:
        aligned_clips[recording_name] = prepare_trimmed_clip(
            signal_frame=signal_frame,
            peak_time_sec=float(selected_peak["time_sec"]),
            before_sec=before_sec,
            after_sec=after_sec,
            normalize_clip=normalize_display,
        )
    else:
        aligned_clips[recording_name] = pd.DataFrame(columns=["aligned_time_sec", "plot_y"])

st.subheader("Alignment Summary")
st.dataframe(prepare_table_frame(pd.DataFrame(summary_rows)), use_container_width=True)

st.subheader("Aligned Comparison")
st.caption("The red vertical line marks the aligned peak at t = 0. The overlay uses the chosen before/after trim window.")
st.markdown(
    build_overlay_svg(
        {recording_name: aligned_clips[recording_name] for recording_name in RECORDING_ORDER if recording_name in aligned_clips},
        title="Aligned left-knee y comparison",
        normalize_display=normalize_display,
    ),
    unsafe_allow_html=True,
)

st.subheader("Per-Recording Detection")
st.caption("Blue shading is the search window. Orange dots are candidate peaks. The red marker/line is the selected first peak.")
for recording_name in RECORDING_ORDER:
    if recording_name not in processed_frames:
        continue
    source_name, csv_path = recording_meta[recording_name]
    detection = detections[recording_name]
    selected_peak = detection["selected_peak"]
    search_start_sec, search_end_sec = search_windows[recording_name]
    selected_peak_time = float(selected_peak["time_sec"]) if selected_peak is not None else None
    candidate_times = [float(value) for value in detection["candidates"]["time_sec"].tolist()] if not detection["candidates"].empty else []
    clip_start_sec = selected_peak_time - before_sec if selected_peak_time is not None else None
    clip_end_sec = selected_peak_time + after_sec if selected_peak_time is not None else None

    st.markdown(
        f"**{recording_name}**  `source={source_name}`  `direction={detection['direction']}`  `status={detection['status']}`  `file={display_path(csv_path)}`"
    )
    if selected_peak is not None:
        st.caption(
            f"selected_peak={selected_peak_time:.3f}s  frame={int(selected_peak['frame_index'])}  "
            f"y={float(selected_peak['y']):.6f}  candidates={len(candidate_times)}"
        )
    else:
        st.caption("No peak selected in the current search window.")

    st.markdown(
        build_detection_svg(
            signal_frame=processed_frames[recording_name],
            title=f"{recording_name} left knee y",
            search_start_sec=search_start_sec,
            search_end_sec=search_end_sec,
            candidate_times=candidate_times,
            selected_peak_time=selected_peak_time,
            clip_start_sec=clip_start_sec,
            clip_end_sec=clip_end_sec,
            normalize_display=normalize_display,
        ),
        unsafe_allow_html=True,
    )
