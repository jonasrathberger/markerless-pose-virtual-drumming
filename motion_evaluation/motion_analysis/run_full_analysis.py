#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/mplcache").resolve()))

from analysis_knee_foot import paired_test
from features import range_metrics
from io_optitrack import load_optitrack_csv
from proxy_dynamic import run_proxy_dynamic
from viz import plot_box_compare

RAW_INPUT_DIR = REPO_ROOT / "thesis_data" / "raw" / "mocap"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "thesis_data" / "motion_analysis"

OPTITRACK_FILES = {
    "air_feet": RAW_INPUT_DIR / "air_forefoot.csv",
    "air_knees": RAW_INPUT_DIR / "air_knees.csv",
    "drums": RAW_INPUT_DIR / "drums.csv",
}

CONDITION_DISPLAY_NAMES = {
    "air_feet": "Air-Forefoot",
    "air_knees": "Air-Knees",
    "drums": "Physical Drumming",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OptiTrack motion analysis and write canonical thesis data outputs.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for canonical CSV outputs.")
    parser.add_argument("--overlay-dir", type=Path, default=None, help="Directory for required proxy overlay PNG inputs.")
    parser.add_argument("--diagnostic-dir", type=Path, default=None, help="Optional directory for non-final diagnostic PNGs.")
    return parser.parse_args()


def prepare_output_dir(output_dir: Path, overlay_dir: Path, diagnostic_dir: Path | None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    for csv_name in (
        "proxy_dynamic_metrics.csv",
        "knee_condition_vs_foot_condition_segments.csv",
        "knee_condition_vs_foot_condition_tests.csv",
        "knee_condition_vs_foot_condition_summary.csv",
    ):
        path = output_dir / csv_name
        if path.exists():
            path.unlink()
    for overlay_name in (
        "proxy_dynamic_overlay_top_drums_L.png",
        "proxy_dynamic_overlay_top_drums_R.png",
    ):
        path = overlay_dir / overlay_name
        if path.exists():
            path.unlink()
    if diagnostic_dir is not None:
        diagnostic_dir.mkdir(parents=True, exist_ok=True)


def load_all() -> dict[str, Any]:
    return {name: load_optitrack_csv(path) for name, path in OPTITRACK_FILES.items()}


def proxy_analysis(
    opti: dict[str, Any],
    *,
    output_dir: Path,
    overlay_dir: Path,
    diagnostic_dir: Path | None,
) -> pd.DataFrame:
    proxy_df = run_proxy_dynamic(
        opti,
        overlay_dir=overlay_dir,
        overlay_recordings={"drums": CONDITION_DISPLAY_NAMES["drums"]},
    )
    proxy_df.to_csv(output_dir / "proxy_dynamic_metrics.csv", index=False)
    return proxy_df


def segment_metrics(series: pd.Series, time_s: pd.Series, window_s: float = 4.0) -> pd.DataFrame:
    t = time_s.to_numpy(dtype=float)
    x = series.to_numpy(dtype=float)
    edges = np.arange(float(t[0]), float(t[-1]) + window_s, window_s)
    rows: list[dict[str, float]] = []
    for index in range(len(edges) - 1):
        lo, hi = edges[index], edges[index + 1]
        mask = (t >= lo) & (t < hi)
        if mask.sum() < 20:
            continue
        metrics = range_metrics(pd.Series(x[mask]))
        rows.append(
            {
                "segment": float(index),
                "t0": lo,
                "t1": hi,
                "peak_to_peak": metrics["peak_to_peak"],
                "p95_p05": metrics["p95_p05"],
            }
        )
    return pd.DataFrame(rows)


def robust_z(values: pd.Series) -> pd.Series:
    median = float(values.median())
    mad = float((values - median).abs().median())
    if mad == 0.0:
        return pd.Series(np.zeros(len(values)), index=values.index, dtype=float)
    return 0.6745 * (values - median).abs() / mad


def knee_toe_analysis(
    opti: dict[str, Any],
    *,
    output_dir: Path,
    diagnostic_dir: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    knees = opti["air_knees"].data
    toes = opti["air_feet"].data
    configs = [
        ("L", "Skeleton 002:LKneeOut|Position|Y", "Skeleton 002:LToeTip|Position|Y"),
        ("R", "Skeleton 002:RKneeOut|Position|Y", "Skeleton 002:RToeTip|Position|Y"),
    ]

    segment_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for side, knee_col, toe_col in configs:
        knee_seg = segment_metrics(knees[knee_col], knees["time_s"], window_s=4.0)
        toe_seg = segment_metrics(toes[toe_col], toes["time_s"], window_s=4.0)
        n = min(len(knee_seg), len(toe_seg))
        if n == 0:
            continue
        knee_seg = knee_seg.iloc[:n].reset_index(drop=True)
        toe_seg = toe_seg.iloc[:n].reset_index(drop=True)

        side_frame = pd.DataFrame(
            {
                "side": side,
                "segment": knee_seg["segment"],
                "t0": knee_seg["t0"],
                "t1": knee_seg["t1"],
                "knee_peak_to_peak": knee_seg["peak_to_peak"],
                "toe_peak_to_peak": toe_seg["peak_to_peak"],
                "knee_p95_p05": knee_seg["p95_p05"],
                "toe_p95_p05": toe_seg["p95_p05"],
            }
        )
        side_frame["knee_robust_z"] = robust_z(side_frame["knee_peak_to_peak"])
        side_frame["toe_robust_z"] = robust_z(side_frame["toe_peak_to_peak"])
        side_frame["keep_segment"] = (side_frame["knee_robust_z"] <= 8.0) & (side_frame["toe_robust_z"] <= 8.0)
        segment_rows.extend(side_frame.to_dict("records"))

        kept = side_frame[side_frame["keep_segment"]].copy()
        if kept.empty:
            kept = side_frame.copy()
        for metric in ("peak_to_peak", "p95_p05"):
            knee_values = kept[f"knee_{metric}"].to_numpy(dtype=float)
            toe_values = kept[f"toe_{metric}"].to_numpy(dtype=float)
            stat = paired_test(knee_values, toe_values)
            test_rows.append(
                {
                    "side": side,
                    "metric": metric,
                    "n_segments": int(len(kept)),
                    "knee_median": float(np.median(knee_values)),
                    "toe_median": float(np.median(toe_values)),
                    **stat,
                }
            )

        summary_rows.append(
            {
                "side": side,
                "n_segments_total": int(len(side_frame)),
                "n_segments_kept": int(len(kept)),
                "knee_raw_median_peak_to_peak": float(side_frame["knee_peak_to_peak"].median()),
                "toe_raw_median_peak_to_peak": float(side_frame["toe_peak_to_peak"].median()),
                "knee_filtered_median_peak_to_peak": float(kept["knee_peak_to_peak"].median()),
                "toe_filtered_median_peak_to_peak": float(kept["toe_peak_to_peak"].median()),
                "knee_filtered_median_p95_p05": float(kept["knee_p95_p05"].median()),
                "toe_filtered_median_p95_p05": float(kept["toe_p95_p05"].median()),
                "winner_by_filtered_peak_to_peak": "knee" if kept["knee_peak_to_peak"].median() >= kept["toe_peak_to_peak"].median() else "toe",
                "winner_by_filtered_p95_p05": "knee" if kept["knee_p95_p05"].median() >= kept["toe_p95_p05"].median() else "toe",
                "excluded_segments": ",".join(str(int(value)) for value in side_frame.loc[~side_frame["keep_segment"], "segment"]),
            }
        )

        if diagnostic_dir is not None:
            plot_box_compare(
                kept["knee_peak_to_peak"],
                kept["toe_peak_to_peak"],
                f"{side} Knee",
                f"{side} Toe",
                diagnostic_dir / f"knee_vs_toe_{side}_peak_to_peak.png",
                f"{side} Peak-to-Peak Range per 4s Segment",
            )

    segments_df = pd.DataFrame(segment_rows)
    tests_df = pd.DataFrame(test_rows)
    summary_df = pd.DataFrame(summary_rows)
    segments_df.to_csv(output_dir / "knee_condition_vs_foot_condition_segments.csv", index=False)
    tests_df.to_csv(output_dir / "knee_condition_vs_foot_condition_tests.csv", index=False)
    summary_df.to_csv(output_dir / "knee_condition_vs_foot_condition_summary.csv", index=False)
    return segments_df, tests_df, summary_df


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    overlay_dir = args.overlay_dir.resolve() if args.overlay_dir else output_dir / "proxy_overlays"
    diagnostic_dir = args.diagnostic_dir.resolve() if args.diagnostic_dir else None
    prepare_output_dir(output_dir, overlay_dir, diagnostic_dir)
    opti = load_all()
    proxy_analysis(opti, output_dir=output_dir, overlay_dir=overlay_dir, diagnostic_dir=diagnostic_dir)
    knee_toe_analysis(opti, output_dir=output_dir, diagnostic_dir=diagnostic_dir)
    print("Full analysis finished.")
    print(f"- Tables: {output_dir}/*.csv")
    print(f"- Overlay inputs: {overlay_dir}/*.png")
    if diagnostic_dir is not None:
        print(f"- Diagnostic figures: {diagnostic_dir}/*.png")


if __name__ == "__main__":
    main()
