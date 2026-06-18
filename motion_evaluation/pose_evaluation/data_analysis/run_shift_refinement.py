from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_analysis.config import COMPARISON_SPECS, DEFAULT_CONFIG, OUTPUT_DIR, build_joint_mappings
from data_analysis.evaluators import PairArtifacts, evaluate_pair
from data_analysis.io_utils import TrialData, build_shared_time_grid, load_joint_frames, resample_trial_data
from data_analysis.normalization import normalize_trial_coordinates
from data_analysis.run_analysis import summarize_metrics


SHIFT_SELECTION_METRICS = (
    "reversal_mean_abs_timing_error_sec",
    "reversal_interval_rmse_sec",
    "pearson_velocity",
    "pearson_angle",
    "pearson_along_axis_velocity",
    "mpjpe_2d",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a non-destructive residual temporal shift experiment on aligned motion trajectories."
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR.parent / "output_shift_refined")
    parser.add_argument("--max-shift-frames", type=int, default=6)
    parser.add_argument("--evaluation-fps", type=float, default=DEFAULT_CONFIG.evaluation_fps)
    parser.add_argument("--enable-plots", action="store_true", help="Generate plots for the refined run. Disabled by default.")
    return parser.parse_args()


def shift_coordinates(coordinates: dict[str, np.ndarray], time_sec: np.ndarray, shift_sec: float) -> dict[str, np.ndarray]:
    if abs(shift_sec) < 1e-12:
        return {joint_id: values.copy() for joint_id, values in coordinates.items()}
    shifted: dict[str, np.ndarray] = {}
    query_time = time_sec + shift_sec
    for joint_id, values in coordinates.items():
        shifted_values = np.full_like(values, np.nan, dtype=float)
        for axis_index in range(values.shape[1]):
            series = values[:, axis_index]
            valid = np.isfinite(series) & np.isfinite(time_sec)
            if np.sum(valid) < 2:
                continue
            valid_time = time_sec[valid]
            valid_values = series[valid]
            shifted_axis = np.interp(query_time, valid_time, valid_values)
            shifted_axis[(query_time < valid_time[0]) | (query_time > valid_time[-1])] = np.nan
            shifted_values[:, axis_index] = shifted_axis
        shifted[joint_id] = shifted_values
    return shifted


def evaluate_comparison(
    wrist_variant: str,
    comparison,
    config,
    shift_sec: float,
    output_plot_dir: Path,
) -> PairArtifacts:
    joint_mappings = build_joint_mappings(wrist_variant)
    joint_frames_a, availability_a, warnings_a = load_joint_frames(comparison.system_a, comparison.recording_a, joint_mappings)
    joint_frames_b, availability_b, warnings_b = load_joint_frames(comparison.system_b, comparison.recording_b, joint_mappings)
    shared_time = build_shared_time_grid(joint_frames_a, joint_frames_b, config.evaluation_fps)
    if shared_time.size == 0:
        warning = {
            "comparison_name": comparison.comparison_name,
            "pair_key": comparison.pair_key,
            "trial_id": comparison.trial_id,
            "wrist_variant": wrist_variant,
            "system_name": "pair",
            "alignment_mode": "body_only",
            "warning": "No shared time window between the selected systems.",
        }
        return PairArtifacts([], [], [warning], [], [])

    trial_a = resample_trial_data(joint_frames_a, availability_a, warnings_a, shared_time)
    trial_b = resample_trial_data(joint_frames_b, availability_b, warnings_b, shared_time)

    reference_trial = trial_a if comparison.reference_system == comparison.system_a else trial_b
    candidate_trial = trial_b if comparison.reference_system == comparison.system_a else trial_a
    candidate_system = comparison.system_b if comparison.reference_system == comparison.system_a else comparison.system_a

    shifted_candidate = TrialData(
        time_sec=candidate_trial.time_sec,
        coordinates=shift_coordinates(candidate_trial.coordinates, candidate_trial.time_sec, shift_sec),
        availability=candidate_trial.availability,
        warnings=candidate_trial.warnings,
    )

    reference_normalized = normalize_trial_coordinates(
        reference_trial.coordinates,
        apply_rotation=config.rotation_normalization,
        origin_priority=config.body_origin_priority,
        scale_priority=config.body_scale_priority,
        normalize_axes=config.normalize_body_axes,
    )
    candidate_normalized = normalize_trial_coordinates(
        shifted_candidate.coordinates,
        apply_rotation=config.rotation_normalization,
        origin_priority=config.body_origin_priority,
        scale_priority=config.body_scale_priority,
        normalize_axes=config.normalize_body_axes,
    )

    artifacts = evaluate_pair(
        comparison_name=comparison.comparison_name,
        pair_key=comparison.pair_key,
        trial_id=comparison.trial_id,
        wrist_variant=wrist_variant,
        reference_system=comparison.reference_system,
        candidate_system=candidate_system,
        time_sec=shared_time,
        reference_normalized=reference_normalized,
        candidate_normalized=candidate_normalized,
        config=config,
        output_plot_dir=output_plot_dir,
    )
    return artifacts


def summarize_shift_metrics(metrics_rows: list[dict[str, object]]) -> dict[str, float]:
    frame = pd.DataFrame(metrics_rows)
    if frame.empty:
        return {metric_name: float("nan") for metric_name in SHIFT_SELECTION_METRICS}
    result: dict[str, float] = {}
    for metric_name in SHIFT_SELECTION_METRICS:
        values = frame.loc[frame["metric_name"] == metric_name, "metric_value"]
        result[metric_name] = float(values.mean()) if not values.empty else float("nan")
    pearson_values = frame.loc[
        frame["metric_name"].isin(("pearson_velocity", "pearson_angle", "pearson_along_axis_velocity")),
        "metric_value",
    ]
    result["pearson_temporal_mean"] = float(pearson_values.mean()) if not pearson_values.empty else float("nan")
    return result


def relative_improvement(baseline: float, candidate: float, higher_is_better: bool) -> float:
    if not np.isfinite(baseline) or not np.isfinite(candidate):
        return 0.0
    denominator = max(abs(baseline), 0.1)
    return (candidate - baseline) / denominator if higher_is_better else (baseline - candidate) / denominator


def add_shift_scores(search_df: pd.DataFrame) -> pd.DataFrame:
    scored = search_df.copy()
    scored["selection_score"] = 0.0
    for pair_key, pair_rows in scored.groupby("pair_key", dropna=False):
        baseline_rows = pair_rows[pair_rows["shift_frames"] == 0]
        if baseline_rows.empty:
            continue
        baseline = baseline_rows.iloc[0]
        for index, row in pair_rows.iterrows():
            score = (
                relative_improvement(baseline["reversal_mean_abs_timing_error_sec"], row["reversal_mean_abs_timing_error_sec"], False)
                + 0.5 * relative_improvement(baseline["reversal_interval_rmse_sec"], row["reversal_interval_rmse_sec"], False)
                + 0.5 * relative_improvement(baseline["pearson_temporal_mean"], row["pearson_temporal_mean"], True)
                + 0.1 * relative_improvement(baseline["mpjpe_2d"], row["mpjpe_2d"], False)
            )
            scored.loc[index, "selection_score"] = float(score)
    return scored


def choose_shifts(scored_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for pair_key, pair_rows in scored_df.groupby("pair_key", dropna=False):
        baseline_rows = pair_rows[pair_rows["shift_frames"] == 0]
        eligible_rows = pair_rows
        if not baseline_rows.empty and np.isfinite(float(baseline_rows.iloc[0]["reversal_mean_abs_timing_error_sec"])):
            baseline_reversal_error = float(baseline_rows.iloc[0]["reversal_mean_abs_timing_error_sec"])
            eligible_rows = pair_rows[pair_rows["reversal_mean_abs_timing_error_sec"] <= baseline_reversal_error]
            if eligible_rows.empty:
                eligible_rows = baseline_rows
        best = eligible_rows.assign(abs_shift_frames=eligible_rows["shift_frames"].abs()).sort_values(
            ["selection_score", "abs_shift_frames"],
            ascending=[False, True],
        ).iloc[0]
        rows.append(
            {
                "comparison_name": best["comparison_name"],
                "pair_key": pair_key,
                "trial_id": best["trial_id"],
                "chosen_shift_frames": int(best["shift_frames"]),
                "chosen_shift_sec": float(best["shift_sec"]),
                "selection_score": float(best["selection_score"]),
                "criterion": (
                    "eligible shifts must not worsen reversal_mean_abs_timing_error_sec; "
                    "among eligible shifts maximize relative improvement in reversal interval RMSE, "
                    "temporal Pearson metrics, and a small MPJPE guard term"
                ),
            }
        )
    return pd.DataFrame(rows)


def run_shift_search(config, output_dir: Path, max_shift_frames: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    search_rows: list[dict[str, object]] = []
    search_config = replace(config, generate_plots=False)
    output_plot_dir = output_dir / "search_plots"
    frame_range = range(-max_shift_frames, max_shift_frames + 1)
    for comparison in COMPARISON_SPECS:
        for shift_frames in frame_range:
            shift_sec = shift_frames / config.evaluation_fps
            artifacts = evaluate_comparison("body", comparison, search_config, shift_sec, output_plot_dir)
            row = {
                "comparison_name": comparison.comparison_name,
                "pair_key": comparison.pair_key,
                "trial_id": comparison.trial_id,
                "shift_frames": int(shift_frames),
                "shift_sec": float(shift_sec),
            }
            row.update(summarize_shift_metrics(artifacts.metrics_rows))
            search_rows.append(row)
    search_df = add_shift_scores(pd.DataFrame(search_rows))
    chosen_df = choose_shifts(search_df)
    return search_df, chosen_df


def run_refined_evaluation(config, output_dir: Path, chosen_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    shift_by_pair = dict(zip(chosen_df["pair_key"], chosen_df["chosen_shift_sec"]))
    output_plot_dir = output_dir / "plots"
    output_plot_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows: list[dict[str, object]] = []
    normalization_rows: list[dict[str, object]] = []
    warning_rows: list[dict[str, object]] = []
    plot_rows: list[dict[str, object]] = []
    bland_altman_rows: list[dict[str, object]] = []

    for wrist_variant in ("body", "hand"):
        for comparison in COMPARISON_SPECS:
            shift_sec = float(shift_by_pair.get(comparison.pair_key, 0.0))
            artifacts = evaluate_comparison(wrist_variant, comparison, config, shift_sec, output_plot_dir)
            metrics_rows.extend(artifacts.metrics_rows)
            normalization_rows.extend(artifacts.normalization_rows)
            warning_rows.extend(artifacts.warning_rows)
            plot_rows.extend(artifacts.plot_rows)
            bland_altman_rows.extend(artifacts.bland_altman_rows)

    metrics_df = pd.DataFrame(metrics_rows)
    if not metrics_df.empty and "system_name" not in metrics_df.columns:
        metrics_df["system_name"] = "pair"
    normalization_df = pd.DataFrame(normalization_rows)
    warnings_df = pd.DataFrame(warning_rows)
    plots_df = pd.DataFrame(plot_rows)
    bland_altman_df = pd.DataFrame(bland_altman_rows)
    trial_summary_df, module_summary_df, domain_summary_df = summarize_metrics(metrics_df)

    metrics_df.to_csv(output_dir / "metrics_long.csv", index=False)
    normalization_df.to_csv(output_dir / "normalization_summary.csv", index=False)
    warnings_df.to_csv(output_dir / "warnings.csv", index=False)
    trial_summary_df.to_csv(output_dir / "summary_by_trial.csv", index=False)
    module_summary_df.to_csv(output_dir / "summary_by_module.csv", index=False)
    domain_summary_df.to_csv(output_dir / "summary_by_domain.csv", index=False)
    bland_altman_df.to_csv(output_dir / "bland_altman.csv", index=False)
    plots_df.to_csv(output_dir / "plot_manifest.csv", index=False)
    return metrics_df, module_summary_df, bland_altman_df


def compare_with_baseline(refined_metrics_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    baseline_path = OUTPUT_DIR / "metrics_long.csv"
    baseline_df = pd.read_csv(baseline_path) if baseline_path.exists() else pd.DataFrame()
    key_metrics = (
        "reversal_mean_abs_timing_error_sec",
        "reversal_interval_rmse_sec",
        "pearson_velocity",
        "pearson_angle",
        "pearson_along_axis_velocity",
        "mpjpe_2d",
        "pck_2d@0.25",
        "pck_2d@0.5",
    )
    keys = ["wrist_variant", "comparison_name", "alignment_mode", "metric_name"]
    baseline_summary = (
        baseline_df[baseline_df["metric_name"].isin(key_metrics)]
        .groupby(keys, dropna=False)["metric_value"]
        .mean()
        .reset_index()
        .rename(columns={"metric_value": "baseline_mean"})
    )
    refined_summary = (
        refined_metrics_df[refined_metrics_df["metric_name"].isin(key_metrics)]
        .groupby(keys, dropna=False)["metric_value"]
        .mean()
        .reset_index()
        .rename(columns={"metric_value": "refined_mean"})
    )
    comparison = baseline_summary.merge(refined_summary, on=keys, how="outer")
    comparison["delta_refined_minus_baseline"] = comparison["refined_mean"] - comparison["baseline_mean"]
    comparison.to_csv(output_dir / "baseline_vs_shift_refined_summary.csv", index=False)
    return comparison


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = replace(
        DEFAULT_CONFIG,
        evaluation_fps=float(args.evaluation_fps),
        generate_plots=bool(args.enable_plots),
    )

    search_df, chosen_df = run_shift_search(config, output_dir, int(args.max_shift_frames))
    search_df.to_csv(output_dir / "shift_search_summary.csv", index=False)
    chosen_df.to_csv(output_dir / "chosen_shifts.csv", index=False)

    refined_metrics_df, _module_summary_df, _bland_altman_df = run_refined_evaluation(config, output_dir, chosen_df)
    comparison_df = compare_with_baseline(refined_metrics_df, output_dir)

    print(f"Wrote residual-shift experiment outputs under {output_dir}")
    print("Chosen shifts:")
    print(chosen_df[["pair_key", "chosen_shift_frames", "chosen_shift_sec", "selection_score"]].to_string(index=False))
    print("Baseline/refined comparison rows:", len(comparison_df))


if __name__ == "__main__":
    main()
