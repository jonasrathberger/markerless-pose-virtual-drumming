from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_analysis import config as analysis_config
from data_analysis.config import (
    COMPARISON_SPECS,
    DEFAULT_CONFIG,
    INPUT_DIR,
    OUTPUT_DIR,
    build_joint_mappings,
)
from data_analysis.evaluators import evaluate_pair
from data_analysis.io_utils import build_shared_time_grid, load_joint_frames, resample_trial_data
from data_analysis.normalization import normalize_trial_coordinates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate aligned motion trajectories in normalized 2D body-centered space.")
    parser.add_argument("--evaluation-fps", type=float, default=DEFAULT_CONFIG.evaluation_fps)
    parser.add_argument("--sides", nargs="+", default=list(DEFAULT_CONFIG.selected_sides))
    parser.add_argument("--joints", nargs="+", default=list(DEFAULT_CONFIG.selected_joints))
    parser.add_argument(
        "--metric-groups",
        nargs="+",
        default=list(DEFAULT_CONFIG.enabled_metric_groups),
        choices=["core_spatial", "upper_limb", "lower_limb", "pearson", "jitter", "bland_altman"],
    )
    parser.add_argument("--rotation-normalization", action="store_true", default=DEFAULT_CONFIG.rotation_normalization)
    parser.add_argument("--disable-body-axis-normalization", action="store_true")
    parser.add_argument("--disable-similarity-alignment", action="store_true")
    parser.add_argument("--disable-torso-similarity-alignment", action="store_true")
    parser.add_argument("--smoothing-method", default=DEFAULT_CONFIG.smoothing_method, choices=["savitzky_golay", "moving_average"])
    parser.add_argument("--smoothing-window-sec", type=float, default=DEFAULT_CONFIG.smoothing_window_sec)
    parser.add_argument("--smoothing-polyorder", type=int, default=DEFAULT_CONFIG.smoothing_polyorder)
    parser.add_argument(
        "--pck-thresholds",
        default=",".join(f"{threshold:g}" for threshold in DEFAULT_CONFIG.pck_thresholds),
        help="Comma-separated normalized-distance thresholds.",
    )
    parser.add_argument("--disable-pearson", action="store_true")
    parser.add_argument("--disable-jitter", action="store_true")
    parser.add_argument("--disable-bland-altman", action="store_true")
    parser.add_argument("--jitter-smoothing-method", default=DEFAULT_CONFIG.jitter_smoothing_method, choices=["same_as_main", "savitzky_golay", "moving_average"])
    parser.add_argument("--jitter-smoothing-window-sec", type=float, default=DEFAULT_CONFIG.jitter_smoothing_window_sec)
    parser.add_argument("--jitter-smoothing-polyorder", type=int, default=DEFAULT_CONFIG.jitter_smoothing_polyorder)
    parser.add_argument("--pearson-min-valid-samples", type=int, default=DEFAULT_CONFIG.pearson_min_valid_samples)
    parser.add_argument("--bland-altman-min-pairs", type=int, default=DEFAULT_CONFIG.bland_altman_min_pairs)
    parser.add_argument("--disable-plots", action="store_true")
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR, help="Aligned CSV input directory.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Directory for evaluation CSV outputs.")
    parser.add_argument("--plot-output-dir", type=Path, default=None, help="Optional directory for diagnostic plot PNGs.")
    return parser.parse_args()


def build_config(args: argparse.Namespace):
    thresholds = tuple(float(value) for value in args.pck_thresholds.split(",") if value.strip())
    return replace(
        DEFAULT_CONFIG,
        evaluation_fps=float(args.evaluation_fps),
        selected_sides=tuple(args.sides),
        selected_joints=tuple(args.joints),
        enabled_metric_groups=tuple(args.metric_groups),
        rotation_normalization=bool(args.rotation_normalization),
        normalize_body_axes=not bool(args.disable_body_axis_normalization),
        similarity_alignment=not bool(args.disable_similarity_alignment),
        torso_similarity_alignment=not bool(args.disable_torso_similarity_alignment),
        smoothing_method=str(args.smoothing_method),
        smoothing_window_sec=float(args.smoothing_window_sec),
        smoothing_polyorder=int(args.smoothing_polyorder),
        pck_thresholds=thresholds or DEFAULT_CONFIG.pck_thresholds,
        enable_pearson_metrics=not bool(args.disable_pearson),
        enable_jitter_metrics=not bool(args.disable_jitter),
        enable_bland_altman=not bool(args.disable_bland_altman),
        jitter_smoothing_method=str(args.jitter_smoothing_method),
        jitter_smoothing_window_sec=float(args.jitter_smoothing_window_sec),
        jitter_smoothing_polyorder=int(args.jitter_smoothing_polyorder),
        pearson_min_valid_samples=int(args.pearson_min_valid_samples),
        bland_altman_min_pairs=int(args.bland_altman_min_pairs),
        generate_plots=bool(args.plot_output_dir) and not bool(args.disable_plots),
    )


OUTPUT_FILES = (
    "metrics_long.csv",
    "normalization_summary.csv",
    "warnings.csv",
    "joint_availability.csv",
    "summary_by_trial.csv",
    "summary_by_module.csv",
    "summary_by_domain.csv",
    "bland_altman.csv",
    "plot_manifest.csv",
)


def prepare_output_dir(output_dir: Path, plot_output_dir: Path | None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for csv_name in OUTPUT_FILES:
        path = output_dir / csv_name
        if path.exists():
            path.unlink()
    if plot_output_dir is not None:
        if plot_output_dir.exists():
            shutil.rmtree(plot_output_dir)
        plot_output_dir.mkdir(parents=True, exist_ok=True)
    else:
        stale_plot_dir = output_dir / "plots"
        if stale_plot_dir.exists():
            shutil.rmtree(stale_plot_dir)


def summarize_metrics(metrics_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if metrics_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if "system_name" not in metrics_df.columns:
        metrics_df = metrics_df.assign(system_name="pair")

    trial_summary = (
        metrics_df.groupby(
            ["comparison_name", "pair_key", "trial_id", "wrist_variant", "normalization_mode", "alignment_mode", "system_name", "metric_family"],
            dropna=False,
        )["metric_value"]
        .mean()
        .reset_index()
        .rename(columns={"metric_value": "mean_metric_value"})
    )

    module_summary = (
        metrics_df.groupby(
            ["comparison_name", "wrist_variant", "alignment_mode", "system_name", "metric_family", "metric_name"],
            dropna=False,
        )["metric_value"]
        .agg(["mean", "median", "min", "max"])
        .reset_index()
        .rename(columns={"mean": "mean_metric_value", "median": "median_metric_value", "min": "min_metric_value", "max": "max_metric_value"})
    )

    domain_metrics = metrics_df.copy()
    domain_metrics["metric_domain"] = domain_metrics["metric_family"].map(
        {
            "upper_limb": "hand_arm",
            "lower_limb": "knee_pedal",
        }
    ).fillna(domain_metrics["metric_family"])
    domain_summary = (
        domain_metrics.groupby(
            ["comparison_name", "pair_key", "trial_id", "wrist_variant", "alignment_mode", "system_name", "metric_domain", "side", "metric_name"],
            dropna=False,
        )["metric_value"]
        .agg(["mean", "median", "min", "max", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_metric_value", "median": "median_metric_value", "min": "min_metric_value", "max": "max_metric_value", "count": "valid_metric_count"})
    )
    return trial_summary, module_summary, domain_summary


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    plot_output_dir = args.plot_output_dir.resolve() if args.plot_output_dir else None
    analysis_config.INPUT_DIR = input_dir
    config = build_config(args)
    prepare_output_dir(output_dir, plot_output_dir)

    metrics_rows: list[dict[str, object]] = []
    normalization_rows: list[dict[str, object]] = []
    warning_rows: list[dict[str, object]] = []
    availability_rows: list[dict[str, object]] = []
    plot_rows: list[dict[str, object]] = []
    bland_altman_rows: list[dict[str, object]] = []

    for wrist_variant in ("body", "hand"):
        joint_mappings = build_joint_mappings(wrist_variant)
        for comparison in COMPARISON_SPECS:
            joint_frames_a, availability_a, warnings_a = load_joint_frames(comparison.system_a, comparison.recording_a, joint_mappings)
            joint_frames_b, availability_b, warnings_b = load_joint_frames(comparison.system_b, comparison.recording_b, joint_mappings)
            shared_time = build_shared_time_grid(joint_frames_a, joint_frames_b, config.evaluation_fps)
            if shared_time.size == 0:
                warning_rows.append(
                    {
                        "comparison_name": comparison.comparison_name,
                        "pair_key": comparison.pair_key,
                        "trial_id": comparison.trial_id,
                        "wrist_variant": wrist_variant,
                        "system_name": "pair",
                        "alignment_mode": "body_only",
                        "warning": "No shared time window between the selected systems.",
                    }
                )
                continue

            trial_a = resample_trial_data(joint_frames_a, availability_a, warnings_a, shared_time)
            trial_b = resample_trial_data(joint_frames_b, availability_b, warnings_b, shared_time)

            for joint_id, available in availability_a.items():
                availability_rows.append(
                    {
                        "comparison_name": comparison.comparison_name,
                        "pair_key": comparison.pair_key,
                        "trial_id": comparison.trial_id,
                        "wrist_variant": wrist_variant,
                        "system_name": comparison.system_a,
                        "joint_id": joint_id,
                        "available": available,
                    }
                )
            for joint_id, available in availability_b.items():
                availability_rows.append(
                    {
                        "comparison_name": comparison.comparison_name,
                        "pair_key": comparison.pair_key,
                        "trial_id": comparison.trial_id,
                        "wrist_variant": wrist_variant,
                        "system_name": comparison.system_b,
                        "joint_id": joint_id,
                        "available": available,
                    }
                )

            reference_trial = trial_a if comparison.reference_system == comparison.system_a else trial_b
            candidate_trial = trial_b if comparison.reference_system == comparison.system_a else trial_a
            candidate_system = comparison.system_b if comparison.reference_system == comparison.system_a else comparison.system_a

            reference_normalized = normalize_trial_coordinates(
                reference_trial.coordinates,
                apply_rotation=config.rotation_normalization,
                origin_priority=config.body_origin_priority,
                scale_priority=config.body_scale_priority,
                normalize_axes=config.normalize_body_axes,
            )
            candidate_normalized = normalize_trial_coordinates(
                candidate_trial.coordinates,
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
                output_plot_dir=plot_output_dir or (output_dir / "plots"),
            )
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
    availability_df = pd.DataFrame(availability_rows)
    plots_df = pd.DataFrame(plot_rows)
    bland_altman_df = pd.DataFrame(bland_altman_rows)
    trial_summary_df, module_summary_df, domain_summary_df = summarize_metrics(metrics_df)

    metrics_df.to_csv(output_dir / "metrics_long.csv", index=False)
    normalization_df.to_csv(output_dir / "normalization_summary.csv", index=False)
    warnings_df.to_csv(output_dir / "warnings.csv", index=False)
    availability_df.to_csv(output_dir / "joint_availability.csv", index=False)
    trial_summary_df.to_csv(output_dir / "summary_by_trial.csv", index=False)
    module_summary_df.to_csv(output_dir / "summary_by_module.csv", index=False)
    domain_summary_df.to_csv(output_dir / "summary_by_domain.csv", index=False)
    bland_altman_df.to_csv(output_dir / "bland_altman.csv", index=False)
    plots_df.to_csv(output_dir / "plot_manifest.csv", index=False)
    print(f"Wrote evaluation outputs under {output_dir}")


if __name__ == "__main__":
    main()
