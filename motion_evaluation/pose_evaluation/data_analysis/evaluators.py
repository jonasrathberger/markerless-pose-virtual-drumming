from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_figures import THESIS_STYLE, apply_thesis_style, save_single_figure, style_axis

from data_analysis.config import EvaluationConfig, SYSTEM_LABELS
from data_analysis.events import detect_events, match_event_times
from data_analysis.metrics import (
    bland_altman_paired_values,
    bland_altman_stats,
    inter_event_interval_rmse,
    mpjpe_2d,
    pck_2d,
    pearson_correlation,
    principal_axis,
    project_onto_axis,
    residual_jitter_stats,
    speed,
)
from data_analysis.normalization import (
    NormalizationResult,
    apply_similarity_transform_to_trial,
    fit_similarity_transform,
)
from data_analysis.smoothing import SmoothingResult


os.environ.setdefault("MPLCONFIGDIR", str((REPO_ROOT / ".cache" / "matplotlib").resolve()))

UPPER_LIMB_ENDPOINT_JOINTS = ("wrist", "pinky_knuckle")


@dataclass(frozen=True)
class PairArtifacts:
    metrics_rows: list[dict[str, object]]
    normalization_rows: list[dict[str, object]]
    warning_rows: list[dict[str, object]]
    plot_rows: list[dict[str, object]]
    bland_altman_rows: list[dict[str, object]]


def evaluate_pair(
    comparison_name: str,
    pair_key: str,
    trial_id: str,
    wrist_variant: str,
    reference_system: str,
    candidate_system: str,
    time_sec: np.ndarray,
    reference_normalized: NormalizationResult,
    candidate_normalized: NormalizationResult,
    config: EvaluationConfig,
    output_plot_dir: Path,
) -> PairArtifacts:
    metrics_rows: list[dict[str, object]] = []
    normalization_rows: list[dict[str, object]] = []
    warning_rows: list[dict[str, object]] = []
    plot_rows: list[dict[str, object]] = []
    bland_altman_rows: list[dict[str, object]] = []

    for system_name, result in ((reference_system, reference_normalized), (candidate_system, candidate_normalized)):
        normalization_rows.append(
            build_normalization_row(
                comparison_name=comparison_name,
                pair_key=pair_key,
                trial_id=trial_id,
                wrist_variant=wrist_variant,
                system_name=system_name,
                result=result,
            )
        )
        for warning in result.warnings:
            warning_rows.append(build_warning_row(comparison_name, pair_key, trial_id, wrist_variant, system_name, warning))

    for alignment_mode in config.alignment_modes:
        if alignment_mode == "body_only":
            aligned_candidate_coords = candidate_normalized.coordinates
            similarity_warnings: tuple[str, ...] = tuple()
        else:
            similarity_joint_ids = config.torso_similarity_joint_ids if alignment_mode == "body_plus_torso_similarity" else None
            similarity_transform = fit_similarity_transform(
                reference_normalized.coordinates,
                candidate_normalized.coordinates,
                joint_ids=similarity_joint_ids,
            )
            aligned_candidate_coords = apply_similarity_transform_to_trial(candidate_normalized.coordinates, similarity_transform)
            similarity_warnings = similarity_transform.warnings
            normalization_rows.append(
                {
                    "comparison_name": comparison_name,
                    "pair_key": pair_key,
                    "trial_id": trial_id,
                    "wrist_variant": wrist_variant,
                    "system_name": candidate_system,
                    "system_label": SYSTEM_LABELS[candidate_system],
                    "normalization_mode": "body_centered_2d",
                    "alignment_mode": alignment_mode,
                    "origin_source": candidate_normalized.origin_source,
                    "scale_source": candidate_normalized.scale_source,
                    "scale_value": candidate_normalized.scale,
                    "rotation_source": candidate_normalized.rotation_source,
                    "rotation_angle_deg": math.degrees(candidate_normalized.rotation_angle_rad),
                    "orientation_x_sign": candidate_normalized.orientation_x_sign,
                    "orientation_y_sign": candidate_normalized.orientation_y_sign,
                    "orientation_source": candidate_normalized.orientation_source,
                    "similarity_scale": similarity_transform.scale,
                    "similarity_rotation_deg": math.degrees(math.atan2(similarity_transform.rotation[1, 0], similarity_transform.rotation[0, 0])),
                    "similarity_translation_x": similarity_transform.translation[0],
                    "similarity_translation_y": similarity_transform.translation[1],
                }
            )
            for warning in similarity_warnings:
                warning_rows.append(build_warning_row(comparison_name, pair_key, trial_id, wrist_variant, candidate_system, warning))

        reference_coords = reference_normalized.coordinates
        candidate_coords = aligned_candidate_coords

        reference_smoothed = smooth_coordinates(reference_coords, time_sec, config)
        candidate_smoothed = smooth_coordinates(candidate_coords, time_sec, config)
        for warning in reference_smoothed.warnings:
            warning_rows.append(build_warning_row(comparison_name, pair_key, trial_id, wrist_variant, reference_system, warning, alignment_mode))
        for warning in candidate_smoothed.warnings:
            warning_rows.append(build_warning_row(comparison_name, pair_key, trial_id, wrist_variant, candidate_system, warning, alignment_mode))

        available_joint_ids = sorted(set(reference_coords) & set(candidate_coords))
        available_joint_ids = [
            joint_id
            for joint_id in available_joint_ids
            if joint_side(joint_id) in config.selected_sides and joint_name(joint_id) in config.selected_joints
        ]

        if "core_spatial" in config.enabled_metric_groups:
            metrics_rows.extend(
                evaluate_core_spatial_metrics(
                    comparison_name,
                    pair_key,
                    trial_id,
                    wrist_variant,
                    alignment_mode,
                    reference_coords,
                    candidate_coords,
                    available_joint_ids,
                    config,
                )
            )
        if config.enable_pearson_metrics and "pearson" in config.enabled_metric_groups:
            pearson_rows, pearson_warnings = evaluate_pearson_metrics(
                comparison_name,
                pair_key,
                trial_id,
                wrist_variant,
                alignment_mode,
                reference_coords,
                candidate_coords,
                reference_smoothed,
                candidate_smoothed,
                available_joint_ids,
                config,
            )
            metrics_rows.extend(pearson_rows)
            warning_rows.extend(
                build_warning_row(comparison_name, pair_key, trial_id, wrist_variant, "pair", warning, alignment_mode)
                for warning in pearson_warnings
            )
        if config.enable_jitter_metrics and "jitter" in config.enabled_metric_groups:
            metrics_rows.extend(
                evaluate_jitter_metrics(
                    comparison_name,
                    pair_key,
                    trial_id,
                    wrist_variant,
                    alignment_mode,
                    reference_system,
                    candidate_system,
                    time_sec,
                    reference_coords,
                    candidate_coords,
                    reference_smoothed,
                    candidate_smoothed,
                    available_joint_ids,
                    config,
                )
            )
        if "upper_limb" in config.enabled_metric_groups:
            upper_rows, upper_warnings, upper_plots = evaluate_upper_limb_metrics(
                comparison_name,
                pair_key,
                trial_id,
                wrist_variant,
                alignment_mode,
                time_sec,
                reference_coords,
                candidate_coords,
                reference_smoothed,
                candidate_smoothed,
                output_plot_dir,
                config,
            )
            metrics_rows.extend(upper_rows)
            warning_rows.extend(
                build_warning_row(comparison_name, pair_key, trial_id, wrist_variant, reference_system, warning, alignment_mode)
                for warning in upper_warnings
            )
            plot_rows.extend(upper_plots)
        if "lower_limb" in config.enabled_metric_groups:
            lower_rows, lower_warnings, lower_plots = evaluate_lower_limb_metrics(
                comparison_name,
                pair_key,
                trial_id,
                wrist_variant,
                alignment_mode,
                time_sec,
                reference_coords,
                candidate_coords,
                reference_smoothed,
                candidate_smoothed,
                output_plot_dir,
                config,
            )
            metrics_rows.extend(lower_rows)
            warning_rows.extend(
                build_warning_row(comparison_name, pair_key, trial_id, wrist_variant, reference_system, warning, alignment_mode)
                for warning in lower_warnings
            )
            plot_rows.extend(lower_plots)
        if config.enable_bland_altman and "bland_altman" in config.enabled_metric_groups:
            ba_rows, ba_warnings = evaluate_bland_altman_metrics(
                comparison_name,
                pair_key,
                trial_id,
                wrist_variant,
                alignment_mode,
                reference_system,
                candidate_system,
                time_sec,
                reference_coords,
                candidate_coords,
                reference_smoothed,
                candidate_smoothed,
                config,
            )
            bland_altman_rows.extend(ba_rows)
            warning_rows.extend(
                build_warning_row(comparison_name, pair_key, trial_id, wrist_variant, "pair", warning, alignment_mode)
                for warning in ba_warnings
            )

    return PairArtifacts(metrics_rows, normalization_rows, warning_rows, plot_rows, bland_altman_rows)


def smooth_coordinates(coordinates: dict[str, np.ndarray], time_sec: np.ndarray, config: EvaluationConfig) -> SmoothingResult:
    from data_analysis.smoothing import smooth_and_differentiate

    return smooth_and_differentiate(
        coordinates=coordinates,
        time_sec=time_sec,
        method=config.smoothing_method,
        window_sec=config.smoothing_window_sec,
        polyorder=config.smoothing_polyorder,
    )


def evaluate_core_spatial_metrics(
    comparison_name: str,
    pair_key: str,
    trial_id: str,
    wrist_variant: str,
    alignment_mode: str,
    reference_coords: dict[str, np.ndarray],
    candidate_coords: dict[str, np.ndarray],
    available_joint_ids: list[str],
    config: EvaluationConfig,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for side in config.selected_sides:
        side_joint_ids = [joint_id for joint_id in available_joint_ids if joint_side(joint_id) == side]
        if not side_joint_ids:
            continue
        side_reference = {joint_id: reference_coords[joint_id] for joint_id in side_joint_ids}
        side_candidate = {joint_id: candidate_coords[joint_id] for joint_id in side_joint_ids}
        rows.append(metric_row(comparison_name, pair_key, trial_id, wrist_variant, side, "all", "core_spatial", "mpjpe_2d", mpjpe_2d(side_reference, side_candidate), alignment_mode))
        for threshold in config.pck_thresholds:
            rows.append(metric_row(comparison_name, pair_key, trial_id, wrist_variant, side, "all", "core_spatial", f"pck_2d@{threshold:g}", pck_2d(side_reference, side_candidate, threshold), alignment_mode))
    if available_joint_ids:
        selected_reference = {joint_id: reference_coords[joint_id] for joint_id in available_joint_ids}
        selected_candidate = {joint_id: candidate_coords[joint_id] for joint_id in available_joint_ids}
        rows.append(
            metric_row(
                comparison_name,
                pair_key,
                trial_id,
                wrist_variant,
                "both",
                "all",
                "core_spatial",
                "mpjpe_2d",
                mpjpe_2d(selected_reference, selected_candidate),
                alignment_mode,
            )
        )
        for threshold in config.pck_thresholds:
            rows.append(metric_row(comparison_name, pair_key, trial_id, wrist_variant, "both", "all", "core_spatial", f"pck_2d@{threshold:g}", pck_2d(selected_reference, selected_candidate, threshold), alignment_mode))
    return rows


def evaluate_pearson_metrics(
    comparison_name: str,
    pair_key: str,
    trial_id: str,
    wrist_variant: str,
    alignment_mode: str,
    reference_coords: dict[str, np.ndarray],
    candidate_coords: dict[str, np.ndarray],
    reference_smoothed: SmoothingResult,
    candidate_smoothed: SmoothingResult,
    available_joint_ids: list[str],
    config: EvaluationConfig,
) -> tuple[list[dict[str, object]], list[str]]:
    rows: list[dict[str, object]] = []
    warnings: list[str] = []

    for joint_id in available_joint_ids:
        side = joint_side(joint_id)
        joint = joint_name(joint_id)
        if joint not in UPPER_LIMB_ENDPOINT_JOINTS:
            continue
        reference_velocity = reference_smoothed.velocities.get(joint_id)
        candidate_velocity = candidate_smoothed.velocities.get(joint_id)
        if reference_velocity is not None and candidate_velocity is not None:
            rows.append(metric_row(comparison_name, pair_key, trial_id, wrist_variant, side, joint, "pearson", "pearson_velocity", pearson_signal(speed(reference_velocity), speed(candidate_velocity), config), alignment_mode))

    for side in config.selected_sides:
        shoulder = f"shoulder_{side}"
        elbow = f"elbow_{side}"
        wrist = f"wrist_{side}"
        if all(joint_id in reference_coords and joint_id in candidate_coords for joint_id in (shoulder, elbow, wrist)):
            angle_signals = (
                ("upper_arm_angle", segment_angle(reference_coords[shoulder], reference_coords[elbow]), segment_angle(candidate_coords[shoulder], candidate_coords[elbow])),
                ("forearm_angle", segment_angle(reference_coords[elbow], reference_coords[wrist]), segment_angle(candidate_coords[elbow], candidate_coords[wrist])),
                ("elbow_angle", internal_angle(reference_coords[shoulder], reference_coords[elbow], reference_coords[wrist]), internal_angle(candidate_coords[shoulder], candidate_coords[elbow], candidate_coords[wrist])),
            )
            for signal_name, reference_signal, candidate_signal in angle_signals:
                rows.append(metric_row(comparison_name, pair_key, trial_id, wrist_variant, side, signal_name, "pearson", "pearson_angle", pearson_signal(reference_signal, candidate_signal, config), alignment_mode))

        knee = f"knee_{side}"
        hip = f"hip_{side}"
        ankle = f"ankle_{side}"
        if knee in reference_coords and knee in candidate_coords:
            axis_vector, _orth_vector, explained_ratio = principal_axis(reference_coords[knee])
            if not np.isfinite(explained_ratio):
                warnings.append(f"Skipped projected knee Pearson metrics for {side}: invalid principal axis.")
            else:
                reference_knee_velocity = reference_smoothed.velocities.get(knee)
                candidate_knee_velocity = candidate_smoothed.velocities.get(knee)
                if reference_knee_velocity is not None and candidate_knee_velocity is not None:
                    rows.append(
                        metric_row(
                            comparison_name,
                            pair_key,
                            trial_id,
                            wrist_variant,
                            side,
                            "knee_along_axis_velocity",
                            "pearson",
                            "pearson_along_axis_velocity",
                            pearson_signal(project_onto_axis(reference_knee_velocity, axis_vector), project_onto_axis(candidate_knee_velocity, axis_vector), config),
                            alignment_mode,
                        )
                    )
        if hip in reference_coords and hip in candidate_coords and knee in reference_coords and knee in candidate_coords:
            rows.append(
                metric_row(
                    comparison_name,
                    pair_key,
                    trial_id,
                    wrist_variant,
                    side,
                    "thigh_angle",
                    "pearson",
                    "pearson_angle",
                    pearson_signal(segment_angle(reference_coords[hip], reference_coords[knee]), segment_angle(candidate_coords[hip], candidate_coords[knee]), config),
                    alignment_mode,
                )
            )
        if hip in reference_coords and hip in candidate_coords and knee in reference_coords and knee in candidate_coords and ankle in reference_coords and ankle in candidate_coords:
            rows.append(
                metric_row(
                    comparison_name,
                    pair_key,
                    trial_id,
                    wrist_variant,
                    side,
                    "knee_angle",
                    "pearson",
                    "pearson_angle",
                    pearson_signal(internal_angle(reference_coords[hip], reference_coords[knee], reference_coords[ankle]), internal_angle(candidate_coords[hip], candidate_coords[knee], candidate_coords[ankle]), config),
                    alignment_mode,
                )
            )
    return rows, warnings


def evaluate_jitter_metrics(
    comparison_name: str,
    pair_key: str,
    trial_id: str,
    wrist_variant: str,
    alignment_mode: str,
    reference_system: str,
    candidate_system: str,
    time_sec: np.ndarray,
    reference_coords: dict[str, np.ndarray],
    candidate_coords: dict[str, np.ndarray],
    reference_smoothed: SmoothingResult,
    candidate_smoothed: SmoothingResult,
    available_joint_ids: list[str],
    config: EvaluationConfig,
) -> list[dict[str, object]]:
    reference_jitter_smoothed = jitter_smoothed_coordinates(reference_coords, reference_smoothed, time_sec, config)
    candidate_jitter_smoothed = jitter_smoothed_coordinates(candidate_coords, candidate_smoothed, time_sec, config)
    rows: list[dict[str, object]] = []

    for system_name, coordinates, smoothed in (
        (reference_system, reference_coords, reference_jitter_smoothed),
        (candidate_system, candidate_coords, candidate_jitter_smoothed),
    ):
        for joint_id in available_joint_ids:
            joint = joint_name(joint_id)
            if joint not in UPPER_LIMB_ENDPOINT_JOINTS:
                continue
            rows.extend(jitter_signal_rows(comparison_name, pair_key, trial_id, wrist_variant, alignment_mode, system_name, joint_side(joint_id), joint, "jitter_x", coordinates[joint_id][:, 0], smoothed.smoothed[joint_id][:, 0]))
            rows.extend(jitter_signal_rows(comparison_name, pair_key, trial_id, wrist_variant, alignment_mode, system_name, joint_side(joint_id), joint, "jitter_y", coordinates[joint_id][:, 1], smoothed.smoothed[joint_id][:, 1]))

        for side in config.selected_sides:
            knee = f"knee_{side}"
            if knee not in coordinates or knee not in smoothed.smoothed:
                continue
            axis_vector, orth_vector, explained_ratio = principal_axis(coordinates[knee])
            if not np.isfinite(explained_ratio):
                continue
            rows.extend(
                jitter_signal_rows(
                    comparison_name,
                    pair_key,
                    trial_id,
                    wrist_variant,
                    alignment_mode,
                    system_name,
                    side,
                    "knee_along_axis",
                    "jitter_along_axis",
                    project_onto_axis(coordinates[knee], axis_vector),
                    project_onto_axis(smoothed.smoothed[knee], axis_vector),
                )
            )
            rows.extend(
                jitter_signal_rows(
                    comparison_name,
                    pair_key,
                    trial_id,
                    wrist_variant,
                    alignment_mode,
                    system_name,
                    side,
                    "knee_orthogonal_axis",
                    "jitter_orthogonal_axis",
                    project_onto_axis(coordinates[knee], orth_vector),
                    project_onto_axis(smoothed.smoothed[knee], orth_vector),
                )
            )
    return rows


def evaluate_bland_altman_metrics(
    comparison_name: str,
    pair_key: str,
    trial_id: str,
    wrist_variant: str,
    alignment_mode: str,
    reference_system: str,
    candidate_system: str,
    time_sec: np.ndarray,
    reference_coords: dict[str, np.ndarray],
    candidate_coords: dict[str, np.ndarray],
    reference_smoothed: SmoothingResult,
    candidate_smoothed: SmoothingResult,
    config: EvaluationConfig,
) -> tuple[list[dict[str, object]], list[str]]:
    rows: list[dict[str, object]] = []
    warnings: list[str] = []

    for side in config.selected_sides:
        for endpoint_joint in selected_upper_limb_endpoints(config):
            endpoint = f"{endpoint_joint}_{side}"
            if endpoint not in reference_coords or endpoint not in candidate_coords:
                continue
            reference_endpoint_velocity = reference_smoothed.velocities.get(endpoint)
            candidate_endpoint_velocity = candidate_smoothed.velocities.get(endpoint)
            if reference_endpoint_velocity is None or candidate_endpoint_velocity is None:
                continue
            reference_events = detect_events(reference_smoothed.smoothed[endpoint][:, 1], reference_endpoint_velocity[:, 1], time_sec, config.event_min_distance_sec, config.event_prominence_ratio)
            candidate_events = detect_events(candidate_smoothed.smoothed[endpoint][:, 1], candidate_endpoint_velocity[:, 1], time_sec, config.event_min_distance_sec, config.event_prominence_ratio)
            rows.extend(
                build_bland_altman_rows_for_signals(
                    comparison_name,
                    pair_key,
                    trial_id,
                    wrist_variant,
                    alignment_mode,
                    reference_system,
                    candidate_system,
                    side,
                    "upper_limb",
                    (
                        (
                            f"{endpoint_joint}_excursion",
                            "cycle_level",
                            compute_excursions(reference_smoothed.smoothed[endpoint][:, 1], reference_events),
                            compute_excursions(candidate_smoothed.smoothed[endpoint][:, 1], candidate_events),
                        ),
                        (
                            f"{endpoint_joint}_peak_speed",
                            "cycle_level",
                            cycle_stat(speed(reference_endpoint_velocity), reference_events.reversal_indices, np.nanmax),
                            cycle_stat(speed(candidate_endpoint_velocity), candidate_events.reversal_indices, np.nanmax),
                        ),
                        (
                            f"{endpoint_joint}_mean_speed",
                            "cycle_level",
                            cycle_stat(speed(reference_endpoint_velocity), reference_events.reversal_indices, np.nanmean),
                            cycle_stat(speed(candidate_endpoint_velocity), candidate_events.reversal_indices, np.nanmean),
                        ),
                    ),
                    config,
                    warnings,
                )
            )

        knee = f"knee_{side}"
        if knee in reference_coords and knee in candidate_coords:
            axis_vector, _orth_vector, explained_ratio = principal_axis(reference_coords[knee])
            reference_knee_velocity = reference_smoothed.velocities.get(knee)
            candidate_knee_velocity = candidate_smoothed.velocities.get(knee)
            if np.isfinite(explained_ratio) and reference_knee_velocity is not None and candidate_knee_velocity is not None:
                reference_along = project_onto_axis(reference_smoothed.smoothed[knee], axis_vector)
                candidate_along = project_onto_axis(candidate_smoothed.smoothed[knee], axis_vector)
                reference_along_velocity = project_onto_axis(reference_knee_velocity, axis_vector)
                candidate_along_velocity = project_onto_axis(candidate_knee_velocity, axis_vector)
                reference_events = detect_events(reference_along, reference_along_velocity, time_sec, config.event_min_distance_sec, config.event_prominence_ratio)
                candidate_events = detect_events(candidate_along, candidate_along_velocity, time_sec, config.event_min_distance_sec, config.event_prominence_ratio)
                rows.extend(
                    build_bland_altman_rows_for_signals(
                        comparison_name,
                        pair_key,
                        trial_id,
                        wrist_variant,
                        alignment_mode,
                        reference_system,
                        candidate_system,
                        side,
                        "lower_limb",
                        (
                            ("knee_along_axis_excursion", "cycle_level", compute_excursions(reference_along, reference_events), compute_excursions(candidate_along, candidate_events)),
                            ("knee_peak_along_axis_velocity", "cycle_level", cycle_stat(np.abs(reference_along_velocity), reference_events.reversal_indices, np.nanmax), cycle_stat(np.abs(candidate_along_velocity), candidate_events.reversal_indices, np.nanmax)),
                            ("knee_mean_abs_along_axis_velocity", "cycle_level", cycle_stat(np.abs(reference_along_velocity), reference_events.reversal_indices, np.nanmean), cycle_stat(np.abs(candidate_along_velocity), candidate_events.reversal_indices, np.nanmean)),
                        ),
                        config,
                        warnings,
                    )
                )
            else:
                warnings.append(f"Skipped knee Bland-Altman metrics for {side}: invalid principal axis or missing velocity.")
    return rows, warnings


def pearson_signal(reference: np.ndarray, candidate: np.ndarray, config: EvaluationConfig) -> float:
    return pearson_correlation(
        reference,
        candidate,
        min_valid_samples=config.pearson_min_valid_samples,
        min_std=config.pearson_min_std,
    )


def jitter_smoothed_coordinates(
    coordinates: dict[str, np.ndarray],
    main_smoothed: SmoothingResult,
    time_sec: np.ndarray,
    config: EvaluationConfig,
) -> SmoothingResult:
    if config.jitter_smoothing_method == "same_as_main":
        return main_smoothed
    from data_analysis.smoothing import smooth_and_differentiate

    return smooth_and_differentiate(
        coordinates=coordinates,
        time_sec=time_sec,
        method=config.jitter_smoothing_method,
        window_sec=config.jitter_smoothing_window_sec,
        polyorder=config.jitter_smoothing_polyorder,
    )


def jitter_signal_rows(
    comparison_name: str,
    pair_key: str,
    trial_id: str,
    wrist_variant: str,
    alignment_mode: str,
    system_name: str,
    side: str,
    joint: str,
    metric_name: str,
    raw_signal: np.ndarray,
    smoothed_signal: np.ndarray,
) -> list[dict[str, object]]:
    stats = residual_jitter_stats(raw_signal, smoothed_signal)
    return [
        metric_row(comparison_name, pair_key, trial_id, wrist_variant, side, joint, "jitter", metric_name, stats["jitter_rms"], alignment_mode, system_name=system_name),
    ]


def build_bland_altman_rows_for_signals(
    comparison_name: str,
    pair_key: str,
    trial_id: str,
    wrist_variant: str,
    alignment_mode: str,
    reference_system: str,
    candidate_system: str,
    side: str,
    variable_family: str,
    signal_specs: tuple[tuple[str, str, np.ndarray, np.ndarray], ...],
    config: EvaluationConfig,
    warnings: list[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for variable_name, analysis_level, reference_values, candidate_values in signal_specs:
        reference_paired, candidate_paired = paired_prefix(reference_values, candidate_values)
        n_pairs = int(np.sum(np.isfinite(reference_paired) & np.isfinite(candidate_paired)))
        if n_pairs < config.bland_altman_min_pairs:
            warnings.append(f"Skipped Bland-Altman {variable_name} for {side}: only {n_pairs} paired samples.")
            continue
        rows.append(
            bland_altman_row(
                comparison_name,
                pair_key,
                trial_id,
                wrist_variant,
                alignment_mode,
                reference_system,
                candidate_system,
                side,
                variable_family,
                variable_name,
                analysis_level,
                reference_paired,
                candidate_paired,
            )
        )
    return rows


def bland_altman_row(
    comparison_name: str,
    pair_key: str,
    trial_id: str,
    wrist_variant: str,
    alignment_mode: str,
    reference_system: str,
    candidate_system: str,
    side: str,
    variable_family: str,
    variable_name: str,
    analysis_level: str,
    reference_values: np.ndarray,
    candidate_values: np.ndarray,
) -> dict[str, object]:
    mean_of_methods, differences = bland_altman_paired_values(reference_values, candidate_values)
    stats = bland_altman_stats(reference_values, candidate_values)
    return {
        "comparison_name": comparison_name,
        "pair_key": pair_key,
        "trial_id": trial_id,
        "wrist_variant": wrist_variant,
        "side": side,
        "reference_system": reference_system,
        "candidate_system": candidate_system,
        "variable_family": variable_family,
        "variable_name": variable_name,
        "analysis_level": analysis_level,
        "normalization_mode": "body_centered_2d",
        "alignment_mode": alignment_mode,
        "n_pairs": int(len(differences)),
        "mean_of_methods": safe_mean(mean_of_methods),
        "mean_difference": safe_mean(differences),
        "bias": stats["bias"],
        "std_diff": stats["sd_diff"],
        "loa_lower": stats["loa_lower"],
        "loa_upper": stats["loa_upper"],
    }


def paired_prefix(reference_values: np.ndarray, candidate_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference_values = np.asarray(reference_values, dtype=float)
    candidate_values = np.asarray(candidate_values, dtype=float)
    length = min(len(reference_values), len(candidate_values))
    if length == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    return reference_values[:length], candidate_values[:length]


def cycle_stat(signal: np.ndarray, event_indices: np.ndarray, reducer) -> np.ndarray:
    if event_indices.size < 2:
        return np.array([], dtype=float)
    values: list[float] = []
    for start, end in zip(event_indices[:-1], event_indices[1:]):
        if end <= start:
            continue
        segment = signal[start:end + 1]
        if not np.isfinite(segment).any():
            continue
        values.append(float(reducer(segment)))
    return np.asarray(values, dtype=float)


def evaluate_upper_limb_metrics(
    comparison_name: str,
    pair_key: str,
    trial_id: str,
    wrist_variant: str,
    alignment_mode: str,
    time_sec: np.ndarray,
    reference_coords: dict[str, np.ndarray],
    candidate_coords: dict[str, np.ndarray],
    reference_smoothed: SmoothingResult,
    candidate_smoothed: SmoothingResult,
    output_plot_dir: Path,
    config: EvaluationConfig,
) -> tuple[list[dict[str, object]], list[str], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    warnings: list[str] = []
    plot_rows: list[dict[str, object]] = []
    endpoint_joints = selected_upper_limb_endpoints(config)
    if not endpoint_joints:
        warnings.append("Skipped upper-limb metrics: selected joints do not include wrist or pinky_knuckle.")
        return rows, warnings, plot_rows

    for side in config.selected_sides:
        for endpoint_joint in endpoint_joints:
            endpoint = f"{endpoint_joint}_{side}"
            if endpoint not in reference_coords or endpoint not in candidate_coords:
                warnings.append(f"Skipped upper-limb metrics for {side} {endpoint_joint}: missing endpoint.")
                continue
            reference_endpoint_velocity = reference_smoothed.velocities.get(endpoint)
            candidate_endpoint_velocity = candidate_smoothed.velocities.get(endpoint)
            if reference_endpoint_velocity is None or candidate_endpoint_velocity is None:
                warnings.append(f"Skipped upper-limb metrics for {side} {endpoint_joint}: missing velocity.")
                continue

            reference_events = detect_events(
                reference_smoothed.smoothed[endpoint][:, 1],
                reference_endpoint_velocity[:, 1],
                time_sec,
                config.event_min_distance_sec,
                config.event_prominence_ratio,
            )
            candidate_events = detect_events(
                candidate_smoothed.smoothed[endpoint][:, 1],
                candidate_endpoint_velocity[:, 1],
                time_sec,
                config.event_min_distance_sec,
                config.event_prominence_ratio,
            )
            rows.extend(
                event_metric_rows(
                    comparison_name,
                    pair_key,
                    trial_id,
                    wrist_variant,
                    side,
                    endpoint_joint,
                    "upper_limb",
                    alignment_mode,
                    time_sec,
                    reference_events,
                    candidate_events,
                    config.event_match_tolerance_sec,
                )
            )

            if config.generate_plots:
                output_path = output_plot_dir / f"{pair_key}__{wrist_variant}__{alignment_mode}__upper_limb_{endpoint_joint}_{side}.png"
                save_joint_overlay_plot(output_path, time_sec, reference_coords[endpoint], candidate_coords[endpoint], title=f"{pair_key} | {endpoint_joint} {side} | {alignment_mode}", ylabel_prefix=endpoint_joint)
                plot_rows.append({"plot_type": "upper_limb_overlay", "plot_path": str(output_path)})
    return rows, warnings, plot_rows


def selected_upper_limb_endpoints(config: EvaluationConfig) -> tuple[str, ...]:
    return tuple(joint for joint in UPPER_LIMB_ENDPOINT_JOINTS if joint in config.selected_joints)


def evaluate_lower_limb_metrics(
    comparison_name: str,
    pair_key: str,
    trial_id: str,
    wrist_variant: str,
    alignment_mode: str,
    time_sec: np.ndarray,
    reference_coords: dict[str, np.ndarray],
    candidate_coords: dict[str, np.ndarray],
    reference_smoothed: SmoothingResult,
    candidate_smoothed: SmoothingResult,
    output_plot_dir: Path,
    config: EvaluationConfig,
) -> tuple[list[dict[str, object]], list[str], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    warnings: list[str] = []
    plot_rows: list[dict[str, object]] = []
    for side in config.selected_sides:
        knee = f"knee_{side}"
        if "knee" not in config.selected_joints:
            warnings.append(f"Skipped lower-limb metrics for {side}: selected joints do not include knee.")
            continue
        if knee not in reference_coords or knee not in candidate_coords:
            warnings.append(f"Skipped lower-limb metrics for {side}: missing knee.")
            continue

        axis_vector, _orth_vector, explained_ratio = principal_axis(reference_coords[knee])
        reference_along = project_onto_axis(reference_coords[knee], axis_vector)
        candidate_along = project_onto_axis(candidate_coords[knee], axis_vector)
        if np.isfinite(explained_ratio) and explained_ratio < config.principal_axis_min_explained_variance:
            warnings.append(
                f"Low knee principal-axis explained variance for {side}: {explained_ratio:.3f}."
            )

        reference_knee_velocity = project_onto_axis(reference_smoothed.velocities[knee], axis_vector)
        candidate_knee_velocity = project_onto_axis(candidate_smoothed.velocities[knee], axis_vector)

        reference_events = detect_events(reference_along, reference_knee_velocity, time_sec, config.event_min_distance_sec, config.event_prominence_ratio)
        candidate_events = detect_events(candidate_along, candidate_knee_velocity, time_sec, config.event_min_distance_sec, config.event_prominence_ratio)
        rows.extend(
            event_metric_rows(
                comparison_name,
                pair_key,
                trial_id,
                wrist_variant,
                side,
                "knee",
                "lower_limb",
                alignment_mode,
                time_sec,
                reference_events,
                candidate_events,
                config.event_match_tolerance_sec,
            )
        )

        reference_excursions = compute_excursions(reference_along, reference_events)
        candidate_excursions = compute_excursions(candidate_along, candidate_events)
        rows.extend(
            [
                metric_row(comparison_name, pair_key, trial_id, wrist_variant, side, "knee", "lower_limb", "excursion_mean_abs_error", abs(safe_mean(candidate_excursions) - safe_mean(reference_excursions)), alignment_mode),
            ]
        )

        if config.generate_plots:
            output_path = output_plot_dir / f"{pair_key}__{wrist_variant}__{alignment_mode}__knee_{side}.png"
            save_knee_axis_plot(output_path, time_sec, reference_along, candidate_along, reference_events, candidate_events, title=f"{pair_key} | knee {side} | {alignment_mode}")
            plot_rows.append({"plot_type": "knee_principal_axis", "plot_path": str(output_path)})
    return rows, warnings, plot_rows


def metric_row(
    comparison_name: str,
    pair_key: str,
    trial_id: str,
    wrist_variant: str,
    side: str,
    joint: str,
    metric_family: str,
    metric_name: str,
    metric_value: float,
    alignment_mode: str,
    system_name: str = "pair",
) -> dict[str, object]:
    return {
        "comparison_name": comparison_name,
        "pair_key": pair_key,
        "trial_id": trial_id,
        "wrist_variant": wrist_variant,
        "system_name": system_name,
        "side": side,
        "joint": joint,
        "metric_family": metric_family,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "normalization_mode": "body_centered_2d",
        "alignment_mode": alignment_mode,
    }


def build_normalization_row(
    comparison_name: str,
    pair_key: str,
    trial_id: str,
    wrist_variant: str,
    system_name: str,
    result: NormalizationResult,
) -> dict[str, object]:
    return {
        "comparison_name": comparison_name,
        "pair_key": pair_key,
        "trial_id": trial_id,
        "wrist_variant": wrist_variant,
        "system_name": system_name,
        "system_label": SYSTEM_LABELS[system_name],
        "normalization_mode": "body_centered_2d",
        "alignment_mode": "body_only",
        "origin_source": result.origin_source,
        "scale_source": result.scale_source,
        "scale_value": result.scale,
        "rotation_source": result.rotation_source,
        "rotation_angle_deg": math.degrees(result.rotation_angle_rad),
        "orientation_x_sign": result.orientation_x_sign,
        "orientation_y_sign": result.orientation_y_sign,
        "orientation_source": result.orientation_source,
        "similarity_scale": float("nan"),
        "similarity_rotation_deg": float("nan"),
        "similarity_translation_x": float("nan"),
        "similarity_translation_y": float("nan"),
    }


def build_warning_row(
    comparison_name: str,
    pair_key: str,
    trial_id: str,
    wrist_variant: str,
    system_name: str,
    warning: str,
    alignment_mode: str | None = None,
) -> dict[str, object]:
    return {
        "comparison_name": comparison_name,
        "pair_key": pair_key,
        "trial_id": trial_id,
        "wrist_variant": wrist_variant,
        "system_name": system_name,
        "alignment_mode": alignment_mode or "body_only",
        "warning": warning,
    }


def joint_side(joint_id: str) -> str:
    return joint_id.rsplit("_", 1)[1]


def joint_name(joint_id: str) -> str:
    return joint_id.rsplit("_", 1)[0]


def segment_angle(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    angle = np.full(len(start), np.nan, dtype=float)
    mask = np.isfinite(start).all(axis=1) & np.isfinite(end).all(axis=1)
    vectors = end[mask] - start[mask]
    angle[mask] = np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0]))
    return angle


def internal_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    angle = np.full(len(a), np.nan, dtype=float)
    mask = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1) & np.isfinite(c).all(axis=1)
    if not np.any(mask):
        return angle
    ba = a[mask] - b[mask]
    bc = c[mask] - b[mask]
    denom = np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1)
    valid = denom > 0
    cos_theta = np.full(len(denom), np.nan, dtype=float)
    cos_theta[valid] = np.sum(ba[valid] * bc[valid], axis=1) / denom[valid]
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle[mask] = np.degrees(np.arccos(cos_theta))
    return angle


def event_metric_rows(
    comparison_name: str,
    pair_key: str,
    trial_id: str,
    wrist_variant: str,
    side: str,
    joint: str,
    metric_family: str,
    alignment_mode: str,
    time_sec: np.ndarray,
    reference_events,
    candidate_events,
    tolerance_sec: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for label, reference_indices, candidate_indices in (
        ("reversal", reference_events.reversal_indices, candidate_events.reversal_indices),
    ):
        reference_times = time_sec[reference_indices] if reference_indices.size else np.array([], dtype=float)
        candidate_times = time_sec[candidate_indices] if candidate_indices.size else np.array([], dtype=float)
        matched = match_event_times(reference_times, candidate_times, tolerance_sec=tolerance_sec)
        rows.extend(
            [
                metric_row(comparison_name, pair_key, trial_id, wrist_variant, side, joint, metric_family, f"{label}_mean_abs_timing_error_sec", safe_mean(np.abs(matched.matched_time_errors_sec)), alignment_mode),
                metric_row(comparison_name, pair_key, trial_id, wrist_variant, side, joint, metric_family, f"{label}_interval_rmse_sec", inter_event_interval_rmse(reference_times, candidate_times), alignment_mode),
            ]
        )
    return rows


def compute_excursions(signal: np.ndarray, events) -> np.ndarray:
    extrema = np.sort(np.unique(np.concatenate([events.maxima_indices, events.minima_indices])))
    excursions: list[float] = []
    for previous, current in zip(extrema[:-1], extrema[1:]):
        if not np.isfinite(signal[previous]) or not np.isfinite(signal[current]):
            continue
        excursions.append(abs(float(signal[current] - signal[previous])))
    return np.asarray(excursions, dtype=float)


def safe_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else float("nan")


def save_joint_overlay_plot(output_path: Path, time_sec: np.ndarray, reference: np.ndarray, candidate: np.ndarray, title: str, ylabel_prefix: str) -> None:
    apply_thesis_style()
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(6.3, 4.2), dpi=THESIS_STYLE.dpi, sharex=True)
    for axis_index, axis_name in enumerate(("x", "y")):
        axes[axis_index].plot(time_sec, reference[:, axis_index], label="Reference", color="#1d4ed8", linewidth=1.8)
        axes[axis_index].plot(time_sec, candidate[:, axis_index], label="Candidate", color="#dc2626", linewidth=1.5)
        axes[axis_index].set_ylabel(f"{ylabel_prefix} {axis_name}")
        style_axis(axes[axis_index], y_grid=True, x_grid=True)
    axes[0].legend(frameon=False)
    axes[-1].set_xlabel("Time (s)")
    figure.suptitle(title, fontsize=THESIS_STYLE.figure_title_size)
    figure.tight_layout()
    save_single_figure(figure, output_path)


def save_knee_axis_plot(output_path: Path, time_sec: np.ndarray, reference_signal: np.ndarray, candidate_signal: np.ndarray, reference_events, candidate_events, title: str) -> None:
    apply_thesis_style()
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6.3, 3.4), dpi=THESIS_STYLE.dpi)
    axis.plot(time_sec, reference_signal, label="Reference along-axis", color="#1d4ed8", linewidth=1.8)
    axis.plot(time_sec, candidate_signal, label="Candidate along-axis", color="#dc2626", linewidth=1.5)
    axis.scatter(time_sec[reference_events.maxima_indices], reference_signal[reference_events.maxima_indices], color="#1d4ed8", s=18)
    axis.scatter(time_sec[candidate_events.maxima_indices], candidate_signal[candidate_events.maxima_indices], color="#dc2626", s=18)
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Along-axis displacement")
    axis.set_title(title, fontsize=THESIS_STYLE.axes_title_size)
    style_axis(axis, y_grid=True, x_grid=True)
    axis.legend(frameon=False)
    figure.tight_layout()
    save_single_figure(figure, output_path)
