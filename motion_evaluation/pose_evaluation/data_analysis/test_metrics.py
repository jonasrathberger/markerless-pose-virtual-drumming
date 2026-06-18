from __future__ import annotations

import math
import unittest
from dataclasses import replace

import numpy as np

from data_analysis.config import DEFAULT_CONFIG, build_joint_mappings
from data_analysis.events import detect_events, match_event_times
from data_analysis.evaluators import evaluate_core_spatial_metrics, evaluate_jitter_metrics, evaluate_pearson_metrics
from data_analysis.metrics import (
    bland_altman_paired_values,
    mpjpe_2d,
    pck_2d,
    pearson_correlation,
    principal_axis,
    residual_jitter_stats,
    trajectory_rmse_2d,
)
from data_analysis.normalization import fit_similarity_transform, normalize_trial_coordinates
from data_analysis.smoothing import SmoothingResult


class MetricsTests(unittest.TestCase):
    def test_joint_mappings_include_pinky_knuckle_by_default(self) -> None:
        joint_ids = {mapping.joint_id for mapping in build_joint_mappings("body")}
        self.assertIn("pinky_knuckle_left", joint_ids)
        self.assertIn("pinky_knuckle_right", joint_ids)
        self.assertIn("pinky_knuckle", DEFAULT_CONFIG.selected_joints)

    def test_body_centered_normalization_uses_shoulder_width(self) -> None:
        coords = {
            "shoulder_left": np.array([[0.0, 0.0], [0.0, 0.0]]),
            "shoulder_right": np.array([[2.0, 0.0], [2.0, 0.0]]),
            "knee_left": np.array([[0.0, -2.0], [0.0, -2.0]]),
        }
        result = normalize_trial_coordinates(coords, apply_rotation=False)
        self.assertEqual(result.origin_source, "shoulders")
        self.assertEqual(result.scale_source, "shoulder_width")
        self.assertAlmostEqual(result.scale, 2.0, places=9)
        np.testing.assert_allclose(result.coordinates["shoulder_left"][0], [-0.5, 0.0])

    def test_body_axis_normalization_flips_image_y_down_coordinates(self) -> None:
        coords = {
            "shoulder_left": np.array([[0.0, 0.0], [0.0, 0.0]]),
            "shoulder_right": np.array([[2.0, 0.0], [2.0, 0.0]]),
            "hip_left": np.array([[0.0, 2.0], [0.0, 2.0]]),
            "hip_right": np.array([[2.0, 2.0], [2.0, 2.0]]),
        }
        result = normalize_trial_coordinates(coords, apply_rotation=False)
        self.assertEqual(result.orientation_y_sign, -1.0)
        self.assertGreater(result.coordinates["shoulder_left"][0, 1], result.coordinates["hip_left"][0, 1])

    def test_body_axis_normalization_flips_left_right_when_needed(self) -> None:
        coords = {
            "shoulder_left": np.array([[2.0, 2.0], [2.0, 2.0]]),
            "shoulder_right": np.array([[0.0, 2.0], [0.0, 2.0]]),
            "hip_left": np.array([[2.0, 0.0], [2.0, 0.0]]),
            "hip_right": np.array([[0.0, 0.0], [0.0, 0.0]]),
        }
        result = normalize_trial_coordinates(coords, apply_rotation=False)
        self.assertEqual(result.orientation_x_sign, -1.0)
        self.assertGreater(result.coordinates["shoulder_right"][0, 0], result.coordinates["shoulder_left"][0, 0])

    def test_normalization_priority_can_prefer_hip_width(self) -> None:
        coords = {
            "shoulder_left": np.array([[0.0, 0.0], [0.0, 0.0]]),
            "shoulder_right": np.array([[2.0, 0.0], [2.0, 0.0]]),
            "hip_left": np.array([[0.0, -1.0], [0.0, -1.0]]),
            "hip_right": np.array([[4.0, -1.0], [4.0, -1.0]]),
        }
        result = normalize_trial_coordinates(
            coords,
            apply_rotation=False,
            scale_priority=("hip_width", "shoulder_width"),
        )
        self.assertEqual(result.scale_source, "hip_width")
        self.assertAlmostEqual(result.scale, 4.0, places=9)

    def test_similarity_transform_recovers_rotated_scaled_points(self) -> None:
        reference = {"joint": np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])}
        candidate = {"joint": np.array([[2.0, 1.0], [2.0, 3.0], [0.0, 1.0]])}
        transform = fit_similarity_transform(reference, candidate)
        aligned = transform.scale * (candidate["joint"] @ transform.rotation) + transform.translation
        np.testing.assert_allclose(aligned, reference["joint"], atol=1e-6)

    def test_similarity_transform_can_fit_selected_joints_only(self) -> None:
        reference = {
            "torso": np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
            "noisy_wrist": np.array([[100.0, 100.0], [100.0, 101.0], [101.0, 100.0]]),
        }
        candidate = {
            "torso": np.array([[2.0, 1.0], [2.0, 3.0], [0.0, 1.0]]),
            "noisy_wrist": np.array([[-100.0, -100.0], [-100.0, -101.0], [-101.0, -100.0]]),
        }
        transform = fit_similarity_transform(reference, candidate, joint_ids=("torso",))
        aligned = transform.scale * (candidate["torso"] @ transform.rotation) + transform.translation
        np.testing.assert_allclose(aligned, reference["torso"], atol=1e-6)

    def test_spatial_metrics_zero_for_identical_inputs(self) -> None:
        reference = {"joint_left": np.array([[0.0, 0.0], [1.0, 1.0]])}
        candidate = {"joint_left": np.array([[0.0, 0.0], [1.0, 1.0]])}
        self.assertAlmostEqual(trajectory_rmse_2d(reference["joint_left"], candidate["joint_left"]), 0.0, places=9)
        self.assertAlmostEqual(mpjpe_2d(reference, candidate), 0.0, places=9)
        self.assertAlmostEqual(pck_2d(reference, candidate, 0.1), 1.0, places=9)

    def test_pck_ignores_invalid_frame_pairs(self) -> None:
        reference = {"joint_left": np.array([[0.0, 0.0], [np.nan, np.nan], [1.0, 1.0]])}
        candidate = {"joint_left": np.array([[0.0, 0.0], [0.0, 0.0], [2.0, 2.0]])}
        self.assertAlmostEqual(pck_2d(reference, candidate, 0.1), 0.5, places=9)

    def test_core_spatial_pck_uses_passed_aligned_coordinates(self) -> None:
        reference = {
            "wrist_left": np.array([[0.0, 0.0], [0.0, 0.0]]),
            "elbow_left": np.array([[0.0, 0.0], [0.0, 0.0]]),
        }
        aligned_candidate = {
            "wrist_left": np.array([[0.05, 0.0], [0.20, 0.0]]),
            "elbow_left": np.array([[0.0, 0.0], [np.nan, np.nan]]),
        }
        config = replace(
            DEFAULT_CONFIG,
            selected_sides=("left",),
            selected_joints=("wrist", "elbow"),
            pck_thresholds=(0.1,),
        )
        rows = evaluate_core_spatial_metrics(
            "test_comparison",
            "test_pair",
            "test_trial",
            "body",
            "body_plus_similarity",
            reference,
            aligned_candidate,
            ["wrist_left", "elbow_left"],
            config,
        )
        both_rows = {row["metric_name"]: row for row in rows if row["side"] == "both"}
        self.assertAlmostEqual(both_rows["mpjpe_2d"]["metric_value"], (0.05 + 0.20 + 0.0) / 3.0, places=9)
        self.assertAlmostEqual(both_rows["pck_2d@0.1"]["metric_value"], 2.0 / 3.0, places=9)
        self.assertEqual(both_rows["pck_2d@0.1"]["alignment_mode"], "body_plus_similarity")

    def test_pearson_velocity_includes_pinky_knuckle_endpoint(self) -> None:
        reference_coords = {
            "pinky_knuckle_left": np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        }
        candidate_coords = {
            "pinky_knuckle_left": np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        }
        smoothed = SmoothingResult(
            smoothed=reference_coords,
            velocities={"pinky_knuckle_left": np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])},
            warnings=(),
        )
        rows, warnings = evaluate_pearson_metrics(
            "test_comparison",
            "test_pair",
            "test_trial",
            "body",
            "body_only",
            reference_coords,
            candidate_coords,
            smoothed,
            smoothed,
            ["pinky_knuckle_left"],
            replace(DEFAULT_CONFIG, selected_sides=("left",), selected_joints=("pinky_knuckle",)),
        )
        self.assertEqual(warnings, [])
        velocity_rows = [row for row in rows if row["metric_name"] == "pearson_velocity"]
        self.assertEqual(len(velocity_rows), 1)
        self.assertEqual(velocity_rows[0]["joint"], "pinky_knuckle")
        self.assertTrue(math.isnan(velocity_rows[0]["metric_value"]))

    def test_jitter_metrics_include_pinky_knuckle_endpoint(self) -> None:
        coordinates = {
            "pinky_knuckle_left": np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]),
            "knee_left": np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        }
        smoothed = SmoothingResult(
            smoothed={
                "pinky_knuckle_left": np.array([[0.0, 0.0], [1.0, 0.5], [2.0, 2.0]]),
                "knee_left": coordinates["knee_left"],
            },
            velocities={},
            warnings=(),
        )
        rows = evaluate_jitter_metrics(
            "test_comparison",
            "test_pair",
            "test_trial",
            "body",
            "body_only",
            "optitrack",
            "apple_vision",
            np.array([0.0, 0.5, 1.0]),
            coordinates,
            coordinates,
            smoothed,
            smoothed,
            ["pinky_knuckle_left", "knee_left"],
            replace(DEFAULT_CONFIG, selected_sides=("left",), selected_joints=("pinky_knuckle", "knee")),
        )
        pinky_rows = [row for row in rows if row["joint"] == "pinky_knuckle"]
        self.assertEqual({row["metric_name"] for row in pinky_rows}, {"jitter_x", "jitter_y"})

    def test_event_matching_reports_missed_and_extra_events(self) -> None:
        time_sec = np.linspace(0.0, 1.0, 11)
        signal = np.array([0.0, 1.0, 0.0, -1.0, 0.0, 1.2, 0.0, -0.9, 0.0, 0.8, 0.0])
        velocity = np.gradient(signal, time_sec)
        reference = detect_events(signal, velocity, time_sec, min_distance_sec=0.1, prominence_ratio=0.1)
        candidate_times = time_sec[reference.maxima_indices[:-1]]
        matched = match_event_times(time_sec[reference.maxima_indices], candidate_times, tolerance_sec=0.11)
        self.assertEqual(matched.missed_count, 1)
        self.assertEqual(matched.extra_count, 0)

    def test_principal_axis_tracks_main_motion_direction(self) -> None:
        points = np.array([[0.0, 0.0], [1.0, 0.1], [2.0, -0.1], [3.0, 0.0]])
        axis, _orth, explained = principal_axis(points)
        self.assertGreater(explained, 0.9)
        self.assertGreater(abs(axis[0]), abs(axis[1]))

    def test_pearson_correlation_for_identical_series(self) -> None:
        values = np.array([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(pearson_correlation(values, values), 1.0, places=9)

    def test_pearson_correlation_rejects_constant_series(self) -> None:
        values = np.array([1.0, 1.0, 1.0, 1.0])
        self.assertTrue(np.isnan(pearson_correlation(values, values)))

    def test_residual_jitter_stats_uses_raw_minus_smoothed_residual(self) -> None:
        raw = np.array([0.0, 1.0, 0.0, 1.0])
        smoothed = np.array([0.5, 0.5, 0.5, 0.5])
        stats = residual_jitter_stats(raw, smoothed)
        self.assertAlmostEqual(stats["jitter_rms"], 0.5, places=9)

    def test_bland_altman_paired_values(self) -> None:
        means, differences = bland_altman_paired_values(np.array([1.0, 2.0]), np.array([2.0, 4.0]))
        np.testing.assert_allclose(means, [1.5, 3.0])
        np.testing.assert_allclose(differences, [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
