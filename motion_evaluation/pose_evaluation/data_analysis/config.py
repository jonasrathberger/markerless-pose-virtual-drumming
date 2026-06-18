from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
THESIS_DATA_DIR = REPO_ROOT / "thesis_data" / "pose_evaluation"
INPUT_DIR = THESIS_DATA_DIR / "aligned_result"
OUTPUT_DIR = THESIS_DATA_DIR
PLOTS_DIR = OUTPUT_DIR / "plots"

SYSTEM_SOURCE_DIRS = {
    "optitrack": "mocap",
    "apple_vision": "pose",
    "mediapipe": "pose",
}

SYSTEM_RECORDINGS = {
    "optitrack": ("air_knees", "drums"),
    "apple_vision": ("apple_vision",),
    "mediapipe": ("mediapipe",),
}

SYSTEM_LABELS = {
    "optitrack": "OptiTrack",
    "apple_vision": "Apple Vision",
    "mediapipe": "MediaPipe",
}


@dataclass(frozen=True)
class JointMapping:
    canonical_name: str
    target: str
    target_variant: str
    side: str
    csv_names_by_system: dict[str, str]
    used_for_normalization: bool = False
    used_for_upper_limb: bool = False
    used_for_lower_limb: bool = False

    @property
    def joint_id(self) -> str:
        return f"{self.canonical_name}_{self.side}"


@dataclass(frozen=True)
class ComparisonSpec:
    comparison_name: str
    trial_id: str
    reference_system: str
    system_a: str
    recording_a: str
    system_b: str
    recording_b: str

    @property
    def pair_key(self) -> str:
        return f"{self.comparison_name}__{self.trial_id}"


@dataclass(frozen=True)
class EvaluationConfig:
    evaluation_fps: float = 60.0
    body_origin_priority: tuple[str, ...] = ("shoulders", "hips", "mean_available")
    body_scale_priority: tuple[str, ...] = ("shoulder_width", "hip_width", "torso_length", "knee_width", "body_extent")
    rotation_normalization: bool = False
    normalize_body_axes: bool = True
    similarity_alignment: bool = True
    torso_similarity_alignment: bool = True
    torso_similarity_joint_ids: tuple[str, ...] = ("shoulder_left", "shoulder_right", "hip_left", "hip_right")
    smoothing_method: str = "savitzky_golay"
    smoothing_window_sec: float = 0.25
    smoothing_polyorder: int = 3
    pck_thresholds: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5)
    selected_sides: tuple[str, ...] = ("left", "right")
    selected_joints: tuple[str, ...] = ("shoulder", "elbow", "wrist", "pinky_knuckle", "hip", "knee", "ankle")
    enabled_metric_groups: tuple[str, ...] = (
        "core_spatial",
        "upper_limb",
        "lower_limb",
        "pearson",
        "jitter",
        "bland_altman",
    )
    enable_pearson_metrics: bool = True
    enable_jitter_metrics: bool = True
    enable_bland_altman: bool = True
    event_min_distance_sec: float = 0.2
    event_prominence_ratio: float = 0.35
    event_match_tolerance_sec: float = 0.15
    principal_axis_min_explained_variance: float = 0.5
    jitter_smoothing_method: str = "same_as_main"
    jitter_smoothing_window_sec: float = 0.25
    jitter_smoothing_polyorder: int = 3
    pearson_min_valid_samples: int = 5
    pearson_min_std: float = 1e-12
    bland_altman_min_pairs: int = 2
    generate_plots: bool = True
    output_note: str = (
        "These metrics compare motion-pattern fidelity after temporal alignment and spatial normalization. "
        "The recordings are separate takes, so the results are not strict synchronized ground-truth tracking errors. "
        "The final metric set is limited to MPJPE/PCK spatial fidelity, selected Pearson dynamic-fidelity metrics, "
        "reversal timing, knee excursion error, selected jitter RMS stability metrics, and Bland-Altman summaries for selected derived variables."
    )

    @property
    def alignment_modes(self) -> tuple[str, ...]:
        modes = ["body_only"]
        if self.similarity_alignment:
            modes.append("body_plus_similarity")
            if self.torso_similarity_alignment:
                modes.append("body_plus_torso_similarity")
        return tuple(modes)


BASE_JOINT_MAPPINGS = (
    JointMapping(
        canonical_name="shoulder",
        target="shoulder",
        target_variant="default",
        side="left",
        csv_names_by_system={
            "optitrack": "shoulder_default_left.csv",
            "apple_vision": "shoulder_default_left.csv",
            "mediapipe": "shoulder_default_left.csv",
        },
        used_for_normalization=True,
        used_for_upper_limb=True,
    ),
    JointMapping(
        canonical_name="shoulder",
        target="shoulder",
        target_variant="default",
        side="right",
        csv_names_by_system={
            "optitrack": "shoulder_default_right.csv",
            "apple_vision": "shoulder_default_right.csv",
            "mediapipe": "shoulder_default_right.csv",
        },
        used_for_normalization=True,
        used_for_upper_limb=True,
    ),
    JointMapping(
        canonical_name="elbow",
        target="elbow",
        target_variant="default",
        side="left",
        csv_names_by_system={
            "optitrack": "elbow_default_left.csv",
            "apple_vision": "elbow_default_left.csv",
            "mediapipe": "elbow_default_left.csv",
        },
        used_for_upper_limb=True,
    ),
    JointMapping(
        canonical_name="elbow",
        target="elbow",
        target_variant="default",
        side="right",
        csv_names_by_system={
            "optitrack": "elbow_default_right.csv",
            "apple_vision": "elbow_default_right.csv",
            "mediapipe": "elbow_default_right.csv",
        },
        used_for_upper_limb=True,
    ),
    JointMapping(
        canonical_name="hip",
        target="hip",
        target_variant="default",
        side="left",
        csv_names_by_system={
            "optitrack": "hip_default_left.csv",
            "apple_vision": "hip_default_left.csv",
            "mediapipe": "hip_default_left.csv",
        },
        used_for_normalization=True,
        used_for_lower_limb=True,
    ),
    JointMapping(
        canonical_name="hip",
        target="hip",
        target_variant="default",
        side="right",
        csv_names_by_system={
            "optitrack": "hip_default_right.csv",
            "apple_vision": "hip_default_right.csv",
            "mediapipe": "hip_default_right.csv",
        },
        used_for_normalization=True,
        used_for_lower_limb=True,
    ),
    JointMapping(
        canonical_name="knee",
        target="knee",
        target_variant="default",
        side="left",
        csv_names_by_system={
            "optitrack": "knee_default_left.csv",
            "apple_vision": "knee_default_left.csv",
            "mediapipe": "knee_default_left.csv",
        },
        used_for_lower_limb=True,
    ),
    JointMapping(
        canonical_name="knee",
        target="knee",
        target_variant="default",
        side="right",
        csv_names_by_system={
            "optitrack": "knee_default_right.csv",
            "apple_vision": "knee_default_right.csv",
            "mediapipe": "knee_default_right.csv",
        },
        used_for_lower_limb=True,
    ),
    JointMapping(
        canonical_name="ankle",
        target="ankle",
        target_variant="default",
        side="left",
        csv_names_by_system={
            "optitrack": "ankle_default_left.csv",
            "apple_vision": "ankle_default_left.csv",
            "mediapipe": "ankle_default_left.csv",
        },
        used_for_lower_limb=True,
    ),
    JointMapping(
        canonical_name="ankle",
        target="ankle",
        target_variant="default",
        side="right",
        csv_names_by_system={
            "optitrack": "ankle_default_right.csv",
            "apple_vision": "ankle_default_right.csv",
            "mediapipe": "ankle_default_right.csv",
        },
        used_for_lower_limb=True,
    ),
    JointMapping(
        canonical_name="pinky_knuckle",
        target="pinky_knuckle",
        target_variant="default",
        side="left",
        csv_names_by_system={
            "optitrack": "pinky_knuckle_proxy_pinky1_left.csv",
            "apple_vision": "pinky_knuckle_default_left.csv",
            "mediapipe": "pinky_knuckle_default_left.csv",
        },
        used_for_upper_limb=True,
    ),
    JointMapping(
        canonical_name="pinky_knuckle",
        target="pinky_knuckle",
        target_variant="default",
        side="right",
        csv_names_by_system={
            "optitrack": "pinky_knuckle_proxy_pinky1_right.csv",
            "apple_vision": "pinky_knuckle_default_right.csv",
            "mediapipe": "pinky_knuckle_default_right.csv",
        },
        used_for_upper_limb=True,
    ),
)

WRIST_VARIANT_MAPPINGS = {
    "body": (
        JointMapping(
            canonical_name="wrist",
            target="wrist",
            target_variant="body",
            side="left",
            csv_names_by_system={
                "optitrack": "wrist_center_left.csv",
                "apple_vision": "wrist_body_left.csv",
                "mediapipe": "wrist_body_left.csv",
            },
            used_for_upper_limb=True,
        ),
        JointMapping(
            canonical_name="wrist",
            target="wrist",
            target_variant="body",
            side="right",
            csv_names_by_system={
                "optitrack": "wrist_center_right.csv",
                "apple_vision": "wrist_body_right.csv",
                "mediapipe": "wrist_body_right.csv",
            },
            used_for_upper_limb=True,
        ),
    ),
    "hand": (
        JointMapping(
            canonical_name="wrist",
            target="wrist",
            target_variant="hand",
            side="left",
            csv_names_by_system={
                "optitrack": "wrist_center_left.csv",
                "apple_vision": "wrist_hand_left.csv",
                "mediapipe": "wrist_hand_left.csv",
            },
            used_for_upper_limb=True,
        ),
        JointMapping(
            canonical_name="wrist",
            target="wrist",
            target_variant="hand",
            side="right",
            csv_names_by_system={
                "optitrack": "wrist_center_right.csv",
                "apple_vision": "wrist_hand_right.csv",
                "mediapipe": "wrist_hand_right.csv",
            },
            used_for_upper_limb=True,
        ),
    ),
}

COMPARISON_SPECS = (
    ComparisonSpec(
        comparison_name="applevision_vs_optitrack",
        trial_id="air_knees",
        reference_system="optitrack",
        system_a="optitrack",
        recording_a="air_knees",
        system_b="apple_vision",
        recording_b="apple_vision",
    ),
    ComparisonSpec(
        comparison_name="applevision_vs_optitrack",
        trial_id="drums",
        reference_system="optitrack",
        system_a="optitrack",
        recording_a="drums",
        system_b="apple_vision",
        recording_b="apple_vision",
    ),
    ComparisonSpec(
        comparison_name="mediapipe_vs_optitrack",
        trial_id="air_knees",
        reference_system="optitrack",
        system_a="optitrack",
        recording_a="air_knees",
        system_b="mediapipe",
        recording_b="mediapipe",
    ),
    ComparisonSpec(
        comparison_name="mediapipe_vs_optitrack",
        trial_id="drums",
        reference_system="optitrack",
        system_a="optitrack",
        recording_a="drums",
        system_b="mediapipe",
        recording_b="mediapipe",
    ),
    ComparisonSpec(
        comparison_name="mediapipe_vs_applevision",
        trial_id="aligned_window",
        reference_system="apple_vision",
        system_a="apple_vision",
        recording_a="apple_vision",
        system_b="mediapipe",
        recording_b="mediapipe",
    ),
)


def build_joint_mappings(wrist_variant: str) -> tuple[JointMapping, ...]:
    return BASE_JOINT_MAPPINGS + WRIST_VARIANT_MAPPINGS[wrist_variant]


DEFAULT_CONFIG = EvaluationConfig()
