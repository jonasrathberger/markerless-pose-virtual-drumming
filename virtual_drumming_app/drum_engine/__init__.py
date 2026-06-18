"""Provider-agnostic drum engine primitives."""

from .apple_vision import apple_vision_landmarks_to_pose_frame
from .hand_hit_detection import (
    HAND_SIDES,
    HandHitDetector,
    HandHitDetectorConfig,
    HandHitEvent,
    target_observation_from_pose_frame,
)
from .midi import MidiOutput, MidiOutputConfig, available_midi_outputs
from .pedal_hit_detection import HitEvent, PedalHitDetector, PedalHitDetectorConfig
from .schema import PoseFrame, PosePoint
from .smoothing import LandmarkSmoother, SmoothingConfig
from .target_classification import (
    DrumTargetClassifier,
    DrumTargetObservation,
    DrumTargetObservationFrame,
    DrumTargetPrediction,
    KnnDrumTargetClassifier,
    TARGET_FEATURE_SCHEMA,
    TARGET_TEMPORAL_FEATURE_SCHEMA,
)

__all__ = [
    "HAND_SIDES",
    "HandHitDetector",
    "HandHitDetectorConfig",
    "HandHitEvent",
    "HitEvent",
    "DrumTargetClassifier",
    "DrumTargetObservation",
    "DrumTargetObservationFrame",
    "DrumTargetPrediction",
    "KnnDrumTargetClassifier",
    "LandmarkSmoother",
    "MidiOutput",
    "MidiOutputConfig",
    "PedalHitDetector",
    "PedalHitDetectorConfig",
    "PoseFrame",
    "PosePoint",
    "SmoothingConfig",
    "TARGET_FEATURE_SCHEMA",
    "TARGET_TEMPORAL_FEATURE_SCHEMA",
    "apple_vision_landmarks_to_pose_frame",
    "available_midi_outputs",
    "target_observation_from_pose_frame",
]
