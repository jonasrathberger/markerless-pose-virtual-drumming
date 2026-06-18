"""MediaPipe Pose plus MediaPipe Hands backend using the current Tasks API."""

from __future__ import annotations

from typing import Any

from models.canonical import HAND_BASE_NAMES, canonical_id
from models.schema import FrameResult, LandmarkObservation
from utils.mediapipe_tasks import (
    HAND_LANDMARKER_URL,
    POSE_LANDMARKER_LITE_URL,
    ensure_model_asset,
    mediapipe_image_from_bgr,
    monotonic_sec_to_timestamp_ms,
)

from .base import BackendUnavailableError, PoseBackend


POSE_INDEX_MAP = {
    0: canonical_id("body", "center", "nose"),
    11: canonical_id("body", "left", "shoulder"),
    12: canonical_id("body", "right", "shoulder"),
    13: canonical_id("body", "left", "elbow"),
    14: canonical_id("body", "right", "elbow"),
    15: canonical_id("body", "left", "wrist"),
    16: canonical_id("body", "right", "wrist"),
    23: canonical_id("body", "left", "hip"),
    24: canonical_id("body", "right", "hip"),
    25: canonical_id("body", "left", "knee"),
    26: canonical_id("body", "right", "knee"),
    27: canonical_id("body", "left", "ankle"),
    28: canonical_id("body", "right", "ankle"),
}


class MediaPipePoseHandsBackend(PoseBackend):
    backend_name = "mediapipe_pose_hands"

    def __init__(
        self,
        *,
        pose_detection_confidence: float = 0.5,
        pose_tracking_confidence: float = 0.5,
        hand_detection_confidence: float = 0.5,
        hand_tracking_confidence: float = 0.5,
        max_num_hands: int = 2,
        pose_model_path: str | None = None,
        hand_model_path: str | None = None,
        allow_model_download: bool = True,
    ) -> None:
        super().__init__()
        self.pose_detection_confidence = pose_detection_confidence
        self.pose_tracking_confidence = pose_tracking_confidence
        self.hand_detection_confidence = hand_detection_confidence
        self.hand_tracking_confidence = hand_tracking_confidence
        self.max_num_hands = max_num_hands
        self.pose_model_path = pose_model_path
        self.hand_model_path = hand_model_path
        self.allow_model_download = allow_model_download
        self.mp = None
        self.pose = None
        self.hands = None

    def initialize(self) -> None:
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python import vision
        except ImportError as exc:
            raise BackendUnavailableError(
                "MediaPipe is not installed. From the repository root, run "
                "`python -m pip install -r requirements.txt`."
            ) from exc

        self.mp = mp
        pose_model_path = ensure_model_asset(
            output_path=self.pose_model_path,
            default_filename="pose_landmarker_lite.task",
            download_url=POSE_LANDMARKER_LITE_URL,
            allow_download=self.allow_model_download,
        )
        hand_model_path = ensure_model_asset(
            output_path=self.hand_model_path,
            default_filename="hand_landmarker.task",
            download_url=HAND_LANDMARKER_URL,
            allow_download=self.allow_model_download,
        )

        pose_options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(pose_model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=self.pose_detection_confidence,
            min_pose_presence_confidence=self.pose_detection_confidence,
            min_tracking_confidence=self.pose_tracking_confidence,
            output_segmentation_masks=False,
        )
        hand_options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(hand_model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=self.max_num_hands,
            min_hand_detection_confidence=self.hand_detection_confidence,
            min_hand_presence_confidence=self.hand_detection_confidence,
            min_tracking_confidence=self.hand_tracking_confidence,
        )
        self.pose = vision.PoseLandmarker.create_from_options(pose_options)
        self.hands = vision.HandLandmarker.create_from_options(hand_options)

    def process_frame(self, frame_bgr, timestamp_monotonic_sec: float) -> FrameResult:
        if self.pose is None or self.hands is None or self.mp is None:
            raise RuntimeError("Backend was not initialized.")

        mp_image = mediapipe_image_from_bgr(self.mp, frame_bgr)
        timestamp_ms = monotonic_sec_to_timestamp_ms(timestamp_monotonic_sec)
        pose_result = self.pose.detect_for_video(mp_image, timestamp_ms)
        hands_result = self.hands.detect_for_video(mp_image, timestamp_ms)

        height, width = frame_bgr.shape[:2]
        landmarks: dict[str, LandmarkObservation] = {}
        self._populate_pose_landmarks(landmarks, pose_result, width, height)
        self._populate_hand_landmarks(landmarks, hands_result, width, height)
        return FrameResult(
            landmarks=landmarks,
            person_id=0,
            backend_metadata={"timestamp_monotonic_sec": timestamp_monotonic_sec},
        )

    def get_configuration(self) -> dict[str, Any]:
        return {
            "pose_detection_confidence": self.pose_detection_confidence,
            "pose_tracking_confidence": self.pose_tracking_confidence,
            "hand_detection_confidence": self.hand_detection_confidence,
            "hand_tracking_confidence": self.hand_tracking_confidence,
            "max_num_hands": self.max_num_hands,
            "pose_model_path": self.pose_model_path,
            "hand_model_path": self.hand_model_path,
            "allow_model_download": self.allow_model_download,
            "mediapipe_api": "tasks-python",
        }

    def shutdown(self) -> None:
        if self.pose is not None and hasattr(self.pose, "close"):
            self.pose.close()
        if self.hands is not None and hasattr(self.hands, "close"):
            self.hands.close()

    @staticmethod
    def _populate_pose_landmarks(
        target: dict[str, LandmarkObservation],
        pose_result,
        width: int,
        height: int,
    ) -> None:
        pose_landmarks = getattr(pose_result, "pose_landmarks", None)
        if not pose_landmarks:
            return
        for index, landmark in enumerate(pose_landmarks[0]):
            landmark_id = POSE_INDEX_MAP.get(index)
            if landmark_id is None:
                continue
            visibility = getattr(landmark, "visibility", None)
            presence = getattr(landmark, "presence", None)
            score = presence if presence is not None else visibility
            target[landmark_id] = LandmarkObservation(
                x_norm=float(landmark.x),
                y_norm=float(landmark.y),
                z_rel=float(landmark.z) if landmark.z is not None else None,
                x_px=float(landmark.x * width),
                y_px=float(landmark.y * height),
                confidence=float(score) if score is not None else None,
                visibility=float(visibility) if visibility is not None else None,
                present=(score is None) or float(score) > 0.0,
            )

    def _populate_hand_landmarks(
        self,
        target: dict[str, LandmarkObservation],
        hands_result,
        width: int,
        height: int,
    ) -> None:
        hand_landmarks = getattr(hands_result, "hand_landmarks", None)
        handedness_entries = getattr(hands_result, "handedness", None)
        if not hand_landmarks or not handedness_entries:
            return

        best_by_side: dict[str, tuple[float, list[Any]]] = {}
        for landmarks_for_hand, handedness in zip(hand_landmarks, handedness_entries):
            primary_label = handedness[0]
            side = (primary_label.category_name or "").lower()
            if side not in {"left", "right"}:
                continue
            score = float(primary_label.score) if primary_label.score is not None else 0.0
            existing = best_by_side.get(side)
            if existing is None or score > existing[0]:
                best_by_side[side] = (score, landmarks_for_hand)

        for side, (score, landmarks_for_hand) in best_by_side.items():
            group_name = f"{side}_hand"
            for index, landmark in enumerate(landmarks_for_hand):
                landmark_name = HAND_BASE_NAMES[index]
                landmark_id = canonical_id(group_name, side, landmark_name)
                target[landmark_id] = LandmarkObservation(
                    x_norm=float(landmark.x),
                    y_norm=float(landmark.y),
                    z_rel=float(landmark.z) if landmark.z is not None else None,
                    x_px=float(landmark.x * width),
                    y_px=float(landmark.y * height),
                    confidence=score,
                    visibility=None,
                    present=True,
                )
