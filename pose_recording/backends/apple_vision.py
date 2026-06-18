"""Best-effort Apple Vision body and hand pose backend via PyObjC."""

from __future__ import annotations

from typing import Any

import cv2

from models.canonical import HAND_BASE_NAMES, canonical_id
from models.schema import FrameResult, LandmarkObservation

from .base import BackendUnavailableError, PoseBackend


VISION_BODY_MAP = {
    "VNHumanBodyPoseObservationJointNameNose": canonical_id("body", "center", "nose"),
    "VNHumanBodyPoseObservationJointNameLeftShoulder": canonical_id("body", "left", "shoulder"),
    "VNHumanBodyPoseObservationJointNameRightShoulder": canonical_id("body", "right", "shoulder"),
    "VNHumanBodyPoseObservationJointNameLeftElbow": canonical_id("body", "left", "elbow"),
    "VNHumanBodyPoseObservationJointNameRightElbow": canonical_id("body", "right", "elbow"),
    "VNHumanBodyPoseObservationJointNameLeftWrist": canonical_id("body", "left", "wrist"),
    "VNHumanBodyPoseObservationJointNameRightWrist": canonical_id("body", "right", "wrist"),
    "VNHumanBodyPoseObservationJointNameLeftHip": canonical_id("body", "left", "hip"),
    "VNHumanBodyPoseObservationJointNameRightHip": canonical_id("body", "right", "hip"),
    "VNHumanBodyPoseObservationJointNameLeftKnee": canonical_id("body", "left", "knee"),
    "VNHumanBodyPoseObservationJointNameRightKnee": canonical_id("body", "right", "knee"),
    "VNHumanBodyPoseObservationJointNameLeftAnkle": canonical_id("body", "left", "ankle"),
    "VNHumanBodyPoseObservationJointNameRightAnkle": canonical_id("body", "right", "ankle"),
}

VISION_HAND_NAME_MAP = {
    "VNHumanHandPoseObservationJointNameWrist": "wrist",
    "VNHumanHandPoseObservationJointNameThumbCMC": "thumb_cmc",
    "VNHumanHandPoseObservationJointNameThumbMP": "thumb_mcp",
    "VNHumanHandPoseObservationJointNameThumbIP": "thumb_ip",
    "VNHumanHandPoseObservationJointNameThumbTip": "thumb_tip",
    "VNHumanHandPoseObservationJointNameIndexMCP": "index_mcp",
    "VNHumanHandPoseObservationJointNameIndexPIP": "index_pip",
    "VNHumanHandPoseObservationJointNameIndexDIP": "index_dip",
    "VNHumanHandPoseObservationJointNameIndexTip": "index_tip",
    "VNHumanHandPoseObservationJointNameMiddleMCP": "middle_mcp",
    "VNHumanHandPoseObservationJointNameMiddlePIP": "middle_pip",
    "VNHumanHandPoseObservationJointNameMiddleDIP": "middle_dip",
    "VNHumanHandPoseObservationJointNameMiddleTip": "middle_tip",
    "VNHumanHandPoseObservationJointNameRingMCP": "ring_mcp",
    "VNHumanHandPoseObservationJointNameRingPIP": "ring_pip",
    "VNHumanHandPoseObservationJointNameRingDIP": "ring_dip",
    "VNHumanHandPoseObservationJointNameRingTip": "ring_tip",
    "VNHumanHandPoseObservationJointNameLittleMCP": "pinky_mcp",
    "VNHumanHandPoseObservationJointNameLittlePIP": "pinky_pip",
    "VNHumanHandPoseObservationJointNameLittleDIP": "pinky_dip",
    "VNHumanHandPoseObservationJointNameLittleTip": "pinky_tip",
}


class AppleVisionBackend(PoseBackend):
    backend_name = "apple_vision"

    def __init__(self) -> None:
        super().__init__()
        self.Foundation = None
        self.Quartz = None
        self.Vision = None
        self.body_joint_map: dict[Any, str] = {}
        self.hand_joint_map: list[tuple[Any, str]] = []

    def initialize(self) -> None:
        try:
            import Foundation
            import Quartz
            import Vision
        except ImportError as exc:
            raise BackendUnavailableError(
                "PyObjC Vision bindings are not installed. Install the `apple-vision` extra "
                "or the required PyObjC framework packages listed in `requirements.txt`."
            ) from exc

        self.Foundation = Foundation
        self.Quartz = Quartz
        self.Vision = Vision
        self.body_joint_map = {
            getattr(Vision, joint_constant): landmark_id
            for joint_constant, landmark_id in VISION_BODY_MAP.items()
            if hasattr(Vision, joint_constant)
        }
        self.hand_joint_map = [
            (getattr(Vision, joint_constant), canonical_name)
            for joint_constant, canonical_name in VISION_HAND_NAME_MAP.items()
            if hasattr(Vision, joint_constant)
        ]

    def process_frame(self, frame_bgr, timestamp_monotonic_sec: float) -> FrameResult:
        if self.Vision is None or self.Quartz is None or self.Foundation is None:
            raise RuntimeError("Backend was not initialized.")

        cg_image = self._frame_to_cgimage(frame_bgr)
        handler = self.Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg_image, None)

        body_request = self.Vision.VNDetectHumanBodyPoseRequest.alloc().init()
        hand_request = self.Vision.VNDetectHumanHandPoseRequest.alloc().init()
        hand_request.setMaximumHandCount_(2)

        ok, error = handler.performRequests_error_([body_request, hand_request], None)
        if not ok:
            message = str(error) if error is not None else "unknown Vision error"
            raise RuntimeError(f"Apple Vision request failed: {message}")

        landmarks: dict[str, LandmarkObservation] = {}
        self._populate_body_landmarks(landmarks, body_request.results(), frame_bgr.shape[1], frame_bgr.shape[0])
        self._populate_hand_landmarks(landmarks, hand_request.results(), frame_bgr.shape[1], frame_bgr.shape[0])
        return FrameResult(
            landmarks=landmarks,
            person_id=0,
            backend_metadata={"timestamp_monotonic_sec": timestamp_monotonic_sec},
        )

    def get_configuration(self) -> dict[str, Any]:
        return {"framework": "Vision", "platform": "macOS", "hands_enabled": True}

    def shutdown(self) -> None:
        self.Vision = None
        self.Quartz = None
        self.Foundation = None

    def _frame_to_cgimage(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        height, width = frame_rgb.shape[:2]
        bytes_per_row = width * 3
        data = self.Foundation.NSData.dataWithBytes_length_(frame_rgb.tobytes(), frame_rgb.nbytes)
        provider = self.Quartz.CGDataProviderCreateWithCFData(data)
        color_space = self.Quartz.CGColorSpaceCreateDeviceRGB()
        bitmap_info = self.Quartz.kCGImageAlphaNone
        return self.Quartz.CGImageCreate(
            width,
            height,
            8,
            24,
            bytes_per_row,
            color_space,
            bitmap_info,
            provider,
            None,
            False,
            self.Quartz.kCGRenderingIntentDefault,
        )

    def _populate_body_landmarks(self, target, observations, width: int, height: int) -> None:
        if not observations:
            return
        observation = observations[0]
        for joint_name, landmark_id in self.body_joint_map.items():
            point, error = observation.recognizedPointForJointName_error_(joint_name, None)
            if point is None or error is not None:
                continue
            target[landmark_id] = LandmarkObservation(
                x_norm=float(point.x()),
                y_norm=float(1.0 - point.y()),
                z_rel=None,
                x_px=float(point.x() * width),
                y_px=float((1.0 - point.y()) * height),
                confidence=float(point.confidence()),
                visibility=float(point.confidence()),
                present=float(point.confidence()) > 0.0,
            )

    def _populate_hand_landmarks(self, target, observations, width: int, height: int) -> None:
        if not observations:
            return

        hand_entries = self._resolve_hand_sides(observations, target)

        for side, observation in hand_entries:
            group_name = f"{side}_hand"
            for vision_name, canonical_name in self.hand_joint_map:
                point, error = observation.recognizedPointForJointName_error_(vision_name, None)
                if point is None or error is not None:
                    continue
                target[canonical_id(group_name, side, canonical_name)] = LandmarkObservation(
                    x_norm=float(point.x()),
                    y_norm=float(1.0 - point.y()),
                    z_rel=None,
                    x_px=float(point.x() * width),
                    y_px=float((1.0 - point.y()) * height),
                    confidence=float(point.confidence()),
                    visibility=None,
                    present=float(point.confidence()) > 0.0,
                )

    def _resolve_hand_sides(
        self,
        observations,
        body_landmarks: dict[str, LandmarkObservation],
    ) -> list[tuple[str, Any]]:
        entries: list[dict[str, Any]] = []
        has_body_wrist_reference = self._has_body_wrist_reference(body_landmarks)

        for observation in observations:
            wrist_x = self._get_hand_wrist_x(observation)
            # Prefer matching hands to the body wrists when the body detector is present.
            # Vision hand chirality can be inverted relative to the subject on front-facing
            # webcam feeds, which produced swapped labels on MacBook cameras.
            side = (
                self._infer_hand_side_from_body(wrist_x, body_landmarks)
                if has_body_wrist_reference
                else self._extract_chirality(observation)
            )
            entries.append({"observation": observation, "side": side, "wrist_x": wrist_x})

        used_sides = {entry["side"] for entry in entries if entry["side"] in {"left", "right"}}
        unresolved_entries = [entry for entry in entries if entry["side"] not in {"left", "right"}]
        for entry in unresolved_entries:
            inferred_side = (
                self._extract_chirality(entry["observation"])
                if has_body_wrist_reference
                else self._infer_hand_side_from_body(entry["wrist_x"], body_landmarks)
            )
            if inferred_side is not None and inferred_side not in used_sides:
                entry["side"] = inferred_side
                used_sides.add(inferred_side)

        unresolved_entries = [entry for entry in entries if entry["side"] not in {"left", "right"}]
        if len(unresolved_entries) == 1 and len(entries) == 2 and len(used_sides) == 1:
            unresolved_entries[0]["side"] = "right" if "left" in used_sides else "left"
            used_sides.add(unresolved_entries[0]["side"])

        unresolved_entries = [entry for entry in entries if entry["side"] not in {"left", "right"}]
        if unresolved_entries:
            unresolved_entries.sort(
                key=lambda entry: 0.5 if entry["wrist_x"] is None else float(entry["wrist_x"])
            )
            for index, entry in enumerate(unresolved_entries):
                # Front-facing webcam fallback: the participant's left hand usually appears
                # on the right side of the image.
                heuristic_side = "right" if index == 0 else "left"
                if heuristic_side in used_sides and len(used_sides) < 2:
                    heuristic_side = "left" if heuristic_side == "right" else "right"
                entry["side"] = heuristic_side
                used_sides.add(heuristic_side)

        return [(entry["side"], entry["observation"]) for entry in entries if entry["side"] in {"left", "right"}]

    def _extract_chirality(self, observation) -> str | None:
        if not hasattr(observation, "chirality"):
            return None
        try:
            raw_value = observation.chirality()
        except Exception:
            return None
        raw_text = str(raw_value).strip().lower()
        if "left" in raw_text:
            return "left"
        if "right" in raw_text:
            return "right"
        if raw_text in {"1", "2"}:
            return "left" if raw_text == "1" else "right"
        return None

    def _infer_hand_side_from_body(
        self,
        wrist_x: float | None,
        body_landmarks: dict[str, LandmarkObservation],
    ) -> str | None:
        if wrist_x is None:
            return None
        left_wrist = body_landmarks.get(canonical_id("body", "left", "wrist"))
        right_wrist = body_landmarks.get(canonical_id("body", "right", "wrist"))
        if left_wrist and left_wrist.present and left_wrist.x_norm is not None:
            if right_wrist and right_wrist.present and right_wrist.x_norm is not None:
                left_distance = abs(wrist_x - float(left_wrist.x_norm))
                right_distance = abs(wrist_x - float(right_wrist.x_norm))
                return "left" if left_distance <= right_distance else "right"
            return "left"
        if right_wrist and right_wrist.present and right_wrist.x_norm is not None:
            return "right"
        return None

    @staticmethod
    def _has_body_wrist_reference(body_landmarks: dict[str, LandmarkObservation]) -> bool:
        left_wrist = body_landmarks.get(canonical_id("body", "left", "wrist"))
        right_wrist = body_landmarks.get(canonical_id("body", "right", "wrist"))
        return any(
            wrist and wrist.present and wrist.x_norm is not None
            for wrist in (left_wrist, right_wrist)
        )

    @staticmethod
    def _get_hand_wrist_x(observation) -> float | None:
        try:
            import Vision
            joint_name = Vision.VNHumanHandPoseObservationJointNameWrist
        except Exception:
            joint_name = "wrist"
        point, error = observation.recognizedPointForJointName_error_(joint_name, None)
        if point is None or error is not None:
            return None
        return float(point.x())
