"""Apple Vision backend that merges 2D hands with 3D body pose depth."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from models.canonical import canonical_id
from models.schema import FrameResult, LandmarkObservation

from .apple_vision import AppleVisionBackend
from .base import BackendUnavailableError


BODY_3D_FALLBACK_CONSTANTS = {
    "VNHumanBodyPose3DObservationJointNameCenterHead": canonical_id("body", "center", "nose"),
    "VNHumanBodyPose3DObservationJointNameHead": canonical_id("body", "center", "nose"),
    "VNHumanBodyPose3DObservationJointNameTopHead": canonical_id("body", "center", "nose"),
    "VNHumanBodyPose3DObservationJointNameLeftShoulder": canonical_id("body", "left", "shoulder"),
    "VNHumanBodyPose3DObservationJointNameRightShoulder": canonical_id("body", "right", "shoulder"),
    "VNHumanBodyPose3DObservationJointNameLeftElbow": canonical_id("body", "left", "elbow"),
    "VNHumanBodyPose3DObservationJointNameRightElbow": canonical_id("body", "right", "elbow"),
    "VNHumanBodyPose3DObservationJointNameLeftWrist": canonical_id("body", "left", "wrist"),
    "VNHumanBodyPose3DObservationJointNameRightWrist": canonical_id("body", "right", "wrist"),
    "VNHumanBodyPose3DObservationJointNameLeftHip": canonical_id("body", "left", "hip"),
    "VNHumanBodyPose3DObservationJointNameRightHip": canonical_id("body", "right", "hip"),
    "VNHumanBodyPose3DObservationJointNameLeftKnee": canonical_id("body", "left", "knee"),
    "VNHumanBodyPose3DObservationJointNameRightKnee": canonical_id("body", "right", "knee"),
    "VNHumanBodyPose3DObservationJointNameLeftAnkle": canonical_id("body", "left", "ankle"),
    "VNHumanBodyPose3DObservationJointNameRightAnkle": canonical_id("body", "right", "ankle"),
}

BODY_3D_CANONICAL_BY_NORMALIZED_NAME = {
    "centerhead": canonical_id("body", "center", "nose"),
    "head": canonical_id("body", "center", "nose"),
    "tophead": canonical_id("body", "center", "nose"),
    "nose": canonical_id("body", "center", "nose"),
    "leftshoulder": canonical_id("body", "left", "shoulder"),
    "rightshoulder": canonical_id("body", "right", "shoulder"),
    "leftelbow": canonical_id("body", "left", "elbow"),
    "rightelbow": canonical_id("body", "right", "elbow"),
    "leftwrist": canonical_id("body", "left", "wrist"),
    "rightwrist": canonical_id("body", "right", "wrist"),
    "lefthip": canonical_id("body", "left", "hip"),
    "righthip": canonical_id("body", "right", "hip"),
    "leftknee": canonical_id("body", "left", "knee"),
    "rightknee": canonical_id("body", "right", "knee"),
    "leftankle": canonical_id("body", "left", "ankle"),
    "rightankle": canonical_id("body", "right", "ankle"),
}

BODY_3D_NAME_PRIORITY = {
    "centerhead": 0,
    "head": 1,
    "tophead": 2,
    "nose": 3,
}


class AppleVision3DBackend(AppleVisionBackend):
    backend_name = "apple_vision_3d"

    def __init__(self, *, body_3d_stride: int = 3) -> None:
        super().__init__()
        if body_3d_stride < 1:
            raise ValueError("`body_3d_stride` must be at least 1.")
        self.body_3d_stride = int(body_3d_stride)
        self.body_3d_request_class = None
        self.body_3d_joint_map: list[tuple[Any, str]] = []
        self.body_request = None
        self.body_3d_request = None
        self.hand_request = None
        self.frame_counter = 0
        self.cached_body_depths: dict[str, tuple[float, float | None]] = {}

    def initialize(self) -> None:
        super().initialize()
        if self.Vision is None:
            raise RuntimeError("Backend was not initialized.")

        self.body_3d_request_class = getattr(self.Vision, "VNDetectHumanBodyPose3DRequest", None)
        if self.body_3d_request_class is None:
            raise BackendUnavailableError(
                "Apple Vision 3D body pose is not available in this PyObjC/Vision runtime. "
                "Use `apple_vision` or upgrade to a macOS release that exposes "
                "`VNDetectHumanBodyPose3DRequest`."
            )

        self.body_3d_joint_map = [
            (getattr(self.Vision, joint_constant), landmark_id)
            for joint_constant, landmark_id in BODY_3D_FALLBACK_CONSTANTS.items()
            if hasattr(self.Vision, joint_constant)
        ]
        self.body_request = self.Vision.VNDetectHumanBodyPoseRequest.alloc().init()
        self.body_3d_request = self._make_body_3d_request()
        self.hand_request = self.Vision.VNDetectHumanHandPoseRequest.alloc().init()
        self.hand_request.setMaximumHandCount_(2)

    def process_frame(self, frame_bgr, timestamp_monotonic_sec: float) -> FrameResult:
        if (
            self.Vision is None
            or self.Quartz is None
            or self.Foundation is None
            or self.body_3d_request_class is None
            or self.body_request is None
            or self.body_3d_request is None
            or self.hand_request is None
        ):
            raise RuntimeError("Backend was not initialized.")

        cg_image = self._frame_to_cgimage(frame_bgr)
        handler = self.Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg_image, None)

        self.frame_counter += 1
        should_run_body_3d = (self.frame_counter % self.body_3d_stride) == 1
        requests = [self.body_request, self.hand_request]
        if should_run_body_3d:
            requests.insert(1, self.body_3d_request)

        ok, error = handler.performRequests_error_(requests, None)
        if not ok:
            message = str(error) if error is not None else "unknown Vision error"
            raise RuntimeError(f"Apple Vision request failed: {message}")

        width = frame_bgr.shape[1]
        height = frame_bgr.shape[0]
        landmarks: dict[str, LandmarkObservation] = {}
        self._populate_body_landmarks(landmarks, self.body_request.results(), width, height)
        if should_run_body_3d:
            self._populate_body_depth_landmarks(landmarks, self.body_3d_request.results())
        else:
            self._apply_cached_body_depths(landmarks)
        self._populate_hand_landmarks(landmarks, self.hand_request.results(), width, height)
        return FrameResult(
            landmarks=landmarks,
            person_id=0,
            backend_metadata={
                "timestamp_monotonic_sec": timestamp_monotonic_sec,
                "body_3d_ran_this_frame": should_run_body_3d,
            },
        )

    def get_configuration(self) -> dict[str, Any]:
        return {
            "framework": "Vision",
            "platform": "macOS",
            "body_pose_dimension": "3d",
            "body_3d_stride": self.body_3d_stride,
            "hands_enabled": True,
            "hand_pose_dimension": "2d",
            "hand_z_source": "matching_body_wrist",
        }

    def shutdown(self) -> None:
        super().shutdown()
        self.body_request = None
        self.body_3d_request = None
        self.hand_request = None
        self.cached_body_depths.clear()

    def _make_body_3d_request(self):
        try:
            return self.body_3d_request_class.alloc().initWithCompletionHandler_(None)
        except Exception as exc:
            raise BackendUnavailableError(
                "Failed to initialize `VNDetectHumanBodyPose3DRequest` in this Vision runtime. "
                "Apple's 3D body pose model could not be loaded on this machine, so "
                "`apple_vision_3d` is not usable here."
            ) from exc

    def _populate_body_depth_landmarks(self, target, observations) -> None:
        if not observations:
            return

        self.cached_body_depths.clear()
        observation = observations[0]
        for joint_name, landmark_id in self._resolve_body_3d_joint_entries(observation):
            point = self._recognized_3d_point(observation, joint_name)
            if point is None:
                continue

            z_rel = self._extract_point3d_z(point)
            if z_rel is None:
                continue

            confidence = self._extract_numeric_attribute(point, "confidence")
            present = (confidence is None) or confidence > 0.0
            self.cached_body_depths[landmark_id] = (z_rel, confidence)
            existing = target.get(landmark_id)
            if existing is None:
                target[landmark_id] = LandmarkObservation(
                    z_rel=z_rel,
                    confidence=confidence,
                    visibility=confidence,
                    present=present,
                )
                continue

            existing.z_rel = z_rel
            if existing.confidence is None and confidence is not None:
                existing.confidence = confidence
            if existing.visibility is None and confidence is not None:
                existing.visibility = confidence
            existing.present = existing.present or present

    def _apply_cached_body_depths(self, target: dict[str, LandmarkObservation]) -> None:
        for landmark_id, (z_rel, confidence) in self.cached_body_depths.items():
            existing = target.get(landmark_id)
            if existing is None:
                continue
            existing.z_rel = z_rel
            if existing.confidence is None and confidence is not None:
                existing.confidence = confidence
            if existing.visibility is None and confidence is not None:
                existing.visibility = confidence

    def _populate_hand_landmarks(self, target, observations, width: int, height: int) -> None:
        if not observations:
            return

        hand_entries = self._resolve_hand_sides(observations, target)

        for side, observation in hand_entries:
            hand_depth = self._body_wrist_depth(target, side)
            group_name = f"{side}_hand"
            for vision_name, canonical_name in self.hand_joint_map:
                point, error = observation.recognizedPointForJointName_error_(vision_name, None)
                if point is None or error is not None:
                    continue
                target[canonical_id(group_name, side, canonical_name)] = LandmarkObservation(
                    x_norm=float(point.x()),
                    y_norm=float(1.0 - point.y()),
                    z_rel=hand_depth,
                    x_px=float(point.x() * width),
                    y_px=float((1.0 - point.y()) * height),
                    confidence=float(point.confidence()),
                    visibility=None,
                    present=float(point.confidence()) > 0.0,
                )

    def _resolve_body_3d_joint_entries(self, observation) -> list[tuple[Any, str]]:
        discovered = self._discover_body_3d_joint_entries(observation)
        if discovered:
            return discovered
        return list(self.body_3d_joint_map)

    def _discover_body_3d_joint_entries(self, observation) -> list[tuple[Any, str]]:
        if not hasattr(observation, "availableJointNames"):
            return []

        try:
            raw_joint_names = list(observation.availableJointNames())
        except Exception:
            return []

        candidates: list[tuple[int, Any, str]] = []
        for joint_name in raw_joint_names:
            normalized_name = self._normalize_joint_name(joint_name)
            landmark_id = BODY_3D_CANONICAL_BY_NORMALIZED_NAME.get(normalized_name)
            if landmark_id is None:
                continue
            candidates.append((BODY_3D_NAME_PRIORITY.get(normalized_name, 10), joint_name, landmark_id))

        candidates.sort(key=lambda entry: entry[0])
        resolved: list[tuple[Any, str]] = []
        used_landmark_ids: set[str] = set()
        for _, joint_name, landmark_id in candidates:
            if landmark_id in used_landmark_ids:
                continue
            resolved.append((joint_name, landmark_id))
            used_landmark_ids.add(landmark_id)
        return resolved

    @staticmethod
    def _recognized_3d_point(observation, joint_name):
        for selector_name in ("recognizedPointForJointName_error_", "recognizedPoint_error_"):
            selector = getattr(observation, selector_name, None)
            if selector is None:
                continue
            try:
                point, error = selector(joint_name, None)
            except Exception:
                continue
            if point is not None and error is None:
                return point
        return None

    @classmethod
    def _extract_point3d_z(cls, point) -> float | None:
        direct_z = cls._extract_numeric_attribute(point, "z")
        if direct_z is not None:
            return direct_z

        position = cls._extract_attribute(point, "position")
        if position is None:
            return None
        return cls._extract_translation_component(position, axis_index=2)

    @staticmethod
    def _body_wrist_depth(
        body_landmarks: dict[str, LandmarkObservation],
        side: str,
    ) -> float | None:
        wrist = body_landmarks.get(canonical_id("body", side, "wrist"))
        if wrist is None or not wrist.present:
            return None
        return wrist.z_rel

    @staticmethod
    def _normalize_joint_name(joint_name: Any) -> str:
        text = str(joint_name)
        text = text.replace("VNHumanBodyPose3DObservationJointName", "")
        text = text.replace("VNHumanBodyPoseObservationJointName", "")
        return re.sub(r"[^a-z0-9]", "", text.lower())

    @classmethod
    def _extract_numeric_attribute(cls, obj: Any, attr_name: str) -> float | None:
        value = cls._extract_attribute(obj, attr_name)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _extract_attribute(obj: Any, attr_name: str):
        attribute = getattr(obj, attr_name, None)
        if attribute is None:
            return None
        try:
            return attribute()
        except TypeError:
            return attribute
        except Exception:
            return None

    @classmethod
    def _extract_translation_component(cls, matrix: Any, axis_index: int) -> float | None:
        columns = cls._extract_attribute(matrix, "columns")
        if columns is not None:
            component = cls._extract_vector_component(columns, vector_index=3, axis_index=axis_index)
            if component is not None:
                return component

        iterable_value = cls._coerce_iterable(matrix)
        if iterable_value is None:
            return None

        for vector_index, inner_axis_index in ((3, axis_index), (axis_index, 3)):
            component = cls._extract_vector_component(
                iterable_value,
                vector_index=vector_index,
                axis_index=inner_axis_index,
            )
            if component is not None:
                return component
        return None

    @classmethod
    def _extract_vector_component(
        cls,
        sequence_like: Any,
        *,
        vector_index: int,
        axis_index: int,
    ) -> float | None:
        outer = cls._coerce_iterable(sequence_like)
        if outer is None or len(outer) <= vector_index:
            return None

        inner = cls._coerce_iterable(outer[vector_index])
        if inner is None or len(inner) <= axis_index:
            return None

        try:
            return float(inner[axis_index])
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _coerce_iterable(value: Any) -> list[Any] | None:
        if value is None or isinstance(value, (str, bytes)):
            return None
        if isinstance(value, Iterable):
            try:
                return list(value)
            except TypeError:
                return None
        return None
