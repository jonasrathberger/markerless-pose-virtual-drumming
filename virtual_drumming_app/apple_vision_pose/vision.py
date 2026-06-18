"""Apple Vision body and hand pose inference."""

from __future__ import annotations

from typing import Any

from .constants import (
    BODY_JOINT_CONSTANTS,
    HAND_JOINT_CONSTANTS,
    TRACKED_BODY_JOINT_CONSTANTS,
    TRACKED_HAND_JOINT_CONSTANTS,
)
from .dependencies import load_opencv
from .hand_tracking import HandIdentityTracker
from .models import Landmark
from .transforms import normalize_landmarks_to_shoulder_center


class AppleVisionPose:
    def __init__(
        self,
        max_hands: int = 2,
        detect_width: int = 640,
        extract_all_landmarks: bool = False,
    ) -> None:
        try:
            import Foundation
            import Quartz
            import Vision
        except ImportError as exc:
            raise RuntimeError(
                "Missing PyObjC Vision bindings. From the repository root, run:\n"
                "  python -m pip install -r requirements.txt"
            ) from exc

        self.Foundation = Foundation
        self.Quartz = Quartz
        self.Vision = Vision
        self.max_hands = max_hands
        self.detect_width = detect_width
        self.extract_all_landmarks = extract_all_landmarks
        body_joint_constants = BODY_JOINT_CONSTANTS if extract_all_landmarks else TRACKED_BODY_JOINT_CONSTANTS
        hand_joint_constants = HAND_JOINT_CONSTANTS if extract_all_landmarks else TRACKED_HAND_JOINT_CONSTANTS
        self.body_joints = self._available_joints(body_joint_constants)
        self.hand_joints = self._available_joints(hand_joint_constants)
        self.body_request = self.Vision.VNDetectHumanBodyPoseRequest.alloc().init()
        self.hand_request = self.Vision.VNDetectHumanHandPoseRequest.alloc().init()
        self.hand_request.setMaximumHandCount_(self.max_hands)
        self.requests = [self.body_request, self.hand_request]
        self.sequence_handler = self.Vision.VNSequenceRequestHandler.alloc().init()
        self.body_all_group = getattr(self.Vision, "VNHumanBodyPoseObservationJointsGroupNameAll", None)
        self.hand_all_group = getattr(self.Vision, "VNHumanHandPoseObservationJointsGroupNameAll", None)
        self.hand_identity_tracker = HandIdentityTracker()

        if not self.body_joints:
            raise RuntimeError("No Apple Vision body pose joint constants are available.")
        if not self.hand_joints:
            raise RuntimeError("No Apple Vision hand pose joint constants are available.")

    def reset_tracking(self) -> None:
        self.sequence_handler = self.Vision.VNSequenceRequestHandler.alloc().init()
        self.hand_identity_tracker = HandIdentityTracker()

    def detect(self, frame_bgr: Any) -> dict[str, Landmark]:
        height, width = frame_bgr.shape[:2]
        detection_frame = self._detection_frame(frame_bgr)
        cg_image = self._frame_to_cgimage(detection_frame)
        handler = self.Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg_image, None)

        ok, error = handler.performRequests_error_(self.requests, None)
        if not ok:
            message = str(error) if error is not None else "unknown Vision error"
            raise RuntimeError(f"Apple Vision request failed: {message}")

        landmarks: dict[str, Landmark] = {}
        self._read_body_landmarks(landmarks, self.body_request.results(), width, height)
        self._read_hand_landmarks(landmarks, self.hand_request.results(), width, height)
        landmarks = self.hand_identity_tracker.assign(landmarks)
        normalize_landmarks_to_shoulder_center(landmarks)
        return landmarks

    def detect_pixel_buffer(self, pixel_buffer: Any, width: int, height: int) -> dict[str, Landmark]:
        try:
            ok, error = self.sequence_handler.performRequests_onCVPixelBuffer_error_(self.requests, pixel_buffer, None)
        except (AttributeError, TypeError):
            handler = self.Vision.VNImageRequestHandler.alloc().initWithCVPixelBuffer_options_(pixel_buffer, None)
            ok, error = handler.performRequests_error_(self.requests, None)

        if not ok:
            message = str(error) if error is not None else "unknown Vision error"
            raise RuntimeError(f"Apple Vision request failed: {message}")

        landmarks: dict[str, Landmark] = {}
        self._read_body_landmarks(landmarks, self.body_request.results(), width, height)
        self._read_hand_landmarks(landmarks, self.hand_request.results(), width, height)
        landmarks = self.hand_identity_tracker.assign(landmarks)
        normalize_landmarks_to_shoulder_center(landmarks)
        return landmarks

    def _available_joints(self, names: list[str]) -> list[tuple[str, Any]]:
        return [(name, getattr(self.Vision, name)) for name in names if hasattr(self.Vision, name)]

    def _frame_to_cgimage(self, frame_bgr: Any) -> Any:
        cv2 = load_opencv()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        height, width = frame_rgb.shape[:2]
        bytes_per_row = width * 3
        data = self.Foundation.NSData.dataWithBytes_length_(frame_rgb.tobytes(), frame_rgb.nbytes)
        provider = self.Quartz.CGDataProviderCreateWithCFData(data)
        color_space = self.Quartz.CGColorSpaceCreateDeviceRGB()
        return self.Quartz.CGImageCreate(
            width,
            height,
            8,
            24,
            bytes_per_row,
            color_space,
            self.Quartz.kCGImageAlphaNone,
            provider,
            None,
            False,
            self.Quartz.kCGRenderingIntentDefault,
        )

    def _detection_frame(self, frame_bgr: Any) -> Any:
        cv2 = load_opencv()
        if self.detect_width <= 0:
            return frame_bgr

        height, width = frame_bgr.shape[:2]
        if width <= self.detect_width:
            return frame_bgr

        detect_height = max(1, round(height * (self.detect_width / width)))
        return cv2.resize(frame_bgr, (self.detect_width, detect_height), interpolation=cv2.INTER_AREA)

    def _read_body_landmarks(
        self,
        target: dict[str, Landmark],
        observations: Any,
        width: int,
        height: int,
    ) -> None:
        if not observations:
            return

        observation = observations[0]
        recognized_points = self._recognized_points_for_group(observation, self.body_all_group)
        for joint_name, vision_joint in self.body_joints:
            landmark = self._read_point(
                observation=observation,
                vision_joint=vision_joint,
                recognized_points=recognized_points,
                group="body",
                joint_name=joint_name,
                width=width,
                height=height,
            )
            if landmark is not None:
                target[f"body:{joint_name}"] = landmark

    def _read_hand_landmarks(
        self,
        target: dict[str, Landmark],
        observations: Any,
        width: int,
        height: int,
    ) -> None:
        if not observations:
            return

        for hand_index, observation in enumerate(observations):
            chirality = self._extract_chirality(observation)
            recognized_points = self._recognized_points_for_group(observation, self.hand_all_group)
            group = f"hand_{hand_index}"
            for joint_name, vision_joint in self.hand_joints:
                landmark = self._read_point(
                    observation=observation,
                    vision_joint=vision_joint,
                    recognized_points=recognized_points,
                    group=group,
                    joint_name=joint_name,
                    width=width,
                    height=height,
                    chirality=chirality,
                )
                if landmark is not None:
                    target[f"{group}:{joint_name}"] = landmark

    def _read_point(
        self,
        *,
        observation: Any,
        vision_joint: Any,
        recognized_points: Any | None,
        group: str,
        joint_name: str,
        width: int,
        height: int,
        chirality: str | None = None,
    ) -> Landmark | None:
        point = self._point_from_recognized_points(recognized_points, vision_joint)
        error = None
        if point is None:
            point, error = observation.recognizedPointForJointName_error_(vision_joint, None)
        if point is None or error is not None:
            return None

        x_norm = float(point.x())
        y_norm = float(1.0 - point.y())
        confidence = float(point.confidence())
        return Landmark(
            group=group,
            joint=joint_name,
            x_norm=x_norm,
            y_norm=y_norm,
            x_rel=None,
            y_rel=None,
            x_px=float(x_norm * width),
            y_px=float(y_norm * height),
            confidence=confidence,
            tracking_present=confidence > 0.0,
            chirality=chirality,
        )

    @staticmethod
    def _recognized_points_for_group(observation: Any, group_key: Any | None) -> Any | None:
        if group_key is None or not hasattr(observation, "recognizedPointsForGroupKey_error_"):
            return None
        try:
            points, error = observation.recognizedPointsForGroupKey_error_(group_key, None)
        except (AttributeError, TypeError):
            return None
        if points is None or error is not None:
            return None
        return points

    @staticmethod
    def _point_from_recognized_points(recognized_points: Any | None, vision_joint: Any) -> Any | None:
        if recognized_points is None:
            return None
        if hasattr(recognized_points, "objectForKey_"):
            return recognized_points.objectForKey_(vision_joint)
        return recognized_points.get(vision_joint)

    @staticmethod
    def _extract_chirality(observation: Any) -> str | None:
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
        return raw_text or None
