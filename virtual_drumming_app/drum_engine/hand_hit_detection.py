"""Hand hit detection from smoothed pose frames."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Iterable

from .schema import PoseFrame, PosePoint
from .target_classification import (
    DrumTargetClassifier,
    DrumTargetHandSample,
    DrumTargetObservation,
    DrumTargetObservationFrame,
    KnnDrumTargetClassifier,
)

HAND_SIDES = ("left", "right")
HAND_JOINT_WRIST = "VNHumanHandPoseObservationJointNameWrist"
HAND_JOINT_THUMB_MP = "VNHumanHandPoseObservationJointNameThumbMP"
HAND_JOINT_MIDDLE_MCP = "VNHumanHandPoseObservationJointNameMiddleMCP"
HAND_JOINT_LITTLE_MCP = "VNHumanHandPoseObservationJointNameLittleMCP"
BODY_LEFT_ELBOW_POINT_ID = "body:VNHumanBodyPoseObservationJointNameLeftElbow"
BODY_RIGHT_ELBOW_POINT_ID = "body:VNHumanBodyPoseObservationJointNameRightElbow"
BODY_LEFT_SHOULDER_POINT_ID = "body:VNHumanBodyPoseObservationJointNameLeftShoulder"
BODY_RIGHT_SHOULDER_POINT_ID = "body:VNHumanBodyPoseObservationJointNameRightShoulder"


@dataclass(frozen=True, slots=True)
class HandHitEvent:
    side: str
    drum: str
    timestamp_seconds: float
    velocity: float
    wrist_x: float
    wrist_y: float
    strike_motion_y: float
    hand_motion_y: float
    forearm_motion_y: float
    context_name: str
    confidence: float


@dataclass(frozen=True, slots=True)
class HandHitDetectorConfig:
    up_threshold: float = 0.012
    down_velocity_threshold: float = 0.25
    min_hit_interval_seconds: float = 0.030
    max_missing_frames: int = 5
    target_history_size: int = 12
    hand_motion_weight: float = 0.6
    forearm_motion_weight: float = 0.4


@dataclass(slots=True)
class _HandState:
    last_motion_y: float | None = None
    last_timestamp_seconds: float | None = None
    lower_motion_y: float | None = None
    armed: bool = False
    last_hit_timestamp_seconds: float | None = None
    missing_frames: int = 0
    history: list[DrumTargetObservationFrame] = field(default_factory=list)


class HandHitDetector:
    def __init__(
        self,
        config: HandHitDetectorConfig | None = None,
        target_classifier: DrumTargetClassifier | None = None,
    ) -> None:
        self.config = config or HandHitDetectorConfig()
        self.target_classifier = target_classifier or KnnDrumTargetClassifier.from_default_model()
        self._states = {side: _HandState() for side in HAND_SIDES}

    def update(self, frame: PoseFrame) -> list[HandHitEvent]:
        samples = {side: _hand_sample(frame, side) for side in HAND_SIDES}
        events: list[HandHitEvent] = []
        for side in HAND_SIDES:
            state = self._states[side]
            sample = samples[side]
            if sample is None:
                self._mark_missing(side, state)
                continue

            event = self._update_hand(
                side=side,
                sample=sample,
                samples=samples,
                timestamp_seconds=frame.timestamp_seconds,
                state=state,
            )
            if event is not None:
                events.append(event)
        return events

    def classify_current_targets(self, frame: PoseFrame) -> dict[str, str]:
        samples = {side: _hand_sample(frame, side) for side in HAND_SIDES}
        predictions: dict[str, str] = {}
        for side, sample in samples.items():
            if sample is None:
                continue
            other_side = "right" if side == "left" else "left"
            strike_motion_y = sample.strike_motion_y(
                hand_weight=self.config.hand_motion_weight,
                forearm_weight=self.config.forearm_motion_weight,
            )
            state = self._states[side]
            prediction = self.target_classifier.classify(
                DrumTargetObservation(
                    side=side,
                    active=sample,
                    other=samples.get(other_side),
                    strike_motion_y=strike_motion_y,
                    timestamp_seconds=frame.timestamp_seconds,
                    history=tuple(state.history),
                )
            )
            if prediction is not None:
                predictions[side] = prediction.drum
        return predictions

    def reset(self) -> None:
        self._states = {side: _HandState() for side in HAND_SIDES}

    def _update_hand(
        self,
        *,
        side: str,
        sample: DrumTargetHandSample,
        samples: dict[str, DrumTargetHandSample | None],
        timestamp_seconds: float,
        state: _HandState,
    ) -> HandHitEvent | None:
        motion_y = sample.strike_motion_y(
            hand_weight=self.config.hand_motion_weight,
            forearm_weight=self.config.forearm_motion_weight,
        )
        if state.last_motion_y is None or state.last_timestamp_seconds is None:
            state.last_motion_y = motion_y
            state.last_timestamp_seconds = timestamp_seconds
            state.lower_motion_y = motion_y
            state.missing_frames = 0
            self._append_history(state, sample, motion_y, timestamp_seconds)
            return None

        previous_motion_y = state.last_motion_y
        dt = timestamp_seconds - state.last_timestamp_seconds
        if dt <= 0.0:
            dt = 1.0 / 60.0
        velocity = (motion_y - previous_motion_y) / dt

        state.missing_frames = 0
        if state.lower_motion_y is None:
            state.lower_motion_y = max(previous_motion_y, motion_y)

        event = self._detect_hit(
            side=side,
            sample=sample,
            samples=samples,
            timestamp_seconds=timestamp_seconds,
            motion_y=motion_y,
            velocity=velocity,
            state=state,
        )

        state.last_motion_y = motion_y
        state.last_timestamp_seconds = timestamp_seconds
        self._append_history(state, sample, motion_y, timestamp_seconds)
        return event

    def _detect_hit(
        self,
        *,
        side: str,
        sample: DrumTargetHandSample,
        samples: dict[str, DrumTargetHandSample | None],
        timestamp_seconds: float,
        motion_y: float,
        velocity: float,
        state: _HandState,
    ) -> HandHitEvent | None:
        assert state.lower_motion_y is not None

        if not state.armed:
            state.lower_motion_y = max(state.lower_motion_y, motion_y)
            if state.lower_motion_y - motion_y >= self.config.up_threshold:
                state.armed = True
            return None

        if velocity >= self.config.down_velocity_threshold:
            state.armed = False
            state.lower_motion_y = motion_y
            if self._cooldown_elapsed(state, timestamp_seconds):
                state.last_hit_timestamp_seconds = timestamp_seconds
                other_side = "right" if side == "left" else "left"
                prediction = self.target_classifier.classify(
                    DrumTargetObservation(
                        side=side,
                        active=sample,
                        other=samples.get(other_side),
                        strike_motion_y=motion_y,
                        timestamp_seconds=timestamp_seconds,
                        strike_velocity=velocity,
                        history=tuple(state.history),
                    )
                )
                if prediction is None:
                    return None
                return HandHitEvent(
                    side=side,
                    drum=prediction.drum,
                    timestamp_seconds=timestamp_seconds,
                    velocity=velocity,
                    wrist_x=sample.wrist_x,
                    wrist_y=sample.wrist_y,
                    strike_motion_y=motion_y,
                    hand_motion_y=sample.hand_motion_y,
                    forearm_motion_y=sample.forearm_motion_y,
                    context_name=prediction.context_name,
                    confidence=prediction.confidence,
                )
            return None

        if motion_y >= state.lower_motion_y - (self.config.up_threshold * 0.25):
            state.armed = False
            state.lower_motion_y = motion_y
        return None

    def _cooldown_elapsed(self, state: _HandState, timestamp_seconds: float) -> bool:
        if state.last_hit_timestamp_seconds is None:
            return True
        return timestamp_seconds - state.last_hit_timestamp_seconds >= self.config.min_hit_interval_seconds

    def _mark_missing(self, side: str, state: _HandState) -> None:
        state.missing_frames += 1
        if state.missing_frames > self.config.max_missing_frames:
            self._states[side] = _HandState()

    def _append_history(
        self,
        state: _HandState,
        sample: DrumTargetHandSample,
        strike_motion_y: float,
        timestamp_seconds: float,
    ) -> None:
        state.history.append(
            DrumTargetObservationFrame(
                active=sample,
                strike_motion_y=strike_motion_y,
                timestamp_seconds=timestamp_seconds,
            )
        )
        if len(state.history) > self.config.target_history_size:
            del state.history[: len(state.history) - self.config.target_history_size]


def target_observation_from_pose_frame(
    frame: PoseFrame,
    *,
    side: str,
    strike_velocity: float = 0.0,
    config: HandHitDetectorConfig | None = None,
    history_frames: Iterable[PoseFrame] = (),
) -> DrumTargetObservation | None:
    resolved_config = config or HandHitDetectorConfig()
    samples = {hand_side: _hand_sample(frame, hand_side) for hand_side in HAND_SIDES}
    active = samples.get(side)
    if active is None:
        return None
    other_side = "right" if side == "left" else "left"
    strike_motion_y = active.strike_motion_y(
        hand_weight=resolved_config.hand_motion_weight,
        forearm_weight=resolved_config.forearm_motion_weight,
    )
    history = _target_observation_history_from_pose_frames(
        history_frames,
        side=side,
        config=resolved_config,
    )
    return DrumTargetObservation(
        side=side,
        active=active,
        other=samples.get(other_side),
        strike_motion_y=strike_motion_y,
        timestamp_seconds=frame.timestamp_seconds,
        strike_velocity=strike_velocity,
        history=history,
    )


def _target_observation_history_from_pose_frames(
    frames: Iterable[PoseFrame],
    *,
    side: str,
    config: HandHitDetectorConfig,
) -> tuple[DrumTargetObservationFrame, ...]:
    history: list[DrumTargetObservationFrame] = []
    for frame in frames:
        sample = _hand_sample(frame, side)
        if sample is None:
            continue
        history.append(
            DrumTargetObservationFrame(
                active=sample,
                strike_motion_y=sample.strike_motion_y(
                    hand_weight=config.hand_motion_weight,
                    forearm_weight=config.forearm_motion_weight,
                ),
                timestamp_seconds=frame.timestamp_seconds,
            )
        )
    return tuple(history)


def _hand_sample(frame: PoseFrame, side: str) -> DrumTargetHandSample | None:
    wrist = _valid_point(frame.points.get(_point_id(side, HAND_JOINT_WRIST)))
    if wrist is None:
        return None
    elbow = _valid_point(frame.points.get(_elbow_point_id(side)))
    if elbow is None:
        return None

    thumb = _valid_point(frame.points.get(_point_id(side, HAND_JOINT_THUMB_MP)))
    middle = _valid_point(frame.points.get(_point_id(side, HAND_JOINT_MIDDLE_MCP)))
    little = _valid_point(frame.points.get(_point_id(side, HAND_JOINT_LITTLE_MCP)))
    if thumb is None or middle is None or little is None:
        return None
    shoulder = _valid_point(frame.points.get(_shoulder_point_id(side)))

    return DrumTargetHandSample(
        wrist_x=float(wrist.x),
        wrist_y=float(wrist.y),
        thumb_mcp_x=float(thumb.x),
        thumb_mcp_y=float(thumb.y),
        middle_mcp_x=float(middle.x),
        middle_mcp_y=float(middle.y),
        little_mcp_x=float(little.x),
        little_mcp_y=float(little.y),
        elbow_x=float(elbow.x),
        elbow_y=float(elbow.y),
        shoulder_x=float(shoulder.x) if shoulder is not None else None,
        shoulder_y=float(shoulder.y) if shoulder is not None else None,
    )


def _valid_point(point: PosePoint | None) -> PosePoint | None:
    if point is None or not point.tracking_present or point.x is None or point.y is None:
        return None
    return point


def _point_id(side: str, joint: str) -> str:
    return f"hand_{side}:{joint}"


def _elbow_point_id(side: str) -> str:
    return BODY_LEFT_ELBOW_POINT_ID if side == "left" else BODY_RIGHT_ELBOW_POINT_ID


def _shoulder_point_id(side: str) -> str:
    return BODY_LEFT_SHOULDER_POINT_ID if side == "left" else BODY_RIGHT_SHOULDER_POINT_ID
