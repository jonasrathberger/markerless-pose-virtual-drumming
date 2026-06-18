"""Pedal hit detection from smoothed pose frames."""

from __future__ import annotations

from dataclasses import dataclass

from .schema import PoseFrame, PosePoint

LEFT_KNEE_POINT_ID = "body:VNHumanBodyPoseObservationJointNameLeftKnee"
RIGHT_KNEE_POINT_ID = "body:VNHumanBodyPoseObservationJointNameRightKnee"
LEFT_HIP_POINT_ID = "body:VNHumanBodyPoseObservationJointNameLeftHip"
RIGHT_HIP_POINT_ID = "body:VNHumanBodyPoseObservationJointNameRightHip"
DEFAULT_PEDAL_POINTS = {
    "left": LEFT_KNEE_POINT_ID,
    "right": RIGHT_KNEE_POINT_ID,
}
DEFAULT_PEDAL_ANCHOR_POINTS = {
    "left": None,
    "right": None,
}


@dataclass(frozen=True, slots=True)
class HitEvent:
    side: str
    pedal_id: str
    point_id: str
    timestamp_seconds: float
    velocity: float
    y: float
    previous_y: float


@dataclass(frozen=True, slots=True)
class PedalHitDetectorConfig:
    up_threshold: float = 0.008
    down_velocity_threshold: float = 0.13
    down_displacement_threshold: float = 0.008
    min_hit_interval_seconds: float = 0.10
    max_missing_frames: int = 5
    min_point_confidence: float = 0.25
    require_anchor: bool = True


@dataclass(slots=True)
class _PedalState:
    last_y: float | None = None
    last_timestamp_seconds: float | None = None
    raised_y: float | None = None
    pressed_y: float | None = None
    downstroke_start_y: float | None = None
    pressed: bool = False
    last_hit_timestamp_seconds: float | None = None
    missing_frames: int = 0


class PedalHitDetector:
    def __init__(
        self,
        config: PedalHitDetectorConfig | None = None,
        pedal_points: dict[str, str] | None = None,
        anchor_points: dict[str, str | None] | None = None,
    ) -> None:
        self.config = config or PedalHitDetectorConfig()
        self.pedal_points = dict(pedal_points or DEFAULT_PEDAL_POINTS)
        self.anchor_points = dict(anchor_points or DEFAULT_PEDAL_ANCHOR_POINTS)
        self._states = {side: _PedalState() for side in self.pedal_points}

    def update(self, frame: PoseFrame) -> list[HitEvent]:
        events: list[HitEvent] = []
        for side, point_id in self.pedal_points.items():
            state = self._states.setdefault(side, _PedalState())
            point = frame.points.get(point_id)
            anchor_id = self.anchor_points.get(side)
            anchor = frame.points.get(anchor_id) if anchor_id is not None else None
            if not _valid_point(point, self.config.min_point_confidence):
                self._mark_missing(side, state)
                continue
            if anchor_id is not None and not _valid_point(anchor, self.config.min_point_confidence):
                if self.config.require_anchor:
                    self._mark_missing(side, state)
                    continue
                anchor = None

            y = _pedal_signal_y(point, anchor)
            if y is None:
                self._mark_missing(side, state)
                continue

            event = self._update_pedal(
                side=side,
                point_id=point_id,
                y=y,
                timestamp_seconds=frame.timestamp_seconds,
                state=state,
            )
            if event is not None:
                events.append(event)
        return events

    def reset(self) -> None:
        self._states = {side: _PedalState() for side in self.pedal_points}

    def _update_pedal(
        self,
        *,
        side: str,
        point_id: str,
        y: float,
        timestamp_seconds: float,
        state: _PedalState,
    ) -> HitEvent | None:
        if state.last_y is None or state.last_timestamp_seconds is None:
            state.last_y = y
            state.last_timestamp_seconds = timestamp_seconds
            state.raised_y = y
            state.missing_frames = 0
            return None

        previous_y = state.last_y
        dt = timestamp_seconds - state.last_timestamp_seconds
        if dt <= 0.0:
            dt = 1.0 / 60.0
        velocity = (y - previous_y) / dt

        state.missing_frames = 0
        if state.raised_y is None:
            state.raised_y = min(previous_y, y)

        event = self._detect_hit(
            side=side,
            point_id=point_id,
            timestamp_seconds=timestamp_seconds,
            y=y,
            previous_y=previous_y,
            velocity=velocity,
            state=state,
        )

        state.last_y = y
        state.last_timestamp_seconds = timestamp_seconds
        return event

    def _detect_hit(
        self,
        *,
        side: str,
        point_id: str,
        timestamp_seconds: float,
        y: float,
        previous_y: float,
        velocity: float,
        state: _PedalState,
    ) -> HitEvent | None:
        assert state.raised_y is not None

        if state.pressed:
            state.pressed_y = max(state.pressed_y if state.pressed_y is not None else y, y)
            if y <= state.pressed_y - self.config.up_threshold:
                state.pressed = False
                state.pressed_y = None
                state.downstroke_start_y = None
                state.raised_y = y
            return None

        state.raised_y = min(state.raised_y, y)
        if velocity <= 0.0:
            state.downstroke_start_y = None
            return None

        if state.downstroke_start_y is None:
            state.downstroke_start_y = previous_y
        else:
            state.downstroke_start_y = min(state.downstroke_start_y, previous_y)

        down_displacement = y - state.downstroke_start_y
        if velocity >= self.config.down_velocity_threshold and down_displacement >= self.config.down_displacement_threshold:
            state.pressed = True
            state.pressed_y = y
            state.downstroke_start_y = None
            if self._cooldown_elapsed(state, timestamp_seconds):
                state.last_hit_timestamp_seconds = timestamp_seconds
                return HitEvent(
                    side=side,
                    pedal_id=f"{side}_pedal",
                    point_id=point_id,
                    timestamp_seconds=timestamp_seconds,
                    velocity=velocity,
                    y=y,
                    previous_y=previous_y,
                )
            return None
        return None

    def _cooldown_elapsed(self, state: _PedalState, timestamp_seconds: float) -> bool:
        if state.last_hit_timestamp_seconds is None:
            return True
        return timestamp_seconds - state.last_hit_timestamp_seconds >= self.config.min_hit_interval_seconds

    def _mark_missing(self, side: str, state: _PedalState) -> None:
        state.missing_frames += 1
        if state.missing_frames > self.config.max_missing_frames:
            self._states[side] = _PedalState()


def _valid_point(point: PosePoint | None, min_confidence: float) -> bool:
    return (
        point is not None
        and point.tracking_present
        and (point.y_raw is not None or point.y is not None)
        and point.confidence >= min_confidence
    )


def _pedal_signal_y(point: PosePoint, anchor: PosePoint | None) -> float | None:
    point_y = point.y_raw if point.y_raw is not None else point.y
    if point_y is None:
        return None
    if anchor is None:
        return float(point_y)
    anchor_y = anchor.y_raw if anchor.y_raw is not None else anchor.y
    if anchor_y is None:
        return None
    return float(point_y) - float(anchor_y)
