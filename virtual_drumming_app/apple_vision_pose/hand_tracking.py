"""Stable hand identity assignment across Vision frames."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

from .models import Landmark

HAND_IDENTITIES = ("hand_left", "hand_right")
HAND_ANCHOR_JOINTS = (
    "VNHumanHandPoseObservationJointNameWrist",
    "VNHumanHandPoseObservationJointNameThumbMP",
    "VNHumanHandPoseObservationJointNameMiddleMCP",
    "VNHumanHandPoseObservationJointNameLittleMCP",
)


@dataclass(slots=True)
class HandCandidate:
    group: str
    chirality: str | None
    center: tuple[float, float]


class HandIdentityTracker:
    def __init__(self) -> None:
        self.previous_centers: dict[str, tuple[float, float]] = {}

    def assign(self, landmarks: dict[str, Landmark]) -> dict[str, Landmark]:
        candidates = self._hand_candidates(landmarks)
        if not candidates:
            self.previous_centers.clear()
            return landmarks

        assignments = self._assign_candidates(candidates)
        if not assignments:
            return landmarks

        stable_landmarks: dict[str, Landmark] = {}
        for key, landmark in landmarks.items():
            if not key.startswith("hand_"):
                stable_landmarks[key] = landmark
                continue

            raw_group, joint = key.split(":", 1)
            stable_group = assignments.get(raw_group)
            if stable_group is None:
                continue

            landmark.group = stable_group
            landmark.chirality = stable_group.removeprefix("hand_")
            stable_landmarks[f"{stable_group}:{joint}"] = landmark

        self._update_previous_centers(stable_landmarks)
        return stable_landmarks

    def _assign_candidates(self, candidates: list[HandCandidate]) -> dict[str, str]:
        best_assignment: dict[str, str] = {}
        best_cost: float | None = None
        identity_count = min(len(candidates), len(HAND_IDENTITIES))
        for identities in permutations(HAND_IDENTITIES, identity_count):
            assignment = {
                candidate.group: identity
                for candidate, identity in zip(candidates[:identity_count], identities, strict=True)
            }
            cost = sum(
                self._assignment_cost(candidate, identity)
                for candidate, identity in zip(candidates[:identity_count], identities, strict=True)
            )
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_assignment = assignment
        return best_assignment

    def _assignment_cost(self, candidate: HandCandidate, identity: str) -> float:
        previous_center = self.previous_centers.get(identity)
        if previous_center is None:
            # Camera images are not mirrored at the Vision/input stage. For a
            # player facing the camera, anatomical left usually appears on the
            # right side of the image.
            preferred_x = 0.65 if identity == "hand_left" else 0.35
            distance_cost = abs(candidate.center[0] - preferred_x) * 0.2
        else:
            distance_cost = _squared_distance(candidate.center, previous_center)
            if distance_cost < 0.01:
                distance_cost -= 0.3

        if candidate.chirality in {"left", "right"}:
            # Vision's hand chirality is reported from camera/image
            # perspective in this setup, while the app prompts use the
            # player's perspective.
            expected_identity = "hand_right" if candidate.chirality == "left" else "hand_left"
            chirality_cost = -0.1 if identity == expected_identity else 0.35
        else:
            chirality_cost = 0.0
        return distance_cost + chirality_cost

    @staticmethod
    def _hand_candidates(landmarks: dict[str, Landmark]) -> list[HandCandidate]:
        groups = sorted({key.split(":", 1)[0] for key in landmarks if key.startswith("hand_")})
        candidates: list[HandCandidate] = []
        for group in groups:
            points = [
                landmarks[f"{group}:{joint}"]
                for joint in HAND_ANCHOR_JOINTS
                if f"{group}:{joint}" in landmarks and landmarks[f"{group}:{joint}"].tracking_present
            ]
            if not points:
                continue
            center = (
                sum(point.x_norm for point in points) / len(points),
                sum(point.y_norm for point in points) / len(points),
            )
            chirality = next((point.chirality for point in points if point.chirality in {"left", "right"}), None)
            candidates.append(HandCandidate(group=group, chirality=chirality, center=center))
        return candidates

    def _update_previous_centers(self, landmarks: dict[str, Landmark]) -> None:
        next_centers: dict[str, tuple[float, float]] = {}
        for group in HAND_IDENTITIES:
            points = [
                landmarks[f"{group}:{joint}"]
                for joint in HAND_ANCHOR_JOINTS
                if f"{group}:{joint}" in landmarks and landmarks[f"{group}:{joint}"].tracking_present
            ]
            if not points:
                continue
            measured = (
                sum(point.x_norm for point in points) / len(points),
                sum(point.y_norm for point in points) / len(points),
            )
            previous = self.previous_centers.get(group)
            if previous is None:
                next_centers[group] = measured
            else:
                next_centers[group] = (
                    (previous[0] * 0.65) + (measured[0] * 0.35),
                    (previous[1] * 0.65) + (measured[1] * 0.35),
                )
        self.previous_centers = next_centers


def _squared_distance(first: tuple[float, float], second: tuple[float, float]) -> float:
    dx = first[0] - second[0]
    dy = first[1] - second[1]
    return (dx * dx) + (dy * dy)
