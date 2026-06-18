"""Drum target classification for hand hits."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol


DEFAULT_KNN_MODEL_PATH = "knn_100.json"
TARGET_DRUMS = ("hi_hat", "snare", "tom_1", "tom_2", "floor_tom", "crash", "ride")
TARGET_HAND_SIDES = ("left", "right")
TARGET_SAMPLE_TYPE = "hand_target_samples_v1"
TARGET_FEATURE_SCHEMA = "hand_target_features_v1"
TARGET_TEMPORAL_FEATURE_SCHEMA = "hand_target_features_temporal_v1"
KNN_MODEL_TYPE = "knn_hand_target_v1"
KNN_DISTANCE_SQUARED_EUCLIDEAN = "squared-euclidean"
KNN_DISTANCE_MANHATTAN = "manhattan"
KNN_DISTANCE_METRICS = (KNN_DISTANCE_SQUARED_EUCLIDEAN, KNN_DISTANCE_MANHATTAN)
TARGET_FEATURE_NAMES = [
    "active_wrist_x",
    "active_wrist_y",
    "mcp_x",
    "mcp_y",
    "mcp_span_length",
    "mcp_span_angle",
    "active_wrist_radius",
    "active_wrist_angle",
    "wrist_to_mcp_length",
    "wrist_to_mcp_angle",
    "elbow_to_wrist_length",
    "elbow_to_wrist_angle",
    "shoulder_to_elbow_length",
    "shoulder_to_elbow_angle",
    "elbow_bend_angle",
    "wrist_hand_angle",
    "hand_motion_y",
    "forearm_motion_y",
    "strike_motion_y",
]
TARGET_TEMPORAL_LAGS = (3, 6, 9)
TARGET_TEMPORAL_WINDOW = 8
TARGET_TEMPORAL_FEATURE_NAMES = TARGET_FEATURE_NAMES + [
    feature_name
    for lag in TARGET_TEMPORAL_LAGS
    for feature_name in (
        f"active_wrist_delta_x_{lag}",
        f"active_wrist_delta_y_{lag}",
        f"mcp_delta_x_{lag}",
        f"mcp_delta_y_{lag}",
        f"strike_motion_delta_{lag}",
        f"pre_hit_direction_angle_{lag}",
    )
] + [
    f"wrist_y_range_{TARGET_TEMPORAL_WINDOW}",
    f"mcp_y_range_{TARGET_TEMPORAL_WINDOW}",
    f"strike_motion_range_{TARGET_TEMPORAL_WINDOW}",
]
TARGET_FEATURE_NAMES_BY_SCHEMA = {
    TARGET_FEATURE_SCHEMA: TARGET_FEATURE_NAMES,
    TARGET_TEMPORAL_FEATURE_SCHEMA: TARGET_TEMPORAL_FEATURE_NAMES,
}


@dataclass(frozen=True, slots=True)
class DrumTargetPrediction:
    drum: str
    context_name: str
    confidence: float


@dataclass(frozen=True, slots=True)
class DrumTargetHandSample:
    wrist_x: float
    wrist_y: float
    thumb_mcp_x: float
    thumb_mcp_y: float
    middle_mcp_x: float
    middle_mcp_y: float
    little_mcp_x: float
    little_mcp_y: float
    elbow_x: float
    elbow_y: float
    shoulder_x: float | None = None
    shoulder_y: float | None = None

    @property
    def mcp_x(self) -> float:
        return (self.thumb_mcp_x + self.middle_mcp_x + self.little_mcp_x) / 3.0

    @property
    def mcp_y(self) -> float:
        return (self.thumb_mcp_y + self.middle_mcp_y + self.little_mcp_y) / 3.0

    @property
    def mcp_span_x(self) -> float:
        return self.little_mcp_x - self.thumb_mcp_x

    @property
    def mcp_span_y(self) -> float:
        return self.little_mcp_y - self.thumb_mcp_y

    @property
    def hand_motion_y(self) -> float:
        return self.mcp_y - self.wrist_y

    @property
    def forearm_motion_y(self) -> float:
        return self.wrist_y - self.elbow_y

    def strike_motion_y(self, *, hand_weight: float, forearm_weight: float) -> float:
        return (self.hand_motion_y * hand_weight) + (self.forearm_motion_y * forearm_weight)


@dataclass(frozen=True, slots=True)
class DrumTargetObservationFrame:
    active: DrumTargetHandSample
    strike_motion_y: float
    timestamp_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class DrumTargetObservation:
    side: str
    active: DrumTargetHandSample
    other: DrumTargetHandSample | None
    strike_motion_y: float
    timestamp_seconds: float = 0.0
    strike_velocity: float = 0.0
    history: tuple[DrumTargetObservationFrame, ...] = ()


class DrumTargetClassifier(Protocol):
    def classify(self, observation: DrumTargetObservation) -> DrumTargetPrediction | None:
        ...


class KnnDrumTargetClassifier:
    def __init__(
        self,
        *,
        feature_names: list[str],
        feature_means: list[float],
        feature_stds: list[float],
        samples: list[dict],
        k: int = 5,
        min_confidence: float = 0.45,
        model_type: str = KNN_MODEL_TYPE,
        feature_schema: str = TARGET_FEATURE_SCHEMA,
        distance_metric: str = KNN_DISTANCE_SQUARED_EUCLIDEAN,
    ) -> None:
        if not feature_names:
            raise ValueError("KNN model has no features.")
        if distance_metric not in KNN_DISTANCE_METRICS:
            raise ValueError("Unsupported KNN distance metric.")
        self.feature_names = feature_names
        self.feature_means = feature_means
        self.feature_stds = [max(float(std), 1e-6) for std in feature_stds]
        self.samples = samples
        self.k = max(1, int(k))
        self.min_confidence = min_confidence
        self.model_type = model_type
        self.feature_schema = feature_schema
        self.distance_metric = distance_metric
        self._normalized_samples = [
            (
                str(sample["drum"]),
                str(sample.get("side", "")),
                _normalize_vector(sample["features"], self.feature_means, self.feature_stds),
            )
            for sample in samples
        ]
        if not self._normalized_samples:
            raise ValueError("KNN model has no samples.")

    @classmethod
    def from_default_model(
        cls,
        *,
        distance_metric: str | None = None,
    ) -> "KnnDrumTargetClassifier":
        default_path = Path(__file__).resolve().parents[1] / DEFAULT_KNN_MODEL_PATH
        return cls.from_path(default_path, distance_metric=distance_metric)

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        distance_metric: str | None = None,
    ) -> "KnnDrumTargetClassifier":
        with Path(path).open("r", encoding="utf-8") as file:
            return cls.from_dict(json.load(file), distance_metric=distance_metric)

    @classmethod
    def from_dict(
        cls,
        data: dict,
        *,
        distance_metric: str | None = None,
    ) -> "KnnDrumTargetClassifier":
        model_type = data.get("model_type")
        if model_type != KNN_MODEL_TYPE:
            raise ValueError("Unsupported KNN hand target model type.")
        if data.get("feature_schema") not in TARGET_FEATURE_NAMES_BY_SCHEMA:
            raise ValueError("Unsupported KNN hand target feature schema.")
        _validate_known_feature_names(
            data.get("feature_names", []),
            feature_schema=str(data["feature_schema"]),
        )
        resolved_distance_metric = str(
            distance_metric or data.get("distance_metric", KNN_DISTANCE_SQUARED_EUCLIDEAN)
        )
        return cls(
            feature_names=list(data["feature_names"]),
            feature_means=[float(value) for value in data["feature_means"]],
            feature_stds=[float(value) for value in data["feature_stds"]],
            samples=list(data["samples"]),
            k=int(data.get("k", 5)),
            min_confidence=float(data.get("min_confidence", 0.45)),
            model_type=str(model_type),
            feature_schema=str(data["feature_schema"]),
            distance_metric=resolved_distance_metric,
        )

    def classify(self, observation: DrumTargetObservation) -> DrumTargetPrediction | None:
        feature_map = build_target_features(observation, feature_schema=self.feature_schema)
        if feature_map is None:
            return None
        vector = [feature_map.get(name, 0.0) for name in self.feature_names]
        normalized = _normalize_vector(vector, self.feature_means, self.feature_stds)
        distances: list[tuple[float, str]] = []
        for drum, _side, sample in self._normalized_samples:
            if _side != observation.side:
                continue
            distances.append((self._distance(normalized, sample), drum))
        if not distances:
            return None
        nearest = sorted(distances, key=lambda item: item[0])[: self.k]
        votes = Counter(drum for _distance, drum in nearest)
        drum, vote_count = votes.most_common(1)[0]
        sorted_votes = votes.most_common(2)
        vote_share = vote_count / len(nearest)
        margin = 1.0
        if len(sorted_votes) > 1:
            margin = (sorted_votes[0][1] - sorted_votes[1][1]) / len(nearest)
        confidence = max(0.0, min(1.0, (vote_share * 0.85) + (margin * 0.15)))
        if confidence < self.min_confidence:
            return None
        return DrumTargetPrediction(drum=drum, context_name=self.model_type, confidence=confidence)

    def _distance(self, left: list[float], right: list[float]) -> float:
        if self.distance_metric == KNN_DISTANCE_MANHATTAN:
            return _manhattan_distance(left, right)
        return _squared_distance(left, right)


def target_feature_names(feature_schema: str = TARGET_FEATURE_SCHEMA) -> list[str]:
    try:
        return list(TARGET_FEATURE_NAMES_BY_SCHEMA[feature_schema])
    except KeyError as exc:
        raise ValueError("Unsupported target sample feature schema.") from exc


def build_target_features(
    observation: DrumTargetObservation,
    *,
    feature_schema: str = TARGET_FEATURE_SCHEMA,
) -> dict[str, float] | None:
    if feature_schema not in TARGET_FEATURE_NAMES_BY_SCHEMA:
        raise ValueError("Unsupported target sample feature schema.")
    active = observation.active
    if active.shoulder_x is None or active.shoulder_y is None:
        return None

    wrist_to_mcp_x = active.mcp_x - active.wrist_x
    wrist_to_mcp_y = active.mcp_y - active.wrist_y
    elbow_to_wrist_x = active.wrist_x - active.elbow_x
    elbow_to_wrist_y = active.wrist_y - active.elbow_y
    shoulder_to_elbow_x = active.elbow_x - active.shoulder_x
    shoulder_to_elbow_y = active.elbow_y - active.shoulder_y

    mcp_span_length = math.hypot(active.mcp_span_x, active.mcp_span_y)
    mcp_span_angle = math.atan2(active.mcp_span_y, active.mcp_span_x)
    active_wrist_radius = math.hypot(active.wrist_x, active.wrist_y)
    active_wrist_angle = math.atan2(active.wrist_y, active.wrist_x)
    wrist_to_mcp_angle = math.atan2(wrist_to_mcp_y, wrist_to_mcp_x)
    elbow_to_wrist_angle = math.atan2(elbow_to_wrist_y, elbow_to_wrist_x)
    shoulder_to_elbow_angle = math.atan2(shoulder_to_elbow_y, shoulder_to_elbow_x)

    features = {
        "active_wrist_x": active.wrist_x,
        "active_wrist_y": active.wrist_y,
        "mcp_x": active.mcp_x,
        "mcp_y": active.mcp_y,
        "mcp_span_length": mcp_span_length,
        "mcp_span_angle": mcp_span_angle,
        "active_wrist_radius": active_wrist_radius,
        "active_wrist_angle": active_wrist_angle,
        "wrist_to_mcp_length": math.hypot(wrist_to_mcp_x, wrist_to_mcp_y),
        "wrist_to_mcp_angle": wrist_to_mcp_angle,
        "elbow_to_wrist_length": math.hypot(elbow_to_wrist_x, elbow_to_wrist_y),
        "elbow_to_wrist_angle": elbow_to_wrist_angle,
        "shoulder_to_elbow_length": math.hypot(shoulder_to_elbow_x, shoulder_to_elbow_y),
        "shoulder_to_elbow_angle": shoulder_to_elbow_angle,
        "elbow_bend_angle": _angle_between(
            active.shoulder_x - active.elbow_x,
            active.shoulder_y - active.elbow_y,
            active.wrist_x - active.elbow_x,
            active.wrist_y - active.elbow_y,
        ),
        "wrist_hand_angle": _signed_angle_delta(elbow_to_wrist_angle, wrist_to_mcp_angle),
        "hand_motion_y": active.hand_motion_y,
        "forearm_motion_y": active.forearm_motion_y,
        "strike_motion_y": observation.strike_motion_y,
    }
    if feature_schema == TARGET_TEMPORAL_FEATURE_SCHEMA:
        features.update(_build_temporal_target_features(observation))
    return features


def target_observation_to_sample_record(
    *,
    drum: str,
    observation: DrumTargetObservation,
    timestamp_seconds: float,
    velocity: float,
    feature_schema: str = TARGET_FEATURE_SCHEMA,
) -> dict | None:
    feature_names = target_feature_names(feature_schema)
    features = build_target_features(observation, feature_schema=feature_schema)
    if features is None:
        return None
    active = observation.active
    return {
        "drum": drum,
        "side": observation.side,
        "timestamp_seconds": float(timestamp_seconds),
        "velocity": float(velocity),
        "features": [features[name] for name in feature_names],
        "observation": {
            "active": {
                "wrist_x": active.wrist_x,
                "wrist_y": active.wrist_y,
                "thumb_mcp_x": active.thumb_mcp_x,
                "thumb_mcp_y": active.thumb_mcp_y,
                "middle_mcp_x": active.middle_mcp_x,
                "middle_mcp_y": active.middle_mcp_y,
                "little_mcp_x": active.little_mcp_x,
                "little_mcp_y": active.little_mcp_y,
                "elbow_x": active.elbow_x,
                "elbow_y": active.elbow_y,
                "shoulder_x": active.shoulder_x,
                "shoulder_y": active.shoulder_y,
            },
            "strike_motion_y": observation.strike_motion_y,
            "strike_velocity": observation.strike_velocity,
        },
    }


def new_target_sample_file(
    samples: list[dict] | None = None,
    *,
    feature_schema: str = TARGET_FEATURE_SCHEMA,
) -> dict:
    return {
        "samples_type": TARGET_SAMPLE_TYPE,
        "feature_schema": feature_schema,
        "feature_names": target_feature_names(feature_schema),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "samples": list(samples or []),
    }


def read_target_sample_file(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)
    _validate_target_sample_file(data)
    return data


def write_target_sample_file(path: str | Path, data: dict) -> None:
    _validate_target_sample_file(data)
    with Path(path).open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)
        file.write("\n")


def load_or_create_target_sample_file(path: str | Path, *, append: bool) -> dict:
    sample_path = Path(path)
    if append and sample_path.exists():
        try:
            return read_target_sample_file(sample_path)
        except ValueError as exc:
            raise ValueError(
                f"Cannot append target samples to {sample_path}: {exc} "
                "Use a different output path or pass --collection-output-mode overwrite."
            ) from exc
    return new_target_sample_file()


def train_knn_model_from_sample_file(
    data: dict,
    *,
    k: int = 5,
    min_confidence: float = 0.45,
) -> dict:
    _validate_target_sample_file(data)
    samples = list(data["samples"])
    if not samples:
        raise ValueError("Cannot train KNN model without samples.")
    feature_names = list(data["feature_names"])
    columns = list(zip(*(sample["features"] for sample in samples)))
    feature_means = [_mean(column) for column in columns]
    feature_stds = [max(_std(column), 1e-6) for column in columns]
    return {
        "model_type": KNN_MODEL_TYPE,
        "feature_schema": data["feature_schema"],
        "distance_metric": KNN_DISTANCE_SQUARED_EUCLIDEAN,
        "feature_names": feature_names,
        "feature_means": feature_means,
        "feature_stds": feature_stds,
        "k": int(k),
        "min_confidence": float(min_confidence),
        "samples": [
            {
                "drum": sample["drum"],
                "side": sample["side"],
                "features": sample["features"],
            }
            for sample in samples
        ],
    }


def write_knn_model(path: str | Path, model: dict) -> None:
    with Path(path).open("w", encoding="utf-8") as file:
        json.dump(model, file, indent=2, sort_keys=True)
        file.write("\n")


def _validate_target_sample_file(data: dict) -> None:
    if data.get("samples_type") != TARGET_SAMPLE_TYPE:
        raise ValueError("Unsupported target sample file type.")
    feature_schema = data.get("feature_schema")
    if feature_schema not in TARGET_FEATURE_NAMES_BY_SCHEMA:
        raise ValueError("Unsupported target sample feature schema.")
    feature_names = target_feature_names(str(feature_schema))
    _validate_known_feature_names(data.get("feature_names", []), feature_schema=str(feature_schema))
    if list(data.get("feature_names", [])) != feature_names:
        raise ValueError("Target sample feature names do not match current schema.")
    for sample in data.get("samples", []):
        if sample.get("side") not in TARGET_HAND_SIDES:
            raise ValueError("Target sample has an unsupported hand side.")
        if sample.get("drum") not in TARGET_DRUMS:
            raise ValueError("Target sample has an unsupported drum.")
        if len(sample.get("features", [])) != len(feature_names):
            raise ValueError("Target sample feature length does not match current schema.")


def _build_temporal_target_features(observation: DrumTargetObservation) -> dict[str, float]:
    features: dict[str, float] = {}
    history = observation.history
    current_frame = DrumTargetObservationFrame(
        active=observation.active,
        strike_motion_y=observation.strike_motion_y,
        timestamp_seconds=observation.timestamp_seconds,
    )
    for lag in TARGET_TEMPORAL_LAGS:
        previous = _lagged_frame(history, lag, fallback=current_frame)
        wrist_dx = observation.active.wrist_x - previous.active.wrist_x
        wrist_dy = observation.active.wrist_y - previous.active.wrist_y
        mcp_dx = observation.active.mcp_x - previous.active.mcp_x
        mcp_dy = observation.active.mcp_y - previous.active.mcp_y
        strike_delta = observation.strike_motion_y - previous.strike_motion_y
        dt = observation.timestamp_seconds - previous.timestamp_seconds
        if dt <= 0.0:
            dt = lag / 60.0
        features[f"active_wrist_delta_x_{lag}"] = wrist_dx
        features[f"active_wrist_delta_y_{lag}"] = wrist_dy
        features[f"mcp_delta_x_{lag}"] = mcp_dx
        features[f"mcp_delta_y_{lag}"] = mcp_dy
        features[f"strike_motion_delta_{lag}"] = strike_delta
        features[f"pre_hit_direction_angle_{lag}"] = math.atan2(mcp_dy, mcp_dx)

    window = list(history[-TARGET_TEMPORAL_WINDOW:]) + [current_frame]
    features[f"wrist_y_range_{TARGET_TEMPORAL_WINDOW}"] = _range(frame.active.wrist_y for frame in window)
    features[f"mcp_y_range_{TARGET_TEMPORAL_WINDOW}"] = _range(frame.active.mcp_y for frame in window)
    features[f"strike_motion_range_{TARGET_TEMPORAL_WINDOW}"] = _range(frame.strike_motion_y for frame in window)
    return features


def _lagged_frame(
    history: tuple[DrumTargetObservationFrame, ...],
    lag: int,
    *,
    fallback: DrumTargetObservationFrame,
) -> DrumTargetObservationFrame:
    if len(history) >= lag:
        return history[-lag]
    if history:
        return history[0]
    return fallback


def _validate_known_feature_names(feature_names: Iterable[str], *, feature_schema: str) -> None:
    unknown = set(feature_names) - set(target_feature_names(feature_schema))
    if unknown:
        raise ValueError("Target sample feature names do not match current schema.")


def _angle_between(ax: float, ay: float, bx: float, by: float) -> float:
    left = math.hypot(ax, ay)
    right = math.hypot(bx, by)
    if left <= 1e-9 or right <= 1e-9:
        return 0.0
    dot = (ax * bx) + (ay * by)
    cosine = max(-1.0, min(1.0, dot / (left * right)))
    return math.acos(cosine)


def _signed_angle_delta(start: float, end: float) -> float:
    return (end - start + math.pi) % (math.pi * 2.0) - math.pi


def _normalize_vector(values: list[float], means: list[float], stds: list[float]) -> list[float]:
    return [(float(value) - mean) / std for value, mean, std in zip(values, means, stds)]


def _squared_distance(left: list[float], right: list[float]) -> float:
    return sum((a - b) * (a - b) for a, b in zip(left, right))


def _manhattan_distance(left: list[float], right: list[float]) -> float:
    return sum(abs(a - b) for a, b in zip(left, right))


def _range(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        return 0.0
    return max(items) - min(items)


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        return 0.0
    return sum(items) / len(items)


def _std(values: Iterable[float]) -> float:
    items = list(values)
    if len(items) <= 1:
        return 0.0
    mean = sum(items) / len(items)
    variance = sum((value - mean) * (value - mean) for value in items) / (len(items) - 1)
    return math.sqrt(variance)
