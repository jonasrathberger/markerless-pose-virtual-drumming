"""Canonical landmark schema and overlay connections."""

from __future__ import annotations

from .schema import LandmarkSpec


def canonical_id(landmark_group: str, side: str, landmark_name: str) -> str:
    return f"{landmark_group}:{side}:{landmark_name}"


BODY_LANDMARK_NAMES = [
    ("nose", "center"),
    ("left_shoulder", "left"),
    ("right_shoulder", "right"),
    ("left_elbow", "left"),
    ("right_elbow", "right"),
    ("left_wrist", "left"),
    ("right_wrist", "right"),
    ("left_hip", "left"),
    ("right_hip", "right"),
    ("left_knee", "left"),
    ("right_knee", "right"),
    ("left_ankle", "left"),
    ("right_ankle", "right"),
]

HAND_BASE_NAMES = [
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
]


CANONICAL_SPECS: list[LandmarkSpec] = []

for landmark_name, side in BODY_LANDMARK_NAMES:
    clean_name = landmark_name.replace("left_", "").replace("right_", "")
    CANONICAL_SPECS.append(
        LandmarkSpec(
            landmark_id=canonical_id("body", side, clean_name),
            landmark_name=clean_name,
            landmark_group="body",
            side=side,
        )
    )

for side in ("left", "right"):
    group_name = f"{side}_hand"
    for landmark_name in HAND_BASE_NAMES:
        CANONICAL_SPECS.append(
            LandmarkSpec(
                landmark_id=canonical_id(group_name, side, landmark_name),
                landmark_name=landmark_name,
                landmark_group=group_name,
                side=side,
            )
        )

CANONICAL_SPEC_BY_ID = {spec.landmark_id: spec for spec in CANONICAL_SPECS}
CORE_LANDMARK_IDS = [spec.landmark_id for spec in CANONICAL_SPECS]

BODY_CONNECTIONS = [
    ("body:center:nose", "body:left:shoulder"),
    ("body:center:nose", "body:right:shoulder"),
    ("body:left:shoulder", "body:right:shoulder"),
    ("body:left:shoulder", "body:left:elbow"),
    ("body:left:elbow", "body:left:wrist"),
    ("body:right:shoulder", "body:right:elbow"),
    ("body:right:elbow", "body:right:wrist"),
    ("body:left:shoulder", "body:left:hip"),
    ("body:right:shoulder", "body:right:hip"),
    ("body:left:hip", "body:right:hip"),
    ("body:left:hip", "body:left:knee"),
    ("body:left:knee", "body:left:ankle"),
    ("body:right:hip", "body:right:knee"),
    ("body:right:knee", "body:right:ankle"),
]

HAND_CONNECTION_PAIRS = [
    ("wrist", "thumb_cmc"),
    ("thumb_cmc", "thumb_mcp"),
    ("thumb_mcp", "thumb_ip"),
    ("thumb_ip", "thumb_tip"),
    ("wrist", "index_mcp"),
    ("index_mcp", "index_pip"),
    ("index_pip", "index_dip"),
    ("index_dip", "index_tip"),
    ("wrist", "middle_mcp"),
    ("middle_mcp", "middle_pip"),
    ("middle_pip", "middle_dip"),
    ("middle_dip", "middle_tip"),
    ("wrist", "ring_mcp"),
    ("ring_mcp", "ring_pip"),
    ("ring_pip", "ring_dip"),
    ("ring_dip", "ring_tip"),
    ("wrist", "pinky_mcp"),
    ("pinky_mcp", "pinky_pip"),
    ("pinky_pip", "pinky_dip"),
    ("pinky_dip", "pinky_tip"),
    ("index_mcp", "middle_mcp"),
    ("middle_mcp", "ring_mcp"),
    ("ring_mcp", "pinky_mcp"),
]

HAND_CONNECTIONS: list[tuple[str, str]] = []
for side in ("left", "right"):
    group_name = f"{side}_hand"
    for start_name, end_name in HAND_CONNECTION_PAIRS:
        HAND_CONNECTIONS.append(
            (
                canonical_id(group_name, side, start_name),
                canonical_id(group_name, side, end_name),
            )
        )

OVERLAY_CONNECTIONS = BODY_CONNECTIONS + HAND_CONNECTIONS

