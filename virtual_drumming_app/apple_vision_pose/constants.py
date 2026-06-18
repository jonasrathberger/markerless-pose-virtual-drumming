"""Constants shared by capture, Vision, and rendering."""

IPHONE_CAMERA_HINTS = ("iphone", "continuity")
MAC_WEBCAM_HINTS = ("facetime", "built-in", "builtin", "isight")

BODY_JOINT_CONSTANTS = [
    "VNHumanBodyPoseObservationJointNameNose",
    "VNHumanBodyPoseObservationJointNameLeftShoulder",
    "VNHumanBodyPoseObservationJointNameRightShoulder",
    "VNHumanBodyPoseObservationJointNameLeftElbow",
    "VNHumanBodyPoseObservationJointNameRightElbow",
    "VNHumanBodyPoseObservationJointNameLeftWrist",
    "VNHumanBodyPoseObservationJointNameRightWrist",
    "VNHumanBodyPoseObservationJointNameLeftHip",
    "VNHumanBodyPoseObservationJointNameRightHip",
    "VNHumanBodyPoseObservationJointNameLeftKnee",
    "VNHumanBodyPoseObservationJointNameRightKnee",
    "VNHumanBodyPoseObservationJointNameLeftAnkle",
    "VNHumanBodyPoseObservationJointNameRightAnkle",
]

HAND_JOINT_CONSTANTS = [
    "VNHumanHandPoseObservationJointNameWrist",
    "VNHumanHandPoseObservationJointNameThumbCMC",
    "VNHumanHandPoseObservationJointNameThumbMP",
    "VNHumanHandPoseObservationJointNameThumbIP",
    "VNHumanHandPoseObservationJointNameThumbTip",
    "VNHumanHandPoseObservationJointNameIndexMCP",
    "VNHumanHandPoseObservationJointNameIndexPIP",
    "VNHumanHandPoseObservationJointNameIndexDIP",
    "VNHumanHandPoseObservationJointNameIndexTip",
    "VNHumanHandPoseObservationJointNameMiddleMCP",
    "VNHumanHandPoseObservationJointNameMiddlePIP",
    "VNHumanHandPoseObservationJointNameMiddleDIP",
    "VNHumanHandPoseObservationJointNameMiddleTip",
    "VNHumanHandPoseObservationJointNameRingMCP",
    "VNHumanHandPoseObservationJointNameRingPIP",
    "VNHumanHandPoseObservationJointNameRingDIP",
    "VNHumanHandPoseObservationJointNameRingTip",
    "VNHumanHandPoseObservationJointNameLittleMCP",
    "VNHumanHandPoseObservationJointNameLittlePIP",
    "VNHumanHandPoseObservationJointNameLittleDIP",
    "VNHumanHandPoseObservationJointNameLittleTip",
]

BODY_CONNECTIONS = [
    ("VNHumanBodyPoseObservationJointNameNose", "VNHumanBodyPoseObservationJointNameLeftShoulder"),
    ("VNHumanBodyPoseObservationJointNameNose", "VNHumanBodyPoseObservationJointNameRightShoulder"),
    ("VNHumanBodyPoseObservationJointNameLeftShoulder", "VNHumanBodyPoseObservationJointNameRightShoulder"),
    ("VNHumanBodyPoseObservationJointNameLeftShoulder", "VNHumanBodyPoseObservationJointNameLeftElbow"),
    ("VNHumanBodyPoseObservationJointNameLeftElbow", "VNHumanBodyPoseObservationJointNameLeftWrist"),
    ("VNHumanBodyPoseObservationJointNameRightShoulder", "VNHumanBodyPoseObservationJointNameRightElbow"),
    ("VNHumanBodyPoseObservationJointNameRightElbow", "VNHumanBodyPoseObservationJointNameRightWrist"),
    ("VNHumanBodyPoseObservationJointNameLeftShoulder", "VNHumanBodyPoseObservationJointNameLeftHip"),
    ("VNHumanBodyPoseObservationJointNameRightShoulder", "VNHumanBodyPoseObservationJointNameRightHip"),
    ("VNHumanBodyPoseObservationJointNameLeftHip", "VNHumanBodyPoseObservationJointNameRightHip"),
    ("VNHumanBodyPoseObservationJointNameLeftHip", "VNHumanBodyPoseObservationJointNameLeftKnee"),
    ("VNHumanBodyPoseObservationJointNameLeftKnee", "VNHumanBodyPoseObservationJointNameLeftAnkle"),
    ("VNHumanBodyPoseObservationJointNameRightHip", "VNHumanBodyPoseObservationJointNameRightKnee"),
    ("VNHumanBodyPoseObservationJointNameRightKnee", "VNHumanBodyPoseObservationJointNameRightAnkle"),
]

HAND_CONNECTIONS = [
    ("VNHumanHandPoseObservationJointNameWrist", "VNHumanHandPoseObservationJointNameThumbCMC"),
    ("VNHumanHandPoseObservationJointNameThumbCMC", "VNHumanHandPoseObservationJointNameThumbMP"),
    ("VNHumanHandPoseObservationJointNameThumbMP", "VNHumanHandPoseObservationJointNameThumbIP"),
    ("VNHumanHandPoseObservationJointNameThumbIP", "VNHumanHandPoseObservationJointNameThumbTip"),
    ("VNHumanHandPoseObservationJointNameWrist", "VNHumanHandPoseObservationJointNameIndexMCP"),
    ("VNHumanHandPoseObservationJointNameIndexMCP", "VNHumanHandPoseObservationJointNameIndexPIP"),
    ("VNHumanHandPoseObservationJointNameIndexPIP", "VNHumanHandPoseObservationJointNameIndexDIP"),
    ("VNHumanHandPoseObservationJointNameIndexDIP", "VNHumanHandPoseObservationJointNameIndexTip"),
    ("VNHumanHandPoseObservationJointNameWrist", "VNHumanHandPoseObservationJointNameMiddleMCP"),
    ("VNHumanHandPoseObservationJointNameMiddleMCP", "VNHumanHandPoseObservationJointNameMiddlePIP"),
    ("VNHumanHandPoseObservationJointNameMiddlePIP", "VNHumanHandPoseObservationJointNameMiddleDIP"),
    ("VNHumanHandPoseObservationJointNameMiddleDIP", "VNHumanHandPoseObservationJointNameMiddleTip"),
    ("VNHumanHandPoseObservationJointNameWrist", "VNHumanHandPoseObservationJointNameRingMCP"),
    ("VNHumanHandPoseObservationJointNameRingMCP", "VNHumanHandPoseObservationJointNameRingPIP"),
    ("VNHumanHandPoseObservationJointNameRingPIP", "VNHumanHandPoseObservationJointNameRingDIP"),
    ("VNHumanHandPoseObservationJointNameRingDIP", "VNHumanHandPoseObservationJointNameRingTip"),
    ("VNHumanHandPoseObservationJointNameWrist", "VNHumanHandPoseObservationJointNameLittleMCP"),
    ("VNHumanHandPoseObservationJointNameLittleMCP", "VNHumanHandPoseObservationJointNameLittlePIP"),
    ("VNHumanHandPoseObservationJointNameLittlePIP", "VNHumanHandPoseObservationJointNameLittleDIP"),
    ("VNHumanHandPoseObservationJointNameLittleDIP", "VNHumanHandPoseObservationJointNameLittleTip"),
]

TRACKED_BODY_JOINT_CONSTANTS = [
    "VNHumanBodyPoseObservationJointNameLeftShoulder",
    "VNHumanBodyPoseObservationJointNameRightShoulder",
    "VNHumanBodyPoseObservationJointNameLeftElbow",
    "VNHumanBodyPoseObservationJointNameRightElbow",
    "VNHumanBodyPoseObservationJointNameLeftWrist",
    "VNHumanBodyPoseObservationJointNameRightWrist",
    "VNHumanBodyPoseObservationJointNameLeftHip",
    "VNHumanBodyPoseObservationJointNameRightHip",
    "VNHumanBodyPoseObservationJointNameLeftKnee",
    "VNHumanBodyPoseObservationJointNameRightKnee",
]

TRACKED_HAND_JOINT_CONSTANTS = [
    "VNHumanHandPoseObservationJointNameWrist",
    "VNHumanHandPoseObservationJointNameThumbMP",
    "VNHumanHandPoseObservationJointNameMiddleMCP",
    "VNHumanHandPoseObservationJointNameLittleMCP",
]

TRACKED_LANDMARK_IDS = {
    "body:VNHumanBodyPoseObservationJointNameLeftShoulder",
    "body:VNHumanBodyPoseObservationJointNameRightShoulder",
    "body:VNHumanBodyPoseObservationJointNameLeftElbow",
    "body:VNHumanBodyPoseObservationJointNameRightElbow",
    "body:VNHumanBodyPoseObservationJointNameLeftWrist",
    "body:VNHumanBodyPoseObservationJointNameRightWrist",
    "body:VNHumanBodyPoseObservationJointNameLeftHip",
    "body:VNHumanBodyPoseObservationJointNameRightHip",
    "body:VNHumanBodyPoseObservationJointNameLeftKnee",
    "body:VNHumanBodyPoseObservationJointNameRightKnee",
    "hand_left:VNHumanHandPoseObservationJointNameWrist",
    "hand_left:VNHumanHandPoseObservationJointNameThumbMP",
    "hand_left:VNHumanHandPoseObservationJointNameMiddleMCP",
    "hand_left:VNHumanHandPoseObservationJointNameLittleMCP",
    "hand_right:VNHumanHandPoseObservationJointNameWrist",
    "hand_right:VNHumanHandPoseObservationJointNameThumbMP",
    "hand_right:VNHumanHandPoseObservationJointNameMiddleMCP",
    "hand_right:VNHumanHandPoseObservationJointNameLittleMCP",
}

TRACKED_CONNECTIONS = [
    ("body:VNHumanBodyPoseObservationJointNameLeftShoulder", "body:VNHumanBodyPoseObservationJointNameRightShoulder"),
    ("body:VNHumanBodyPoseObservationJointNameLeftShoulder", "body:VNHumanBodyPoseObservationJointNameLeftElbow"),
    ("body:VNHumanBodyPoseObservationJointNameRightShoulder", "body:VNHumanBodyPoseObservationJointNameRightElbow"),
    ("body:VNHumanBodyPoseObservationJointNameLeftElbow", "body:VNHumanBodyPoseObservationJointNameLeftWrist"),
    ("body:VNHumanBodyPoseObservationJointNameRightElbow", "body:VNHumanBodyPoseObservationJointNameRightWrist"),
    ("body:VNHumanBodyPoseObservationJointNameLeftShoulder", "body:VNHumanBodyPoseObservationJointNameLeftHip"),
    ("body:VNHumanBodyPoseObservationJointNameRightShoulder", "body:VNHumanBodyPoseObservationJointNameRightHip"),
    ("body:VNHumanBodyPoseObservationJointNameLeftHip", "body:VNHumanBodyPoseObservationJointNameRightHip"),
    ("body:VNHumanBodyPoseObservationJointNameLeftHip", "body:VNHumanBodyPoseObservationJointNameLeftKnee"),
    ("body:VNHumanBodyPoseObservationJointNameRightHip", "body:VNHumanBodyPoseObservationJointNameRightKnee"),
    ("hand_left:VNHumanHandPoseObservationJointNameWrist", "hand_left:VNHumanHandPoseObservationJointNameThumbMP"),
    ("hand_left:VNHumanHandPoseObservationJointNameWrist", "hand_left:VNHumanHandPoseObservationJointNameMiddleMCP"),
    ("hand_left:VNHumanHandPoseObservationJointNameWrist", "hand_left:VNHumanHandPoseObservationJointNameLittleMCP"),
    ("hand_right:VNHumanHandPoseObservationJointNameWrist", "hand_right:VNHumanHandPoseObservationJointNameThumbMP"),
    ("hand_right:VNHumanHandPoseObservationJointNameWrist", "hand_right:VNHumanHandPoseObservationJointNameMiddleMCP"),
    ("hand_right:VNHumanHandPoseObservationJointNameWrist", "hand_right:VNHumanHandPoseObservationJointNameLittleMCP"),
]
