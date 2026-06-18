"""Factory registry for selectable backends."""

from __future__ import annotations

from .apple_vision import AppleVisionBackend
from .apple_vision_3d import AppleVision3DBackend
from .base import PoseBackend
from .mediapipe_pose_hands import MediaPipePoseHandsBackend


BACKEND_REGISTRY: dict[str, type[PoseBackend]] = {
    MediaPipePoseHandsBackend.backend_name: MediaPipePoseHandsBackend,
    AppleVisionBackend.backend_name: AppleVisionBackend,
    AppleVision3DBackend.backend_name: AppleVision3DBackend,
}


def list_backends() -> list[str]:
    return sorted(BACKEND_REGISTRY.keys())


def create_backend(name: str, **kwargs) -> PoseBackend:
    backend_class = BACKEND_REGISTRY.get(name)
    if backend_class is None:
        available = ", ".join(list_backends())
        raise ValueError(f"Unknown backend '{name}'. Available backends: {available}")
    return backend_class(**kwargs)
