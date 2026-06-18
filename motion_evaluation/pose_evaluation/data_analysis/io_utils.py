from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from data_analysis import config as analysis_config
from data_analysis.config import JointMapping, SYSTEM_SOURCE_DIRS


@dataclass(frozen=True)
class TrialData:
    time_sec: np.ndarray
    coordinates: dict[str, np.ndarray]
    availability: dict[str, bool]
    warnings: tuple[str, ...]


def load_joint_frames(
    system_name: str,
    recording_id: str,
    joint_mappings: tuple[JointMapping, ...],
) -> tuple[dict[str, pd.DataFrame], dict[str, bool], tuple[str, ...]]:
    warnings: list[str] = []
    source_dir = analysis_config.INPUT_DIR / SYSTEM_SOURCE_DIRS[system_name] / recording_id
    joint_frames: dict[str, pd.DataFrame] = {}
    availability: dict[str, bool] = {}

    for mapping in joint_mappings:
        csv_name = mapping.csv_names_by_system[system_name]
        csv_path = source_dir / csv_name
        if not csv_path.exists():
            availability[mapping.joint_id] = False
            warnings.append(f"Missing {mapping.joint_id} for {system_name}:{recording_id}.")
            continue
        frame = pd.read_csv(csv_path)
        for column in ("time_sec", "x", "y", "valid"):
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce") if column != "valid" else frame[column]
        if "valid" in frame.columns:
            frame["valid"] = frame["valid"].fillna("").astype(str).str.lower().eq("true")
            frame = frame[frame["valid"]]
        frame = frame.dropna(subset=["time_sec", "x", "y"]).sort_values("time_sec").reset_index(drop=True)
        joint_frames[mapping.joint_id] = frame
        availability[mapping.joint_id] = not frame.empty

    return joint_frames, availability, tuple(dict.fromkeys(warnings))


def resample_trial_data(
    joint_frames: dict[str, pd.DataFrame],
    availability: dict[str, bool],
    warnings: tuple[str, ...],
    time_grid: np.ndarray,
) -> TrialData:
    coordinates: dict[str, np.ndarray] = {}
    for joint_id, frame in joint_frames.items():
        coordinates[joint_id] = interpolate_joint(frame, time_grid)
    return TrialData(time_sec=time_grid, coordinates=coordinates, availability=availability, warnings=warnings)


def build_time_grid(joint_frames: dict[str, pd.DataFrame], evaluation_fps: float) -> np.ndarray:
    start_candidates = []
    end_candidates = []
    for frame in joint_frames.values():
        if frame.empty:
            continue
        start_candidates.append(float(frame["time_sec"].min()))
        end_candidates.append(float(frame["time_sec"].max()))
    if not start_candidates or not end_candidates:
        return np.array([], dtype=float)
    start_time_sec = max(start_candidates)
    end_time_sec = min(end_candidates)
    if end_time_sec <= start_time_sec:
        return np.array([start_time_sec], dtype=float)
    step = 1.0 / evaluation_fps
    return np.arange(start_time_sec, end_time_sec + (step / 2.0), step, dtype=float)


def build_shared_time_grid(
    joint_frames_a: dict[str, pd.DataFrame],
    joint_frames_b: dict[str, pd.DataFrame],
    evaluation_fps: float,
) -> np.ndarray:
    start_candidates = []
    end_candidates = []
    for frame_group in (joint_frames_a, joint_frames_b):
        for frame in frame_group.values():
            if frame.empty:
                continue
            start_candidates.append(float(frame["time_sec"].min()))
            end_candidates.append(float(frame["time_sec"].max()))
    if not start_candidates or not end_candidates:
        return np.array([], dtype=float)
    start_time_sec = max(start_candidates)
    end_time_sec = min(end_candidates)
    if end_time_sec <= start_time_sec:
        return np.array([start_time_sec], dtype=float)
    step = 1.0 / evaluation_fps
    return np.arange(start_time_sec, end_time_sec + (step / 2.0), step, dtype=float)


def interpolate_joint(frame: pd.DataFrame, time_grid: np.ndarray) -> np.ndarray:
    if frame.empty or time_grid.size == 0:
        return np.full((time_grid.size, 2), np.nan, dtype=float)
    source_time = frame["time_sec"].to_numpy(dtype=float)
    source_x = frame["x"].to_numpy(dtype=float)
    source_y = frame["y"].to_numpy(dtype=float)
    coords = np.full((time_grid.size, 2), np.nan, dtype=float)
    if len(source_time) < 2:
        return coords

    coords[:, 0] = np.interp(time_grid, source_time, source_x)
    coords[:, 1] = np.interp(time_grid, source_time, source_y)

    valid_mask = (time_grid >= source_time[0]) & (time_grid <= source_time[-1])
    coords[~valid_mask] = np.nan
    return coords
