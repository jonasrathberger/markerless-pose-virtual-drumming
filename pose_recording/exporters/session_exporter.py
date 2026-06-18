"""Buffered exporters for JSON, CSV, Parquet, snapshots, and video files."""

from __future__ import annotations

import json
import logging
import queue
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from config.defaults import DEFAULT_EXPORT_BUFFER_ROWS, DEFAULT_VIDEO_CODEC, DEFAULT_VIDEO_QUEUE_SIZE
from models.canonical import CANONICAL_SPEC_BY_ID, CORE_LANDMARK_IDS
from models.schema import FrameResult, LANDMARK_EXPORT_COLUMNS
from utils.io import ensure_directory
from utils.time_utils import wallclock_iso


LOGGER = logging.getLogger(__name__)

PARQUET_SCHEMA = pa.schema(
    [
        ("session_id", pa.string()),
        ("model_name", pa.string()),
        ("frame_index", pa.int64()),
        ("timestamp_monotonic_sec", pa.float64()),
        ("timestamp_wallclock_iso", pa.string()),
        ("image_width", pa.int64()),
        ("image_height", pa.int64()),
        ("person_id", pa.int64()),
        ("landmark_name", pa.string()),
        ("landmark_group", pa.string()),
        ("side", pa.string()),
        ("x_norm", pa.float64()),
        ("y_norm", pa.float64()),
        ("z_rel", pa.float64()),
        ("x_px", pa.float64()),
        ("y_px", pa.float64()),
        ("confidence", pa.float64()),
        ("visibility", pa.float64()),
        ("tracking_present", pa.bool_()),
    ]
)


class AsyncVideoWriter:
    """Decouple video encoding from frame processing."""

    def __init__(self, output_path: Path, fps: float, frame_size: tuple[int, int]):
        self.output_path = output_path
        self.fps = fps
        self.frame_size = frame_size
        self.queue: queue.Queue[Any] = queue.Queue(maxsize=DEFAULT_VIDEO_QUEUE_SIZE)
        self.writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*DEFAULT_VIDEO_CODEC),
            fps,
            frame_size,
        )
        if not self.writer.isOpened():
            raise RuntimeError(f"Unable to open video writer for {output_path}.")
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.dropped_frames = 0
        self.thread.start()

    def write(self, frame_bgr) -> None:
        try:
            self.queue.put_nowait(frame_bgr.copy())
        except queue.Full:
            self.dropped_frames += 1

    def close(self) -> None:
        self.stop_event.set()
        self.thread.join()
        self.writer.release()

    def _run(self) -> None:
        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                frame = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue
            self.writer.write(frame)
            self.queue.task_done()


@dataclass(slots=True)
class SessionExporter:
    session_id: str
    model_name: str
    session_dir: Path
    command_line_settings: dict[str, Any]
    model_configuration: dict[str, Any]
    notes: str = ""
    target_fps: int = 30
    camera_name: str | None = None
    save_raw_video: bool = False
    save_annotated_video: bool = False
    save_parquet: bool = True
    row_buffer: list[dict[str, Any]] = field(default_factory=list)
    export_buffer_rows: int = DEFAULT_EXPORT_BUFFER_ROWS
    csv_path: Path = field(init=False)
    parquet_path: Path = field(init=False)
    metadata_path: Path = field(init=False)
    preview_log_path: Path = field(init=False)
    raw_video_path: Path = field(init=False)
    annotated_video_path: Path = field(init=False)
    preview_log_handle: Any = field(init=False)
    csv_header_written: bool = field(init=False, default=False)
    parquet_writer: pq.ParquetWriter | None = field(init=False, default=None)
    raw_video_writer: AsyncVideoWriter | None = field(init=False, default=None)
    annotated_video_writer: AsyncVideoWriter | None = field(init=False, default=None)
    recorded_frame_count: int = field(init=False, default=0)
    recording_started_at: str | None = field(init=False, default=None)
    recording_frame_timestamps: list[float] = field(init=False, default_factory=list)
    capture_failures: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.session_dir = ensure_directory(self.session_dir)
        self.csv_path = self.session_dir / "landmarks.csv"
        self.parquet_path = self.session_dir / "landmarks.parquet"
        self.metadata_path = self.session_dir / "metadata.json"
        self.preview_log_path = self.session_dir / "preview_log.txt"
        self.raw_video_path = self.session_dir / "raw_video.mp4"
        self.annotated_video_path = self.session_dir / "annotated_video.mp4"
        self.preview_log_handle = self.preview_log_path.open("a", encoding="utf-8")
        self.csv_header_written = False
        self.parquet_writer: pq.ParquetWriter | None = None
        self.raw_video_writer: AsyncVideoWriter | None = None
        self.annotated_video_writer: AsyncVideoWriter | None = None
        self.recorded_frame_count = 0
        self.recording_started_at: str | None = None
        self.recording_frame_timestamps: list[float] = []
        self.capture_failures = 0

    def open_video_writers(self, width: int, height: int) -> None:
        frame_size = (width, height)
        if self.save_raw_video and self.raw_video_writer is None:
            self.raw_video_writer = AsyncVideoWriter(self.raw_video_path, self.target_fps, frame_size)
        if self.save_annotated_video and self.annotated_video_writer is None:
            self.annotated_video_writer = AsyncVideoWriter(
                self.annotated_video_path,
                self.target_fps,
                frame_size,
            )

    def log_event(self, message: str) -> None:
        self.preview_log_handle.write(f"{wallclock_iso()} {message}\n")
        self.preview_log_handle.flush()

    def append_frame(
        self,
        *,
        result: FrameResult,
        frame_index: int,
        timestamp_monotonic_sec: float,
        timestamp_wallclock_iso: str,
        image_width: int,
        image_height: int,
        raw_frame_bgr=None,
        annotated_frame_bgr=None,
    ) -> None:
        rows = self._frame_result_to_rows(
            result=result,
            frame_index=frame_index,
            timestamp_monotonic_sec=timestamp_monotonic_sec,
            timestamp_wallclock_iso=timestamp_wallclock_iso,
            image_width=image_width,
            image_height=image_height,
        )
        self.row_buffer.extend(rows)
        self.recorded_frame_count += 1
        self.recording_frame_timestamps.append(timestamp_monotonic_sec)
        if raw_frame_bgr is not None and self.raw_video_writer is not None:
            self.raw_video_writer.write(raw_frame_bgr)
        if annotated_frame_bgr is not None and self.annotated_video_writer is not None:
            self.annotated_video_writer.write(annotated_frame_bgr)
        if len(self.row_buffer) >= self.export_buffer_rows:
            self.flush_rows()

    def flush_rows(self) -> None:
        if not self.row_buffer:
            return

        dataframe = pd.DataFrame(self.row_buffer, columns=LANDMARK_EXPORT_COLUMNS)
        dataframe.to_csv(
            self.csv_path,
            mode="a",
            header=not self.csv_header_written,
            index=False,
            encoding="utf-8",
        )
        self.csv_header_written = True

        if self.save_parquet:
            table = self._dataframe_to_arrow_table(dataframe)
            if self.parquet_writer is None:
                self.parquet_writer = pq.ParquetWriter(self.parquet_path, PARQUET_SCHEMA)
            self.parquet_writer.write_table(table)

        self.row_buffer.clear()

    def save_snapshot(self, frame_bgr, snapshot_name: str) -> Path:
        snapshots_dir = ensure_directory(self.session_dir / "snapshots")
        output_path = snapshots_dir / snapshot_name
        cv2.imwrite(str(output_path), frame_bgr)
        return output_path

    def note_capture_failure(self) -> None:
        self.capture_failures += 1

    def write_metadata(self, *, resolution: tuple[int, int], actual_mean_fps: float | None = None) -> None:
        metadata = {
            "session_id": self.session_id,
            "model_name": self.model_name,
            "recording_start_time": self.recording_started_at,
            "camera_name": self.camera_name,
            "resolution": {"width": resolution[0], "height": resolution[1]},
            "target_fps": self.target_fps,
            "actual_mean_fps": actual_mean_fps,
            "dropped_frame_count": self.capture_failures + self._video_writer_drop_count(),
            "command_line_settings": self.command_line_settings,
            "model_configuration": self.model_configuration,
            "notes": self.notes,
            "recorded_frame_count": self.recorded_frame_count,
            "preview_log_path": self.preview_log_path.name,
        }
        self.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def close(self, *, resolution: tuple[int, int]) -> None:
        self.flush_rows()
        actual_mean_fps = self._compute_actual_mean_fps()
        self.write_metadata(resolution=resolution, actual_mean_fps=actual_mean_fps)
        if self.parquet_writer is not None:
            self.parquet_writer.close()
        if self.raw_video_writer is not None:
            self.raw_video_writer.close()
        if self.annotated_video_writer is not None:
            self.annotated_video_writer.close()
        self.preview_log_handle.close()

    def _compute_actual_mean_fps(self) -> float | None:
        if len(self.recording_frame_timestamps) < 2:
            return None
        elapsed = self.recording_frame_timestamps[-1] - self.recording_frame_timestamps[0]
        if elapsed <= 0:
            return None
        return (len(self.recording_frame_timestamps) - 1) / elapsed

    def _video_writer_drop_count(self) -> int:
        raw_drop_count = self.raw_video_writer.dropped_frames if self.raw_video_writer else 0
        annotated_drop_count = self.annotated_video_writer.dropped_frames if self.annotated_video_writer else 0
        return raw_drop_count + annotated_drop_count

    def _frame_result_to_rows(
        self,
        *,
        result: FrameResult,
        frame_index: int,
        timestamp_monotonic_sec: float,
        timestamp_wallclock_iso: str,
        image_width: int,
        image_height: int,
    ) -> list[dict[str, Any]]:
        ordered_ids = list(CORE_LANDMARK_IDS)
        extra_ids = sorted(landmark_id for landmark_id in result.landmarks.keys() if landmark_id not in CANONICAL_SPEC_BY_ID)
        ordered_ids.extend(extra_ids)

        rows: list[dict[str, Any]] = []
        for landmark_id in ordered_ids:
            spec = CANONICAL_SPEC_BY_ID.get(landmark_id)
            if spec is None:
                parts = landmark_id.split(":")
                if len(parts) == 3:
                    group_name, side, landmark_name = parts
                else:
                    group_name, side, landmark_name = "unknown", "unknown", landmark_id
            else:
                group_name = spec.landmark_group
                side = spec.side
                landmark_name = spec.landmark_name

            observation = result.landmarks.get(landmark_id)
            x_norm = observation.x_norm if observation else None
            y_norm = observation.y_norm if observation else None
            x_px = observation.x_px if observation else None
            y_px = observation.y_px if observation else None
            if x_px is None and x_norm is not None:
                x_px = x_norm * image_width
            if y_px is None and y_norm is not None:
                y_px = y_norm * image_height

            rows.append(
                {
                    "session_id": self.session_id,
                    "model_name": self.model_name,
                    "frame_index": frame_index,
                    "timestamp_monotonic_sec": timestamp_monotonic_sec,
                    "timestamp_wallclock_iso": timestamp_wallclock_iso,
                    "image_width": image_width,
                    "image_height": image_height,
                    "person_id": result.person_id,
                    "landmark_name": landmark_name,
                    "landmark_group": group_name,
                    "side": side,
                    "x_norm": x_norm,
                    "y_norm": y_norm,
                    "z_rel": observation.z_rel if observation else None,
                    "x_px": x_px,
                    "y_px": y_px,
                    "confidence": observation.confidence if observation else None,
                    "visibility": observation.visibility if observation else None,
                    "tracking_present": observation.present if observation else False,
                }
            )
        return rows

    @staticmethod
    def _dataframe_to_arrow_table(dataframe: pd.DataFrame) -> pa.Table:
        arrays = []
        for field in PARQUET_SCHEMA:
            values = dataframe[field.name].tolist()
            arrays.append(pa.array(values, type=field.type))
        return pa.Table.from_arrays(arrays, schema=PARQUET_SCHEMA)
