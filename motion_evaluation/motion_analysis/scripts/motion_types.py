from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(slots=True)
class MotionRecording:
    """Container for one parsed motion recording."""

    source: str
    path: Path
    metadata: dict[str, str]
    data: pd.DataFrame

    @property
    def n_frames(self) -> int:
        return len(self.data)

    @property
    def columns(self) -> list[str]:
        return list(self.data.columns)

    def summary(self) -> dict[str, Any]:
        time_col = "time_s"
        if time_col in self.data:
            start = float(self.data[time_col].iloc[0])
            end = float(self.data[time_col].iloc[-1])
            duration = end - start
        else:
            start = float("nan")
            end = float("nan")
            duration = float("nan")

        return {
            "source": self.source,
            "path": str(self.path),
            "n_frames": self.n_frames,
            "n_columns": len(self.columns),
            "time_start_s": start,
            "time_end_s": end,
            "duration_s": duration,
        }

