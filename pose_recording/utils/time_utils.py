"""Timestamp and session-id helpers."""

from __future__ import annotations

import re
import time
from datetime import datetime


def monotonic_time_sec() -> float:
    return time.perf_counter()


def wallclock_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def sanitize_slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned.lower() or "session"


def make_session_id(session_name: str | None = None) -> str:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    if not session_name:
        return timestamp
    return f"{sanitize_slug(session_name)}_{timestamp}"

