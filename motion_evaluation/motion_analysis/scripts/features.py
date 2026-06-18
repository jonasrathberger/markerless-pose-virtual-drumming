from __future__ import annotations

import numpy as np
import pandas as pd


def range_metrics(series: pd.Series) -> dict[str, float]:
    x = series.to_numpy(dtype=float)
    return {
        "peak_to_peak": float(np.nanmax(x) - np.nanmin(x)),
        "iqr": float(np.nanpercentile(x, 75) - np.nanpercentile(x, 25)),
        "std": float(np.nanstd(x)),
        "p95_p05": float(np.nanpercentile(x, 95) - np.nanpercentile(x, 5)),
    }


def velocity_acceleration(series: pd.Series, time_s: pd.Series) -> pd.DataFrame:
    x = series.to_numpy(dtype=float)
    t = time_s.to_numpy(dtype=float)
    v = np.gradient(x, t)
    a = np.gradient(v, t)
    return pd.DataFrame({"time_s": t, "value": x, "velocity": v, "acceleration": a})


def _local_extrema_indices(y: np.ndarray, mode: str = "minima") -> np.ndarray:
    if len(y) < 3:
        return np.array([], dtype=int)

    prev = y[:-2]
    curr = y[1:-1]
    nxt = y[2:]

    if mode == "minima":
        mask = (curr < prev) & (curr <= nxt)
    else:
        mask = (curr > prev) & (curr >= nxt)

    idx = np.flatnonzero(mask) + 1
    return idx


def _prominence(y: np.ndarray, i: int, half_window: int = 5, mode: str = "minima") -> float:
    lo = max(0, i - half_window)
    hi = min(len(y), i + half_window + 1)
    neighborhood = y[lo:hi]
    if len(neighborhood) == 0:
        return 0.0

    if mode == "minima":
        return float(np.max(neighborhood) - y[i])
    return float(y[i] - np.min(neighborhood))


def detect_hits(
    y: pd.Series,
    time_s: pd.Series,
    min_prominence: float = 0.02,
    min_distance_s: float = 0.05,
    mode: str = "minima",
) -> pd.DataFrame:
    yv = y.to_numpy(dtype=float)
    tv = time_s.to_numpy(dtype=float)

    valid = np.isfinite(yv) & np.isfinite(tv)
    yv = yv[valid]
    tv = tv[valid]

    if len(yv) < 3:
        return pd.DataFrame(columns=["idx", "time_s", "y", "prominence"])

    idx_candidates = _local_extrema_indices(yv, mode=mode)
    if len(idx_candidates) == 0:
        return pd.DataFrame(columns=["idx", "time_s", "y", "prominence"])

    dt = float(np.nanmedian(np.diff(tv)))
    min_distance = int(max(1, round(min_distance_s / dt))) if dt > 0 else 1

    accepted: list[int] = []
    prominences: list[float] = []

    for idx in idx_candidates:
        prom = _prominence(yv, idx, half_window=max(3, min_distance), mode=mode)
        if prom < min_prominence:
            continue

        if accepted and (idx - accepted[-1]) < min_distance:
            # Keep the stronger peak within refractory window.
            if prom > prominences[-1]:
                accepted[-1] = idx
                prominences[-1] = prom
            continue

        accepted.append(int(idx))
        prominences.append(float(prom))

    if not accepted:
        return pd.DataFrame(columns=["idx", "time_s", "y", "prominence"])

    accepted_arr = np.array(accepted, dtype=int)
    return pd.DataFrame(
        {
            "idx": accepted_arr,
            "time_s": tv[accepted_arr],
            "y": yv[accepted_arr],
            "prominence": np.array(prominences, dtype=float),
        }
    )
