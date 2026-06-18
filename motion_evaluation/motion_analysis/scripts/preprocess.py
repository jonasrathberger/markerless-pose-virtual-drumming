from __future__ import annotations

import numpy as np
import pandas as pd


def infer_sampling_stats(time_s: pd.Series) -> dict[str, float]:
    t = time_s.to_numpy(dtype=float)
    dt = np.diff(t)
    dt = dt[dt > 0]
    if len(dt) == 0:
        return {
            "mean_dt_s": float("nan"),
            "effective_hz": float("nan"),
            "std_dt_s": float("nan"),
            "min_dt_s": float("nan"),
            "max_dt_s": float("nan"),
        }

    mean_dt = float(np.mean(dt))
    return {
        "mean_dt_s": mean_dt,
        "effective_hz": float(1.0 / mean_dt),
        "std_dt_s": float(np.std(dt)),
        "min_dt_s": float(np.min(dt)),
        "max_dt_s": float(np.max(dt)),
    }


def zscore_normalize(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        s = out[col]
        mu = s.mean()
        sigma = s.std(ddof=0)
        if np.isfinite(sigma) and sigma > 0:
            out[col] = (s - mu) / sigma
    return out


def _rolling_lowpass(series: pd.Series, sample_hz: float, cutoff_hz: float) -> pd.Series:
    # Very lightweight fallback if scipy is unavailable.
    if sample_hz <= 0 or cutoff_hz <= 0:
        return series
    window = int(max(3, round(sample_hz / cutoff_hz)))
    if window % 2 == 0:
        window += 1
    return series.rolling(window=window, center=True, min_periods=1).mean()


def lowpass_filter(
    series: pd.Series,
    sample_hz: float,
    cutoff_hz: float = 6.0,
    order: int = 4,
) -> pd.Series:
    if sample_hz <= 0 or cutoff_hz <= 0 or cutoff_hz >= sample_hz / 2:
        return series

    x = series.to_numpy(dtype=float)
    nan_mask = np.isnan(x)
    if np.any(nan_mask):
        valid_idx = np.flatnonzero(~nan_mask)
        if len(valid_idx) < 2:
            return series
        x[nan_mask] = np.interp(np.flatnonzero(nan_mask), valid_idx, x[valid_idx])

    try:
        from scipy.signal import butter, filtfilt  # type: ignore

        b, a = butter(order, cutoff_hz / (sample_hz / 2), btype="low")
        y = filtfilt(b, a, x)
        return pd.Series(y, index=series.index)
    except Exception:
        # If scipy is unavailable, use a centered rolling average fallback.
        return _rolling_lowpass(pd.Series(x, index=series.index), sample_hz, cutoff_hz)


def resample_time_series(
    data: pd.DataFrame,
    target_time: np.ndarray,
    value_columns: list[str],
    time_col: str = "time_s",
) -> pd.DataFrame:
    src_time = data[time_col].to_numpy(dtype=float)
    out = pd.DataFrame({time_col: target_time})

    for col in value_columns:
        y = data[col].to_numpy(dtype=float)
        valid = np.isfinite(src_time) & np.isfinite(y)
        if valid.sum() < 2:
            out[col] = np.nan
            continue
        out[col] = np.interp(target_time, src_time[valid], y[valid])

    return out


def align_by_cross_correlation(
    reference: np.ndarray,
    estimate: np.ndarray,
    sample_hz: float,
    max_lag_s: float = 1.0,
) -> dict[str, float]:
    ref = np.asarray(reference, dtype=float)
    est = np.asarray(estimate, dtype=float)

    n = min(len(ref), len(est))
    if n < 3:
        return {"lag_samples": 0, "lag_ms": 0.0, "corr_peak": float("nan")}

    ref = ref[:n]
    est = est[:n]

    mask = np.isfinite(ref) & np.isfinite(est)
    ref = ref[mask]
    est = est[mask]
    n = min(len(ref), len(est))
    if n < 3:
        return {"lag_samples": 0, "lag_ms": 0.0, "corr_peak": float("nan")}

    ref = ref[:n] - np.mean(ref[:n])
    est = est[:n] - np.mean(est[:n])

    corr = np.correlate(ref, est, mode="full")
    lags = np.arange(-n + 1, n)

    max_lag = int(max_lag_s * sample_hz)
    mask_lag = (lags >= -max_lag) & (lags <= max_lag)
    corr = corr[mask_lag]
    lags = lags[mask_lag]

    best_idx = int(np.argmax(corr))
    best_lag_samples = int(lags[best_idx])

    return {
        "lag_samples": best_lag_samples,
        "lag_ms": (best_lag_samples / sample_hz) * 1000.0,
        "corr_peak": float(corr[best_idx]),
    }
