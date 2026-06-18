from __future__ import annotations

import numpy as np
import pandas as pd

from features import detect_hits

def _linear_fit(x: np.ndarray, y: np.ndarray, fit_intercept: bool = True) -> tuple[float, float]:
    if fit_intercept:
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        denom = np.sum((x - x_mean) ** 2)
        if denom == 0:
            a = 0.0
        else:
            a = float(np.sum((x - x_mean) * (y - y_mean)) / denom)
        b = float(y_mean - a * x_mean)
    else:
        denom = np.sum(x**2)
        a = float(np.sum(x * y) / denom) if denom != 0 else 0.0
        b = 0.0
    return a, b


def _metrics(y_true: np.ndarray, y_hat: np.ndarray) -> dict[str, float]:
    resid = y_true - y_hat
    sse = float(np.sum(resid**2))
    mae = float(np.mean(np.abs(resid)))
    rmse = float(np.sqrt(np.mean(resid**2)))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - sse / ss_tot) if ss_tot > 0 else float("nan")
    return {"r2": r2, "rmse": rmse, "mae": mae}


def fit_linear_proxy(
    predictor: pd.Series,
    target: pd.Series,
    fit_intercept: bool = True,
) -> dict[str, float]:
    x = predictor.to_numpy(dtype=float)
    y = target.to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    a, b = _linear_fit(x, y, fit_intercept=fit_intercept)
    y_hat = a * x + b

    out = _metrics(y, y_hat)
    out.update({"coef": float(a), "intercept": float(b)})
    return out


def predict_linear(predictor: pd.Series, coef: float, intercept: float) -> pd.Series:
    return predictor * coef + intercept


def cross_validate_proxy(
    predictor: pd.Series,
    target: pd.Series,
    n_splits: int = 5,
) -> pd.DataFrame:
    x = predictor.to_numpy(dtype=float)
    y = target.to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    n = len(x)
    if n < n_splits + 2:
        return pd.DataFrame(columns=["fold", "r2", "rmse", "mae"])

    fold_size = n // (n_splits + 1)
    rows: list[dict[str, float]] = []

    for fold in range(1, n_splits + 1):
        split = fold * fold_size
        train_x = x[:split]
        train_y = y[:split]
        test_x = x[split : split + fold_size]
        test_y = y[split : split + fold_size]
        if len(test_x) == 0:
            continue

        a, b = _linear_fit(train_x, train_y, fit_intercept=True)
        y_hat = a * test_x + b

        m = _metrics(test_y, y_hat)
        rows.append({"fold": float(fold), **m})

    return pd.DataFrame(rows)


def event_match_metrics(
    true_times_s: np.ndarray,
    pred_times_s: np.ndarray,
    tolerance_ms: float = 50.0,
) -> dict[str, float]:
    tol = tolerance_ms / 1000.0
    true_times = np.asarray(true_times_s, dtype=float)
    pred_times = np.asarray(pred_times_s, dtype=float)

    used_pred = np.zeros(len(pred_times), dtype=bool)
    deltas: list[float] = []
    tp = 0

    for t in true_times:
        if len(pred_times) == 0:
            break
        diff = np.abs(pred_times - t)
        idx = int(np.argmin(diff))
        if diff[idx] <= tol and not used_pred[idx]:
            used_pred[idx] = True
            tp += 1
            deltas.append(float(pred_times[idx] - t))

    fp = int((~used_pred).sum())
    fn = int(len(true_times) - tp)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    onset_mae_ms = float(np.mean(np.abs(deltas)) * 1000.0) if deltas else float("nan")
    onset_jitter_ms = float(np.std(deltas) * 1000.0) if deltas else float("nan")

    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "onset_mae_ms": onset_mae_ms,
        "onset_jitter_ms": onset_jitter_ms,
    }


def wrist_relative_proxy(
    pinky_y: pd.Series,
    wrist_y: pd.Series,
    k: float,
) -> pd.Series:
    """
    User-defined proxy:
    proxy = pinky + k * (pinky - wrist)
    """
    return pinky_y + float(k) * (pinky_y - wrist_y)


def curve_metrics(target: pd.Series, proxy: pd.Series) -> dict[str, float]:
    y = target.to_numpy(dtype=float)
    yhat = proxy.to_numpy(dtype=float)
    valid = np.isfinite(y) & np.isfinite(yhat)
    y = y[valid]
    yhat = yhat[valid]
    return _metrics(y, yhat)


def evaluate_wrist_relative_proxy(
    pinky_y: pd.Series,
    wrist_y: pd.Series,
    target_y: pd.Series,
    time_s: pd.Series,
    k: float,
    min_prom_scale: float = 0.05,
    tolerance_ms: float = 50.0,
) -> dict[str, float]:
    proxy = wrist_relative_proxy(pinky_y, wrist_y, k)
    curve = curve_metrics(target_y, proxy)

    prom_target = max(0.001, min_prom_scale * float(target_y.std()))
    prom_proxy = max(0.001, min_prom_scale * float(proxy.std()))
    hits_true = detect_hits(target_y, time_s, min_prominence=prom_target, mode="minima")
    hits_proxy = detect_hits(proxy, time_s, min_prominence=prom_proxy, mode="minima")
    events = event_match_metrics(
        hits_true["time_s"].to_numpy(),
        hits_proxy["time_s"].to_numpy(),
        tolerance_ms=tolerance_ms,
    )

    return {
        "k": float(k),
        **curve,
        "n_hits_true": float(len(hits_true)),
        "n_hits_proxy": float(len(hits_proxy)),
        **events,
    }


def grid_search_wrist_multiplier(
    pinky_y: pd.Series,
    wrist_y: pd.Series,
    target_y: pd.Series,
    time_s: pd.Series,
    k_values: np.ndarray,
    min_prom_scale: float = 0.05,
    tolerance_ms: float = 50.0,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for k in k_values:
        rows.append(
            evaluate_wrist_relative_proxy(
                pinky_y=pinky_y,
                wrist_y=wrist_y,
                target_y=target_y,
                time_s=time_s,
                k=float(k),
                min_prom_scale=min_prom_scale,
                tolerance_ms=tolerance_ms,
            )
        )
    return pd.DataFrame(rows)
