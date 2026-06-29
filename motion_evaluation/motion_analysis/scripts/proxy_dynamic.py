from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from analysis_proxy import (
    curve_metrics,
    event_match_metrics,
    grid_search_wrist_multiplier,
    wrist_relative_proxy,
)
from features import detect_hits

MIN_PROM_SCALE = 0.05
TOLERANCE_MS = 50.0
W_RMSE = 0.50
W_F1 = 0.30
W_ONSET = 0.20
RNG_SEED = 42

SIDE_CFG = {
    "L": {
        "pinky": "Skeleton 002:LPinky1|Position|Y",
        "wrist": "Skeleton 002:LWristOut|Position|Y",
        "index": "Skeleton 002:LIndex1|Position|Y",
        "handout": "Skeleton 002:LHandOut|Position|Y",
        "stick": "LStick:Tip|Position|Y",
    },
    "R": {
        "pinky": "Skeleton 002:RPinky1|Position|Y",
        "wrist": "Skeleton 002:RWristOut|Position|Y",
        "index": "Skeleton 002:RIndex1|Position|Y",
        "handout": "Skeleton 002:RHandOut|Position|Y",
        "stick": "RStick:Tip|Position|Y",
    },
}

FEATURE_COLS = [
    "pinky_y", "wrist_y", "index_y", "handout_y",
    "pinky_minus_wrist", "index_minus_wrist", "handout_minus_wrist", "pinky_minus_index",
    "pinky_y_vel", "wrist_y_vel", "index_y_vel", "handout_y_vel",
    "pinky_minus_wrist_vel", "index_minus_wrist_vel", "handout_minus_wrist_vel", "pinky_minus_index_vel",
    "pinky_y_acc", "wrist_y_acc", "index_y_acc", "handout_y_acc",
    "pinky_minus_wrist_acc", "index_minus_wrist_acc", "handout_minus_wrist_acc", "pinky_minus_index_acc",
]

TEMPORAL_SEED_COLS = [
    "pinky_y", "wrist_y", "index_y", "handout_y",
    "pinky_minus_wrist", "index_minus_wrist", "handout_minus_wrist", "pinky_minus_index",
]

METRIC_COLUMNS = [
    "recording", "side", "model_name", "model_family", "target_form", "split",
    "r2", "rmse", "mae", "precision", "recall", "f1", "onset_mae_ms", "onset_jitter_ms",
    "composite_score",
]


def evaluate_series(target_y: pd.Series, pred_y: pd.Series, time_s: pd.Series) -> dict[str, float]:
    curve = curve_metrics(target_y, pred_y)

    prom_target = max(0.001, MIN_PROM_SCALE * float(target_y.std()))
    prom_pred = max(0.001, MIN_PROM_SCALE * float(pred_y.std()))

    hits_true = detect_hits(target_y, time_s, min_prominence=prom_target, mode="minima")
    hits_pred = detect_hits(pred_y, time_s, min_prominence=prom_pred, mode="minima")

    events = event_match_metrics(
        hits_true["time_s"].to_numpy(dtype=float),
        hits_pred["time_s"].to_numpy(dtype=float),
        tolerance_ms=TOLERANCE_MS,
    )

    return {
        "r2": float(curve["r2"]),
        "rmse": float(curve["rmse"]),
        "mae": float(curve["mae"]),
        "precision": float(events["precision"]),
        "recall": float(events["recall"]),
        "f1": float(events["f1"]),
        "onset_mae_ms": float(events["onset_mae_ms"]),
        "onset_jitter_ms": float(events["onset_jitter_ms"]),
    }


def split_time_ordered(df: pd.DataFrame, train_frac: float = 0.6, val_frac: float = 0.2) -> dict[str, pd.DataFrame]:
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    n_test = n - n_train - n_val
    if n_train < 100 or n_val < 50 or n_test < 50:
        raise ValueError(f"Not enough frames for split: n={n}, train={n_train}, val={n_val}, test={n_test}")

    return {
        "train": df.iloc[:n_train].copy(),
        "val": df.iloc[n_train:n_train + n_val].copy(),
        "test": df.iloc[n_train + n_val:].copy(),
    }


def add_kinematic_features(df: pd.DataFrame, side: str) -> pd.DataFrame:
    cfg = SIDE_CFG[side]
    out = pd.DataFrame(index=df.index)
    out["time_s"] = df["time_s"]

    out["pinky_y"] = df[cfg["pinky"]]
    out["wrist_y"] = df[cfg["wrist"]]
    out["index_y"] = df[cfg["index"]]
    out["handout_y"] = df[cfg["handout"]]
    out["stick_y"] = df[cfg["stick"]]

    out["pinky_minus_wrist"] = out["pinky_y"] - out["wrist_y"]
    out["index_minus_wrist"] = out["index_y"] - out["wrist_y"]
    out["handout_minus_wrist"] = out["handout_y"] - out["wrist_y"]
    out["pinky_minus_index"] = out["pinky_y"] - out["index_y"]

    t = out["time_s"].to_numpy(dtype=float)
    for col in [
        "pinky_y", "wrist_y", "index_y", "handout_y",
        "pinky_minus_wrist", "index_minus_wrist", "handout_minus_wrist", "pinky_minus_index",
    ]:
        x = out[col].to_numpy(dtype=float)
        v = np.gradient(x, t)
        a = np.gradient(v, t)
        out[f"{col}_vel"] = v
        out[f"{col}_acc"] = a

    return out


def add_temporal_context_features(
    df: pd.DataFrame,
    seed_cols: list[str],
    lags: tuple[int, ...] = (1, 2, 4),
    roll_windows: tuple[int, ...] = (3, 7),
    ewm_span: int = 5,
) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    extra_cols: list[str] = []

    for col in seed_cols:
        s = out[col]

        for lag in lags:
            c = f"{col}_lag_{lag}"
            out[c] = s.shift(lag)
            extra_cols.append(c)

        for win in roll_windows:
            c_mean = f"{col}_rollmean_{win}"
            c_std = f"{col}_rollstd_{win}"
            out[c_mean] = s.rolling(window=win, min_periods=1).mean()
            out[c_std] = s.rolling(window=win, min_periods=2).std()
            extra_cols.extend([c_mean, c_std])

        c_ewm = f"{col}_ewm_{ewm_span}"
        out[c_ewm] = s.ewm(span=ewm_span, adjust=False).mean()
        extra_cols.append(c_ewm)

    return out, extra_cols


def maybe_ema_smooth(y_hat: np.ndarray, alpha: float | None) -> np.ndarray:
    arr = np.asarray(y_hat, dtype=float)
    if alpha is None:
        return arr
    if not np.isfinite(alpha) or alpha <= 0.0 or alpha >= 1.0:
        return arr
    return pd.Series(arr).ewm(alpha=alpha, adjust=False).mean().to_numpy(dtype=float)


def _minmax_penalize(values: pd.Series) -> pd.Series:
    x = values.to_numpy(dtype=float)
    finite = np.isfinite(x)
    if finite.sum() == 0:
        return pd.Series(np.ones(len(values), dtype=float), index=values.index)

    lo = float(np.nanmin(x[finite]))
    hi = float(np.nanmax(x[finite]))
    x2 = np.where(finite, x, hi)

    if hi == lo:
        norm = np.zeros_like(x2, dtype=float)
    else:
        norm = (x2 - lo) / (hi - lo)
    return pd.Series(norm, index=values.index)


def compute_composite(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    one_minus_f1 = 1.0 - out["f1"].clip(lower=0.0, upper=1.0)
    n_rmse = _minmax_penalize(out["rmse"])
    n_f1 = _minmax_penalize(one_minus_f1)
    n_onset = _minmax_penalize(out["onset_mae_ms"])
    out["composite_score"] = W_RMSE * n_rmse + W_F1 * n_f1 + W_ONSET * n_onset
    return out


def _rank_candidates_by_composite(rows: list[dict[str, float]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    scored = compute_composite(df)
    return scored.sort_values(["composite_score", "rmse"], ascending=[True, True]).reset_index(drop=True)


def _fit_and_predict(estimator, x_train, y_train, x_pred):
    estimator.fit(x_train, y_train)
    return estimator.predict(x_pred)


def _rmse_on_finite(y_true: np.ndarray, y_pred: np.ndarray, min_points: int = 50) -> float:
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(valid.sum()) < min_points:
        return float("nan")
    return float(np.sqrt(np.mean((y_true[valid] - y_pred[valid]) ** 2)))


def _build_k_models() -> list[tuple[str, Pipeline]]:
    models: list[tuple[str, Pipeline]] = []
    for alpha in [0.1, 1.0, 10.0]:
        models.append((
            f"ridge_k_alpha_{alpha:g}",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=alpha)),
            ]),
        ))
    for n_knots, alpha in product([5, 9], [0.1, 1.0, 10.0]):
        models.append((
            f"spline_ridge_k_knots_{n_knots}_alpha_{alpha:g}",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("spline", SplineTransformer(n_knots=n_knots, degree=3, include_bias=False)),
                ("ridge", Ridge(alpha=alpha)),
            ]),
        ))
    for max_depth, min_leaf in product([6, 12], [20, 80]):
        models.append((
            f"rf_k_depth_{max_depth}_leaf_{min_leaf}",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("rf", RandomForestRegressor(
                    n_estimators=180,
                    max_depth=max_depth,
                    min_samples_leaf=min_leaf,
                    random_state=RNG_SEED,
                    n_jobs=-1,
                )),
            ]),
        ))
    return models


def _build_gb_models() -> list[tuple[str, Pipeline]]:
    models: list[tuple[str, Pipeline]] = []
    for lr, depth in product([0.05, 0.1], [2, 3]):
        models.append((
            f"gbr_tip_lr_{lr}_depth_{depth}",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("gbr", GradientBoostingRegressor(
                    n_estimators=260,
                    learning_rate=lr,
                    max_depth=depth,
                    random_state=RNG_SEED,
                )),
            ]),
        ))
    return models


def _build_advanced_models() -> list[tuple[str, Pipeline]]:
    models: list[tuple[str, Pipeline]] = []
    for lr, depth, leaf_nodes in product([0.03, 0.06], [4, 6], [31, 63]):
        models.append((
            f"hgbr_tip_lr_{lr}_depth_{depth}_leaf_{leaf_nodes}",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("hgbr", HistGradientBoostingRegressor(
                    loss="squared_error",
                    learning_rate=lr,
                    max_depth=depth,
                    max_leaf_nodes=leaf_nodes,
                    min_samples_leaf=20,
                    l2_regularization=0.05,
                    max_iter=420,
                    random_state=RNG_SEED,
                )),
            ]),
        ))
    for max_depth, min_leaf in product([12, 20], [2, 8]):
        models.append((
            f"et_tip_depth_{max_depth}_leaf_{min_leaf}",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("et", ExtraTreesRegressor(
                    n_estimators=420,
                    max_depth=max_depth,
                    min_samples_leaf=min_leaf,
                    random_state=RNG_SEED,
                    n_jobs=-1,
                )),
            ]),
        ))
    models.append((
        "stack_tip_gbr_rf_et",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("stack", StackingRegressor(
                estimators=[
                    ("gbr", GradientBoostingRegressor(
                        n_estimators=280, learning_rate=0.05, max_depth=3, random_state=RNG_SEED,
                    )),
                    ("rf", RandomForestRegressor(
                        n_estimators=280, max_depth=14, min_samples_leaf=8, random_state=RNG_SEED, n_jobs=-1,
                    )),
                    ("et", ExtraTreesRegressor(
                        n_estimators=280, max_depth=16, min_samples_leaf=4, random_state=RNG_SEED, n_jobs=-1,
                    )),
                ],
                final_estimator=Ridge(alpha=1.0),
                passthrough=True,
                n_jobs=-1,
            )),
        ]),
    ))
    return models


def _tune_fixed_k(val_df: pd.DataFrame) -> tuple[float, float]:
    k_coarse = np.arange(-2.0, 30.5, 0.5)
    coarse = grid_search_wrist_multiplier(
        pinky_y=val_df["pinky_y"], wrist_y=val_df["wrist_y"], target_y=val_df["stick_y"],
        time_s=val_df["time_s"], k_values=k_coarse, min_prom_scale=MIN_PROM_SCALE, tolerance_ms=TOLERANCE_MS,
    )
    best_rmse = coarse.sort_values(["rmse", "mae"], ascending=[True, True]).iloc[0]
    best_f1 = coarse.sort_values(["f1", "onset_mae_ms", "rmse"], ascending=[False, True, True]).iloc[0]
    centers = sorted({float(best_rmse["k"]), float(best_f1["k"])})
    k_refined = np.unique(np.concatenate([np.arange(c - 1.0, c + 1.0001, 0.1) for c in centers]))
    refined = grid_search_wrist_multiplier(
        pinky_y=val_df["pinky_y"], wrist_y=val_df["wrist_y"], target_y=val_df["stick_y"],
        time_s=val_df["time_s"], k_values=k_refined, min_prom_scale=MIN_PROM_SCALE, tolerance_ms=TOLERANCE_MS,
    )
    full = pd.concat([coarse, refined], ignore_index=True).drop_duplicates(subset=["k"])
    k_best_rmse = float(full.sort_values(["rmse", "mae"], ascending=[True, True]).iloc[0]["k"])
    k_best_f1 = float(full.sort_values(["f1", "onset_mae_ms", "rmse"], ascending=[False, True, True]).iloc[0]["k"])
    return k_best_rmse, k_best_f1


def _dynamic_k_targets(frames: list[pd.DataFrame]) -> None:
    eps = 1e-4
    for d in frames:
        den = d["pinky_minus_wrist"].to_numpy(dtype=float)
        numer = (d["stick_y"] - d["pinky_y"]).to_numpy(dtype=float)
        k_true = np.full(len(d), np.nan, dtype=float)
        good = np.isfinite(den) & np.isfinite(numer) & (np.abs(den) > eps)
        k_true[good] = numer[good] / den[good]
        d["k_true"] = np.clip(k_true, -30.0, 30.0)


def _process_pair(
    recording: str,
    side: str,
    raw_df: pd.DataFrame,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    feat_df = add_kinematic_features(raw_df, side=side)
    feat_df, temporal_extra_cols = add_temporal_context_features(feat_df, TEMPORAL_SEED_COLS)
    advanced_feature_cols = FEATURE_COLS + temporal_extra_cols

    splits = split_time_ordered(feat_df)
    val_df = splits["val"]
    test_df = splits["test"]
    train_fit_df = splits["train"].iloc[::2].copy()

    rows: list[dict[str, Any]] = []
    test_preds: dict[str, np.ndarray] = {
        "time_s": test_df["time_s"].to_numpy(dtype=float),
        "target": test_df["stick_y"].to_numpy(dtype=float),
    }

    def add_rows(model_name: str, family: str, target_form: str, splits_eval: list[tuple[str, pd.Series]]) -> None:
        for split_name, y_hat_s, split_df in splits_eval:
            m = evaluate_series(split_df["stick_y"], y_hat_s, split_df["time_s"])
            rows.append({
                "recording": recording, "side": side, "model_name": model_name,
                "model_family": family, "target_form": target_form, "split": split_name, **m,
            })

    k_best_rmse, k_best_f1 = _tune_fixed_k(val_df)
    for model_name, k_star in [("fixed_k_best_rmse", k_best_rmse), ("fixed_k_best_f1", k_best_f1)]:
        evals = []
        for split_name, split_df in [("val", val_df), ("test", test_df)]:
            y_hat = wrist_relative_proxy(split_df["pinky_y"], split_df["wrist_y"], k_star)
            evals.append((split_name, y_hat, split_df))
            if split_name == "test":
                test_preds[model_name] = y_hat.to_numpy(dtype=float)
        add_rows(model_name, "fixed_k", "k(t)", evals)

    _dynamic_k_targets([train_fit_df, val_df, test_df])
    train_k_mask = train_fit_df[FEATURE_COLS + ["k_true"]].notna().all(axis=1)
    if int(train_k_mask.sum()) == 0:
        raise ValueError(f"No valid k_true training rows for {recording} {side}.")
    x_train_k = train_fit_df.loc[train_k_mask, FEATURE_COLS].to_numpy(dtype=float)
    y_train_k = train_fit_df.loc[train_k_mask, "k_true"].to_numpy(dtype=float)
    k_fallback = float(np.nanmedian(y_train_k)) if len(y_train_k) else 0.0
    if not np.isfinite(k_fallback):
        k_fallback = 0.0

    y_true_val = val_df["stick_y"].to_numpy(dtype=float)
    val_pinky = val_df["pinky_y"].to_numpy(dtype=float)
    val_pmw = val_df["pinky_minus_wrist"].to_numpy(dtype=float)
    val_x = val_df[FEATURE_COLS].to_numpy(dtype=float)

    best_k_name, best_k_model, best_k_rmse = None, None, np.inf
    for name, model in _build_k_models():
        try:
            k_val_hat = _fit_and_predict(model, x_train_k, y_train_k, val_x)
            rmse_val = _rmse_on_finite(y_true_val, val_pinky + k_val_hat * val_pmw)
            if np.isfinite(rmse_val) and rmse_val < best_k_rmse:
                best_k_rmse, best_k_name, best_k_model = rmse_val, name, model
        except Exception:
            continue

    if best_k_model is None:
        best_k_name = "fallback_constant_k_from_train_median"
    else:
        best_k_model.fit(x_train_k, y_train_k)

    k_evals = []
    for split_name, split_df in [("val", val_df), ("test", test_df)]:
        x_split = split_df[FEATURE_COLS].to_numpy(dtype=float)
        k_hat = np.full(len(split_df), k_fallback) if best_k_model is None else best_k_model.predict(x_split)
        y_hat = split_df["pinky_y"].to_numpy(dtype=float) + k_hat * split_df["pinky_minus_wrist"].to_numpy(dtype=float)
        y_hat_s = pd.Series(y_hat, index=split_df.index)
        k_evals.append((split_name, y_hat_s, split_df))
        if split_name == "test":
            test_preds[best_k_name] = y_hat
    add_rows(best_k_name, "dynamic_k_model", "k(t)", k_evals)

    train_tip_mask = train_fit_df[FEATURE_COLS + ["stick_y"]].notna().all(axis=1)
    if int(train_tip_mask.sum()) == 0:
        raise ValueError(f"No valid stick_y training rows for {recording} {side}.")
    x_train_tip = train_fit_df.loc[train_tip_mask, FEATURE_COLS].to_numpy(dtype=float)
    y_train_tip = train_fit_df.loc[train_tip_mask, "stick_y"].to_numpy(dtype=float)
    tip_fallback = float(np.nanmedian(y_train_tip)) if len(y_train_tip) else 0.0
    if not np.isfinite(tip_fallback):
        tip_fallback = 0.0
    x_val_tip = val_df[FEATURE_COLS].to_numpy(dtype=float)
    y_true_val_tip = val_df["stick_y"].to_numpy(dtype=float)

    best_tip_name, best_tip_model, best_tip_rmse = None, None, np.inf
    for name, model in _build_gb_models():
        try:
            y_val_hat = _fit_and_predict(model, x_train_tip, y_train_tip, x_val_tip)
            rmse_val = _rmse_on_finite(y_true_val_tip, y_val_hat)
            if np.isfinite(rmse_val) and rmse_val < best_tip_rmse:
                best_tip_rmse, best_tip_name, best_tip_model = rmse_val, name, model
        except Exception:
            continue

    if best_tip_model is None:
        best_tip_name = "fallback_constant_tip_from_train_median"
    else:
        best_tip_model.fit(x_train_tip, y_train_tip)

    tip_evals = []
    for split_name, split_df in [("val", val_df), ("test", test_df)]:
        y_hat = (
            np.full(len(split_df), tip_fallback)
            if best_tip_model is None
            else best_tip_model.predict(split_df[FEATURE_COLS].to_numpy(dtype=float))
        )
        y_hat_s = pd.Series(y_hat, index=split_df.index)
        tip_evals.append((split_name, y_hat_s, split_df))
        if split_name == "test":
            test_preds[best_tip_name] = np.asarray(y_hat, dtype=float)
    add_rows(best_tip_name, "direct_tip_model", "tip_y", tip_evals)

    x_train_adv = train_fit_df.loc[train_tip_mask, advanced_feature_cols].to_numpy(dtype=float)
    x_val_adv = val_df[advanced_feature_cols].to_numpy(dtype=float)
    smoothing_alphas = [None, 0.35, 0.55]

    adv_candidate_rows: list[dict[str, float]] = []
    adv_meta: dict[int, dict[str, object]] = {}
    for base_name, model_template in _build_advanced_models():
        try:
            model = clone(model_template)
            y_val_hat_raw = _fit_and_predict(model, x_train_adv, y_train_tip, x_val_adv)
            for alpha in smoothing_alphas:
                alpha_tag = "raw" if alpha is None else f"ema_{alpha:.2f}".replace(".", "p")
                model_name = f"{base_name}_{alpha_tag}"
                y_val_hat = maybe_ema_smooth(y_val_hat_raw, alpha=alpha)
                m_val = evaluate_series(val_df["stick_y"], pd.Series(y_val_hat, index=val_df.index), val_df["time_s"])
                cand_id = len(adv_candidate_rows)
                adv_meta[cand_id] = {"model_template": model_template, "model_name": model_name, "alpha": alpha}
                adv_candidate_rows.append({"candidate_id": cand_id, "model_name": model_name, **m_val})
        except Exception:
            continue

    ranked_adv = _rank_candidates_by_composite(adv_candidate_rows)
    best_adv_name, best_adv_model, best_adv_alpha = "fallback_constant_tip_eventaware", None, None
    adv_fallback = tip_fallback
    if not ranked_adv.empty:
        best_adv_id = int(ranked_adv.iloc[0]["candidate_id"])
        best_adv_meta = adv_meta[best_adv_id]
        best_adv_name = str(best_adv_meta["model_name"])
        best_adv_alpha = best_adv_meta["alpha"]
        best_adv_model = clone(best_adv_meta["model_template"])
        best_adv_model.fit(x_train_adv, y_train_tip)

    adv_evals = []
    for split_name, split_df in [("val", val_df), ("test", test_df)]:
        if best_adv_model is None:
            y_hat = np.full(len(split_df), adv_fallback, dtype=float)
        else:
            y_hat_raw = best_adv_model.predict(split_df[advanced_feature_cols].to_numpy(dtype=float))
            y_hat = maybe_ema_smooth(y_hat_raw, alpha=best_adv_alpha)
        y_hat_s = pd.Series(y_hat, index=split_df.index)
        adv_evals.append((split_name, y_hat_s, split_df))
        if split_name == "test":
            test_preds[best_adv_name] = np.asarray(y_hat, dtype=float)
    add_rows(best_adv_name, "advanced_supervised_model", "tip_y", adv_evals)

    return rows, test_preds


def _plot_overlay(
    recording: str,
    side: str,
    test_preds: dict[str, np.ndarray],
    test_rows: pd.DataFrame,
    out_path: Path,
    display_title: str,
) -> None:
    top = test_rows.sort_values("composite_score", ascending=True).head(3)
    t = test_preds["time_s"]

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.grid(True)
    ax.plot(t, test_preds["target"], label="stick_tip_true", linewidth=2.0, color="black")
    for _, row in top.iterrows():
        name = row["model_name"]
        if name in test_preds:
            ax.plot(t, test_preds[name], label=f"{name} (score={row['composite_score']:.3f})", linewidth=1.2)
    ax.set_title(f"{display_title} {side} test overlay: top composite models")
    ax.set_xlabel("time_s")
    ax.set_ylabel("Y")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_proxy_dynamic(
    opti: dict[str, Any],
    *,
    overlay_dir: Path,
    overlay_recordings: dict[str, str],
) -> pd.DataFrame:
    all_rows: list[dict[str, Any]] = []
    test_preds_by_pair: dict[tuple[str, str], dict[str, np.ndarray]] = {}

    for recording, rec in opti.items():
        df = rec.data if hasattr(rec, "data") else rec
        for side in ("L", "R"):
            rows, test_preds = _process_pair(recording, side, df)
            all_rows.extend(rows)
            test_preds_by_pair[(recording, side)] = test_preds

    metrics_df = pd.DataFrame(all_rows)

    scored = [compute_composite(g) for _, g in metrics_df.groupby(["recording", "side", "split"], sort=False)]
    metrics_scored = pd.concat(scored, ignore_index=True)[METRIC_COLUMNS]

    test_scored = metrics_scored[metrics_scored["split"] == "test"]
    for recording, display_name in overlay_recordings.items():
        for side in ("L", "R"):
            pair_rows = test_scored[(test_scored["recording"] == recording) & (test_scored["side"] == side)]
            preds = test_preds_by_pair.get((recording, side))
            if pair_rows.empty or preds is None:
                continue
            _plot_overlay(
                recording, side, preds, pair_rows,
                overlay_dir / f"proxy_dynamic_overlay_top_{recording}_{side}.png",
                display_name,
            )

    return metrics_scored
