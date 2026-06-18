from __future__ import annotations

import itertools

import numpy as np


def cohen_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    denom = np.std(diff, ddof=1)
    if denom == 0 or np.isnan(denom):
        return float("nan")
    return float(np.mean(diff) / denom)


def _sign_flip_p_value(diff: np.ndarray, n_iter: int = 20000, seed: int = 42) -> tuple[float, float]:
    diff = diff[np.isfinite(diff)]
    n = len(diff)
    if n == 0:
        return float("nan"), float("nan")

    observed = float(np.mean(diff))

    # Exact test for small n.
    if n <= 18:
        means = []
        for signs in itertools.product([-1.0, 1.0], repeat=n):
            means.append(float(np.mean(diff * np.array(signs))))
        means_arr = np.abs(np.array(means))
        p = float(np.mean(means_arr >= abs(observed)))
        return observed, p

    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_iter, n))
    perm = np.mean(signs * diff[None, :], axis=1)
    p = float(np.mean(np.abs(perm) >= abs(observed)))
    return observed, p


def bootstrap_ci_mean_diff(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    diff = a_arr - b_arr
    diff = diff[np.isfinite(diff)]

    if len(diff) == 0:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boot = np.mean(diff[idx], axis=1)
    lo = float(np.quantile(boot, alpha / 2))
    hi = float(np.quantile(boot, 1 - alpha / 2))
    return lo, hi


def paired_test(a: np.ndarray, b: np.ndarray) -> dict[str, float | str]:
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    diff = a_arr - b_arr

    if np.isfinite(diff).sum() < 3:
        return {
            "test": "insufficient_data",
            "stat": float("nan"),
            "p_value": float("nan"),
            "effect_size_d": cohen_d_paired(a_arr, b_arr),
            "mean_diff_ci_low": float("nan"),
            "mean_diff_ci_high": float("nan"),
        }

    stat, p_val = _sign_flip_p_value(diff)
    ci_lo, ci_hi = bootstrap_ci_mean_diff(a_arr, b_arr)

    return {
        "test": "paired_sign_flip_permutation",
        "stat": float(stat),
        "p_value": float(p_val),
        "effect_size_d": cohen_d_paired(a_arr, b_arr),
        "mean_diff_ci_low": ci_lo,
        "mean_diff_ci_high": ci_hi,
    }
