from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp/mplcache").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from thesis_figures import apply_thesis_style, figure_size, save_single_figure, style_axis

apply_thesis_style()


def _ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _boxplot_with_labels(ax, values: list[pd.Series], labels: list[str]) -> None:
    try:
        ax.boxplot(values, tick_labels=labels, showmeans=True)
    except TypeError:
        ax.boxplot(values, labels=labels, showmeans=True)


def plot_proxy_regression(
    predictor: pd.Series,
    target: pd.Series,
    predicted: pd.Series,
    out_path: str | Path,
    title: str = "Proxy Regression",
) -> None:
    _ensure_parent(out_path)

    valid = predictor.notna() & target.notna() & predicted.notna()
    predictor = predictor[valid]
    target = target[valid]
    predicted = predicted[valid]

    fig, ax = plt.subplots(figsize=figure_size("wide", aspect=0.70))
    ax.scatter(predictor, target, s=8, alpha=0.35, label="Observed")
    order = np.argsort(predictor.to_numpy(dtype=float))
    ax.plot(predictor.iloc[order], predicted.iloc[order], color="red", linewidth=2.0, label="Model")
    ax.set_title(title)
    ax.set_xlabel("Predictor")
    ax.set_ylabel("Target")
    ax.legend()
    style_axis(ax)
    fig.tight_layout()
    save_single_figure(fig, Path(out_path))


def plot_time_overlay(
    time_s: pd.Series,
    reference: pd.Series,
    estimate: pd.Series,
    out_path: str | Path,
    title: str = "Aligned Time-Series",
) -> None:
    _ensure_parent(out_path)

    fig, ax = plt.subplots(figsize=figure_size("wide", aspect=0.44))
    ax.plot(time_s, reference, label="Reference", linewidth=1.2)
    ax.plot(time_s, estimate, label="Estimate", linewidth=1.0, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Value")
    ax.legend()
    style_axis(ax, x_grid=True)
    fig.tight_layout()
    save_single_figure(fig, Path(out_path))


def plot_box_compare(
    values_a: pd.Series,
    values_b: pd.Series,
    label_a: str,
    label_b: str,
    out_path: str | Path,
    title: str,
) -> None:
    _ensure_parent(out_path)

    fig, ax = plt.subplots(figsize=figure_size("wide", aspect=0.62))
    _boxplot_with_labels(ax, [values_a.dropna(), values_b.dropna()], [label_a, label_b])
    ax.set_title(title)
    ax.set_ylabel("Value")
    style_axis(ax)
    fig.tight_layout()
    save_single_figure(fig, Path(out_path))
