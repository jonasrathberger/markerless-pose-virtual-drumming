"""Shared Matplotlib style for thesis figures.

The thesis PDF uses Latin Modern fonts. A full LaTeX toolchain is not
available in all project environments, so this module prefers installed
Latin Modern/CMU fonts and falls back to STIX/DejaVu serif fonts while
keeping LaTeX-like sizes and math rendering.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True, slots=True)
class ThesisFigureStyle:
    font_family: tuple[str, ...] = (
        "Latin Modern Roman",
        "LM Roman 10",
        "CMU Serif",
        "STIX Two Text",
        "STIXGeneral",
        "DejaVu Serif",
    )
    base_font_size: float = 10.0
    small_font_size: float = 8.5
    annotation_font_size: float = 8.5
    axes_title_size: float = 10.0
    figure_title_size: float = 11.0
    label_size: float = 10.0
    tick_size: float = 8.5
    legend_size: float = 8.8
    line_width: float = 1.2
    grid_width: float = 0.45
    spine_width: float = 0.6
    dpi: int = 300
    facecolor: str = "white"
    text_color: str = "#202020"
    muted_text_color: str = "#4b5563"
    grid_color: str = "#d9d9d9"
    light_grid_color: str = "#ededed"
    spine_color: str = "#8a8a8a"
    blue: str = "#2f6f9f"
    orange: str = "#d0812c"
    green: str = "#5f8f3f"
    red: str = "#c62828"
    purple: str = "#6a1b9a"


THESIS_STYLE = ThesisFigureStyle()


def configure_matplotlib_cache(cache_dir: Path | None = None) -> Path:
    """Set writable Matplotlib/font cache directories before pyplot import."""
    resolved_cache = (cache_dir or Path(".cache/matplotlib")).resolve()
    resolved_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(resolved_cache))
    xdg_cache = resolved_cache.parent.resolve()
    xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))
    return resolved_cache


def apply_thesis_style(*, cache_dir: Path | None = None, style: ThesisFigureStyle = THESIS_STYLE) -> None:
    """Apply the shared thesis Matplotlib style."""
    configure_matplotlib_cache(cache_dir)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": style.facecolor,
            "axes.facecolor": style.facecolor,
            "savefig.facecolor": style.facecolor,
            "savefig.edgecolor": style.facecolor,
            "font.family": "serif",
            "font.serif": list(style.font_family),
            "font.size": style.base_font_size,
            "text.color": style.text_color,
            "axes.labelcolor": style.text_color,
            "axes.edgecolor": style.spine_color,
            "axes.linewidth": style.spine_width,
            "axes.titlesize": style.axes_title_size,
            "axes.titleweight": "regular",
            "axes.labelsize": style.label_size,
            "xtick.labelsize": style.tick_size,
            "ytick.labelsize": style.tick_size,
            "xtick.color": style.text_color,
            "ytick.color": style.text_color,
            "legend.fontsize": style.legend_size,
            "legend.frameon": False,
            "figure.titlesize": style.figure_title_size,
            "figure.titleweight": "regular",
            "axes.grid": False,
            "grid.color": style.grid_color,
            "grid.linewidth": style.grid_width,
            "grid.alpha": 1.0,
            "lines.linewidth": style.line_width,
            "patch.linewidth": style.spine_width,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "mathtext.fontset": "stix",
            "mathtext.default": "regular",
        }
    )


def figure_size(kind: str = "single", *, aspect: float | None = None) -> tuple[float, float]:
    """Return standard figure dimensions in inches."""
    widths = {
        "single": 6.3,
        "wide": 6.3,
        "half": 3.15,
        "tall": 6.3,
        "dashboard": 6.3,
    }
    default_aspects = {
        "single": 0.62,
        "wide": 0.52,
        "half": 0.78,
        "tall": 0.78,
        "dashboard": 0.68,
    }
    width = widths.get(kind, widths["single"])
    resolved_aspect = default_aspects.get(kind, default_aspects["single"]) if aspect is None else aspect
    return width, width * resolved_aspect


def style_axis(
    axis,
    *,
    y_grid: bool = True,
    x_grid: bool = False,
    hide_top_right: bool = True,
    hide_y_tick_labels: bool = False,
    style: ThesisFigureStyle = THESIS_STYLE,
) -> None:
    """Apply shared axis styling."""
    axis.set_facecolor(style.facecolor)
    axis.grid(False)
    if y_grid:
        axis.grid(axis="y", color=style.grid_color, linewidth=style.grid_width)
    if x_grid:
        axis.grid(axis="x", color=style.light_grid_color, linewidth=style.grid_width)
    axis.set_axisbelow(True)
    if hide_top_right:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(style.spine_color)
    axis.spines["bottom"].set_color(style.spine_color)
    axis.spines["left"].set_linewidth(style.spine_width)
    axis.spines["bottom"].set_linewidth(style.spine_width)
    axis.tick_params(labelsize=style.tick_size, colors=style.text_color)
    if hide_y_tick_labels:
        axis.set_yticklabels([])
        axis.tick_params(axis="y", length=0)


def finish_figure(
    figure,
    title: str | None = None,
    *,
    top: float = 0.88,
    left: float = 0.10,
    right: float = 0.985,
    bottom: float = 0.13,
    hspace: float = 0.35,
    wspace: float = 0.25,
    style: ThesisFigureStyle = THESIS_STYLE,
) -> None:
    """Set figure background, spacing, and optional title."""
    figure.patch.set_facecolor(style.facecolor)
    figure.subplots_adjust(left=left, right=right, bottom=bottom, top=top, hspace=hspace, wspace=wspace)
    if title:
        figure.suptitle(title, fontsize=style.figure_title_size, y=0.975)


def save_single_figure(
    figure,
    output_path: Path,
    *,
    close: bool = True,
    dpi: int = THESIS_STYLE.dpi,
    pad_inches: float = 0.04,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, facecolor=THESIS_STYLE.facecolor, bbox_inches="tight", pad_inches=pad_inches)
    if close:
        close_figure(figure)
    return output_path


def save_multi_format_figure(
    figure,
    base_path: Path,
    *,
    formats: Iterable[str],
    close: bool = False,
    dpi: int = THESIS_STYLE.dpi,
    pad_inches: float = 0.04,
) -> list[Path]:
    saved = []
    base_path.parent.mkdir(parents=True, exist_ok=True)
    for file_format in formats:
        path = base_path.with_suffix(f".{file_format}")
        figure.savefig(path, dpi=dpi, facecolor=THESIS_STYLE.facecolor, bbox_inches="tight", pad_inches=pad_inches)
        saved.append(path)
    if close:
        close_figure(figure)
    return saved


def close_figure(figure) -> None:
    import matplotlib.pyplot as plt

    plt.close(figure)
