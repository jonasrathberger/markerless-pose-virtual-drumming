"""Shared thesis figure styling and generation helpers."""

from .style import (
    THESIS_STYLE,
    apply_thesis_style,
    close_figure,
    figure_size,
    finish_figure,
    save_multi_format_figure,
    save_single_figure,
    style_axis,
)
from .data import REPO_ROOT, THESIS_DATA_ROOT, THESIS_OUTPUT_ROOT, resolve_input_dir

__all__ = [
    "REPO_ROOT",
    "THESIS_STYLE",
    "THESIS_DATA_ROOT",
    "THESIS_OUTPUT_ROOT",
    "apply_thesis_style",
    "close_figure",
    "figure_size",
    "finish_figure",
    "save_multi_format_figure",
    "save_single_figure",
    "resolve_input_dir",
    "style_axis",
]
