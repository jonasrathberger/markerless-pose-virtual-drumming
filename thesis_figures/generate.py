"""Unified entrypoint for regenerating thesis figures."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

from thesis_figures.data import REPO_ROOT, THESIS_DATA_ROOT, THESIS_OUTPUT_ROOT, resolve_input_dir

VIRTUAL_APP_DIR = REPO_ROOT / "virtual_drumming_app"
MOTION_EVALUATION_DIR = REPO_ROOT / "motion_evaluation"
POSE_EVALUATION_DIR = MOTION_EVALUATION_DIR / "pose_evaluation"
MOTION_ANALYSIS_DIR = MOTION_EVALUATION_DIR / "motion_analysis"
DEFAULT_OUTPUT_ROOT = THESIS_OUTPUT_ROOT
DEFAULT_INPUT_ROOT = THESIS_DATA_ROOT
SECTION_DEPENDENCIES = {
    "virtual-drumming": ("matplotlib", "numpy"),
    "pose-evaluation": ("matplotlib", "numpy", "pandas"),
    "motion-analysis": ("matplotlib", "numpy", "pandas"),
    "motion-evaluation": ("matplotlib", "numpy", "pandas"),
}


def _ensure_import_paths() -> None:
    for path in (REPO_ROOT, VIRTUAL_APP_DIR, MOTION_EVALUATION_DIR, POSE_EVALUATION_DIR, MOTION_ANALYSIS_DIR / "scripts"):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def _check_dependencies(sections: tuple[str, ...]) -> None:
    missing = sorted(
        {
            package
            for section in sections
            for package in SECTION_DEPENDENCIES[section]
            if importlib.util.find_spec(package) is None
        }
    )
    if not missing:
        return

    package_list = " ".join(missing)
    raise SystemExit(
        "Missing Python package(s) required for thesis figure generation: "
        f"{', '.join(missing)}\n\n"
        "Install them in the active environment, for example:\n"
        f"  python -m pip install {package_list}\n\n"
        "The virtual drumming requirements also include Matplotlib:\n"
        "  python -m pip install -r virtual_drumming_app/requirements.txt"
    )


def _input_dir(input_root: Path, relative_dir: str, fallback: Path, *, required: tuple[str, ...] = ()) -> Path:
    return resolve_input_dir(relative_dir, fallback, required=required, input_root=input_root)


def generate_virtual_drumming_figures(input_root: Path, output_root: Path, formats: tuple[str, ...]) -> list[Path]:
    _ensure_import_paths()
    from drum_engine.export_thesis_graphics import export_graphics

    return export_graphics(
        metrics_dir=_input_dir(
            input_root,
            "virtual_drumming",
            input_root / "virtual_drumming",
            required=("summary_metrics.csv",),
        ),
        output_dir=output_root / "virtual_drumming",
        formats=formats,
    )


def generate_pose_evaluation_figures(input_root: Path, output_root: Path) -> list[Path]:
    _ensure_import_paths()
    from pose_evaluation.data_alignment.generate_thesis_alignment_figures import (
        generate_figures as generate_alignment_figures,
    )
    from pose_evaluation.data_analysis.generate_thesis_result_figures import generate_figures

    generated = generate_alignment_figures(
        prepared_dir=_input_dir(
            input_root,
            "pose_evaluation/prepared_result",
            POSE_EVALUATION_DIR / "data_preparation" / "result",
            required=(
                "mocap/air_knees/knee_default_left.csv",
                "mocap/drums/knee_default_left.csv",
                "pose/apple_vision/knee_default_left.csv",
                "pose/mediapipe/knee_default_left.csv",
            ),
        ),
        aligned_dir=_input_dir(
            input_root,
            "pose_evaluation/aligned_result",
            POSE_EVALUATION_DIR / "data_alignment" / "aligned_result",
            required=(
                "mocap/air_knees/wrist_center_right.csv",
                "mocap/drums/wrist_center_right.csv",
                "pose/apple_vision/wrist_body_right.csv",
                "pose/mediapipe/wrist_body_right.csv",
            ),
        ),
        output_dir=output_root / "pose_evaluation",
    )
    generated.extend(generate_figures(
        input_dir=_input_dir(
            input_root,
            "pose_evaluation",
            POSE_EVALUATION_DIR / "data_analysis" / "output",
            required=("summary_by_domain.csv", "summary_by_module.csv", "bland_altman.csv"),
        ),
        output_dir=output_root / "pose_evaluation",
        alignment_result_dir=_input_dir(
            input_root,
            "pose_evaluation/aligned_result",
            POSE_EVALUATION_DIR / "data_alignment" / "aligned_result",
            required=(
                "mocap/air_knees/wrist_center_right.csv",
                "mocap/drums/wrist_center_right.csv",
                "pose/apple_vision/wrist_body_right.csv",
                "pose/mediapipe/wrist_body_right.csv",
            ),
        ),
    ))
    return generated


def generate_motion_analysis_figures(input_root: Path, output_root: Path) -> list[Path]:
    _ensure_import_paths()
    from motion_analysis.generate_results_section_figures import (
        DEFAULT_OVERLAY_DIR,
        DEFAULT_RESULTS_DIR,
        generate_results_section_figures,
    )

    return generate_results_section_figures(
        results_dir=_input_dir(
            input_root,
            "motion_analysis",
            DEFAULT_RESULTS_DIR,
            required=(
                "proxy_dynamic_metrics.csv",
                "knee_condition_vs_foot_condition_segments.csv",
                "knee_condition_vs_foot_condition_tests.csv",
                "knee_condition_vs_foot_condition_summary.csv",
            ),
        ),
        output_dir=output_root / "motion_analysis",
        overlay_dir=_input_dir(
            input_root,
            "motion_analysis/proxy_overlays",
            DEFAULT_OVERLAY_DIR,
            required=(
                "proxy_dynamic_overlay_top_drums_L.png",
                "proxy_dynamic_overlay_top_drums_R.png",
            ),
        ),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate thesis figures with the shared style.")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Root directory containing thesis figure input CSVs and auxiliary files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for generated thesis figures.",
    )
    parser.add_argument(
        "--sections",
        nargs="+",
        choices=("virtual-drumming", "pose-evaluation", "motion-analysis", "motion-evaluation"),
        default=("virtual-drumming", "pose-evaluation", "motion-analysis"),
        help="Figure sections to regenerate. motion-evaluation expands to pose-evaluation and motion-analysis.",
    )
    parser.add_argument(
        "--virtual-formats",
        nargs="+",
        default=("png",),
        choices=("png", "pdf", "svg"),
        help="Formats for virtual-drumming figures.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sections = tuple(args.sections)
    if "motion-evaluation" in sections:
        sections = tuple(
            section
            for section in ("pose-evaluation", "motion-analysis", *sections)
            if section != "motion-evaluation"
        )
        sections = tuple(dict.fromkeys(sections))
    _check_dependencies(sections)
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    generated: list[Path] = []
    if "virtual-drumming" in sections:
        generated.extend(generate_virtual_drumming_figures(input_root, output_root, tuple(args.virtual_formats)))
    if "pose-evaluation" in sections:
        generated.extend(generate_pose_evaluation_figures(input_root, output_root))
    if "motion-analysis" in sections:
        generated.extend(generate_motion_analysis_figures(input_root, output_root))

    print(f"Generated {len(generated)} thesis figure files under {output_root}.")
    for path in generated:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
