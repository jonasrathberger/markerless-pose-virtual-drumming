"""Analyze target-classification features with a Random Forest."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from .target_classification import read_target_sample_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze drum-target features with a Random Forest classifier.")
    parser.add_argument("samples_path", help="Input target sample JSON path.")
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Held-out test fraction used for metrics.",
    )
    parser.add_argument("--trees", type=int, default=500, help="Number of Random Forest trees.")
    parser.add_argument("--max-depth", type=int, default=0, help="Maximum tree depth. Use 0 for unlimited.")
    parser.add_argument("--top", type=int, default=25, help="Number of top features to print.")
    parser.add_argument("--random-state", type=int, default=13, help="Random seed.")
    parser.add_argument(
        "--label-mode",
        choices=("side-drum", "drum"),
        default="side-drum",
        help="Predict side:drum labels or drum labels only.",
    )
    parser.add_argument("--output-json", default=None, metavar="PATH", help="Optional machine-readable report path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = analyze_target_features(
            args.samples_path,
            test_size=args.test_size,
            trees=max(1, args.trees),
            max_depth=args.max_depth if args.max_depth > 0 else None,
            top=max(1, args.top),
            random_state=args.random_state,
            label_mode=args.label_mode,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(format_report(report, top=max(1, args.top)))
    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(report, file, indent=2, sort_keys=True)
            file.write("\n")
    return 0


def analyze_target_features(
    samples_path: str | Path,
    *,
    test_size: float,
    trees: int,
    max_depth: int | None,
    top: int,
    random_state: int,
    label_mode: str,
) -> dict:
    sklearn = _load_sklearn()
    data = read_target_sample_file(samples_path)
    feature_names = list(data["feature_names"])
    samples = list(data["samples"])
    if len(samples) < 2:
        raise ValueError("Need at least two samples for Random Forest analysis.")

    labels = [_sample_label(sample, label_mode=label_mode) for sample in samples]
    if len(set(labels)) < 2:
        raise ValueError("Need at least two target classes for Random Forest analysis.")
    x = [list(sample["features"]) for sample in samples]

    stratify = labels if _can_stratify(labels, test_size=test_size) else None
    x_train, x_test, y_train, y_test = sklearn["train_test_split"](
        x,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    classifier = sklearn["RandomForestClassifier"](
        n_estimators=trees,
        max_depth=max_depth,
        random_state=random_state,
        class_weight="balanced",
    )
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    labels_sorted = sorted(set(labels))
    matrix = sklearn["confusion_matrix"](y_test, y_pred, labels=labels_sorted)
    importances = feature_importance_rows(feature_names, classifier.feature_importances_)

    report = {
        "samples_path": str(samples_path),
        "feature_schema": data["feature_schema"],
        "label_mode": label_mode,
        "sample_count": len(samples),
        "train_count": len(x_train),
        "test_count": len(x_test),
        "class_count": len(labels_sorted),
        "random_forest": {
            "trees": trees,
            "max_depth": max_depth,
            "random_state": random_state,
        },
        "metrics": {
            "accuracy": float(sklearn["accuracy_score"](y_test, y_pred)),
            "macro_f1": float(sklearn["f1_score"](y_test, y_pred, average="macro", zero_division=0)),
            "weighted_f1": float(sklearn["f1_score"](y_test, y_pred, average="weighted", zero_division=0)),
        },
        "classes": labels_sorted,
        "confusion_matrix": matrix.tolist(),
        "feature_importances": importances,
        "top_features": importances[:top],
        "group_importances": group_importance_rows(importances),
    }
    return report


def feature_importance_rows(feature_names: list[str], importances: Any) -> list[dict]:
    rows = [
        {
            "feature": name,
            "group": feature_group(name),
            "importance": float(importance),
        }
        for name, importance in zip(feature_names, importances)
    ]
    return sorted(rows, key=lambda row: row["importance"], reverse=True)


def group_importance_rows(feature_rows: list[dict]) -> list[dict]:
    totals: dict[str, float] = defaultdict(float)
    for row in feature_rows:
        totals[str(row["group"])] += float(row["importance"])
    return [
        {"group": group, "importance": importance}
        for group, importance in sorted(totals.items(), key=lambda item: item[1], reverse=True)
    ]


def feature_group(feature_name: str) -> str:
    if feature_name.startswith(("active_wrist_delta_", "mcp_delta_", "strike_motion_delta_")):
        return "temporal_deltas"
    if feature_name.startswith("pre_hit_direction_angle_"):
        return "temporal_direction"
    if feature_name.endswith("_range_8"):
        return "temporal_ranges"
    if feature_name in {"active_wrist_x", "active_wrist_y", "mcp_x", "mcp_y"}:
        return "static_position"
    if feature_name.startswith("active_wrist_"):
        return "static_position"
    if feature_name.startswith(("mcp_span_", "wrist_to_mcp_", "wrist_hand_angle")):
        return "hand_shape"
    if feature_name.startswith(("elbow_to_wrist_", "shoulder_to_elbow_", "elbow_bend_angle")):
        return "arm_geometry"
    if feature_name in {"hand_motion_y", "forearm_motion_y", "strike_motion_y"}:
        return "static_motion"
    return "other"


def format_report(report: dict, *, top: int) -> str:
    lines = [
        "Random Forest Feature Analysis",
        f"samples: {report['sample_count']} | train: {report['train_count']} | test: {report['test_count']}",
        f"schema: {report['feature_schema']} | labels: {report['label_mode']} | classes: {report['class_count']}",
        (
            "metrics: "
            f"accuracy={report['metrics']['accuracy']:.4f} "
            f"macro_f1={report['metrics']['macro_f1']:.4f} "
            f"weighted_f1={report['metrics']['weighted_f1']:.4f}"
        ),
        "",
        "Top feature groups:",
    ]
    for row in report["group_importances"]:
        lines.append(f"  {row['importance']:.5f}  {row['group']}")

    lines.extend(["", f"Top {top} features:"])
    for row in report["top_features"][:top]:
        lines.append(f"  {row['importance']:.5f}  {row['feature']}  [{row['group']}]")

    lines.extend(["", "Confusion matrix:", "  labels: " + ", ".join(report["classes"])])
    for label, values in zip(report["classes"], report["confusion_matrix"]):
        formatted = " ".join(str(value).rjust(4) for value in values)
        lines.append(f"  {label.rjust(16)} {formatted}")
    return "\n".join(lines)


def _sample_label(sample: dict, *, label_mode: str) -> str:
    if label_mode == "drum":
        return str(sample["drum"])
    return f"{sample['side']}:{sample['drum']}"


def _can_stratify(labels: list[str], *, test_size: float) -> bool:
    counts: dict[str, int] = defaultdict(int)
    for label in labels:
        counts[label] += 1
    class_count = len(counts)
    test_count = round(len(labels) * test_size)
    train_count = len(labels) - test_count
    return min(counts.values()) >= 2 and test_count >= class_count and train_count >= class_count


def _load_sklearn() -> dict[str, Any]:
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
        from sklearn.model_selection import train_test_split
    except ImportError as exc:
        raise RuntimeError(
            "Missing scikit-learn. From the repository root, run:\n"
            "  python -m pip install -r requirements.txt"
        ) from exc
    return {
        "RandomForestClassifier": RandomForestClassifier,
        "accuracy_score": accuracy_score,
        "confusion_matrix": confusion_matrix,
        "f1_score": f1_score,
        "train_test_split": train_test_split,
    }


if __name__ == "__main__":
    raise SystemExit(main())
