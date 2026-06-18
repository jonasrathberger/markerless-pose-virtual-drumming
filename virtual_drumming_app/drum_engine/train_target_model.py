"""Train a KNN hand-target model from collected target samples."""

from __future__ import annotations

import argparse

from .target_classification import (
    read_target_sample_file,
    train_knn_model_from_sample_file,
    write_knn_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a KNN hand-target model from collected samples.")
    parser.add_argument("samples_path", help="Input hand_target_samples_v1 JSON path.")
    parser.add_argument("model_path", help="Output KNN model JSON path.")
    parser.add_argument("--k", type=int, default=5, help="K nearest neighbors to use at runtime.")
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.45,
        help="Minimum runtime confidence required to emit a KNN prediction.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = read_target_sample_file(args.samples_path)
    model = train_knn_model_from_sample_file(
        data,
        k=max(1, args.k),
        min_confidence=args.min_confidence,
    )
    write_knn_model(args.model_path, model)
    print(f"Wrote KNN model with {len(model['samples'])} samples to {args.model_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
