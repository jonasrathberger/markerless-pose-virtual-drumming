"""Export thesis-ready evaluation metrics for the 100-static KNN model."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .evaluate_recording import (
    EvaluationEvent,
    MatchMode,
    build_report,
    load_midi_reference_events,
    replay_video_analysis,
)
from .target_classification import KnnDrumTargetClassifier

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VIDEO_PATH = REPO_ROOT / "thesis_data" / "virtual_drumming" / "evaluation_60bpm.mp4"
DEFAULT_MODEL_PATH = Path("knn_100.json")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "thesis_data" / "virtual_drumming"
DEFAULT_MIDI_PATHS = (
    Path("computer.mid"),
    Path("human.mid"),
)
DRUM_ORDER = (
    "hi_hat",
    "snare",
    "tom_1",
    "tom_2",
    "floor_tom",
    "crash",
    "ride",
    "right_pedal",
    "left_pedal",
)


@dataclass(frozen=True, slots=True)
class EventMatch:
    reference_index: int | None
    detected_index: int | None
    reference: EvaluationEvent | None
    detected: EvaluationEvent | None
    expected_video_time_seconds: float | None
    delta_seconds: float | None


@dataclass(frozen=True, slots=True)
class PairMatchResult:
    matches: list[EventMatch]
    false_negatives: list[EventMatch]
    false_positives: list[EventMatch]


def match_event_pairs(
    reference_events: list[EvaluationEvent],
    detected_events: list[EvaluationEvent],
    *,
    offset_seconds: float,
    scale: float,
    tolerance_seconds: float,
    mode: MatchMode,
) -> PairMatchResult:
    detected_by_key: dict[tuple[str, ...], list[tuple[float, int]]] = defaultdict(list)
    for index, event in enumerate(detected_events):
        detected_by_key[_match_key(event, mode)].append((event.time_seconds, index))
    for events in detected_by_key.values():
        events.sort()

    used_detected: set[int] = set()
    matches: list[EventMatch] = []
    false_negatives: list[EventMatch] = []
    for reference_index, reference in enumerate(reference_events):
        expected_time = offset_seconds + (reference.time_seconds * scale)
        candidates = detected_by_key.get(_match_key(reference, mode), [])
        candidate_index = bisect.bisect_left(candidates, (expected_time - tolerance_seconds, -1))
        best: tuple[float, int] | None = None
        while candidate_index < len(candidates) and candidates[candidate_index][0] <= expected_time + tolerance_seconds:
            detected_time, detected_index = candidates[candidate_index]
            if detected_index not in used_detected:
                delta = detected_time - expected_time
                if best is None or abs(delta) < abs(best[0]):
                    best = (delta, detected_index)
            candidate_index += 1
        if best is None:
            false_negatives.append(
                EventMatch(
                    reference_index=reference_index,
                    detected_index=None,
                    reference=reference,
                    detected=None,
                    expected_video_time_seconds=expected_time,
                    delta_seconds=None,
                )
            )
            continue
        delta, detected_index = best
        used_detected.add(detected_index)
        matches.append(
            EventMatch(
                reference_index=reference_index,
                detected_index=detected_index,
                reference=reference,
                detected=detected_events[detected_index],
                expected_video_time_seconds=expected_time,
                delta_seconds=delta,
            )
        )

    false_positives = [
        EventMatch(
            reference_index=None,
            detected_index=detected_index,
            reference=None,
            detected=event,
            expected_video_time_seconds=None,
            delta_seconds=None,
        )
        for detected_index, event in enumerate(detected_events)
        if detected_index not in used_detected
    ]
    return PairMatchResult(matches=matches, false_negatives=false_negatives, false_positives=false_positives)


def build_full_confusion_matrix(
    pair_result: PairMatchResult,
    *,
    drum_order: tuple[str, ...] = DRUM_ORDER,
) -> tuple[list[str], list[str], dict[str, dict[str, int]]]:
    rows = _ordered_labels(
        [match.reference.drum for match in pair_result.matches if match.reference is not None]
        + [match.reference.drum for match in pair_result.false_negatives if match.reference is not None],
        drum_order,
    ) + ["extra"]
    columns = _ordered_labels(
        [match.detected.drum for match in pair_result.matches if match.detected is not None]
        + [match.detected.drum for match in pair_result.false_positives if match.detected is not None],
        drum_order,
    ) + ["missed"]
    matrix = _empty_matrix(rows, columns)
    for match in pair_result.matches:
        assert match.reference is not None and match.detected is not None
        matrix[match.reference.drum][match.detected.drum] += 1
    for match in pair_result.false_negatives:
        assert match.reference is not None
        matrix[match.reference.drum]["missed"] += 1
    for match in pair_result.false_positives:
        assert match.detected is not None
        matrix["extra"][match.detected.drum] += 1
    return rows, columns, matrix


def build_classification_confusion_matrix(
    pair_result: PairMatchResult,
    *,
    drum_order: tuple[str, ...] = DRUM_ORDER,
) -> tuple[list[str], list[str], dict[str, dict[str, int]]]:
    labels = _ordered_labels(
        [match.reference.drum for match in pair_result.matches if match.reference is not None]
        + [match.detected.drum for match in pair_result.matches if match.detected is not None],
        drum_order,
    )
    matrix = _empty_matrix(labels, labels)
    for match in pair_result.matches:
        assert match.reference is not None and match.detected is not None
        matrix[match.reference.drum][match.detected.drum] += 1
    return labels, labels, matrix


def export_metrics(
    *,
    video_path: Path,
    model_path: Path,
    midi_paths: list[Path],
    output_dir: Path,
    json_report_path: Path | None,
    tolerance_seconds: float,
    progress_every: int,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    classifier = KnnDrumTargetClassifier.from_path(model_path)
    replay = replay_video_analysis(
        video_path,
        progress_every=progress_every,
        collect_signals=False,
        target_classifier=classifier,
    )

    summary_rows: list[dict[str, object]] = []
    json_report: dict[str, object] = {
        "video_path": str(video_path),
        "model_path": str(model_path),
        "tolerance_seconds": tolerance_seconds,
        "detected_events": len(replay.detected_events),
        "references": {},
    }

    for midi_path in midi_paths:
        reference_name = midi_path.stem
        safe_name = _safe_name(reference_name)
        reference_events = load_midi_reference_events(midi_path)
        report = build_report(
            reference_events,
            replay.detected_events,
            tolerance_seconds=tolerance_seconds,
        )
        reference_report = {
            "midi_path": str(midi_path),
            "report": report,
        }
        json_report["references"][reference_name] = reference_report

        for mode in ("type", "drum"):
            mode_report = report["modes"][mode]
            stats = mode_report["stats"]
            summary_rows.append(
                {
                    "reference_name": reference_name,
                    "midi_path": str(midi_path),
                    "model_path": str(model_path),
                    "match_mode": mode,
                    "reference_events": report["reference_events"],
                    "detected_events": report["detected_events"],
                    "tp": stats["tp"],
                    "fp": stats["fp"],
                    "fn": stats["fn"],
                    "precision": _format_float(stats["precision"]),
                    "recall": _format_float(stats["recall"]),
                    "f1": _format_float(stats["f1"]),
                    "offset_seconds": _format_float(mode_report["offset_seconds"]),
                    "scale": _format_float(mode_report["scale"]),
                    "tolerance_seconds": _format_float(tolerance_seconds),
                }
            )

        type_report = report["modes"]["type"]
        pairs = match_event_pairs(
            reference_events,
            replay.detected_events,
            offset_seconds=float(type_report["offset_seconds"]),
            scale=float(type_report["scale"]),
            tolerance_seconds=tolerance_seconds,
            mode="type",
        )
        rows, columns, matrix = build_full_confusion_matrix(pairs)
        write_matrix_csv(output_dir / f"confusion_full_{safe_name}.csv", rows, columns, matrix)
        rows, columns, matrix = build_classification_confusion_matrix(pairs)
        write_matrix_csv(output_dir / f"confusion_classification_{safe_name}.csv", rows, columns, matrix)
        write_matched_events_csv(output_dir / f"matched_events_{safe_name}.csv", pairs)

    write_summary_csv(output_dir / "summary_metrics.csv", summary_rows)
    if json_report_path is not None:
        json_report_path.parent.mkdir(parents=True, exist_ok=True)
        with json_report_path.open("w", encoding="utf-8") as file:
            json.dump(json_report, file, indent=2, sort_keys=True)
            file.write("\n")
    return json_report


def write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "reference_name",
        "midi_path",
        "model_path",
        "match_mode",
        "reference_events",
        "detected_events",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "offset_seconds",
        "scale",
        "tolerance_seconds",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_matrix_csv(
    path: Path,
    rows: list[str],
    columns: list[str],
    matrix: dict[str, dict[str, int]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["reference_drum", *columns])
        for row in rows:
            writer.writerow([row, *[matrix[row][column] for column in columns]])


def write_matched_events_csv(path: Path, pairs: PairMatchResult) -> None:
    fields = [
        "status",
        "reference_index",
        "detected_index",
        "midi_time_seconds",
        "expected_video_time_seconds",
        "detected_time_seconds",
        "delta_ms",
        "reference_type",
        "reference_drum",
        "reference_note",
        "detected_type",
        "detected_drum",
        "detected_side",
    ]
    keyed_rows = []
    for match in pairs.matches:
        keyed_rows.append((_matched_event_sort_key(match), _matched_event_row("matched", match)))
    for match in pairs.false_negatives:
        keyed_rows.append((_matched_event_sort_key(match), _matched_event_row("missed", match)))
    for match in pairs.false_positives:
        keyed_rows.append((_matched_event_sort_key(match), _matched_event_row("extra", match)))
    rows = [row for _key, row in sorted(keyed_rows, key=lambda item: item[0])]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _matched_event_sort_key(match: EventMatch) -> tuple[float, int]:
    if match.expected_video_time_seconds is not None:
        return (match.expected_video_time_seconds, 0)
    if match.detected is not None:
        return (match.detected.time_seconds, 1)
    return (0.0, 2)


def _matched_event_row(status: str, match: EventMatch) -> dict[str, object]:
    reference = match.reference
    detected = match.detected
    return {
        "status": status,
        "reference_index": "" if match.reference_index is None else match.reference_index,
        "detected_index": "" if match.detected_index is None else match.detected_index,
        "midi_time_seconds": "" if reference is None else _format_float(reference.time_seconds),
        "expected_video_time_seconds": (
            "" if match.expected_video_time_seconds is None else _format_float(match.expected_video_time_seconds)
        ),
        "detected_time_seconds": "" if detected is None else _format_float(detected.time_seconds),
        "delta_ms": "" if match.delta_seconds is None else _format_float(match.delta_seconds * 1000.0),
        "reference_type": "" if reference is None else reference.event_type,
        "reference_drum": "" if reference is None else reference.drum,
        "reference_note": "" if reference is None or reference.note is None else reference.note,
        "detected_type": "" if detected is None else detected.event_type,
        "detected_drum": "" if detected is None else detected.drum,
        "detected_side": "" if detected is None or detected.side is None else detected.side,
    }


def _ordered_labels(labels: list[str], preferred_order: tuple[str, ...]) -> list[str]:
    present = set(labels)
    ordered = [label for label in preferred_order if label in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def _empty_matrix(rows: list[str], columns: list[str]) -> dict[str, dict[str, int]]:
    return {row: {column: 0 for column in columns} for row in rows}


def _match_key(event: EvaluationEvent, mode: MatchMode) -> tuple[str, ...]:
    if mode == "type":
        return (event.event_type,)
    return (event.event_type, event.drum)


def _safe_name(name: str) -> str:
    return "".join(character if character.isalnum() or character in ("-", "_") else "_" for character in name)


def _format_float(value: float) -> str:
    return f"{value:.6f}"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export thesis evaluation metrics for the 100-static KNN model.")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO_PATH, help="Evaluation video path.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="KNN hand target model path.")
    parser.add_argument("--midi", type=Path, nargs="+", default=list(DEFAULT_MIDI_PATHS), help="Reference MIDI path(s).")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for CSV outputs.")
    parser.add_argument("--json-report", type=Path, default=None, help="Optional generated JSON debug report path.")
    parser.add_argument("--tolerance", type=float, default=0.120, help="Match tolerance in seconds.")
    parser.add_argument("--progress-every", type=int, default=300, help="Print replay progress every N frames; 0 disables.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = export_metrics(
        video_path=args.video,
        model_path=args.model,
        midi_paths=list(args.midi),
        output_dir=args.output_dir,
        json_report_path=args.json_report,
        tolerance_seconds=args.tolerance,
        progress_every=args.progress_every,
    )
    print(f"Wrote thesis metrics for {len(report['references'])} reference MIDI files to {args.output_dir}.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
