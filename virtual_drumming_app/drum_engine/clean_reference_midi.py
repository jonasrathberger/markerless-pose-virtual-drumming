"""Clean human-recorded reference MIDI files for evaluator use."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

APP_EQUIVALENT_NOTE_REMAP = {
    40: 38,
    46: 42,
    55: 49,
}

APP_COMPATIBLE_NOTES = frozenset({36, 38, 42, 43, 44, 45, 48, 49, 51})


@dataclass(frozen=True, slots=True)
class MidiCleanStats:
    input_positive_note_ons: int
    output_positive_note_ons: int
    dropped_leading_note_ons: int
    dropped_trailing_note_ons: int
    dropped_duplicate_kicks: int
    remapped_note_ons: int
    unsupported_note_ons: int


@dataclass(frozen=True, slots=True)
class _PositiveNoteOn:
    absolute_tick: int
    track_index: int
    message_index: int
    note: int
    remapped_note: int
    velocity: int


def clean_reference_midi(
    input_path: Path,
    output_path: Path,
    *,
    drop_first_note_ons: int = 4,
    drop_last_note_ons: int = 2,
    duplicate_kick_ticks: int = 30,
    duplicate_kick_low_velocity: int = 45,
) -> MidiCleanStats:
    """Write a cleaned copy of a reference MIDI while preserving retained timings."""
    mido = _load_mido()
    midi = mido.MidiFile(input_path)
    positive_note_ons = _positive_note_ons(midi)
    drop_ids: set[tuple[int, int]] = set()

    leading = positive_note_ons[:drop_first_note_ons]
    trailing = positive_note_ons[len(positive_note_ons) - drop_last_note_ons :] if drop_last_note_ons else []
    drop_ids.update((event.track_index, event.message_index) for event in leading)
    drop_ids.update((event.track_index, event.message_index) for event in trailing)

    duplicate_ids = _duplicate_kick_note_on_ids(
        [event for event in positive_note_ons if (event.track_index, event.message_index) not in drop_ids],
        max_tick_delta=duplicate_kick_ticks,
        low_velocity_threshold=duplicate_kick_low_velocity,
    )
    drop_ids.update(duplicate_ids)

    output = mido.MidiFile(type=midi.type, ticks_per_beat=midi.ticks_per_beat, charset=midi.charset)
    remapped_note_ons = 0
    unsupported_note_ons = 0
    output_positive_note_ons = 0
    for track_index, track in enumerate(midi.tracks):
        output_track = mido.MidiTrack()
        output.tracks.append(output_track)
        pending_dropped_note_offs: dict[tuple[int | None, int], int] = {}
        absolute_tick = 0
        last_written_tick = 0
        for message_index, message in enumerate(track):
            absolute_tick += message.time
            if _is_positive_note_on(message):
                message_id = (track_index, message_index)
                if message_id in drop_ids:
                    key = (_message_channel(message), message.note)
                    pending_dropped_note_offs[key] = pending_dropped_note_offs.get(key, 0) + 1
                    continue
                remapped_note = _remap_note(message.note)
                if remapped_note != message.note:
                    remapped_note_ons += 1
                if remapped_note not in APP_COMPATIBLE_NOTES:
                    unsupported_note_ons += 1
                output_positive_note_ons += 1
                output_message = message.copy(note=remapped_note, time=absolute_tick - last_written_tick)
            elif _is_note_off(message):
                key = (_message_channel(message), message.note)
                if pending_dropped_note_offs.get(key, 0) > 0:
                    pending_dropped_note_offs[key] -= 1
                    continue
                remapped_note = _remap_note(message.note)
                output_message = message.copy(note=remapped_note, time=absolute_tick - last_written_tick)
            else:
                output_message = message.copy(time=absolute_tick - last_written_tick)
            output_track.append(output_message)
            last_written_tick = absolute_tick

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.save(output_path)
    return MidiCleanStats(
        input_positive_note_ons=len(positive_note_ons),
        output_positive_note_ons=output_positive_note_ons,
        dropped_leading_note_ons=len(leading),
        dropped_trailing_note_ons=len(trailing),
        dropped_duplicate_kicks=len(duplicate_ids),
        remapped_note_ons=remapped_note_ons,
        unsupported_note_ons=unsupported_note_ons,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    stats = clean_reference_midi(
        args.input,
        args.output,
        drop_first_note_ons=args.drop_first_note_ons,
        drop_last_note_ons=args.drop_last_note_ons,
        duplicate_kick_ticks=args.duplicate_kick_ticks,
        duplicate_kick_low_velocity=args.duplicate_kick_low_velocity,
    )
    print(f"Wrote {args.output}")
    print(f"input_positive_note_ons={stats.input_positive_note_ons}")
    print(f"output_positive_note_ons={stats.output_positive_note_ons}")
    print(f"dropped_leading_note_ons={stats.dropped_leading_note_ons}")
    print(f"dropped_trailing_note_ons={stats.dropped_trailing_note_ons}")
    print(f"dropped_duplicate_kicks={stats.dropped_duplicate_kicks}")
    print(f"remapped_note_ons={stats.remapped_note_ons}")
    print(f"unsupported_note_ons={stats.unsupported_note_ons}")
    return 0


def _positive_note_ons(midi: Any) -> list[_PositiveNoteOn]:
    events: list[_PositiveNoteOn] = []
    for track_index, track in enumerate(midi.tracks):
        absolute_tick = 0
        for message_index, message in enumerate(track):
            absolute_tick += message.time
            if not _is_positive_note_on(message):
                continue
            events.append(
                _PositiveNoteOn(
                    absolute_tick=absolute_tick,
                    track_index=track_index,
                    message_index=message_index,
                    note=message.note,
                    remapped_note=_remap_note(message.note),
                    velocity=message.velocity,
                )
            )
    return sorted(events, key=lambda event: (event.absolute_tick, event.track_index, event.message_index))


def _duplicate_kick_note_on_ids(
    events: list[_PositiveNoteOn],
    *,
    max_tick_delta: int,
    low_velocity_threshold: int,
) -> set[tuple[int, int]]:
    drop_ids: set[tuple[int, int]] = set()
    previous_kick: _PositiveNoteOn | None = None
    for event in events:
        if event.remapped_note != 36:
            continue
        if previous_kick is None or event.absolute_tick - previous_kick.absolute_tick > max_tick_delta:
            previous_kick = event
            continue
        weaker, stronger = _weaker_and_stronger(previous_kick, event)
        if weaker.velocity <= low_velocity_threshold:
            drop_ids.add((weaker.track_index, weaker.message_index))
            previous_kick = stronger
        else:
            previous_kick = event
    return drop_ids


def _weaker_and_stronger(
    first: _PositiveNoteOn,
    second: _PositiveNoteOn,
) -> tuple[_PositiveNoteOn, _PositiveNoteOn]:
    if first.velocity < second.velocity:
        return first, second
    return second, first


def _remap_note(note: int) -> int:
    return APP_EQUIVALENT_NOTE_REMAP.get(note, note)


def _is_positive_note_on(message: Any) -> bool:
    return message.type == "note_on" and message.velocity > 0


def _is_note_off(message: Any) -> bool:
    return message.type == "note_off" or (message.type == "note_on" and message.velocity == 0)


def _message_channel(message: Any) -> int | None:
    return getattr(message, "channel", None)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean a human-recorded reference MIDI file.")
    parser.add_argument("input", type=Path, help="Input MIDI file.")
    parser.add_argument("output", type=Path, help="Output cleaned MIDI file.")
    parser.add_argument("--drop-first-note-ons", type=int, default=4)
    parser.add_argument("--drop-last-note-ons", type=int, default=2)
    parser.add_argument("--duplicate-kick-ticks", type=int, default=30)
    parser.add_argument("--duplicate-kick-low-velocity", type=int, default=45)
    return parser.parse_args(argv)


def _load_mido():
    try:
        import mido
    except ImportError as exc:
        raise RuntimeError(
            "Missing MIDI dependencies. From the repository root, run:\n"
            "  python -m pip install -r requirements.txt"
        ) from exc
    return mido


if __name__ == "__main__":
    raise SystemExit(main())
