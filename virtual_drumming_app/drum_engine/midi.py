"""MIDI output for drum-engine hit events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .hand_hit_detection import HandHitEvent
from .pedal_hit_detection import HitEvent

MIDI_NOTE_MAP = {
    "kick": 36,
    "right_pedal": 36,
    "snare": 38,
    "floor_tom": 43,
    "pedal_hi_hat": 44,
    "left_pedal": 44,
    "tom_2": 45,
    "tom_1": 48,
    "crash": 49,
    "ride": 51,
    "hi_hat": 42,
}


@dataclass(frozen=True, slots=True)
class MidiOutputConfig:
    port_name: str | None = None
    channel: int = 9
    velocity_scale: float = 96.0
    min_velocity: int = 35
    max_velocity: int = 127


@dataclass(frozen=True, slots=True)
class MidiMessage:
    type: str
    note: int
    velocity: int
    channel: int


class MidiOutput:
    def __init__(self, config: MidiOutputConfig | None = None, port: Any | None = None) -> None:
        self.config = config or MidiOutputConfig()
        self._mido = None
        if port is None:
            self._mido = _load_mido()
            self.port = self._mido.open_output(self.config.port_name)
        else:
            self.port = port

    def send_hand_hit(self, event: HandHitEvent) -> bool:
        return self.send_drum_hit(event.drum, event.velocity)

    def send_pedal_hit(self, event: HitEvent) -> bool:
        return self.send_drum_hit(event.pedal_id, event.velocity)

    def send_drum_hit(self, drum: str, velocity: float) -> bool:
        note = MIDI_NOTE_MAP.get(drum)
        if note is None:
            return False
        midi_velocity = velocity_to_midi(
            velocity,
            scale=self.config.velocity_scale,
            minimum=self.config.min_velocity,
            maximum=self.config.max_velocity,
        )
        self.port.send(self._message("note_on", note=note, velocity=midi_velocity))
        self.port.send(self._message("note_off", note=note, velocity=0))
        return True

    def close(self) -> None:
        close = getattr(self.port, "close", None)
        if close is not None:
            close()

    def _message(self, message_type: str, *, note: int, velocity: int) -> Any:
        if self._mido is not None:
            return self._mido.Message(message_type, note=note, velocity=velocity, channel=self.config.channel)
        return MidiMessage(type=message_type, note=note, velocity=velocity, channel=self.config.channel)


def available_midi_outputs() -> list[str]:
    return list(_load_mido().get_output_names())


def velocity_to_midi(
    velocity: float,
    *,
    scale: float = 96.0,
    minimum: int = 35,
    maximum: int = 127,
) -> int:
    if velocity <= 0.0:
        return 0
    return max(minimum, min(maximum, round(abs(velocity) * scale)))


def _load_mido() -> Any:
    try:
        import mido
    except ImportError as exc:
        raise RuntimeError(
            "Missing MIDI dependencies. From the repository root, run:\n"
            "  python -m pip install -r requirements.txt"
        ) from exc
    return mido
