"""Event construction and the stdout JSON-lines writer.

This is the contract boundary. Everything upstream of here deals in pixels,
boxes and tracks; everything downstream sees only events matching
schema/event.schema.json. Nothing about the model, the camera or the polygon
crosses it.

Two rules this file enforces:

stdout carries events and nothing else. Every diagnostic, warning and licence
banner goes to stderr, so `python perception/perceive.py | python consumer.py`
works with no filtering. A stray print() to stdout corrupts the stream.

timestamp_us is monotonic, never wall clock. The schema is explicit that NTP
corrections must not be able to reorder or duplicate events, and every duration
rule in the engine depends on it. wall_time is carried alongside for humans and
for SOC correlation only.

Both clocks exist because neither one answers both questions. A monotonic
microsecond count orders events correctly across an NTP step but says nothing
about when anything happened; a wall clock reads as a date but can jump
backwards mid-run. So the monotonic value is the one every rule keys on, and
the wall clock is a derived convenience that is never used for sequencing.

`wall_time_at` is how a monotonic reading becomes a wall clock one. It converts
through a single offset captured once at import, NOT by reading the clock at
call time -- see the note on `_WALL_OFFSET_US` for why that distinction decides
whether a past span gets the right date.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import uuid
from datetime import datetime, timezone

# Sources the schema defines as carrying no perception uncertainty. Events from
# these are facts, and are emitted with confidence exactly 1.0.
CERTAIN_SOURCES = {"robot_bus", "machine_bus", "access_control", "timer", "engine"}


def monotonic_us() -> int:
    """Microseconds from the monotonic device clock."""
    return time.monotonic_ns() // 1000


def _iso(dt: datetime) -> str:
    """The one wall-clock format this layer writes. Millisecond precision."""
    return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def wall_time_now() -> str:
    return _iso(datetime.now(timezone.utc))


def _capture_wall_offset() -> int:
    """Wall-clock microseconds minus monotonic microseconds, read back to back.

    Captured ONCE, at import, and every later conversion derives from it.

    The alternative -- calling the clock when a record is written -- is wrong
    for anything that describes the past, and an action interval always does:
    it is written when it CLOSES, which is `close_after` windows after the
    action itself ended. Reading the wall clock there would date a span that
    ran at 14:02:10 to 14:02:14 instead, and the error grows with every
    configured delay in the pipeline. Converting through a fixed offset keeps
    start and end at the moments the recogniser actually observed them.

    Taking it once is also what makes the offset honest about its own job: it
    is a label for monotonic readings, so it must not move underneath them. A
    re-read would fold an NTP correction into the conversion and let two
    records written seconds apart disagree about when the same microsecond was,
    which is the exact reordering `timestamp_us` exists to prevent.
    """
    mono = monotonic_us()
    wall = int(time.time() * 1_000_000)
    return wall - mono


# Read at import, i.e. at process start, before any event or log record exists.
_WALL_OFFSET_US = _capture_wall_offset()


def wall_offset_us() -> int:
    """The offset in use, for tests and diagnostics."""
    return _WALL_OFFSET_US


def wall_time_at(timestamp_us: int) -> str:
    """ISO wall clock for a MONOTONIC microsecond reading.

    Same format as `wall_time_now`, and deliberately the same precision: a
    millisecond is already finer than any span this is used on is accurate to.
    Two readings converted this way differ by exactly their monotonic
    difference, to the millisecond, because they share one offset -- so a
    derived duration still matches the monotonic fields it came from.
    """
    seconds = (int(timestamp_us) + _WALL_OFFSET_US) / 1_000_000
    return _iso(datetime.fromtimestamp(seconds, timezone.utc))


# What `subject` says when nothing identified the person. Not a placeholder for
# a name: it is the complete, correct answer for an anonymous detection, and it
# is what every event carried before identity existed.
ANONYMOUS_SUBJECT = {"class": "human"}


def subject_of(subject_for, track_id: str | None) -> dict:
    """Resolve the `subject` block for a track through an optional resolver.

    `subject_for` is a callable taking a track_id and returning a subject dict
    -- in production `IdentityStage.subject_for`, in tests a stub. It is passed
    in rather than imported so that the stages which emit these events
    (`actions_stage.py`, `ppe_stage.py`, `action_intervals.py`) stay free of
    any dependency on the identity layer, and testable without one.

    `None` means no identity stage is configured, which is the default and
    stays the default: naming people is a heavier proposition, legally and
    operationally, than counting them. A track with no resolver, and a
    track_id of None (an unattributed PPE box), both get the anonymous subject.

    A fresh dict every time. Callers put this straight into an event and
    `emit.build` is free to filter it, so a shared mutable default would let
    one event's subject leak into the next one's.
    """
    if subject_for is None or track_id is None:
        return dict(ANONYMOUS_SUBJECT)
    return subject_for(track_id) or dict(ANONYMOUS_SUBJECT)


class EventEmitter:
    """Builds schema-valid events and writes them as JSON lines.

    `integrity` is off by default. The chain implemented here is the seq and
    prev_hash half of what the schema specifies; the TPM-keyed signature is
    applied at log-write time and is not this process's job.
    """

    def __init__(self, sensor_id: str, stream=None, integrity: bool = False) -> None:
        self.sensor_id = sensor_id
        self.stream = stream if stream is not None else sys.stdout
        self.integrity = integrity
        self.seq = 0
        self.prev_hash = "0" * 64
        self.count = 0

    def build(
        self,
        observation: str,
        confidence: float,
        source: str = "camera",
        zone_id: str | None = None,
        track_id: str | None = None,
        value=None,
        unit: str | None = None,
        subject: dict | None = None,
        confidence_components: dict | None = None,
        sensor_id: str | None = None,
        timestamp_us: int | None = None,
    ) -> dict:
        if source in CERTAIN_SOURCES:
            # Not a correction of a nearly-right number: a bus read, a badge or
            # a timer has no perception uncertainty by construction, and the
            # engine relies on that being unconditional.
            confidence = 1.0

        event = {
            "event_id": str(uuid.uuid4()),
            "timestamp_us": timestamp_us if timestamp_us is not None else monotonic_us(),
            "wall_time": wall_time_now(),
            "source": source,
            "sensor_id": sensor_id or self.sensor_id,
            "observation": observation,
            "confidence": round(float(confidence), 4),
        }
        if track_id is not None:
            event["track_id"] = track_id
        if zone_id is not None:
            event["zone_id"] = zone_id
        if value is not None:
            event["value"] = value
        if unit is not None:
            event["unit"] = unit
        if subject:
            event["subject"] = {k: v for k, v in subject.items() if v is not None}
        if confidence_components is not None:
            event["confidence_components"] = confidence_components

        if self.integrity:
            event["integrity"] = {"seq": self.seq, "prev_hash": self.prev_hash}
            self.seq += 1
            self.prev_hash = hashlib.sha256(
                json.dumps(event, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()

        return event

    def emit(self, event: dict) -> dict:
        self.stream.write(json.dumps(event, separators=(",", ":")) + "\n")
        self.stream.flush()  # a downstream engine is reading this live
        self.count += 1
        return event


def log(message: str) -> None:
    """Diagnostics go to stderr. Never stdout: that channel is the contract."""
    print(message, file=sys.stderr, flush=True)
