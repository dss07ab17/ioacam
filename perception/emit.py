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


def wall_time_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


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
