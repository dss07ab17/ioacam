"""Object tracks, wrist attribution, and object_at_station events.

Shared by `perceive.py` (production stdout) and `preview_pose.py` (demo log).
Association lives in `association.py`; this module owns tracking, event
construction, and the edge-triggered emission policy for the engine feed.
"""

from __future__ import annotations

from typing import Optional, Sequence

from association import HeldObject, associate
from emit import EventEmitter
from tracking import MultiClassTracker, Track


def build_object_at_station_event(
    emitter: EventEmitter,
    held: HeldObject,
    persistence: float,
    timestamp_us: Optional[int] = None,
) -> dict:
    """Schema-shaped object_at_station event. Same builder for both tools.

    Confidence is raw detector score times track persistence — not the
    calibrated person composition in confidence.py. Quality is person-box
    height and no temperature has been fitted for objects.
    """
    return emitter.build(
        observation="object_at_station",
        confidence=round(held.score * persistence, 4),
        value=held.label,
        # Attributed to the person holding it, or omitted when unheld.
        # Dropping unattributed objects would make "no object" and
        # "unattributed object" the same stream.
        track_id=held.held_by,
        subject={"class": "object"},
        timestamp_us=timestamp_us,
    )


class ObjectStage:
    """Track non-person detections and attribute them to wrists.

    `mode="edge"` (production): emit when a track first appears or its
    attribution changes. `mode="all"` (demo log): one event per held object
    per call, matching the previous preview_pose logging cadence.
    """

    def __init__(
        self,
        object_classes: Sequence[str],
        margin: float = 0.25,
        min_wrist_score: float = 0.30,
        persistence_window: int = 30,
        mode: str = "edge",
    ) -> None:
        if not object_classes:
            raise ValueError(
                "no object classes configured. Add a 'classes' list to the "
                "detector block — the list is a site decision, not a constant "
                "in the source."
            )
        if mode not in ("edge", "all"):
            raise ValueError(f"unknown object emission mode {mode!r}")
        self.object_classes = list(object_classes)
        self.margin = float(margin)
        self.min_wrist_score = float(min_wrist_score)
        self.persistence_window = int(persistence_window)
        self.mode = mode
        self.tracker = MultiClassTracker()
        # object_track_id -> last emitted held_by (None means unattributed)
        self._announced: dict[str, Optional[str]] = {}

    def process(
        self,
        object_detections: Sequence,
        poses: dict,
        emitter: EventEmitter,
        timestamp_us: Optional[int] = None,
    ) -> tuple[list[HeldObject], list[dict]]:
        """Update tracks, associate wrists, return (held, events)."""
        object_tracks = self.tracker.update(object_detections)
        held_objects = associate(
            [(t.track_id, t.detection) for t in object_tracks],
            poses,
            margin=self.margin,
            min_wrist_score=self.min_wrist_score,
        )
        by_id = {t.track_id: t for t in object_tracks}
        events = self._events_for(held_objects, by_id, emitter, timestamp_us)

        live = {h.object_track_id for h in held_objects}
        for gone in list(self._announced):
            if gone not in live:
                del self._announced[gone]

        return held_objects, events

    def _events_for(
        self,
        held_objects: list[HeldObject],
        by_id: dict[str, Track],
        emitter: EventEmitter,
        timestamp_us: Optional[int],
    ) -> list[dict]:
        events: list[dict] = []
        for held in held_objects:
            track = by_id[held.object_track_id]
            persistence = track.persistence(self.persistence_window)
            if self.mode == "all":
                events.append(
                    build_object_at_station_event(
                        emitter, held, persistence, timestamp_us
                    )
                )
                continue

            prev = self._announced.get(held.object_track_id, _MISSING)
            if prev is _MISSING or prev != held.held_by:
                events.append(
                    build_object_at_station_event(
                        emitter, held, persistence, timestamp_us
                    )
                )
                self._announced[held.object_track_id] = held.held_by
        return events


_MISSING = object()
