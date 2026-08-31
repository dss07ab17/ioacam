"""A minimal greedy IoU tracker, enough to mint a stable track_id.

Deliberately not ByteTrack. The schema needs a track_id that is stable while
one subject stays continuously visible to one sensor, and the confidence model
needs frame counts per subject; both are satisfied by greedy IoU association on
a single fixed camera. ByteTrack (MIT, so no licence problem) is the intended
replacement when occlusion and crossing subjects start costing identity swaps.

The failure mode to know about: two people crossing will swap ids here. That
matters for per-actor workflow attribution, which is why the README defers
concurrent instances until a correlation key exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from detectors.base import Detection


def iou(a: Detection, b: Detection) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    track_id: str
    detection: Detection
    age: int = 0            # frames since first seen
    hits: int = 1           # frames in which it was matched
    misses: int = 0         # consecutive frames unmatched
    seen: list[bool] = field(default_factory=lambda: [True])

    def persistence(self, window: int = 30) -> float:
        """Fraction of the recent window in which this track was detected.

        This is the schema's `persistence` component, and it deliberately
        measures DETECTION stability rather than how long the subject has been
        in a zone. Measuring zone-state age instead scores every entry event
        near zero, because at the instant of entry the subject has by
        definition only just arrived -- which crushes the confidence of the
        single most operationally important event the layer emits.

        A subject detected in 29 of the last 30 frames is solid whether they
        entered a second ago or a minute ago. One detected in 12 of 30 is
        flickering, and that is what should lower confidence.
        """
        recent = self.seen[-window:]
        return sum(recent) / len(recent) if recent else 0.0


class IouTracker:
    def __init__(self, iou_threshold: float = 0.30, max_misses: int = 15,
                 prefix: str = "trk", history_window: int = 30) -> None:
        self.iou_threshold = iou_threshold
        self.max_misses = max_misses
        self.prefix = prefix
        self.history_window = history_window
        self.tracks: list[Track] = []
        self._next_id = 1

    def update(self, detections: Sequence[Detection]) -> list[Track]:
        """Associate, then age out. Returns the tracks matched this frame."""
        unmatched = list(detections)
        matched: list[Track] = []

        # Highest-IoU pairs first, so a confident overlap is never stolen by a
        # marginal one that happened to be considered earlier.
        pairs = sorted(
            (
                (iou(t.detection, d), ti, di)
                for ti, t in enumerate(self.tracks)
                for di, d in enumerate(detections)
            ),
            reverse=True,
        )
        used_tracks: set[int] = set()
        used_dets: set[int] = set()
        for score, ti, di in pairs:
            if score < self.iou_threshold or ti in used_tracks or di in used_dets:
                continue
            used_tracks.add(ti)
            used_dets.add(di)
            track = self.tracks[ti]
            track.detection = detections[di]
            track.hits += 1
            track.misses = 0
            track.seen.append(True)
            matched.append(track)

        for ti, track in enumerate(self.tracks):
            track.age += 1
            if ti not in used_tracks:
                track.misses += 1
                track.seen.append(False)
            del track.seen[:-self.history_window]

        for di, det in enumerate(detections):
            if di in used_dets:
                continue
            track = Track(track_id=f"{self.prefix}-{self._next_id:04d}", detection=det)
            self._next_id += 1
            self.tracks.append(track)
            matched.append(track)

        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]
        return matched

    def lost_tracks(self) -> list[Track]:
        """Tracks currently unmatched but not yet aged out.

        A subject who walks out of frame while inside a zone never produces a
        'left' transition from geometry, because there is no detection to test.
        The caller uses this list to close those zones out explicitly, so the
        engine never sees a person who entered and never left.
        """
        return [t for t in self.tracks if t.misses > 0]
