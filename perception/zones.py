"""Zone polygons and membership, with hysteresis.

Two decisions here are the difference between a usable event stream and an
unusable one.

Polygons are stored NORMALISED (0..1) by default. A laptop webcam that
negotiates 1280x720 today and 640x480 tomorrow would silently move every zone
boundary if the polygon were in pixels. Normalised coordinates survive that,
and survive swapping the camera for the board's sensor later.

Membership is hysteretic. A raw per-frame point-in-polygon test on a subject
standing on the boundary flips state at the detector's noise frequency, which
would emit dozens of enter/leave pairs per second. The engine has no way to
tell those from real traffic, so the debounce belongs here, at the source.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Zone:
    zone_id: str
    polygon: list[tuple[float, float]]
    normalised: bool = True

    def contains(self, point: tuple[float, float], frame_w: int, frame_h: int) -> bool:
        """Ray-casting point-in-polygon, evaluated in pixel space."""
        px, py = point
        if self.normalised:
            px, py = px / max(1, frame_w), py / max(1, frame_h)
        inside = False
        n = len(self.polygon)
        for i in range(n):
            x1, y1 = self.polygon[i]
            x2, y2 = self.polygon[(i + 1) % n]
            if (y1 > py) != (y2 > py):
                x_cross = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
                if px < x_cross:
                    inside = not inside
        return inside

    def pixel_polygon(self, frame_w: int, frame_h: int) -> list[tuple[int, int]]:
        if not self.normalised:
            return [(int(x), int(y)) for x, y in self.polygon]
        return [(int(x * frame_w), int(y * frame_h)) for x, y in self.polygon]


def load_zones(zone_cfgs: list[dict]) -> list[Zone]:
    zones = []
    for z in zone_cfgs:
        polygon = [(float(p[0]), float(p[1])) for p in z["polygon"]]
        if len(polygon) < 3:
            raise ValueError(f"Zone {z['zone_id']!r} needs at least 3 points, got {len(polygon)}")
        normalised = z.get("coordinates", "normalized") in ("normalized", "normalised")
        if normalised and any(not (0.0 <= c <= 1.0) for pt in polygon for c in pt):
            raise ValueError(
                f"Zone {z['zone_id']!r} is declared normalised but has a coordinate "
                'outside 0..1. Set "coordinates": "pixel" if that was intended.'
            )
        zones.append(Zone(zone_id=z["zone_id"], polygon=polygon, normalised=normalised))
    if not zones:
        raise ValueError("No zones declared. Perception with no zone emits nothing useful.")
    return zones


@dataclass
class Membership:
    """Debounced in/out state for one (track, zone) pair.

    `announced` records whether the entry was actually EMITTED, which is not
    the same as whether it happened. An entry suppressed by the min_confidence
    gate still flips `inside`, and without this flag the later exit is emitted
    unpaired -- the engine then sees a person leaving a zone it never saw them
    enter, which it can only classify as unknown. The emitted stream has to be
    balanced per track, independently of what the geometry knows.
    """

    inside: bool = False
    inside_streak: int = 0
    outside_streak: int = 0
    frames_in_state: int = 0
    announced: bool = False
    last_confidence: float = 0.0

    def update(self, raw_inside: bool, enter_frames: int, exit_frames: int) -> str | None:
        """Feed one frame's raw test. Returns 'entered', 'left' or None.

        Only a transition returns a value, so the caller emits on edges. A
        subject who stands in a zone for ten minutes produces one event, not
        eighteen thousand.
        """
        if raw_inside:
            self.inside_streak += 1
            self.outside_streak = 0
        else:
            self.outside_streak += 1
            self.inside_streak = 0

        self.frames_in_state += 1
        if not self.inside and self.inside_streak >= enter_frames:
            self.inside, self.frames_in_state = True, self.inside_streak
            return "entered"
        if self.inside and self.outside_streak >= exit_frames:
            self.inside, self.frames_in_state = False, self.outside_streak
            return "left"
        return None
