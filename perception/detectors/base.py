"""What a detector must provide, and nothing more.

The whole point of this seam is that the licence of the model backend is a
one-line config change, not a rewrite. See perception/LICENCE-NOTES.md: the
default backend is Apache-2.0 precisely so that the AGPL backend never becomes
load-bearing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass(frozen=True)
class Detection:
    """One detected box in frame pixel coordinates.

    `score` is the model's RAW output. Nothing in this class is calibrated;
    calibration happens once, in confidence.py, so there is exactly one place
    where a raw number becomes a confidence.
    """

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    label: str = "person"

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def ground_point(self) -> tuple[float, float]:
        """Bottom-centre of the box: where the subject meets the floor.

        Zone membership is tested here rather than at the box centroid. A
        person standing just outside a zone leans over it constantly; their
        feet do not. Using the centroid produces a stream of spurious
        enter/leave pairs that the engine cannot distinguish from real ones.
        """
        return ((self.x1 + self.x2) / 2.0, self.y2)


class Detector(Protocol):
    """Backends implement this and are selected by name in the config."""

    name: str
    licence: str

    def detect(self, frame) -> Sequence[Detection]:
        """Return person detections for one BGR frame."""
        ...
