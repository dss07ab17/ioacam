"""What a face detector must provide, and nothing more.

Same shape as perception/detectors/base.py: a plain box dataclass with raw,
uncalibrated scores, and a Protocol so the backend is a config choice rather
than a hardwired import.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Sequence


@dataclass(frozen=True)
class FaceBox:
    """One detected face in frame pixel coordinates.

    `score` is the model's RAW output, uncalibrated -- same convention as
    Detection.score in detectors/base.py.
    """

    x1: float
    y1: float
    x2: float
    y2: float
    score: float

    # Landmarks in frame pixel coordinates, if the backend produces them.
    # Either five points (left eye, right eye, nose, left mouth, right mouth)
    # or two (both eyes). RFB-320 emits neither, so this stays None and the
    # face has to get its landmarks elsewhere -- see identity/landmarks.py.
    #
    # It is declared here rather than in a subclass because it is the seam
    # that lets a landmark-producing detector (SCRFD, RetinaFace) replace
    # RFB-320 without any change to the identity layer: alignment reads this
    # field first and only falls back if it is empty.
    landmarks: Optional[tuple[tuple[float, float], ...]] = None

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    def crop(self, frame):
        """Return the BGR pixel crop for this box, clamped to the frame."""
        h, w = frame.shape[:2]
        x1 = max(0, int(self.x1))
        y1 = max(0, int(self.y1))
        x2 = min(w, int(self.x2))
        y2 = min(h, int(self.y2))
        return frame[y1:y2, x1:x2]


class FaceDetector(Protocol):
    """Backends implement this and are selected by name in config."""

    name: str
    licence: str

    def detect(self, frame) -> Sequence[FaceBox]:
        """Return face detections for one BGR frame."""
        ...