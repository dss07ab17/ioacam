"""A detector with no model behind it. Scripted boxes, deterministic.

Exists so the geometry, tracking, hysteresis, calibration and emission logic
can be tested without a camera, without weights and without a GPU -- which is
what makes perception/test_perception.py runnable in CI.
"""

from __future__ import annotations

from .base import Detection


class StubDetector:
    name = "stub"
    licence = "n/a (no model)"

    def __init__(self, script: list[list[dict]] | None = None) -> None:
        self.script = script or []
        self.frame_index = 0

    def detect(self, frame) -> list[Detection]:
        boxes = self.script[self.frame_index] if self.frame_index < len(self.script) else []
        self.frame_index += 1
        return [Detection(**b) for b in boxes]
