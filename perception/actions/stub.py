"""Deterministic recogniser for tests. Never ships.

Scores come from a table the test supplies, so the tube, abstention and fusion
logic can be exercised without a model, a GPU, or a single frame of video.
"""

from __future__ import annotations

from typing import Optional, Sequence


class StubRecognizer:
    name = "stub"
    licence = "n/a (test stub)"

    def __init__(
        self,
        scores: Optional[dict] = None,
        classes: Sequence[str] = ("standing", "walking", "reaching", "tightening"),
        num_frames: int = 16,
        input_size: int = 224,
        label: str = "stub",
    ) -> None:
        # {track_id: {class: score}}
        self.scores = scores or {}
        self.classes = list(classes)
        self.num_frames = num_frames
        self.input_size = input_size
        self.name = label

    def infer(self, tube) -> dict[str, float]:
        return dict(self.scores.get(tube.track_id, {}))
