"""Ultralytics YOLO backend. AGPL-3.0. OPT-IN ONLY. DO NOT SHIP.

Read perception/LICENCE-NOTES.md before enabling this.

Ultralytics' code AND its pretrained weights are AGPL-3.0. In a product that is
distributed to customers, that obliges you to release the complete corresponding
source of the combined work under AGPL, which for iOACAM means the engine, the
policies and the site integrations. Ultralytics' own commercial licence is the
only other lawful route.

Running this backend in a separate process does NOT neutralise that. Process
separation is a contested mitigation, not a settled one, and it is not a defence
worth betting a product on. The seam in this directory exists so the AGPL path
stays a bench-comparison tool, never a dependency.

Kept because it is genuinely useful for one job: A/B-ing YOLOX's recall against
a well-tuned YOLO11 on YOUR footage before committing to the permissive model.
That is an internal R&D use, which AGPL permits, since nothing is distributed.
Gated behind an explicit acknowledgement so it cannot be switched on by accident.
"""

from __future__ import annotations

import sys

from .base import Detection

COCO_PERSON_CLASS = 0

_BANNER = """
+---------------------------------------------------------------------------+
|  AGPL-3.0 BACKEND ACTIVE -- ultralytics                                    |
|  Internal evaluation only. This must not be present in a shipped build.    |
|  Default to the 'yolox-onnx' backend (Apache-2.0) for anything delivered.  |
|  See perception/LICENCE-NOTES.md                                           |
+---------------------------------------------------------------------------+
"""


class UltralyticsDetector:
    name = "ultralytics"
    licence = "AGPL-3.0 -- NOT SHIPPABLE without a commercial licence from Ultralytics"

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        score_threshold: float = 0.30,
        acknowledge_agpl: bool = False,
        **_ignored,
    ) -> None:
        if not acknowledge_agpl:
            raise PermissionError(
                "The ultralytics backend is AGPL-3.0 and is disabled by default.\n"
                "To use it for internal evaluation only, set\n"
                '  "detector": { "backend": "ultralytics", "acknowledge_agpl": true }\n'
                "in the config. Read perception/LICENCE-NOTES.md first."
            )
        print(_BANNER, file=sys.stderr)
        from ultralytics import YOLO  # imported lazily: never a hard dependency

        self.model = YOLO(model_path)
        self.score_threshold = float(score_threshold)

    def detect(self, frame) -> list[Detection]:
        results = self.model.predict(
            frame, classes=[COCO_PERSON_CLASS], conf=self.score_threshold, verbose=False
        )
        out = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            out.append(Detection(x1=x1, y1=y1, x2=x2, y2=y2, score=float(box.conf[0])))
        return out
