"""Action recognition stage: pose tubes → PoseC3D → abstention → events.

Shared by `perceive.py` and `preview_pose.py`. PoseC3D is expensive
(~130–185 ms); inference runs on an `--action-every` cadence, not per frame.

Abstentions are emitted as `action_recognised` with value `"unknown"` so the
engine can route them for review rather than treating silence as "nothing
happened".
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional

from emit import EventEmitter

_ACTIONS = Path(__file__).resolve().parent / "actions"
if str(_ACTIONS) not in sys.path:
    sys.path.insert(0, str(_ACTIONS))

from base import AbstentionPolicy, ActionScore  # noqa: E402
from posec3d import NTU60_CLASSES  # noqa: E402
from posetube import PoseFrameRecord, PoseTubeExtractor  # noqa: E402

DEFAULT_POSEC3D = "perception/models/posec3d_pose_only.onnx"
EXPORT_POSEC3D = "python3 perception/tools/export_posec3d.py"


def require_action_model(model_path: str) -> None:
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"PoseC3D ONNX not found at {model_path}.\n"
            f"Export it with:  {EXPORT_POSEC3D}"
        )


def resolve_classes(classes) -> list[str]:
    if classes is None or classes == "ntu60":
        return list(NTU60_CLASSES)
    if isinstance(classes, str):
        raise ValueError(
            f"unknown class vocabulary {classes!r}; use 'ntu60' or a list of names"
        )
    return list(classes)


def build_action_recognised_event(
    emitter: EventEmitter,
    result: ActionScore,
    timestamp_us: Optional[int] = None,
) -> dict:
    """Schema-shaped action_recognised event, including abstentions.

    A decided recognition carries the class label. An abstention carries
    value \"unknown\" with the model's best-class confidence: that is
    information the engine can queue for review, not a silent drop.
    """
    if result.decided:
        value = result.label
        confidence = result.confidence
    else:
        value = "unknown"
        confidence = result.confidence
    return emitter.build(
        observation="action_recognised",
        confidence=confidence,
        value=value,
        track_id=result.track_id,
        subject={"class": "human"},
        timestamp_us=timestamp_us,
    )


class ActionsStage:
    """Rolling tubes + cadence-gated PoseC3D + abstention policy."""

    def __init__(self, cfg: dict) -> None:
        backend = cfg.get("backend", "posec3d")
        every_s = float(cfg.get("every_s", cfg.get("action_every", 1.0)))
        self.every_s = max(0.0, every_s)

        tube_cfg = cfg.get("tube") or {}
        self.extractor = PoseTubeExtractor(
            window_s=float(tube_cfg.get("window_s", 1.5)),
            num_frames=int(cfg.get("num_frames", tube_cfg.get("num_frames", 32))),
            heatmap_size=int(
                cfg.get("heatmap_size", tube_cfg.get("heatmap_size", 56))
            ),
            num_keypoints=int(
                cfg.get("num_keypoints", tube_cfg.get("num_keypoints", 17))
            ),
            box_margin=float(tube_cfg.get("box_margin", 0.1)),
            sigma=float(tube_cfg.get("sigma", 0.6)),
            min_keypoint_score=float(tube_cfg.get("min_keypoint_score", 0.15)),
        )

        abstention = cfg.get("abstention") or {}
        self.policy = AbstentionPolicy.from_config(
            {
                "min_confidence": cfg.get(
                    "min_confidence", abstention.get("min_confidence", 0.55)
                ),
                "min_margin": cfg.get(
                    "min_margin", abstention.get("min_margin", 0.15)
                ),
                "per_class_confidence": abstention.get("per_class_confidence"),
                "temperature": abstention.get("temperature", 1.0),
            }
        )

        classes = resolve_classes(cfg.get("classes", "ntu60"))

        if backend == "stub":
            from stub import StubRecognizer

            self.recognizer = StubRecognizer(
                classes=classes
                if cfg.get("classes") not in (None, "ntu60")
                else ["standing", "walking", "reaching"],
                num_frames=self.extractor.num_frames,
                input_size=int(cfg.get("input_size", 224)),
            )
        elif backend == "posec3d":
            from posec3d import PoseC3DRecognizer

            model_path = cfg.get("model_path", DEFAULT_POSEC3D)
            require_action_model(model_path)
            self.recognizer = PoseC3DRecognizer(
                model_path=model_path,
                classes=classes,
                num_frames=self.extractor.num_frames,
                heatmap_size=self.extractor.heatmap_size,
                num_keypoints=self.extractor.num_keypoints,
                device=cfg.get("device", "cpu"),
                rgb_branch=bool(cfg.get("rgb_branch", False)),
                input_size=int(cfg.get("input_size", 224)),
            )
        else:
            raise ValueError(
                f"unknown action backend {backend!r}; known: stub, posec3d"
            )

        self.actions: dict[str, tuple[ActionScore, float]] = {}
        self._last_infer = -1e9
        self.runs = 0
        self.tubes = 0

    @property
    def name(self) -> str:
        return getattr(self.recognizer, "name", "actions")

    def push(
        self,
        timestamp_us: int,
        keypoints: dict,
        boxes: dict,
        image=None,
    ) -> None:
        self.extractor.push(
            PoseFrameRecord(
                timestamp_us=timestamp_us,
                keypoints=keypoints,
                boxes=boxes,
                image=image,
            )
        )

    def maybe_infer(
        self,
        timestamp_us: int,
        force: bool = False,
        now: Optional[float] = None,
    ) -> list[ActionScore]:
        """Run PoseC3D for ready tracks when the cadence allows.

        Returns the decisions produced this call (including abstentions).
        """
        clock = time.perf_counter() if now is None else now
        if not force and (clock - self._last_infer) < self.every_s:
            return []

        self._last_infer = clock
        self.runs += 1
        ready = self.extractor.tracks_ready(timestamp_us)
        produced: list[ActionScore] = []

        for track_id in ready:
            tube = self.extractor.extract(track_id, timestamp_us)
            if tube is None:
                continue
            self.tubes += 1
            raw = self.recognizer.infer(tube)
            result = self.policy.decide(
                track_id, raw, backend=self.recognizer.name
            )
            result.window_start_us = tube.window_start_us
            result.window_end_us = tube.window_end_us
            self.actions[track_id] = (result, tube.coverage)
            produced.append(result)

        for gone in set(self.actions) - set(ready):
            self.actions.pop(gone, None)

        return produced

    def build_events(
        self,
        emitter: EventEmitter,
        results: list[ActionScore],
        timestamp_us: Optional[int] = None,
    ) -> list[dict]:
        return [
            build_action_recognised_event(emitter, r, timestamp_us)
            for r in results
        ]
