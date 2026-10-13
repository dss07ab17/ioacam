"""Pose estimation stage shared by perceive.py and preview_pose.py.

Fail loudly when the stage is enabled but the weights are missing: a stage
that is configured yet quietly inactive is the failure mode wiring this into
the production path exists to fix.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional

_ACTIONS = Path(__file__).resolve().parent / "actions"
if str(_ACTIONS) not in sys.path:
    sys.path.insert(0, str(_ACTIONS))


DEFAULT_RTMPOSE = "perception/models/rtmpose_t.onnx"
EXPORT_RTMPOSE = "python3 perception/tools/export_rtmpose.py"


def require_pose_model(model_path: str) -> None:
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"pose model not found at {model_path}.\n"
            f"Export it with:  {EXPORT_RTMPOSE}"
        )


def build_pose_estimator(cfg: dict):
    """Build from a pose config block. backend stub needs no weights."""
    backend = cfg.get("backend", "rtmpose")

    if backend == "stub":
        from posetube import StubPoseEstimator

        return StubPoseEstimator(score=float(cfg.get("score", 0.9)))

    if backend == "rtmpose":
        from rtmpose import RTMPoseEstimator

        model_path = cfg.get("model_path", DEFAULT_RTMPOSE)
        require_pose_model(model_path)
        return RTMPoseEstimator(
            model_path=model_path,
            input_size=tuple(cfg.get("input_size", (192, 256))),
            simcc_split_ratio=float(cfg.get("simcc_split_ratio", 2.0)),
            num_keypoints=int(cfg.get("num_keypoints", 17)),
            device=cfg.get("device", "cpu"),
            max_batch=int(cfg.get("max_batch", 8)),
            runtime=cfg.get("runtime", "auto"),
        )

    raise ValueError(
        f"unknown pose backend {backend!r}; known: stub, rtmpose"
    )


class PoseStage:
    """Thin wrapper so both tools call the same estimate path."""

    def __init__(self, cfg: Optional[dict] = None, estimator: Any = None) -> None:
        if estimator is not None:
            self.estimator = estimator
        else:
            self.estimator = build_pose_estimator(cfg or {})

    @property
    def name(self) -> str:
        return getattr(self.estimator, "name", "pose")

    def estimate(self, frame, boxes: dict) -> dict:
        """{track_id: (K,3) x,y,score}."""
        if not boxes:
            return {}
        return self.estimator.estimate(frame, boxes)


def person_boxes_from_tracks(tracks) -> dict:
    """Map live person tracks to the box dict RTMPose expects."""
    return {
        t.track_id: (t.detection.x1, t.detection.y1, t.detection.x2, t.detection.y2)
        for t in tracks
    }
