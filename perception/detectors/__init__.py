"""Backend registry. The config names a backend; nothing else knows which one."""

from __future__ import annotations

from .base import Detection, Detector

BACKENDS = ("yolox-onnx", "yolo26", "yolo26-ppe", "ultralytics", "stub")

# Backends that detect PPE rather than people. They are selected by the `ppe`
# config block and run as a SECOND detector alongside the person one; naming
# one under `detector.backend` would leave the pipeline with no person boxes,
# no tracks and no zone events, which is why `perceive.py` rejects that.
PPE_BACKENDS = ("yolo26-ppe",)


def build_detector(cfg: dict) -> Detector:
    backend = cfg.get("backend", "yolox-onnx")

    if backend == "yolox-onnx":
        from .yolox_onnx import YoloxOnnxDetector

        return YoloxOnnxDetector(
            model_path=cfg.get("model_path", "perception/models/yolox_tiny.onnx"),
            input_size=tuple(cfg.get("input_size", (416, 416))),
            score_threshold=cfg.get("score_threshold", 0.30),
            nms_threshold=cfg.get("nms_threshold", 0.45),
            p6=cfg.get("p6", False),
            # Person alone unless the site asks for more. Objects are opt-in
            # because every extra class is another stream of detections whose
            # false-positive rate nobody has measured yet.
            classes=cfg.get("classes", ("person",)),
            class_thresholds=cfg.get("class_thresholds"),
        )

    if backend == "yolo26":
        from .yolo26 import Yolo26Detector

        # Deliberately the same keys as yolox-onnx, read with the same
        # defaults, so switching a site's config is one string. The two that
        # cannot mean the same thing are passed anyway and ignored by the
        # backend: `nms_threshold` has no referent in a one-to-one head, and
        # `p6` is a YOLOX head-layout flag. See yolo26.py -- raising on them
        # would make the one-line switch a two-line one for no gain.
        return Yolo26Detector(
            model_path=cfg.get("model_path", "perception/models/yolo26n.onnx"),
            # 640, not YOLOX's 416: this is the size the checkpoint was trained
            # and exported at, and the graph's input axis is static.
            input_size=tuple(cfg.get("input_size", (640, 640))),
            score_threshold=cfg.get("score_threshold", 0.30),
            classes=cfg.get("classes", ("person",)),
            class_thresholds=cfg.get("class_thresholds"),
            device=cfg.get("device", "cpu"),
            nms_threshold=cfg.get("nms_threshold"),
            p6=cfg.get("p6"),
        )

    if backend == "yolo26-ppe":
        from .ppe import DEFAULT_PPE_CLASSES, DEFAULT_PPE_MODEL, PpeDetector

        # Same decode, same keys, same meanings as `yolo26` -- only the class
        # list the head's class_id is read against differs. The defaults are
        # the four classes that map onto a schema observation; the model knows
        # six more and `detectors/ppe.py` says why they are off.
        return PpeDetector(
            model_path=cfg.get("model_path", DEFAULT_PPE_MODEL),
            input_size=tuple(cfg.get("input_size", (640, 640))),
            # 0.40 rather than yolo26's 0.30, and it is only a fallback: every
            # class this backend can emit has a measured floor of its own in
            # PPE_DEFAULT_THRESHOLDS, and the NO-* ones are deliberately well
            # below this. See ppe.py on why the asymmetry is the point.
            score_threshold=cfg.get("score_threshold", 0.40),
            classes=cfg.get("classes", DEFAULT_PPE_CLASSES),
            class_thresholds=cfg.get("class_thresholds"),
            device=cfg.get("device", "cpu"),
        )

    if backend == "ultralytics":
        from .ultralytics_yolo import UltralyticsDetector

        return UltralyticsDetector(
            model_path=cfg.get("model_path", "yolo11n.pt"),
            score_threshold=cfg.get("score_threshold", 0.30),
            acknowledge_agpl=cfg.get("acknowledge_agpl", False),
        )

    if backend == "stub":
        from .stub import StubDetector

        return StubDetector(script=cfg.get("script"))

    raise ValueError(f"Unknown detector backend {backend!r}. Known: {', '.join(BACKENDS)}")


__all__ = ["Detection", "Detector", "build_detector", "BACKENDS", "PPE_BACKENDS"]
