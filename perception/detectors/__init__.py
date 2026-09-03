"""Backend registry. The config names a backend; nothing else knows which one."""

from __future__ import annotations

from .base import Detection, Detector

BACKENDS = ("yolox-onnx", "ultralytics", "stub")


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


__all__ = ["Detection", "Detector", "build_detector", "BACKENDS"]
