"""Face detector backend registry. The config names a backend."""

from __future__ import annotations

from .base import FaceBox, FaceDetector

FACE_BACKENDS = ("rfb-320-onnx",)


def build_face_detector(cfg: dict) -> FaceDetector:
    backend = cfg.get("backend", "rfb-320-onnx")

    if backend == "rfb-320-onnx":
        from .rfb_onnx import RfbOnnxFaceDetector

        return RfbOnnxFaceDetector(
            model_path=cfg.get(
                "model_path", "perception/models/version-RFB-320-int8.onnx"
            ),
            score_threshold=cfg.get("score_threshold", 0.70),
            nms_threshold=cfg.get("nms_threshold", 0.30),
        )

    raise ValueError(
        f"Unknown face backend {backend!r}. Known: {', '.join(FACE_BACKENDS)}"
    )


__all__ = ["FaceBox", "FaceDetector", "build_face_detector", "FACE_BACKENDS"]
