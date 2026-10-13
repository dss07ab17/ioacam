"""Ultra-Light-Fast-Generic-Face-Detector-1MB (RFB-320), int8, via onnxruntime.

Model:   version-RFB-320-int8.onnx
Runtime: onnxruntime (CPUExecutionProvider)

Preprocessing, thresholds and NMS are ported 1:1 from the working C++
implementation (FaceEngine::detectFaces) so this backend produces the same
boxes on the same frame: RGB, resized to 320x240, normalised as
(pixel - 127) / 128, score threshold 0.70, NMS IoU 0.30. The exported graph
already decodes anchors internally -- `boxes` comes out normalised to [0,1]
in x1,y1,x2,y2, so no separate prior-box math is needed here, same as the
C++ version.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import onnxruntime as ort

from .base import FaceBox

DET_INPUT_W = 320
DET_INPUT_H = 240
DET_SCORE_THRESHOLD = 0.70
DET_NMS_IOU_THRESHOLD = 0.30


class RfbOnnxFaceDetector:
    name = "rfb-320-onnx"
    licence = "Ultra-Light-Fast-Generic-Face-Detector-1MB (MIT), onnxruntime"

    def __init__(
        self,
        model_path: str,
        score_threshold: float = DET_SCORE_THRESHOLD,
        nms_threshold: float = DET_NMS_IOU_THRESHOLD,
        session: ort.InferenceSession | None = None,
    ) -> None:
        # `session` is a test seam, same pattern as `net=None` on
        # YoloxOnnxDetector: lets decode logic be exercised with a scripted
        # session and no weights on disk.
        if session is None:
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"RFB-320 face detector weights not found at {model_path}.\n"
                    "Copy version-RFB-320-int8.onnx into perception/models/."
                )
            session = ort.InferenceSession(
                model_path, providers=["CPUExecutionProvider"]
            )
        self.session = session
        self.model_path = model_path
        self.score_threshold = float(score_threshold)
        self.nms_threshold = float(nms_threshold)

        self._input_name = session.get_inputs()[0].name
        self._output_names = [o.name for o in session.get_outputs()]

    # -- preprocessing -------------------------------------------------

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (DET_INPUT_W, DET_INPUT_H))
        normalised = (resized.astype(np.float32) - 127.0) / 128.0
        chw = normalised.transpose(2, 0, 1)[None]  # 1,3,240,320
        return np.ascontiguousarray(chw)

    # -- inference -------------------------------------------------------

    def detect(self, frame: np.ndarray) -> list[FaceBox]:
        h, w = frame.shape[:2]
        blob = self._preprocess(frame)

        outputs = self.session.run(self._output_names, {self._input_name: blob})
        by_name = dict(zip(self._output_names, outputs))
        # Model exports "scores" [1,N,2] and "boxes" [1,N,4], normalised
        # x1,y1,x2,y2 in [0,1]. Same defensive shape check as the C++ code,
        # in case a future export swaps output order.
        scores_t = by_name.get("scores")
        boxes_t = by_name.get("boxes")
        if scores_t is None or boxes_t is None:
            scores_t, boxes_t = outputs[0], outputs[1]
            if scores_t.shape[-1] == 4:
                scores_t, boxes_t = boxes_t, scores_t

        scores = scores_t[0][:, 1]  # class-1 (face) probability
        boxes = boxes_t[0]

        keep_mask = scores >= self.score_threshold
        if not np.any(keep_mask):
            return []

        kept_boxes = boxes[keep_mask]
        kept_scores = scores[keep_mask]

        # Denormalise to pixel coords for cv2.dnn.NMSBoxes (needs x,y,w,h).
        xywh = np.empty_like(kept_boxes)
        xywh[:, 0] = kept_boxes[:, 0] * w
        xywh[:, 1] = kept_boxes[:, 1] * h
        xywh[:, 2] = (kept_boxes[:, 2] - kept_boxes[:, 0]) * w
        xywh[:, 3] = (kept_boxes[:, 3] - kept_boxes[:, 1]) * h

        idxs = cv2.dnn.NMSBoxes(
            xywh.tolist(), kept_scores.tolist(),
            self.score_threshold, self.nms_threshold,
        )
        if len(idxs) == 0:
            return []

        out: list[FaceBox] = []
        for i in np.array(idxs).reshape(-1):
            x, y, bw, bh = xywh[i]
            out.append(
                FaceBox(
                    x1=float(np.clip(x, 0, w)),
                    y1=float(np.clip(y, 0, h)),
                    x2=float(np.clip(x + bw, 0, w)),
                    y2=float(np.clip(y + bh, 0, h)),
                    score=float(kept_scores[i]),
                )
            )
        return out
