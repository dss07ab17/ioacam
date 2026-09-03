"""YOLOX-s via OpenCV's DNN module. Apache-2.0 end to end, no extra deps.

Model:   YOLOX (Megvii)      Apache-2.0
Weights: yolox_tiny.onnx     Apache-2.0, published by the same project
Runtime: cv2.dnn             Apache-2.0 (OpenCV 4.5+)

This is the default backend and the one intended to ship. It needs no
onnxruntime and no torch: OpenCV is already a dependency of the capture path,
so the permissive option is also the lightest one.

YOLOX was chosen over the alternatives because it is already the model named in
the README's target pipeline, it exports to ONNX by design, and RKNN-Toolkit2
conversions of it are well trodden. That last point still has to be confirmed on
the actual board before anyone commits to it -- as the README says, a model that
will not convert costs nothing to discover in week one and everything in month
four.
"""

from __future__ import annotations

import os
from typing import Sequence

import cv2
import numpy as np

from .base import Detection

COCO_PERSON_CLASS = 0

# The 80 COCO classes, in the order the head emits them. The graph has always
# produced all of these -- this backend simply threw 79 of them away. Order is
# the only thing mapping a column to a meaning, so it is not editable.
COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)
CLASS_INDEX = {name: i for i, name in enumerate(COCO_CLASSES)}


class YoloxOnnxDetector:
    name = "yolox-onnx"
    licence = "Apache-2.0 (YOLOX model, YOLOX weights, OpenCV runtime)"

    def __init__(
        self,
        model_path: str,
        input_size: tuple[int, int] = (416, 416),
        score_threshold: float = 0.30,
        nms_threshold: float = 0.45,
        p6: bool = False,
        classes: Sequence[str] = ("person",),
        class_thresholds: dict[str, float] | None = None,
    ) -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"YOLOX weights not found at {model_path}.\n"
                "Fetch them with:  python perception/tools/fetch_model.py"
            )
        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.input_size = (int(input_size[0]), int(input_size[1]))  # (h, w)
        self.score_threshold = float(score_threshold)
        self.nms_threshold = float(nms_threshold)
        self.p6 = bool(p6)

        # Which of the 80 the site cares about. Defaults to person alone, so
        # every existing caller keeps exactly the behaviour it had; a site that
        # wants objects names them in config rather than editing this file.
        unknown = [c for c in classes if c not in CLASS_INDEX]
        if unknown:
            raise ValueError(
                f"not COCO classes: {unknown}. This model knows only its 80 "
                f"training classes -- a site-specific object (a torque wrench, "
                f"a specific tool) needs a detector trained on it, and asking "
                f"for it here would silently return nothing."
            )
        if not classes:
            raise ValueError("classes is empty: the detector would return nothing")
        self.classes = tuple(classes)

        # Per class, because one threshold cannot serve both. A person fills
        # the frame and scores 0.95; a phone is 40 pixels across and scores
        # 0.45 when it is really there. One number either floods the log with
        # phantom objects or never sees a real one.
        self.class_thresholds = {
            name: float((class_thresholds or {}).get(name, score_threshold))
            for name in self.classes
        }
        self._grids: np.ndarray | None = None
        self._strides: np.ndarray | None = None

    # -- preprocessing -----------------------------------------------------

    def _preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        """Letterbox to input_size, pad with 114, keep BGR, no /255.

        This matches YOLOX's own export contract exactly. The released ONNX
        folds normalisation into the graph, so scaling here would silently
        halve the detector's accuracy rather than fail loudly.
        """
        ih, iw = self.input_size
        padded = np.full((ih, iw, 3), 114, dtype=np.uint8)
        r = min(ih / frame.shape[0], iw / frame.shape[1])
        nh, nw = int(frame.shape[0] * r), int(frame.shape[1] * r)
        padded[:nh, :nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        blob = padded.transpose(2, 0, 1)[None].astype(np.float32)
        return np.ascontiguousarray(blob), r

    # -- decoding ----------------------------------------------------------

    def _anchor_grid(self) -> tuple[np.ndarray, np.ndarray]:
        """Grid centres and strides for the flattened YOLOX head.

        Cached: it depends only on input_size, and rebuilding it per frame is
        measurable at 30 fps on a laptop CPU.
        """
        if self._grids is not None and self._strides is not None:
            return self._grids, self._strides
        ih, iw = self.input_size
        strides = [8, 16, 32, 64] if self.p6 else [8, 16, 32]
        grids, expanded = [], []
        for stride in strides:
            hs, ws = ih // stride, iw // stride
            xv, yv = np.meshgrid(np.arange(ws), np.arange(hs))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            expanded.append(np.full((1, grid.shape[1], 1), stride))
        self._grids = np.concatenate(grids, 1).astype(np.float32)
        self._strides = np.concatenate(expanded, 1).astype(np.float32)
        return self._grids, self._strides

    def _decode(self, raw: np.ndarray) -> np.ndarray:
        """[1, N, 85] raw head output -> [N, 85] with boxes in input pixels."""
        preds = raw[0].astype(np.float32)
        grids, strides = self._anchor_grid()
        if preds.shape[0] != grids.shape[1]:
            raise ValueError(
                f"Model emitted {preds.shape[0]} anchors, grid expects "
                f"{grids.shape[1]}. Check input_size and the p6 flag in the config."
            )
        preds[:, :2] = (preds[:, :2] + grids[0]) * strides[0]
        preds[:, 2:4] = np.exp(preds[:, 2:4]) * strides[0]
        return preds

    # -- inference ---------------------------------------------------------

    def detect(self, frame: np.ndarray) -> list[Detection]:
        blob, ratio = self._preprocess(frame)
        self.net.setInput(blob)
        preds = self._decode(self.net.forward())

        h, w = frame.shape[:2]
        out: list[Detection] = []

        # Score every enabled class at once, then drop the anchors that clear
        # no threshold at all. Almost all of the ~3500 anchors are background,
        # so this collapses the work before any per-class handling: scoring
        # and converting boxes once for ten classes rather than ten times.
        columns = [5 + CLASS_INDEX[name] for name in self.classes]
        thresholds = np.array([self.class_thresholds[n] for n in self.classes],
                              dtype=np.float32)
        class_scores = preds[:, columns] * preds[:, 4:5]        # anchors x classes
        surviving = np.any(class_scores >= thresholds, axis=1)
        if not np.any(surviving):
            return []
        boxes = preds[surviving, :4]
        class_scores = class_scores[surviving]

        # cx,cy,w,h (letterboxed) -> x,y,w,h (original frame pixels)
        xywh = np.empty_like(boxes)
        xywh[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2.0) / ratio
        xywh[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2.0) / ratio
        xywh[:, 2] = boxes[:, 2] / ratio
        xywh[:, 3] = boxes[:, 3] / ratio
        xywh_list = xywh.tolist()

        # NMS *within* a class, never across. Cross-class suppression would be
        # actively wrong here: a backpack sits inside its wearer's box and a
        # phone inside a hand, so one global pass would delete exactly the
        # objects this was turned on to find.
        for ci, name in enumerate(self.classes):
            threshold = float(thresholds[ci])
            scores = class_scores[:, ci]
            keep = np.nonzero(scores >= threshold)[0]
            if keep.size == 0:
                continue

            idxs = cv2.dnn.NMSBoxes(
                [xywh_list[i] for i in keep], scores[keep].tolist(),
                threshold, self.nms_threshold,
            )
            if len(idxs) == 0:
                continue

            for j in np.array(idxs).reshape(-1):
                i = keep[j]
                x, y, bw, bh = xywh[i]
                out.append(
                    Detection(
                        x1=float(np.clip(x, 0, w)),
                        y1=float(np.clip(y, 0, h)),
                        x2=float(np.clip(x + bw, 0, w)),
                        y2=float(np.clip(y + bh, 0, h)),
                        score=float(scores[i]),
                        label=name,
                    )
                )
        return out
