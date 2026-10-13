"""YOLO26 via onnxruntime. NMS-free, end to end. AGPL-3.0 -- read the notes.

Model:   YOLO26 (Ultralytics)   AGPL-3.0, or Ultralytics' commercial licence
Weights: yolo26n.onnx           AGPL-3.0, same terms as the code
Runtime: onnxruntime            MIT

## Licence, before anything else

YOLO26 is an Ultralytics model and carries **the same AGPL-3.0 terms as the
`ultralytics` backend next to it** -- code and pretrained weights alike. Every
argument in `perception/LICENCE-NOTES.md` against shipping Ultralytics YOLO11
applies here unchanged, including the one that matters most: converting the
checkpoint to ONNX and dropping the `ultralytics` package does not change the
licence of the weights running inside the graph. This backend does not require
`acknowledge_agpl`, so nothing stops it being switched on -- which is exactly
why it says so here, in its `licence` attribute, and on stderr at construction.
`yolox-onnx` remains the default and the shippable path.

What this backend *is* good for is the same job the `ultralytics` one has:
measuring, on your own footage and your own machines, whether a modern
NMS-free detector buys enough recall over YOLOX to justify buying a licence.

## Why onnxruntime and not the ultralytics package

For the same reason `actions/rtmpose.py` does not import MMPose. The inference
path across this whole layer is onnxruntime and OpenCV, and that is what makes
the RK3568 port a conversion job rather than a rewrite: RKNN-Toolkit2 takes
ONNX. Pulling torch onto the runtime path to get `model.predict()` would end
that, permanently, and it would happen a long way from the person who later
has to make the board work. `ultralytics` is needed by
`perception/tools/export_yolo26.py`, once, offline, and nowhere else.

## The contract

Export with `nms=False` (`perception/tools/export_yolo26.py` does):

    input    images    (1, 3, 640, 640) float32, RGB, 0..1, letterboxed
    output   output0   (1, 300, 6)      [x1, y1, x2, y2, score, class_id]

The boxes are already xyxy in *input-image* pixels -- letterboxed 640x640
space, not the original frame -- and already deduplicated.

### The checkpoint has two heads, and `yolo predict` uses the other one

YOLO26 is trained with both a one-to-many head (YOLOv8-style, needs NMS) and a
one-to-one head (NMS-free). The published `.pt` ships `head.end2end = False`,
so `YOLO("yolo26n.pt").predict(...)` runs the **one-to-many** head plus NMS.
`export(nms=False)` turns `end2end` on and serialises the **one-to-one** head,
which is what this backend runs.

They are different predictors sharing a backbone and they give different
answers -- on `export_yolo26.py`'s synthetic frame, 0.187 against 0.118 on the
same box, plus a third box only one of them finds. Neither is wrong. But it
means benchmarking `yolo predict` against YOLOX and concluding anything about
*this* backend is measuring a model you are not running. Compare like for like:
either `Yolo26Detector` against `YoloxOnnxDetector`, or ultralytics loading the
exported `.onnx` (which does use the one-to-one head).

## Why the postprocess is not YOLOX's with the NMS deleted

YOLO26's detection head is trained with one-to-one assignment, so it emits at
most one box per object and needs no NMS. That changes three things, and
getting any of them wrong produces plausible boxes rather than an error:

1. **No anchor grid, no exponentials.** YOLOX emits ~3500 anchor-relative
   `(dx, dy, log w, log h)` tuples that must be un-shifted by grid centre and
   un-scaled by stride. YOLO26's end-to-end head emits final pixel corners.
   There is nothing to decode geometrically -- decoding it *again* would move
   every box.
2. **No objectness column.** YOLOX's score is `objectness * class_score`, two
   columns multiplied. YOLO26 emits a single confidence and its class id.
   Multiplying by anything here would quietly halve every score, and the only
   symptom would be a detector that has become mysteriously conservative.
3. **Fixed-length, padded output.** The head always returns `max_det` (300)
   rows. The tail is padding -- zero or near-zero confidence, often zero boxes
   -- and is removed by the confidence threshold, not by a count. Rows arrive
   sorted by confidence descending.

The one thing that is *not* different is what happens after decoding: class
filtering and per-class thresholds come from `detectors/vocabulary.py` and are
the same code YOLOX runs through `detectors/coco.py`, so a config's `classes`
and `class_thresholds` mean exactly the same thing under either backend.

## The class list is a parameter, and the decode is not

`vocabulary=` selects which trained class list the emitted `class_id` is read
against. It defaults to COCO, which is what the stock checkpoint emits. The PPE
fine-tune in `detectors/ppe.py` is the same head over ten different classes and
is served by this same decode with that one argument changed -- see the note
there for why a second checkpoint is a second detector rather than more classes
on this one.
"""

from __future__ import annotations

import os
import sys
from typing import Sequence

import cv2
import numpy as np

from .base import Detection
from .coco import CLASS_INDEX, COCO, COCO_CLASSES, resolve_classes  # noqa: F401
from .vocabulary import Vocabulary

# What `export_yolo26.py --nms false` produces, and what the decode assumes.
END2END_COLUMNS = 6

_warned: set[str] = set()


def _banner(backend: str, note: str) -> str:
    lines = [
        f"AGPL-3.0 BACKEND ACTIVE -- {backend}",
        "Ultralytics weights keep their licence after an ONNX export.",
        *note.splitlines(),
        "See perception/LICENCE-NOTES.md",
    ]
    width = max(len(line) for line in lines)
    rule = "+" + "-" * (width + 4) + "+"
    body = "\n".join(f"|  {line.ljust(width)}  |" for line in lines)
    return f"\n{rule}\n{body}\n{rule}\n"


def _warn_once(backend: str, note: str) -> None:
    """Print the AGPL banner on the first construction of a backend in a process.

    Once rather than every time. A pipeline builds one detector at startup, so
    in the only case that matters this is identical to printing unconditionally
    -- but a test that builds a dozen detectors would otherwise bury its own
    output under a dozen banners, and a warning nobody can read is not a
    warning. `perceive.py` also logs `detector.licence` on every run, so the
    licence of the running model is in the log of every session regardless.

    Keyed by backend rather than a single flag, because the PPE detector runs
    ALONGSIDE the person one and carries its own licence line. One banner
    suppressing the other would leave a second AGPL model running unannounced.
    """
    if backend not in _warned:
        _warned.add(backend)
        print(_banner(backend, note), file=sys.stderr)


class _OnnxRuntimeSession:
    """onnxruntime, wrapped so the detector only knows about arrays.

    Same shape as `actions/rtmpose.py`'s wrapper, for the same reason: the
    thing under test is the decode, and a session that is one `run(blob)`
    method can be replaced by twelve lines in a test file.
    """

    def __init__(self, model_path: str, device: str = "cpu") -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "the yolo26 backend needs onnxruntime (pip install onnxruntime).\n"
                "Unlike yolox-onnx it cannot fall back to cv2.dnn: OpenCV's DNN "
                "importer does not accept the end-to-end head's ops."
            ) from exc

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(model_path, providers=providers)
        spec = self.session.get_inputs()[0]
        self.input_name = spec.name
        self.input_shape = spec.shape

    def expect_input(self, model_path: str, width: int, height: int) -> None:
        """Fail at load if the config's input_size is not the graph's.

        `export_yolo26.py` exports with `dynamic=False`, so the input axis is
        static and a mismatch is fatal -- but onnxruntime's own message names
        two tensor shapes and no cause. The likely cause is one specific thing:
        a config switched from `yolox-onnx` to `yolo26` while keeping YOLOX's
        `input_size: [416, 416]`, so say that.
        """
        shape = self.input_shape
        if len(shape) != 4 or not all(isinstance(d, int) for d in shape[2:]):
            return                                   # dynamic axes: nothing to check
        got_h, got_w = int(shape[2]), int(shape[3])
        if (got_h, got_w) != (height, width):
            raise ValueError(
                f"{os.path.basename(model_path)} takes {got_w}x{got_h} input, but "
                f"the config asks for {width}x{height}.\n"
                "The graph's input axis is static (export_yolo26.py uses "
                "dynamic=False deliberately -- RKNN wants static shapes).\n"
                "If this config was switched over from yolox-onnx, it still has "
                "YOLOX's input_size; set detector.input_size to "
                f"[{got_w}, {got_h}] or drop the key to take the default."
            )

    def run(self, blob: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input_name: blob})[0]


class Yolo26Detector:
    name = "yolo26"
    licence = (
        "AGPL-3.0 (Ultralytics YOLO26 code and weights) -- NOT SHIPPABLE without "
        "a commercial licence from Ultralytics; see perception/LICENCE-NOTES.md"
    )
    agpl_note = (
        "Internal evaluation only. Default to 'yolox-onnx' for anything delivered."
    )
    weights_hint = (
        "Export them with:  python perception/tools/export_yolo26.py --download\n"
        "(ultralytics is needed for that one offline step, not at run time.)"
    )

    def __init__(
        self,
        model_path: str,
        input_size: tuple[int, int] = (640, 640),
        score_threshold: float = 0.30,
        classes: Sequence[str] = ("person",),
        class_thresholds: dict[str, float] | None = None,
        device: str = "cpu",
        session=None,
        # Which trained class list the head's `class_id` column is read
        # against. COCO for the stock checkpoint; `detectors/ppe.py` passes its
        # own for the ten-class fine-tune. Nothing else about the decode moves.
        vocabulary: Vocabulary = COCO,
        # Accepted and ignored so one config runs under either backend without
        # editing. `nms_threshold` has no meaning for a one-to-one head, and
        # `p6` is a YOLOX head-layout flag. Silently accepting them is the
        # lesser evil: the alternative is a TypeError when someone flips
        # `backend` on an existing site config, which is the one change this
        # seam exists to make free.
        nms_threshold: float | None = None,
        p6: bool | None = None,
        **_ignored,
    ) -> None:
        if session is None and not os.path.exists(model_path):
            raise FileNotFoundError(
                f"{self.name} weights not found at {model_path}.\n" + self.weights_hint
            )
        _warn_once(self.name, self.agpl_note)

        self.vocabulary = vocabulary
        self.model_path = model_path
        # (w, h). Note this is the opposite order to YoloxOnnxDetector's (h, w).
        # Both are square in every shipped config, so the difference has never
        # bitten; it is spelled out because a non-square input would make it
        # matter and nothing would raise.
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.score_threshold = float(score_threshold)
        self.device = device

        self.classes, self.class_thresholds = vocabulary.resolve(
            classes, self.score_threshold, class_thresholds
        )
        # Index -> name for the enabled classes only. The head emits a class id
        # from the model's full class list, so this doubles as the class
        # filter: a lookup miss is a class the site did not ask for.
        index = vocabulary.index
        self.enabled = {index[name]: name for name in self.classes}
        self._threshold_by_id = {
            index[name]: self.class_thresholds[name] for name in self.classes
        }
        self._floor = min(self.class_thresholds.values())

        self._session = session
        self._checked = False

    # -- runtime -----------------------------------------------------------

    def _load(self):
        if self._session is None:
            session = _OnnxRuntimeSession(self.model_path, self.device)
            session.expect_input(self.model_path, *self.input_size)
            self._session = session
        return self._session

    # -- preprocessing -----------------------------------------------------

    def _preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, float, float, float]:
        """Letterbox to input_size, pad 114, RGB, 0..1 float. Returns the mapping.

        Three things differ from YOLOX's `_preprocess`, and all three are the
        export's contract rather than a preference:

        - **Centred padding.** Ultralytics' `LetterBox` splits the pad between
          both sides; YOLOX puts the whole pad bottom-right. Getting this wrong
          does not fail -- it offsets every box by half the pad, which on a
          16:9 frame at 640 is 140 px vertically.
        - **RGB.** Frames arrive BGR from OpenCV and the model was trained RGB.
        - **/255.** YOLOX's released graph folds normalisation in and takes
          0..255; the Ultralytics export does not and takes 0..1.
        """
        iw, ih = self.input_size
        h0, w0 = frame.shape[:2]
        r = min(ih / h0, iw / w0)
        nh, nw = round(h0 * r), round(w0 * r)
        pad_x, pad_y = (iw - nw) / 2.0, (ih - nh) / 2.0

        padded = np.full((ih, iw, 3), 114, dtype=np.uint8)
        top, left = int(round(pad_y - 0.1)), int(round(pad_x - 0.1))
        padded[top:top + nh, left:left + nw] = cv2.resize(
            frame, (nw, nh), interpolation=cv2.INTER_LINEAR
        )
        rgb = padded[:, :, ::-1].astype(np.float32) / 255.0
        blob = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])
        return blob, r, float(left), float(top)

    # -- decoding ----------------------------------------------------------

    def _decode(
        self, raw: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(1, N, 6) end-to-end output -> boxes (input px), scores, class ids.

        No NMS and no geometric decode: the head has already done both. All
        this does is check the layout is the one the export promised, split the
        columns, and drop the padding rows.
        """
        arr = np.asarray(raw)
        if arr.ndim == 3 and arr.shape[0] == 1 and arr.shape[2] == END2END_COLUMNS:
            preds = arr[0].astype(np.float32)
        elif arr.ndim == 2 and arr.shape[1] == END2END_COLUMNS:
            preds = arr.astype(np.float32)          # a squeezed batch axis
        else:
            raise ValueError(self._layout_error(arr))

        boxes = preds[:, :4]
        scores = preds[:, 4]
        class_ids = preds[:, 5].astype(np.int64)

        # The padding rows. Filtering on the lowest enabled threshold rather
        # than on a row count is what makes this robust to an export with a
        # different max_det: the contract is "confidence-sorted, padded", not
        # "exactly 300 real detections".
        keep = scores >= self._floor
        return boxes[keep], scores[keep], class_ids[keep]

    def _layout_error(self, arr: np.ndarray) -> str:
        """Say which wrong export this is, because the shape identifies it."""
        shape = tuple(arr.shape)
        channels = 4 + len(self.vocabulary.names)
        one_to_many = len(shape) == 3 and shape[1] == channels
        detail = (
            f"That is the one-to-many head: (1, {channels}, 8400) raw xywh + "
            "class scores, which needs NMS and is not what this backend "
            "decodes.\n"
            "Re-export with nms=False:\n"
            "  python perception/tools/export_yolo26.py --download\n"
            if one_to_many
            else
            "Expected the end-to-end head's (1, max_det, 6).\n"
            "An export made with nms=True embeds NMS and emits a different "
            "layout again; this backend wants the NMS-free one.\n"
        )
        return (
            f"yolo26: model emitted {shape}, which this backend cannot decode.\n"
            + detail
            + "Decoding the wrong layout does not crash -- it returns boxes in "
            "the wrong places with confident-looking scores."
        )

    # -- inference ---------------------------------------------------------

    def detect(self, frame: np.ndarray) -> list[Detection]:
        blob, ratio, pad_x, pad_y = self._preprocess(frame)
        boxes, scores, class_ids = self._decode(self._load().run(blob))

        h, w = frame.shape[:2]
        out: list[Detection] = []
        for (x1, y1, x2, y2), score, class_id in zip(boxes, scores, class_ids):
            name = self.enabled.get(int(class_id))
            if name is None:                       # a class this site did not ask for
                continue
            if score < self._threshold_by_id[int(class_id)]:
                continue
            # Un-letterbox: remove the pad, then the resize.
            out.append(
                Detection(
                    x1=float(np.clip((x1 - pad_x) / ratio, 0, w)),
                    y1=float(np.clip((y1 - pad_y) / ratio, 0, h)),
                    x2=float(np.clip((x2 - pad_x) / ratio, 0, w)),
                    y2=float(np.clip((y2 - pad_y) / ratio, 0, h)),
                    score=float(score),
                    label=name,
                )
            )
        return out
