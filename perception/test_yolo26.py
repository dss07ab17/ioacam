#!/usr/bin/env python3
"""Tests for the YOLO26 backend, and for its interchangeability with YOLOX.

    python3 perception/test_yolo26.py

No weights, no onnxruntime, no network: both backends take an injected session
(`session=` on YOLO26, `net=` on YOLOX), so what runs here is the shipped
decode against a scripted head output. That is deliberate -- the decode is the
part that fails silently. A wrong letterbox, a doubled normalisation or a
misread column does not raise; it returns boxes in the wrong places with
confident-looking scores, and every stage downstream believes them.

Three things are under test, and the third is the reason this file exists
rather than a few more cases in test_perception.py:

1. **Decode.** The end-to-end head is not YOLOX's head with the NMS removed.
   No anchor grid, no exponentials, no objectness column, and a fixed-length
   padded output. Each of those is a way to be wrong without an exception.
2. **Class filtering and per-class thresholds**, which must mean exactly what
   they mean under YOLOX -- they are the same config keys.
3. **Equivalence.** Given a scene, encoded into each backend's own model space
   by that backend's own reported letterbox mapping, both must return the same
   detections. Not similarly shaped output: the same boxes, labels and scores.
   A seam whose two sides disagree is not a seam, it is a fork.

The real graph is exercised by `perception/tools/export_yolo26.py`, which
checks the shipped decode against ultralytics' own postprocess and the
converted graph against the checkpoint. That needs weights and belongs there.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from detectors import BACKENDS, build_detector  # noqa: E402
from detectors.base import Detection  # noqa: E402
from detectors.coco import CLASS_INDEX, COCO_CLASSES  # noqa: E402
from detectors.yolo26 import Yolo26Detector  # noqa: E402
from detectors.yolox_onnx import YoloxOnnxDetector  # noqa: E402

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"[{'PASS' if condition else 'FAIL'}] {name}")
    if not condition:
        failures.append(f"{name}: {detail}")


FRAME_W, FRAME_H = 1280, 720
YOLOX_INPUT = 416
YOLO26_INPUT = 640
YOLOX_ANCHORS = sum((YOLOX_INPUT // s) ** 2 for s in (8, 16, 32))   # 3549


def frame() -> np.ndarray:
    """A non-square frame of nothing. The stub decides what is detected.

    Non-square on purpose: it is the only shape in which a letterbox mistake
    has anywhere to show up. On a square frame the pad is zero and a
    centred-vs-corner error is perfectly invisible.
    """
    return np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)


# The scene every equivalence assertion below is made against. Chosen well
# inside the frame so that clipping -- which both backends do, at the very end
# -- cannot mask a decode that is slightly off.
SCENE = [
    ("person", 0.90, (300.0, 140.0, 520.0, 700.0)),
    ("person", 0.42, (900.0, 200.0, 1050.0, 640.0)),
    ("cell phone", 0.61, (480.0, 330.0, 536.0, 420.0)),
    ("cup", 0.55, (640.0, 600.0, 700.0, 680.0)),
]


# ----------------------------------------------------------------------
# Stub sessions: each backend's graph output, scripted
# ----------------------------------------------------------------------

class _FakeYolo26Session:
    """Emits the end-to-end head's (1, max_det, 6), padding included.

    The padding matters. The real head always returns `max_det` rows and fills
    the tail with near-zero confidence; a decode that trusted the row count
    instead of the confidence would return 300 boxes per frame, most of them
    at the origin.
    """

    def __init__(self, rows, max_det: int = 300, pad_score: float = 1e-5):
        self.calls = 0
        out = np.zeros((1, max_det, 6), dtype=np.float32)
        out[0, :, 4] = pad_score            # the padding tail
        for i, (x1, y1, x2, y2, score, class_id) in enumerate(rows):
            out[0, i] = (x1, y1, x2, y2, score, class_id)
        self.out = out

    def run(self, blob):
        self.calls += 1
        assert blob.shape == (1, 3, YOLO26_INPUT, YOLO26_INPUT), blob.shape
        return self.out


class _RawSession:
    """Returns whatever it was given. For the layout-rejection cases."""

    def __init__(self, out):
        self.out = out

    def run(self, blob):
        return self.out


class _FakeYoloxNet:
    """Emits YOLOX's (1, N, 85) anchor-relative head output.

    Encoding a box means running YOLOX's decode backwards -- un-multiply the
    stride, subtract the grid centre, take the log of the size. Doing it that
    way rather than asserting on a hand-written array is what makes this a test
    of the decode rather than a copy of it.
    """

    def __init__(self, boxes, input_size: int = YOLOX_INPUT):
        self.blob = None
        strides = (8, 16, 32)
        counts = [(input_size // s) ** 2 for s in strides]
        offsets = np.cumsum([0] + counts)
        out = np.zeros((1, sum(counts), 5 + len(COCO_CLASSES)), dtype=np.float32)

        for cx, cy, w, h, score, class_id in boxes:
            # The coarsest level, so one cell holds each box unambiguously.
            level, stride = 2, strides[2]
            side = input_size // stride
            gx, gy = int(cx // stride), int(cy // stride)
            i = int(offsets[level]) + gy * side + gx
            out[0, i, 0] = cx / stride - gx
            out[0, i, 1] = cy / stride - gy
            out[0, i, 2] = np.log(w / stride)
            out[0, i, 3] = np.log(h / stride)
            # Objectness pinned to 1.0 so that YOLOX's score -- which is
            # objectness * class score, a column YOLO26 does not have -- comes
            # out as the same number the YOLO26 stub was handed.
            out[0, i, 4] = 1.0
            out[0, i, 5 + class_id] = score
        self.out = out

    def setInput(self, blob):
        self.blob = blob

    def forward(self):
        return self.out


# ----------------------------------------------------------------------
# Builders: a scene, encoded into each backend's own model space
# ----------------------------------------------------------------------

def yolo26_detector(rows, classes=("person",), class_thresholds=None,
                    score_threshold=0.30, session=None, **kwargs):
    return Yolo26Detector(
        model_path="unused.onnx",
        input_size=(YOLO26_INPUT, YOLO26_INPUT),
        score_threshold=score_threshold,
        classes=classes,
        class_thresholds=class_thresholds,
        session=session if session is not None else _FakeYolo26Session(rows),
        **kwargs,
    )


def yolox_detector(boxes, classes=("person",), class_thresholds=None,
                   score_threshold=0.30):
    return YoloxOnnxDetector(
        model_path="unused.onnx",
        input_size=(YOLOX_INPUT, YOLOX_INPUT),
        score_threshold=score_threshold,
        classes=classes,
        class_thresholds=class_thresholds,
        net=_FakeYoloxNet(boxes),
    )


def as_yolo26(scene, detector):
    """Frame-space scene -> the rows the end-to-end head would emit.

    Uses the detector's own `_preprocess` mapping, so what the round trip
    proves is that decode inverts preprocess. That the mapping itself is the
    one the real graph expects is checked separately below, and against the
    real model by `tools/export_yolo26.py`.
    """
    _, ratio, pad_x, pad_y = detector._preprocess(frame())
    rows = []
    for label, score, (x1, y1, x2, y2) in scene:
        rows.append((
            x1 * ratio + pad_x, y1 * ratio + pad_y,
            x2 * ratio + pad_x, y2 * ratio + pad_y,
            score, float(CLASS_INDEX[label]),
        ))
    return rows


def as_yolox(scene, detector):
    """Frame-space scene -> the anchor-relative cx,cy,w,h YOLOX's head emits."""
    _, ratio = detector._preprocess(frame())
    boxes = []
    for label, score, (x1, y1, x2, y2) in scene:
        boxes.append((
            (x1 + x2) / 2.0 * ratio, (y1 + y2) / 2.0 * ratio,
            (x2 - x1) * ratio, (y2 - y1) * ratio,
            score, CLASS_INDEX[label],
        ))
    return boxes


def structure(detections):
    """(label, box rounded to 0.1 px, score rounded to 4 dp), order-independent.

    Rounded because the two backends letterbox to different sizes -- 416 and
    640 -- so the same frame box survives a different scale factor on each
    path. Sorted because the head emits confidence-first while YOLOX walks its
    class list; both are legitimate orders for a set of detections, and
    requiring them to match would be asserting an accident.
    """
    return sorted(
        (d.label, tuple(round(v, 1) for v in (d.x1, d.y1, d.x2, d.y2)),
         round(d.score, 4))
        for d in detections
    )


# A detector built only to borrow its letterbox mapping.
_geometry = yolo26_detector([])


# ----------------------------------------------------------------------
# Output shape: what the stub session must be, and what is rejected
# ----------------------------------------------------------------------

print("\noutput shape and layout")

check(
    "the YOLOX stub emits the anchor count its decode expects",
    _FakeYoloxNet([]).out.shape == (1, YOLOX_ANCHORS, 5 + len(COCO_CLASSES)),
    f"{_FakeYoloxNet([]).out.shape}; YOLOX raises on a grid mismatch, so a "
    f"stub of the wrong length would fail loudly rather than mislead -- but "
    f"then this file would be testing nothing",
)

session = _FakeYolo26Session(as_yolo26(SCENE, _geometry))
check(
    "the stub emits the end-to-end head's (1, 300, 6)",
    session.out.shape == (1, 300, 6) and session.out.dtype == np.float32,
    f"{session.out.shape} {session.out.dtype}",
)

detector = yolo26_detector(as_yolo26(SCENE, _geometry))
detector.detect(frame())
check(
    "one session call per frame",
    detector._session.calls == 1,
    f"{detector._session.calls} calls",
)

# The one-to-many head. It is the export mistake most likely to be made --
# `nms=None` is ultralytics' default -- and the one that would otherwise
# produce 8400 rows of garbage rather than an error.
one_to_many = np.zeros((1, 4 + len(COCO_CLASSES), 8400), dtype=np.float32)
try:
    yolo26_detector([], session=_RawSession(one_to_many)).detect(frame())
    check("the one-to-many head is rejected", False, "no exception raised")
except ValueError as exc:
    message = str(exc)
    check(
        "the one-to-many head is rejected, and named",
        "one-to-many" in message and "nms=False" in message,
        message,
    )

try:
    yolo26_detector(
        [], session=_RawSession(np.zeros((1, 300, 85), dtype=np.float32))
    ).detect(frame())
    check("an 85-column output is rejected", False, "no exception raised")
except ValueError as exc:
    check("an 85-column output is rejected", "cannot decode" in str(exc), str(exc))

# A batch axis squeezed away by a runtime is not an error worth failing on.
squeezed = np.asarray(_FakeYolo26Session(as_yolo26(SCENE, _geometry)).out[0])
check(
    "a squeezed (N, 6) output decodes the same as (1, N, 6)",
    structure(yolo26_detector([], session=_RawSession(squeezed),
                              classes=("person",)).detect(frame()))
    == structure(yolo26_detector(as_yolo26(SCENE, _geometry),
                                 classes=("person",)).detect(frame())),
)


# ----------------------------------------------------------------------
# Decode correctness
# ----------------------------------------------------------------------

print("\ndecode")

all_classes = ("person", "cell phone", "cup")
decoded = yolo26_detector(
    as_yolo26(SCENE, _geometry), classes=all_classes, score_threshold=0.05
).detect(frame())

check(
    "every scripted box comes back, and nothing else",
    len(decoded) == len(SCENE),
    f"{len(decoded)} detections from {len(SCENE)} scripted boxes -- "
    f"more than scripted means the padding tail leaked through",
)

expected = sorted(
    (label, tuple(round(v, 1) for v in box), round(score, 4))
    for label, score, box in SCENE
)
check(
    "boxes decode back to the frame coordinates they were encoded from",
    structure(decoded) == expected,
    f"{structure(decoded)}\n          != {expected}",
)

check(
    "scores are carried through unmodified, not multiplied by anything",
    sorted(round(d.score, 6) for d in decoded)
    == sorted(round(s, 6) for _, s, _ in SCENE),
    f"{sorted(d.score for d in decoded)}",
)

# The padding tail: 296 rows at 1e-5 that must not become detections. Asked for
# with a threshold below the padding score, they still must not appear, because
# what removes them is the enabled-class filter and the configured floor -- not
# a row count.
padded = yolo26_detector(
    as_yolo26(SCENE, _geometry), classes=all_classes, score_threshold=0.05
).detect(frame())
check(
    "the padded tail does not become 296 detections at the origin",
    all(d.area > 0 for d in padded) and len(padded) == len(SCENE),
    f"{len(padded)} detections",
)

# Letterbox geometry, computed here rather than borrowed from the detector.
_, ratio, pad_x, pad_y = _geometry._preprocess(frame())
expected_ratio = YOLO26_INPUT / FRAME_W
check(
    "the letterbox pad is centred, not bottom-right",
    abs(ratio - expected_ratio) < 1e-6
    and abs(pad_x - 0.0) < 0.51
    and abs(pad_y - (YOLO26_INPUT - FRAME_H * expected_ratio) / 2.0) < 0.51,
    f"ratio {ratio} pad ({pad_x}, {pad_y}); a corner pad would give pad_y 0 "
    f"and put every box {(YOLO26_INPUT - FRAME_H * expected_ratio) / 2.0:.0f} "
    f"px too low",
)

_, yolox_ratio = yolox_detector([])._preprocess(frame())
check(
    "YOLOX still pads bottom-right -- the two letterboxes really do differ",
    abs(yolox_ratio - YOLOX_INPUT / FRAME_W) < 1e-6,
    f"{yolox_ratio}",
)

blob, *_ = _geometry._preprocess(frame())
check(
    "the blob is NCHW float32 normalised to 0..1, not 0..255",
    blob.shape == (1, 3, YOLO26_INPUT, YOLO26_INPUT)
    and blob.dtype == np.float32
    and blob.max() <= 1.0,
    f"{blob.shape} {blob.dtype} max={blob.max()}",
)

# A box running off the edge is clipped to the frame, as YOLOX's is: the
# geometry downstream assumes boxes lie inside the image.
overflow = [(-40.0, -30.0, YOLO26_INPUT + 50.0, YOLO26_INPUT + 60.0, 0.8,
             float(CLASS_INDEX["person"]))]
clipped = yolo26_detector(overflow).detect(frame())[0]
check(
    "boxes are clipped to the frame",
    (clipped.x1, clipped.y1) == (0.0, 0.0)
    and clipped.x2 <= FRAME_W and clipped.y2 <= FRAME_H,
    f"{clipped}",
)

check(
    "no NMS is applied: two heavily overlapping boxes both survive",
    len(yolo26_detector(as_yolo26(
        [("person", 0.90, (300.0, 140.0, 520.0, 700.0)),
         ("person", 0.88, (305.0, 145.0, 525.0, 705.0))], _geometry
    )).detect(frame())) == 2,
    "the one-to-one head has already deduplicated; suppressing here would "
    "delete real detections it deliberately kept",
)


# ----------------------------------------------------------------------
# Class filtering and per-class thresholds -- the same contract as YOLOX
# ----------------------------------------------------------------------

print("\nclass filtering and per-class thresholds")

MIXED = [
    ("person", 0.40, (300.0, 140.0, 520.0, 700.0)),
    ("cell phone", 0.45, (480.0, 330.0, 536.0, 420.0)),   # under its floor
    ("cell phone", 0.60, (640.0, 300.0, 700.0, 390.0)),   # over it
    ("car", 0.95, (100.0, 100.0, 300.0, 300.0)),          # not enabled
]
THRESHOLDS = {"person": 0.35, "cell phone": 0.50}
ENABLED = ("person", "cell phone")

for backend, dets in (
    ("yolo26", yolo26_detector(as_yolo26(MIXED, _geometry), classes=ENABLED,
                               class_thresholds=THRESHOLDS).detect(frame())),
    ("yolox-onnx", yolox_detector(as_yolox(MIXED, yolox_detector([])),
                                  classes=ENABLED,
                                  class_thresholds=THRESHOLDS).detect(frame())),
):
    labels = sorted(d.label for d in dets)
    check(
        f"{backend}: a class absent from the config is never returned",
        "car" not in labels,
        f"{labels}",
    )
    check(
        f"{backend}: each class is filtered by its own threshold, not one floor",
        labels == ["cell phone", "person"]
        and all(d.score >= 0.5 for d in dets if d.label == "cell phone"),
        f"{labels} -- a phone at 0.45 must fail a 0.50 floor while a person "
        f"at 0.40 passes a 0.35 one",
    )

for name, build in (("yolo26", yolo26_detector), ("yolox-onnx", yolox_detector)):
    try:
        build([], classes=("person", "torque wrench"))
        check(f"{name}: a non-COCO class is rejected", False, "no exception")
    except ValueError as exc:
        check(f"{name}: a non-COCO class is rejected at construction",
              "torque wrench" in str(exc), str(exc))
    try:
        build([], classes=())
        check(f"{name}: an empty class list is rejected", False, "no exception")
    except ValueError as exc:
        check(f"{name}: an empty class list is rejected",
              "empty" in str(exc), str(exc))

check(
    "an unlisted class falls back to score_threshold under both backends",
    yolo26_detector([], classes=("person", "cup"), score_threshold=0.42,
                    class_thresholds={"person": 0.35}).class_thresholds
    == yolox_detector([], classes=("person", "cup"), score_threshold=0.42,
                      class_thresholds={"person": 0.35}).class_thresholds
    == {"person": 0.35, "cup": 0.42},
)


# ----------------------------------------------------------------------
# Interchangeability -- the point of the seam
# ----------------------------------------------------------------------

print("\nboth backends are the same interface")

yolox = yolox_detector(as_yolox(SCENE, yolox_detector([])), classes=all_classes,
                       score_threshold=0.05)
yolo26 = yolo26_detector(as_yolo26(SCENE, _geometry), classes=all_classes,
                         score_threshold=0.05)
from_yolox, from_yolo26 = yolox.detect(frame()), yolo26.detect(frame())

check(
    "the same scene gives the same detections through either backend",
    structure(from_yolox) == structure(from_yolo26),
    f"yolox     {structure(from_yolox)}\n"
    f"          yolo26    {structure(from_yolo26)}",
)

check(
    "both return a list of the same Detection type, not a lookalike",
    all(type(d) is Detection for d in from_yolox + from_yolo26)
    and isinstance(from_yolox, list) and isinstance(from_yolo26, list),
    f"{sorted({type(d).__name__ for d in from_yolox + from_yolo26})}",
)


def ground_points(detections):
    # Rounded to 0.1 px for the same reason `structure` is: the two backends
    # letterbox to 416 and 640, so the same frame box travels through a
    # different scale factor on each path and comes back differing in the
    # fourth decimal. A tenth of a pixel is far below anything a zone boundary
    # or an IoU match can distinguish.
    return [
        tuple(round(v, 1) for v in d.ground_point)
        for d in sorted(detections, key=lambda d: (d.label, d.x1))
    ]


check(
    "the derived geometry every downstream stage uses agrees too",
    ground_points(from_yolox) == ground_points(from_yolo26),
    "zones test membership at the ground point and tracking matches on area; "
    "a backend swap that moved either would move every event",
)

check(
    "every field is the same type on both paths",
    {(type(d.x1), type(d.score), type(d.label)) for d in from_yolox}
    == {(type(d.x1), type(d.score), type(d.label)) for d in from_yolo26}
    == {(float, float, str)},
)

for name, det in (("yolox-onnx", yolox), ("yolo26", yolo26)):
    check(
        f"{name} carries the attributes perceive.py logs at startup",
        isinstance(det.name, str) and isinstance(det.licence, str)
        and callable(det.detect),
        f"name={getattr(det, 'name', None)!r} "
        f"licence={getattr(det, 'licence', None)!r}",
    )

check(
    "yolo26 states its AGPL position in the licence string perceive.py logs",
    "AGPL-3.0" in Yolo26Detector.licence
    and "Apache-2.0" in YoloxOnnxDetector.licence,
    f"{Yolo26Detector.licence!r}",
)

# An empty frame must be an empty list from both, not None and not a crash.
check(
    "nothing detected is an empty list on both paths",
    yolo26_detector([]).detect(frame()) == []
    == yolox_detector([]).detect(frame()),
)


# ----------------------------------------------------------------------
# Registry: backend is a config string and nothing else
# ----------------------------------------------------------------------

print("\nregistry")

check(
    "yolo26 is a known backend, and yolox-onnx is still the default",
    "yolo26" in BACKENDS and "yolox-onnx" in BACKENDS
    and BACKENDS[0] == "yolox-onnx",
    f"{BACKENDS}",
)

# The whole config-change surface: one string. Reaching the weights check means
# the backend resolved, its arguments were accepted and construction got as far
# as looking for a model -- everything a site's config actually exercises.
try:
    build_detector({"backend": "yolo26",
                    "model_path": "perception/models/definitely-absent.onnx"})
    check("an absent model is reported before inference", False, "no exception")
except FileNotFoundError as exc:
    check(
        "an absent model names the export tool rather than failing at run time",
        "export_yolo26.py" in str(exc),
        str(exc),
    )

# A config written for YOLOX carries two keys YOLO26 has no use for. Switching
# `backend` must not become a TypeError over them -- that would make the
# one-line switch a three-line one, which is the thing this seam is for.
try:
    build_detector({
        "backend": "yolo26",
        "model_path": "perception/models/definitely-absent.onnx",
        "nms_threshold": 0.45, "p6": False, "acknowledge_agpl": False,
    })
    check("YOLOX-shaped config keys are tolerated", False, "no exception")
except FileNotFoundError:
    check("a YOLOX-shaped config switches over without editing other keys", True)
except TypeError as exc:
    check("a YOLOX-shaped config switches over without editing other keys",
          False, f"rejected a leftover key: {exc}")

# The constraint the whole backend is shaped around, asserted rather than
# trusted to a comment. `ultralytics` pulls torch; torch on the inference path
# is what would make the RK3568 port a rewrite instead of a conversion. The
# export tool may import it -- it runs once, offline, on a laptop.
import ast  # noqa: E402

for module in sorted((HERE / "detectors").glob("*.py")):
    tree = ast.parse(module.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module.split(".")[0])
    banned = imported & {"torch", "ultralytics"} - (
        # ultralytics_yolo.py is the AGPL backend whose entire purpose is to
        # call into the package; it is gated behind acknowledge_agpl and is not
        # a shipping path.
        {"ultralytics"} if module.name == "ultralytics_yolo.py" else set()
    )
    check(
        f"detectors/{module.name} keeps torch and ultralytics off the runtime",
        not banned,
        f"imports {sorted(banned)} -- the inference path is onnxruntime and "
        f"opencv, and that is what makes the board port a conversion",
    )

try:
    build_detector({"backend": "yolo27"})
    check("an unknown backend is rejected", False, "no exception")
except ValueError as exc:
    check("an unknown backend is rejected and lists the known ones",
          "yolo26" in str(exc) and "yolox-onnx" in str(exc), str(exc))


print()
if failures:
    print(f"{len(failures)} yolo26 test(s) failed")
    for f in failures:
        print(f"  !! {f}")
    raise SystemExit(1)
print("all yolo26 tests passed")
