#!/usr/bin/env python3
"""Convert a YOLO26 checkpoint to an NMS-free ONNX graph. Once, offline.

    python3 perception/tools/export_yolo26.py --download
    python3 perception/tools/export_yolo26.py \\
        --checkpoint perception/models/yolo26n.pt \\
        --out perception/models/yolo26n.onnx

## Licence: read this before running it

YOLO26 is an Ultralytics model. Its code **and its pretrained weights** are
AGPL-3.0, and exporting to ONNX does not change that -- the weights inside the
graph carry the same terms they did inside the `.pt`. Running this tool
downloads AGPL weights onto your machine and installs an AGPL package to do it.

That is lawful for the job this backend exists to do: private, internal
evaluation, which is not distribution and does not trigger §13. It is not
lawful for a delivered product without Ultralytics' commercial licence. The
full argument is in `perception/LICENCE-NOTES.md`; the default backend stays
`yolox-onnx`, Apache-2.0 end to end.

## Why an exporter at all, rather than fetch_model.py

Because Ultralytics publishes `.pt` checkpoints, not ONNX. `fetch_model.py`
downloads YOLOX's ONNX releases directly, so there is nothing to convert.

## Why `nms=False`, specifically

`model.export(format="onnx")` accepts three settings, and they produce three
different graphs. Only one of them is the model this repo wants:

    nms=False   the one-to-one head:  (1, 300, 6)  [x1 y1 x2 y2 conf cls]
                NMS-free, end to end. What detectors/yolo26.py decodes.
    nms=None    the one-to-many head: (1, 84, 8400) raw xywh + class scores.
                Needs NMS after it, which is the thing YOLO26 removed.
    nms=True    NMS baked into the graph as ONNX ops. Runs, but embeds exactly
                the operator RKNN-Toolkit2 is least likely to accept, and
                hides the threshold inside the file where no config can reach.

The default here is `nms=False`. The whole reason to look at YOLO26 rather
than YOLOX is that the NMS-free head removes a tuned threshold and a
board-hostile op from the pipeline; exporting it with NMS re-attached throws
that away and leaves only the licence problem.

## Why ONNX rather than shipping the .pt

Same reason as `export_rtmpose.py` and `export_posec3d.py`. RKNN-Toolkit2 takes
ONNX, so ONNX is the only format that can be tried on the RK3568 NPU without a
second conversion project -- and `perception/detectors/yolo26.py` deliberately
does not import `ultralytics`, so torch never reaches the inference path or the
board.

## Verification

The export is checked before it is called a success, and by more than "the file
exists". A graph that loads and returns 300 rows of plausible numbers is the
normal appearance of a wrong letterbox, a missing /255, or a decode reading the
one-to-many head -- none of which raise. So, with onnxruntime installed:

  1. the graph is loaded and its real input and output shapes printed;
  2. **decode check** -- the graph through `Yolo26Detector` (the shipped path)
     against the same graph through ultralytics' own postprocess, on a
     non-square frame, to sub-pixel tolerance;
  3. **export check** -- the graph through onnxruntime against the original
     `.pt` through torch, on a square frame, to a numeric tolerance.

Two checks rather than one because they fail for different reasons: (2) tests
the pre/post code in this repo with the arithmetic held constant, (3) tests the
conversion with the code held constant. See `verify` for why (3) uses a square
frame -- ultralytics letterboxes a `.pt` rectangularly and the exported graph's
input is static, so a non-square frame would compare two different inputs and
report a conversion failure that is really a preprocessing difference.

As the README puts it: a model that will not convert costs nothing to discover
now and everything in month four.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent          # perception/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

DEFAULT_VARIANT = "yolo26n"
DEFAULT_OUT = str(ROOT / "models" / "yolo26n.onnx")
IMGSZ = 640
MAX_DET = 300

# Variant, approximate ONNX size, note. Sizes are the .onnx, not the .pt.
VARIANTS = {
    "yolo26n": (10, "default: the only one worth trying on the board first"),
    "yolo26s": (38, "accuracy reference on a laptop"),
    "yolo26m": (78, "slower again; measure before assuming it is needed"),
    "yolo26l": (98, "not a realistic RK3568 candidate"),
    "yolo26x": (218, "not a realistic RK3568 candidate"),
}

_LICENCE_WARNING = """
+---------------------------------------------------------------------------+
|  AGPL-3.0 -- YOLO26 code and pretrained weights are Ultralytics' AGPL-3.0. |
|  An ONNX export does not relicense them. Internal evaluation only unless   |
|  you hold an Ultralytics commercial licence.                               |
|  See perception/LICENCE-NOTES.md                                           |
+---------------------------------------------------------------------------+
"""


def synthetic_frame(w: int = 1280, h: int = 720) -> np.ndarray:
    """A crude standing figure on a noisy ground, drawn rather than photographed.

    The comparison in `verify` is between two implementations on the same
    pixels, so the pixels do not have to contain a real person -- they only
    have to make the network produce *something*, and disagreement is
    detectable whether or not the box is over anything. Drawn rather than
    shipped so the check needs no test asset, and non-square so a letterbox
    error has somewhere to show up.
    """
    import cv2

    rng = np.random.RandomState(0)
    img = np.clip(
        np.full((h, w, 3), 130, dtype=np.int16) + rng.randint(-8, 8, (h, w, 3)),
        0, 255,
    ).astype(np.uint8)
    skin, shirt, trousers = (120, 150, 190), (150, 90, 60), (70, 70, 80)
    cv2.circle(img, (620, 190), 40, skin, -1)                    # head
    cv2.rectangle(img, (576, 235), (664, 420), shirt, -1)        # torso
    cv2.line(img, (584, 250), (532, 355), shirt, 22)             # arms
    cv2.line(img, (532, 355), (518, 445), skin, 18)
    cv2.line(img, (656, 250), (712, 355), shirt, 22)
    cv2.line(img, (712, 355), (726, 445), skin, 18)
    cv2.line(img, (598, 420), (588, 560), trousers, 28)          # legs
    cv2.line(img, (588, 560), (584, 690), trousers, 24)
    cv2.line(img, (642, 420), (652, 560), trousers, 28)
    cv2.line(img, (652, 560), (656, 690), trousers, 24)
    return img


def export(checkpoint: str, out_path: str, imgsz: int = IMGSZ,
           opset: int = 16, max_det: int = MAX_DET) -> str:
    """Run the Ultralytics exporter and put the result where the config expects.

    `ultralytics` is imported inside the function, and only here. Nothing under
    `perception/detectors/` may import it -- see that package's notes.
    """
    from ultralytics import YOLO

    model = YOLO(checkpoint)
    produced = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        # The whole point. See the module docstring: nms=None gives the
        # one-to-many head and nms=True bakes NMS into the graph.
        nms=False,
        max_det=max_det,
        # A fixed 1x3x640x640 input. Dynamic axes are a downstream liability
        # here: RKNN wants static shapes, and the detector letterboxes every
        # frame to one size anyway, so a dynamic axis buys nothing and costs
        # the conversion.
        dynamic=False,
        half=False,
        simplify=True,
        verbose=False,
    )

    produced = str(produced)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if os.path.abspath(produced) != os.path.abspath(out_path):
        os.replace(produced, out_path)
    return out_path


def describe(out_path: str) -> tuple[bool, int]:
    """Print the graph's real input/output shapes. Returns (looks right, max_det)."""
    import onnxruntime as ort

    session = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    inp = session.get_inputs()[0]
    outs = session.get_outputs()
    print(f"  input   {inp.name:>8}  {inp.shape}")
    for o in outs:
        print(f"  output  {o.name:>8}  {o.shape}")

    shape = outs[0].shape
    ok = len(outs) == 1 and len(shape) == 3 and shape[2] == 6
    if not ok:
        print(
            "  !! that is not the end-to-end head. (1, max_det, 6) is what "
            "detectors/yolo26.py decodes; anything else was exported with the "
            "wrong `nms` setting. Do not use this file.",
            file=sys.stderr,
        )
    return ok, (shape[1] if ok and isinstance(shape[1], int) else MAX_DET)


def _pairs(result, COCO_CLASSES):
    """Ultralytics Results -> [(label, score, [x1,y1,x2,y2])], confidence-sorted."""
    rows = [
        (COCO_CLASSES[int(c)], float(sc), [float(v) for v in b])
        for c, sc, b in zip(
            result.boxes.cls.tolist(),
            result.boxes.conf.tolist(),
            result.boxes.xyxy.tolist(),
        )
    ]
    return sorted(rows, key=lambda r: -r[1])


def _ours(out_path: str, frame, imgsz: int, conf: float):
    """The shipped runtime path: onnxruntime + numpy, no ultralytics."""
    from detectors.coco import COCO_CLASSES
    from detectors.yolo26 import Yolo26Detector

    dets = Yolo26Detector(
        model_path=out_path,
        input_size=(imgsz, imgsz),
        # Far below any config default on purpose. What is under test is
        # agreement on every box the model emits, including the weak ones; a
        # 0.30 threshold would hide a disagreement behind a filter.
        score_threshold=conf,
        classes=COCO_CLASSES,
    ).detect(frame)
    return sorted(
        [(d.label, d.score, [d.x1, d.y1, d.x2, d.y2]) for d in dets],
        key=lambda r: -r[1],
    )


def verify_decode(out_path: str, imgsz: int = IMGSZ, conf: float = 0.02) -> bool:
    """Our pre/post code against ultralytics' own, on the *same* graph.

    Identical weights and identical arithmetic on both sides, so the only thing
    that can differ is the code this repo wrote: the letterbox, the /255, the
    column layout, the un-letterbox. That is why the tolerance is sub-pixel
    rather than approximate -- anything above float noise here is a bug, not
    drift.

    The frame is deliberately non-square. Ultralytics splits a letterbox pad
    between both sides and YOLOX puts it all bottom-right; a decode that copied
    YOLOX's would be off by half the pad, which is 140 px vertically on 16:9 at
    640 -- and completely invisible on a square test frame.
    """
    from ultralytics import YOLO

    from detectors.coco import COCO_CLASSES

    frame = synthetic_frame(1280, 720)
    ours = _ours(out_path, frame, imgsz, conf)
    theirs = _pairs(
        YOLO(out_path).predict(
            frame, imgsz=imgsz, conf=conf, max_det=MAX_DET, verbose=False
        )[0],
        COCO_CLASSES,
    )

    print(f"  detections on 1280x720: ours {len(ours)}, ultralytics {len(theirs)}")
    if len(ours) != len(theirs):
        print(
            "  !! different number of detections from the same graph. That is a "
            "decode or threshold difference, not numeric drift.",
            file=sys.stderr,
        )
        return False
    if not ours:
        print(
            "  !! neither found anything, so nothing was actually compared. "
            "Re-run against a real frame before trusting this export.",
            file=sys.stderr,
        )
        return False

    worst_box = max(
        max(abs(a - b) for a, b in zip(o[2], t[2])) for o, t in zip(ours, theirs)
    )
    worst_score = max(abs(o[1] - t[1]) for o, t in zip(ours, theirs))
    mismatched = sum(o[0] != t[0] for o, t in zip(ours, theirs))
    print(f"    worst box disagreement:    {worst_box:.4f} px")
    print(f"    worst score disagreement:  {worst_score:.2e}")
    print(f"    labels differing:          {mismatched} / {len(ours)}")

    ok = worst_box <= 0.5 and worst_score <= 1e-4 and mismatched == 0
    if not ok:
        print(
            "  !! the shipped decode disagrees with ultralytics' postprocess on "
            "the same weights. A box offset near half the letterbox pad is a "
            "centred-vs-corner padding bug; a uniform score difference is a "
            "missing or doubled /255. Do not ship it.",
            file=sys.stderr,
        )
    return ok


def verify_export(checkpoint: str, out_path: str, imgsz: int = IMGSZ,
                  floor: float = 1e-3) -> bool:
    """The converted graph against the checkpoint, tensor to tensor.

    Same shape as the check in `export_rtmpose.py` and `export_posec3d.py`: one
    input, both runtimes, compare the raw output. Here that output is the
    (1, 300, 6) one-to-one head, so boxes, scores and class ids are all
    compared directly.

    ## Why this does not go through `model.predict()`

    Because YOLO26's `.pt` has **two heads, and predict uses the other one**.
    The checkpoint ships with `head.end2end = False`, so
    `YOLO(ckpt).predict(...)` runs the *one-to-many* head followed by NMS,
    exactly like YOLOv8. `export(nms=False)` turns `end2end` on and serialises
    the *one-to-one* head, which is the NMS-free model this repo wants and the
    only one `detectors/yolo26.py` decodes.

    They are two different predictors sharing a backbone, and they disagree.
    On this tool's synthetic frame the one-to-many head scores 0.187 where the
    one-to-one head scores 0.118, and it finds a third box the other does not.
    Neither is broken. But it means `yolo predict model=yolo26n.pt` is **not**
    a reference for this backend, and anyone A/B-ing YOLO26 against YOLOX that
    way is measuring a model they are not running. The comparison below flips
    `end2end` on the torch module first, so both sides are the same head.

    ## Why rows below `floor` are excluded

    The head always emits `max_det` rows and the tail is padding at ~1e-5
    confidence. Rows that close together reorder freely between two runtimes,
    which shows up as a 369 px "box disagreement" between two boxes that are
    both noise. The comparison stops where the padding starts.
    """
    import torch
    from ultralytics import YOLO

    from detectors.yolo26 import Yolo26Detector

    # Preprocess once, with the shipped code, and hand both runtimes the same
    # tensor. A preprocessing difference is verify_decode's job, not this one.
    blob, _, _, _ = Yolo26Detector(
        model_path=out_path, input_size=(imgsz, imgsz)
    )._preprocess(synthetic_frame(1280, 720))

    model = YOLO(checkpoint).model.eval().float()
    head = model.model[-1]
    head.end2end = True          # the one-to-one head -- see the docstring
    head.export = True           # return the tensor, not (tensor, feature dict)
    with torch.no_grad():
        reference = model(torch.from_numpy(blob))
    reference = (
        reference[0] if isinstance(reference, (list, tuple)) else reference
    ).numpy()[0]

    import onnxruntime as ort

    session = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    got = session.run(None, {session.get_inputs()[0].name: blob})[0][0]

    if reference.shape != got.shape:
        print(f"  !! torch emitted {reference.shape}, onnx {got.shape}",
              file=sys.stderr)
        return False

    keep = reference[:, 4] >= floor
    rows = int(keep.sum())
    if rows == 0:
        print(
            f"  !! no row scored above {floor}, so nothing was compared. "
            "Re-run against a real frame.",
            file=sys.stderr,
        )
        return False

    box_drift = float(np.abs(reference[keep, :4] - got[keep, :4]).max())
    score_drift = float(np.abs(reference[keep, 4] - got[keep, 4]).max())
    class_changed = int((reference[keep, 5] != got[keep, 5]).sum())
    print(f"  rows above {floor}: {rows} / {reference.shape[0]}  "
          f"(the rest is padding, excluded)")
    print(f"    box drift vs torch:        {box_drift:.4f} px")
    print(f"    score drift vs torch:      {score_drift:.2e}")
    print(f"    class ids changed:         {class_changed} / {rows}")

    ok = box_drift <= 1.0 and score_drift <= 1e-4 and class_changed == 0
    if not ok:
        print(
            "  !! the conversion changed the model's output. Try --opset 17, or "
            "re-export without simplify. Do not ship it.",
            file=sys.stderr,
        )
    return ok


def verify(checkpoint: str, out_path: str, imgsz: int = IMGSZ) -> bool:
    """Both checks. They fail for different reasons and neither covers the other.

    `verify_decode` holds the arithmetic constant and tests this repo's code.
    `verify_export` holds the code constant and tests the conversion. An export
    that passes one and fails the other is a real and distinguishable defect,
    which is the whole reason they are separate.
    """
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        print(
            "  !! onnxruntime not installed -- export written but NOT verified.\n"
            "     Install it and re-run with --verify-only before trusting this\n"
            "     graph. An unverified export is exactly the kind of thing that\n"
            "     runs and returns plausible nonsense.",
            file=sys.stderr,
        )
        return False

    print("  [1/2] shipped decode vs ultralytics' postprocess, same graph")
    decode_ok = verify_decode(out_path, imgsz)
    print("  [2/2] exported graph vs the checkpoint, same input tensor")
    export_ok = verify_export(checkpoint, out_path, imgsz)
    return decode_ok and export_ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", default=None,
                        help="a .pt checkpoint; defaults to perception/models/<variant>.pt")
    parser.add_argument("--variant", default=DEFAULT_VARIANT, choices=sorted(VARIANTS))
    parser.add_argument("--out", default=None)
    parser.add_argument("--imgsz", type=int, default=IMGSZ)
    parser.add_argument("--opset", type=int, default=16)
    parser.add_argument("--max-det", type=int, default=MAX_DET)
    parser.add_argument("--download", action="store_true",
                        help="let ultralytics fetch the checkpoint if it is absent")
    parser.add_argument("--verify-only", action="store_true",
                        help="check an existing .onnx against the checkpoint, "
                             "exporting nothing")
    args = parser.parse_args()

    checkpoint = args.checkpoint or str(ROOT / "models" / f"{args.variant}.pt")
    out_path = args.out or str(ROOT / "models" / f"{args.variant}.onnx")

    print(_LICENCE_WARNING, file=sys.stderr)

    if not os.path.exists(checkpoint):
        if not args.download:
            size_mb, note = VARIANTS[args.variant]
            print(
                f"checkpoint not found: {checkpoint}\n"
                f"  re-run with --download to let ultralytics fetch "
                f"{args.variant}.pt (~{size_mb} MB as ONNX -- {note})",
                file=sys.stderr,
            )
            return 1
        # Ultralytics resolves an unknown path whose *name* is one of its
        # release assets and downloads to that path -- so the checkpoint lands
        # in perception/models/ beside the ONNX, not in the repo root.
        print(f"fetching {args.variant}.pt -> {checkpoint}", file=sys.stderr)

    if not args.verify_only:
        print(f"exporting {checkpoint} (nms=False, imgsz={args.imgsz})", file=sys.stderr)
        try:
            export(checkpoint, out_path, imgsz=args.imgsz, opset=args.opset,
                   max_det=args.max_det)
        except ImportError as exc:
            print(
                f"export needs the ultralytics package: {exc}\n"
                "  pip install ultralytics   (AGPL-3.0; export only, never at "
                "run time -- see perception/LICENCE-NOTES.md)",
                file=sys.stderr,
            )
            return 1
        size_mb = os.path.getsize(out_path) / (1 << 20)
        print(f"saved {out_path}  ({size_mb:.1f} MB)")
        print(f"sha256 {hashlib.sha256(Path(out_path).read_bytes()).hexdigest()}")

    print("\ngraph:")
    shape_ok, max_det = describe(out_path)

    print("\nverification:")
    agrees = verify(checkpoint, out_path, imgsz=args.imgsz)

    print()
    print("contract for perception/detectors/yolo26.py:")
    print(f"  input    1x3x{args.imgsz}x{args.imgsz} float32, RGB, 0..1, "
          f"letterboxed with centred 114 padding")
    print(f"  output   (1, {max_det}, 6)  [x1, y1, x2, y2, score, class_id]")
    print("           xyxy in INPUT pixels, confidence-sorted, tail is padding")
    print("  decode   threshold and un-letterbox. No NMS: the head is one-to-one.")
    print()
    relative = os.path.relpath(out_path, ROOT.parent).replace(os.sep, "/")
    print("switch to it with:  \"detector\": { \"backend\": \"yolo26\", "
          f"\"model_path\": \"{relative}\" }}")
    print("  -- and read perception/LICENCE-NOTES.md before you do.")
    return 0 if (shape_ok and agrees) else 2


if __name__ == "__main__":
    raise SystemExit(main())
