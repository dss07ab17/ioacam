"""Tests for the RTMPose backend.

Run with:  python3 perception/actions/test_rtmpose.py

Most of this needs no weights and no runtime. What is under test is the part
that is fiddly and that a model swap must not break: the crop geometry, the
SimCC decode, and the mapping back to frame coordinates. Those are where a
pose backend goes wrong silently -- every one of them can be off and still
return seventeen plausible-looking keypoints.

The last section runs the real network if this machine can: an exported
`.onnx` through onnxruntime for preference, otherwise the `.pth` through the
rebuilt architecture. It is skipped with a note rather than failing when
neither is present, because the checkpoint is a build artefact and CI will not
have it.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = HERE.parent / "models"
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from rtmpose import (  # noqa: E402
    BBOX_PADDING,
    RTMPoseEstimator,
    bbox_to_center_scale,
    crop_person,
    decode_simcc,
)
from posetube import PoseFrameRecord, PoseTubeExtractor  # noqa: E402

INPUT_W, INPUT_H = 192, 256
NUM_KEYPOINTS = 17
X_BINS, Y_BINS = 384, 512
SPLIT = 2.0

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"[{'PASS' if condition else 'FAIL'}] {name}")
    if not condition:
        failures.append(f"{name}: {detail}")


def synthetic_person(w=640, h=480):
    """A crude figure, drawn rather than photographed.

    Enough for the shape and range assertions this file makes, and -- as it
    turns out -- enough for the real network to find a whole skeleton on, which
    is what the plausibility check at the bottom relies on.
    """
    import cv2

    rng = np.random.RandomState(0)
    img = np.full((h, w, 3), 130, dtype=np.uint8)
    img = np.clip(img.astype(np.int16) + rng.randint(-8, 8, (h, w, 3)), 0, 255)
    img = img.astype(np.uint8)

    skin, shirt, trousers = (120, 150, 190), (150, 90, 60), (70, 70, 80)
    cv2.circle(img, (320, 120), 22, skin, -1)                 # head
    cv2.rectangle(img, (296, 145), (344, 250), shirt, -1)     # torso
    cv2.line(img, (300, 155), (270, 215), shirt, 12)          # arms
    cv2.line(img, (270, 215), (262, 265), skin, 10)
    cv2.line(img, (340, 155), (372, 215), shirt, 12)
    cv2.line(img, (372, 215), (380, 265), skin, 10)
    cv2.line(img, (308, 250), (302, 340), trousers, 16)       # legs
    cv2.line(img, (302, 340), (300, 425), trousers, 14)
    cv2.line(img, (332, 250), (338, 340), trousers, 16)
    cv2.line(img, (338, 340), (340, 425), trousers, 14)
    return img


PERSON_BOX = (255.0, 95.0, 390.0, 445.0)


# ----------------------------------------------------------------------
# Crop geometry -- what the network is shown
# ----------------------------------------------------------------------

(cx, cy), (bw, bh) = bbox_to_center_scale((100.0, 100.0, 200.0, 300.0))
check(
    "the crop rectangle keeps the box's centre",
    (cx, cy) == (150.0, 200.0),
    f"{(cx, cy)}",
)
check(
    "the crop rectangle is forced to 3:4",
    abs(bw / bh - INPUT_W / INPUT_H) < 1e-6,
    f"{bw}x{bh} is {bw / bh:.4f}, wanted 0.75 -- anything else stretches the "
    f"person, and limb geometry is exactly what the action model reads",
)
check(
    "the long side carries the 1.25 padding",
    abs(bh - 200.0 * BBOX_PADDING) < 1e-6,
    f"{bh}; RTMPose was trained on boxes with this much context",
)

# A wide box (person lying down, or a bad detection) must grow in height, not
# get squashed into the tall input.
_, (wide_w, wide_h) = bbox_to_center_scale((0.0, 0.0, 400.0, 100.0))
check(
    "a wide box grows vertically rather than being squashed",
    abs(wide_w / wide_h - 0.75) < 1e-6 and wide_w >= 400.0,
    f"{wide_w}x{wide_h}",
)


frame = np.zeros((480, 640, 3), dtype=np.uint8)
frame[200, 300] = (0, 0, 255)
crop, centre, scale = crop_person(frame, PERSON_BOX)
check(
    "the crop is exactly the network's input size",
    crop.shape == (INPUT_H, INPUT_W, 3),
    f"{crop.shape}",
)

# The crop is a scale-and-translate, so a frame point has one exact home in it.
px = (300 - centre[0]) * scale + INPUT_W / 2
py = (200 - centre[1]) * scale + INPUT_H / 2
patch = crop[int(py) - 1:int(py) + 2, int(px) - 1:int(px) + 2, 2]
check(
    "a known frame pixel lands where the affine says it should",
    patch.max() > 0,
    f"marked pixel not found near ({px:.1f}, {py:.1f}) in the crop",
)

# A box running off the edge of the frame is the normal case, not the corner
# case: the 1.25 padding pushes it off whenever someone stands near an edge.
edge_crop, _, _ = crop_person(frame, (-40.0, -30.0, 60.0, 220.0))
check(
    "a box overhanging the frame still yields a full-size crop",
    edge_crop.shape == (INPUT_H, INPUT_W, 3),
    "warpAffine zero-pads; a slice-and-resize would shrink the crop and shift "
    "every keypoint decoded from it",
)


# ----------------------------------------------------------------------
# SimCC decode -- argmax per axis, peak value as confidence
# ----------------------------------------------------------------------

def bins(peaks_x, peaks_y, amp_x=1.0, amp_y=1.0, sigma=4.0):
    """Gaussian bin scores peaked where we say, in the layout the head emits."""
    ax = np.arange(X_BINS, dtype=np.float32)
    ay = np.arange(Y_BINS, dtype=np.float32)
    sx = np.stack([
        a * np.exp(-((ax - p) ** 2) / (2 * sigma ** 2))
        for p, a in zip(peaks_x, np.broadcast_to(amp_x, len(peaks_x)))
    ])
    sy = np.stack([
        a * np.exp(-((ay - p) ** 2) / (2 * sigma ** 2))
        for p, a in zip(peaks_y, np.broadcast_to(amp_y, len(peaks_y)))
    ])
    return sx.astype(np.float32), sy.astype(np.float32)


want_x = np.arange(NUM_KEYPOINTS) * 17 + 40
want_y = np.arange(NUM_KEYPOINTS) * 23 + 60
sx, sy = bins(want_x, want_y)
coords, scores = decode_simcc(sx, sy, SPLIT)

check(
    "coordinates are the argmax bin divided by the split ratio",
    np.allclose(coords[:, 0], want_x / SPLIT) and np.allclose(coords[:, 1], want_y / SPLIT),
    f"{coords[:2].tolist()} vs {(want_x[:2] / SPLIT).tolist()}; the bins are "
    f"at twice the input resolution, and forgetting to divide scales every "
    f"keypoint by two without raising anything",
)
check(
    "the decoded score is the peak value",
    np.allclose(scores, 1.0),
    f"{scores[:3]}",
)

# One axis confident, the other not. The joint is only as well localised as its
# worse axis -- taking the larger peak would report confidence that is not there.
sx, sy = bins(want_x, want_y, amp_x=0.9, amp_y=0.3)
_, mixed = decode_simcc(sx, sy, SPLIT)
check(
    "the score is the weaker of the two axis peaks",
    np.allclose(mixed, 0.3, atol=1e-6),
    f"{mixed[:3]} -- an occluded limb is typically confident on one axis only",
)

sx, sy = bins(want_x, want_y, amp_x=1.4, amp_y=1.4)
_, hot = decode_simcc(sx, sy, SPLIT)
check(
    "an out-of-range peak is clamped to 1.0",
    np.all(hot <= 1.0) and np.allclose(hot, 1.0),
    f"{hot[:3]} -- posetube.py multiplies blob amplitude by this, so a value "
    f"above 1 would draw an uncertain joint brighter than a certain one",
)

sx, sy = bins(want_x, want_y, amp_x=-0.5, amp_y=-0.5)
_, rejected = decode_simcc(sx, sy, SPLIT)
check(
    "a joint the model rejects scores zero, not a negative",
    np.all(rejected == 0.0),
    f"{rejected[:3]}",
)


# ----------------------------------------------------------------------
# End to end against a stub session -- the contract, without a runtime
# ----------------------------------------------------------------------

class _FakeSession:
    """Emits SimCC bins peaked at fixed crop positions. Runs anywhere."""

    dynamic_batch = True

    def __init__(self, peaks_x, peaks_y, amp=1.0, x_bins=X_BINS, y_bins=Y_BINS):
        self.peaks_x, self.peaks_y, self.amp = peaks_x, peaks_y, amp
        self.x_bins, self.y_bins = x_bins, y_bins
        self.batches: list[int] = []

    def run(self, blob):
        self.batches.append(blob.shape[0])
        ax = np.arange(self.x_bins, dtype=np.float32)
        ay = np.arange(self.y_bins, dtype=np.float32)
        sx = np.stack([self.amp * np.exp(-((ax - p) ** 2) / 32.0) for p in self.peaks_x])
        sy = np.stack([self.amp * np.exp(-((ay - p) ** 2) / 32.0) for p in self.peaks_y])
        n = blob.shape[0]
        return (
            np.repeat(sx[None], n, axis=0).astype(np.float32),
            np.repeat(sy[None], n, axis=0).astype(np.float32),
        )


# Peak every keypoint at the centre of the crop; it must decode to the centre
# of the padded box in frame coordinates, whatever the box was.
centre_x = np.full(NUM_KEYPOINTS, X_BINS // 2)
centre_y = np.full(NUM_KEYPOINTS, Y_BINS // 2)
est = RTMPoseEstimator(
    model_path="unused.onnx", session=_FakeSession(centre_x, centre_y, amp=0.8)
)
img = synthetic_person()
out = est.estimate(img, {"trk-A": PERSON_BOX})
kpts = out["trk-A"]

check(
    "estimate returns one (17, 3) array per track",
    set(out) == {"trk-A"} and kpts.shape == (NUM_KEYPOINTS, 3),
    f"{ {k: v.shape for k, v in out.items()} }",
)
box_cx = (PERSON_BOX[0] + PERSON_BOX[2]) / 2
box_cy = (PERSON_BOX[1] + PERSON_BOX[3]) / 2
check(
    "a keypoint at the centre of the crop decodes to the centre of the box",
    np.allclose(kpts[:, 0], box_cx, atol=0.6) and np.allclose(kpts[:, 1], box_cy, atol=0.6),
    f"{kpts[0, :2]} vs ({box_cx}, {box_cy}); the frame mapping has to invert "
    f"the crop affine exactly, or every keypoint is offset by a constant "
    f"nothing downstream can see",
)
check(
    "the model's confidence reaches the caller unaltered",
    np.allclose(kpts[:, 2], 0.8, atol=1e-6),
    f"{kpts[:2, 2]} -- a backend that flattened this would silently delete the "
    f"uncertainty weighting that makes the heatmap action model work",
)

# The same peaks under a box twice the size must land in the same place: the
# mapping back is per-box, not a fixed constant.
big = (box_cx - 200, box_cy - 260, box_cx + 200, box_cy + 260)
big_kpts = est.estimate(img, {"trk-B": big})["trk-B"]
check(
    "the frame mapping tracks the box, not the frame",
    np.allclose(big_kpts[:, 0], box_cx, atol=0.6) and np.allclose(big_kpts[:, 1], box_cy, atol=0.6),
    f"{big_kpts[0, :2]}",
)

many = {f"trk-{i}": PERSON_BOX for i in range(10)}
session = _FakeSession(centre_x, centre_y)
batched = RTMPoseEstimator(model_path="unused.onnx", session=session, max_batch=8)
result = batched.estimate(img, many)
check(
    "every tracked person is estimated, in batched session calls",
    len(result) == 10 and session.batches == [8, 2],
    f"{len(result)} results in batches {session.batches}; top-down pose is one "
    f"forward pass per person, so batching is the difference between one "
    f"session call and ten on a busy frame",
)

degenerate = RTMPoseEstimator(
    model_path="unused.onnx", session=_FakeSession(centre_x, centre_y)
)
check(
    "a zero-area box yields no keypoints rather than an invented person",
    degenerate.estimate(img, {"trk-A": (10.0, 10.0, 10.0, 10.0)}) == {},
    "posetube already treats a missing track as a gap and reports it through "
    "coverage; keypoints for a crop of nothing would be worse than silence",
)
check("no boxes means no session call at all", degenerate.estimate(img, {}) == {})


# A graph with the wrong bin count must be rejected at the boundary. It cannot
# be caught later: the keypoints it produces are all in range and all wrong by
# a constant factor.
wrong = RTMPoseEstimator(
    model_path="unused.onnx",
    session=_FakeSession(centre_x, centre_y, x_bins=192, y_bins=256),
)
rejected_shape = False
try:
    wrong.estimate(img, {"trk-A": PERSON_BOX})
except ValueError as exc:
    rejected_shape = "x-bins" in str(exc)
check(
    "a model with the wrong number of bins is rejected at the boundary",
    rejected_shape,
    "a split-ratio mismatch does not crash -- it scales every keypoint by the "
    "ratio of the two bin counts and returns a perfectly plausible skeleton",
)


# ----------------------------------------------------------------------
# The seam: these keypoints have to build a pose tube
# ----------------------------------------------------------------------

def tube_for(confidence: float):
    est = RTMPoseEstimator(
        model_path="unused.onnx",
        session=_FakeSession(centre_x, centre_y, amp=confidence),
    )
    pex = PoseTubeExtractor(window_s=1.5, num_frames=24, heatmap_size=56)
    for i in range(20):
        box = (PERSON_BOX[0] + i, PERSON_BOX[1], PERSON_BOX[2] + i, PERSON_BOX[3])
        pex.push(
            PoseFrameRecord(
                timestamp_us=i * (1_000_000 // 15),
                keypoints=est.estimate(img, {"trk-A": box}),
                boxes={"trk-A": box},
            )
        )
    return pex.extract("trk-A", pex.buffer[-1].timestamp_us)


tube = tube_for(1.0)
check(
    "RTMPose output drops straight into the pose tube extractor",
    tube is not None and tube.heatmaps.shape == (17, 24, 56, 56),
    "the backend satisfies posetube.PoseEstimator, so it replaces the stub "
    "with no other change",
)

# Not an equality against the score: the gaussian is sampled on integer grid
# cells, so a joint between cells peaks below its own confidence. The property
# that matters is proportionality -- half the confidence, half the blob.
faint = tube_for(0.5)
check(
    "blob amplitude carries the model's confidence into the volume",
    abs(float(faint.heatmaps.max()) * 2.0 - float(tube.heatmaps.max())) < 1e-6
    and float(tube.heatmaps.max()) <= 1.0,
    f"{float(faint.heatmaps.max()):.4f} at 0.5 confidence against "
    f"{float(tube.heatmaps.max()):.4f} at 1.0 -- this proportionality is the "
    f"property the whole pose-to-heatmap design exists for",
)
check(
    "mean keypoint score is reported from the model, not assumed",
    abs(faint.mean_keypoint_score - 0.5) < 1e-3,
    f"{faint.mean_keypoint_score}",
)


# ----------------------------------------------------------------------
# The real network, if this machine has it
# ----------------------------------------------------------------------

def real_session_path():
    """Prefer the exported graph; fall back to the checkpoint through torch."""
    onnx = MODELS / "rtmpose_t.onnx"
    if onnx.exists():
        try:
            import onnxruntime  # noqa: F401

            return str(onnx), "onnxruntime"
        except ImportError:
            pass
    pth = sorted(MODELS.glob("rtmpose*.pth"))
    if pth:
        try:
            import torch  # noqa: F401

            return str(pth[0]), "torch (checkpoint, not the shipping path)"
        except ImportError:
            pass
    return None, None


model_path, runtime = real_session_path()
if model_path is None:
    print(
        "[SKIP] real weights -- no rtmpose_t.onnx with onnxruntime, and no "
        ".pth with torch.\n"
        "       Export with: python3 perception/tools/export_rtmpose.py"
    )
else:
    print(f"       running the real network via {runtime}")
    real = RTMPoseEstimator(model_path=model_path)
    real_kpts = real.estimate(synthetic_person(), {"trk-A": PERSON_BOX})["trk-A"]

    check(
        "the real model returns (17, 3)",
        real_kpts.shape == (NUM_KEYPOINTS, 3),
        f"{real_kpts.shape}",
    )
    check(
        "every score is in 0..1",
        bool(np.all(real_kpts[:, 2] >= 0.0) and np.all(real_kpts[:, 2] <= 1.0)),
        f"range {float(real_kpts[:, 2].min()):.3f}..{float(real_kpts[:, 2].max()):.3f}",
    )
    check(
        "every keypoint lands inside the padded crop region",
        bool(
            np.all(real_kpts[:, 0] > PERSON_BOX[0] - 60)
            and np.all(real_kpts[:, 0] < PERSON_BOX[2] + 60)
            and np.all(real_kpts[:, 1] > PERSON_BOX[1] - 60)
            and np.all(real_kpts[:, 1] < PERSON_BOX[3] + 60)
        ),
        f"x {real_kpts[:, 0].min():.0f}..{real_kpts[:, 0].max():.0f}  "
        f"y {real_kpts[:, 1].min():.0f}..{real_kpts[:, 1].max():.0f}",
    )

    # COCO order: 0 nose, 5/6 shoulders, 11/12 hips, 15/16 ankles. On a figure
    # standing up these are in a known vertical order, and they are only in it
    # if the crop, the decode and the mapping back are all right at once. A
    # transposed axis or a missed split ratio passes every check above and
    # fails this one.
    nose_y = real_kpts[0, 1]
    shoulder_y = real_kpts[[5, 6], 1].mean()
    hip_y = real_kpts[[11, 12], 1].mean()
    ankle_y = real_kpts[[15, 16], 1].mean()
    check(
        "the skeleton is anatomically ordered on a standing figure",
        nose_y < shoulder_y < hip_y < ankle_y,
        f"nose {nose_y:.0f}, shoulders {shoulder_y:.0f}, hips {hip_y:.0f}, "
        f"ankles {ankle_y:.0f}",
    )
    check(
        "the left/right shoulders are on opposite sides of the torso",
        real_kpts[5, 0] > real_kpts[6, 0],
        f"left {real_kpts[5, 0]:.0f}, right {real_kpts[6, 0]:.0f} -- COCO's "
        f"left is the subject's left, which is image-right for a facing camera",
    )

    # Confidence has to mean something, or the tube's amplitude weighting is
    # decoration. Noise is not a person and must not score like one.
    noise = np.random.RandomState(1).randint(0, 255, (480, 640, 3)).astype(np.uint8)
    noise_kpts = real.estimate(noise, {"trk-A": PERSON_BOX})["trk-A"]
    check(
        "a figure scores higher than noise",
        float(real_kpts[:, 2].mean()) > float(noise_kpts[:, 2].mean()) + 0.1,
        f"figure {float(real_kpts[:, 2].mean()):.3f} vs noise "
        f"{float(noise_kpts[:, 2].mean()):.3f}",
    )

    # The cv2.dnn fallback is only worth having if it is the same model. It is
    # a different implementation of the same graph, so this is a real check --
    # and it is the claim the "no new dependency" fallback rests on.
    onnx_path = MODELS / "rtmpose_t.onnx"
    if onnx_path.exists():
        cv2_kpts = RTMPoseEstimator(
            model_path=str(onnx_path), runtime="cv2"
        ).estimate(synthetic_person(), {"trk-A": PERSON_BOX})["trk-A"]
        check(
            "the cv2.dnn fallback decodes the same keypoints as onnxruntime",
            bool(
                np.allclose(cv2_kpts[:, :2], real_kpts[:, :2], atol=0.5)
                and np.allclose(cv2_kpts[:, 2], real_kpts[:, 2], atol=1e-3)
            ),
            f"max position drift {np.abs(cv2_kpts[:, :2] - real_kpts[:, :2]).max():.3f}px, "
            f"max score drift {np.abs(cv2_kpts[:, 2] - real_kpts[:, 2]).max():.5f}",
        )
    else:
        print("[SKIP] cv2.dnn cross-check -- no exported .onnx to compare with")


print()
if failures:
    print(f"{len(failures)} rtmpose test(s) failed")
    for f in failures:
        print(f"  !! {f}")
    raise SystemExit(1)
print("all rtmpose tests passed")
