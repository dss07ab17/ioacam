"""Tests for the action recognition scaffolding.

Run with:  python3 perception/actions/test_actions.py

No model, no GPU, no video. What is under test is the part that is fiddly and
that a model swap must not break: tube geometry, the abstention rules, and
stream fusion.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from base import AbstentionPolicy, ActionScore  # noqa: E402
from fusion import StreamFusion  # noqa: E402
from stub import StubRecognizer  # noqa: E402
from posetube import (  # noqa: E402
    COCO_KEYPOINTS,
    PoseFrameRecord,
    PoseTubeExtractor,
    StubPoseEstimator,
)

S = 1_000_000
failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"[{'PASS' if condition else 'FAIL'}] {name}")
    if not condition:
        failures.append(f"{name}: {detail}")


def frames(n=30, w=640, h=480, start_us=0, step_us=S // 15, boxes_for=("trk-A",)):
    out = []
    for i in range(n):
        img = np.full((h, w, 3), 40, dtype=np.uint8)
        boxes = {}
        for tid in boxes_for:
            x = 100 + i * 3
            boxes[tid] = (x, 150.0, x + 60, 400.0)
        out.append(FrameRecord(start_us + i * step_us, img, boxes))
    return out


# ----------------------------------------------------------------------
# Abstention -- the rules that stop silent misses
# ----------------------------------------------------------------------

pol = AbstentionPolicy(min_confidence=0.55, min_margin=0.15)

clear = pol.decide("trk-A", {"tightening": 6.0, "reaching": 1.0, "standing": 0.5})
check("a clear winner is accepted", clear.decided and clear.label == "tightening", clear.reason)

low = pol.decide("trk-A", {"tightening": 1.0, "reaching": 0.9, "standing": 0.8})
check(
    "a weak best score abstains",
    low.abstained,
    "an unfamiliar action comes back as the nearest known class with a "
    "mediocre score; without this it would be silently recorded as normal",
)

# Isolate the margin rule: a low confidence floor so the threshold test cannot
# be what fires. Two classes split the mass almost evenly, so neither is a
# recognition -- the model is undecided, and picking the higher one would
# record a coin flip as a fact. On a real floor the confusable pairs are
# exactly the ones that matter (tightening vs inspecting).
margin_only = AbstentionPolicy(min_confidence=0.30, min_margin=0.15)
tie = margin_only.decide("trk-A", {"tightening": 5.0, "inspecting": 4.93, "standing": -5.0})
check(
    "two near-tied classes abstain on margin, even above the confidence floor",
    tie.abstained and "undecided" in tie.reason,
    tie.reason,
)

clear_margin = margin_only.decide("trk-A", {"tightening": 5.0, "inspecting": 2.0, "standing": -5.0})
check(
    "a clear margin passes the same policy",
    clear_margin.decided,
    clear_margin.reason,
)

per_class = AbstentionPolicy(
    min_confidence=0.55, min_margin=0.05, per_class={"tightening": 0.95}
)
strict = per_class.decide("trk-A", {"tightening": 3.0, "standing": 0.5})
check(
    "a per-class threshold can be stricter than the default",
    strict.abstained,
    "fine-grained classes are far less separable than coarse ones, and one "
    "global threshold either admits noise or discards good detections",
)

empty = pol.decide("trk-A", {})
check("no scores at all abstains", empty.abstained and empty.label is None)


# Temperature scaling must move confidence without changing the winner.
hot = AbstentionPolicy(min_confidence=0.0, min_margin=0.0, temperature=1.0)
cold = AbstentionPolicy(min_confidence=0.0, min_margin=0.0, temperature=4.0)
scores = {"a": 4.0, "b": 1.0, "c": 0.5}
check(
    "temperature softens confidence without changing the decision",
    hot.decide("t", scores).label == cold.decide("t", scores).label
    and cold.decide("t", scores).confidence < hot.decide("t", scores).confidence,
    "same correction as everywhere else: the number driving a response has to "
    "mean what it says",
)


# ----------------------------------------------------------------------
# Fusion
# ----------------------------------------------------------------------

fuse = StreamFusion()


def sc(label, conf, abstained=False, backend="x"):
    return ActionScore(
        track_id="trk-A", label=label, confidence=conf,
        abstained=abstained, backend=backend,
    )


agree = fuse.fuse(sc("tightening", 0.80, backend="rgb"), sc("tightening", 0.70, backend="skel"))
check(
    "agreement raises confidence above either stream's weighted blend",
    agree.decided and agree.label == "tightening" and agree.confidence > 0.76,
    f"{agree.confidence} -- {agree.reason}",
)

disagree = fuse.fuse(sc("tightening", 0.88, backend="rgb"), sc("reaching", 0.81, backend="skel"))
check(
    "outright disagreement abstains rather than picking the louder stream",
    disagree.abstained,
    "at least one stream is wrong and nothing says which; acting on the "
    "higher score makes two streams less reliable than one",
)

one_out = fuse.fuse(
    sc("tightening", 0.90, backend="rgb"),
    sc(None, 0.20, abstained=True, backend="skel"),
)
check(
    "one stream abstaining discounts the other rather than ignoring it",
    one_out.decided and one_out.confidence < 0.90,
    f"{one_out.confidence} -- a stream declining to answer is evidence "
    f"against the one that did, not neutral",
)

both_out = fuse.fuse(
    sc(None, 0.2, abstained=True, backend="rgb"),
    sc(None, 0.3, abstained=True, backend="skel"),
)
check("both abstaining stays abstained", both_out.abstained)

solo = fuse.fuse(sc("walking", 0.77, backend="rgb"), None)
check(
    "a single available stream passes through unmodified",
    solo.confidence == 0.77,
    "a fusion bonus from one opinion would be manufactured confidence",
)




# ----------------------------------------------------------------------
# Pose tubes -- the heatmap volume PoseC3D consumes
# ----------------------------------------------------------------------

pose_est = StubPoseEstimator(score=0.9)


def pose_frames(n=30, step_us=S // 15, track="trk-A", drift=3.0, score=0.9,
                boxes_for=("trk-A",)):
    est = StubPoseEstimator(score=score)
    out = []
    for i in range(n):
        img = np.full((480, 640, 3), 40, dtype=np.uint8)
        boxes = {t: (100 + i * drift, 150.0, 160 + i * drift, 400.0) for t in boxes_for}
        out.append(PoseFrameRecord(i * step_us, est.estimate(img, boxes), boxes, img))
    return out


pex = PoseTubeExtractor(window_s=1.5, num_frames=24, heatmap_size=56)
for r in pose_frames():
    pex.push(r)
pnow = pex.buffer[-1].timestamp_us
pt = pex.extract("trk-A", pnow)

check("a pose tube is extracted", pt is not None)
check(
    "the volume is keypoints x frames x H x W",
    pt.heatmaps.shape == (17, 24, 56, 56),
    str(pt.heatmaps.shape),
)
check(
    "heatmap values stay in 0..1",
    float(pt.heatmaps.min()) >= 0.0 and float(pt.heatmaps.max()) <= 1.0,
    f"{pt.heatmaps.min()}..{pt.heatmaps.max()}",
)
check(
    "every keypoint channel carries signal when pose is confident",
    all(pt.heatmaps[k].max() > 0.5 for k in range(17)),
    "a channel that is entirely zero means the keypoint was dropped",
)


# The property that makes heatmaps beat skeleton graphs on noisy pose: an
# uncertain joint must be FAINT, not indistinguishable from a certain one.
strong = PoseTubeExtractor(window_s=1.5, num_frames=8, heatmap_size=32)
for r in pose_frames(n=20, score=0.95):
    strong.push(r)
weak = PoseTubeExtractor(window_s=1.5, num_frames=8, heatmap_size=32)
for r in pose_frames(n=20, score=0.25):
    weak.push(r)
t_strong = strong.extract("trk-A", strong.buffer[-1].timestamp_us)
t_weak = weak.extract("trk-A", weak.buffer[-1].timestamp_us)
check(
    "low-confidence keypoints produce fainter blobs",
    t_weak.heatmaps.max() < t_strong.heatmaps.max() * 0.5,
    f"weak peak {t_weak.heatmaps.max():.3f} vs strong {t_strong.heatmaps.max():.3f}; "
    f"this is what lets the convolution downweight bad pose without any "
    f"special-casing",
)

# A keypoint the estimator did not find must contribute nothing at all,
# rather than asserting a joint position there is no evidence for.
dropped = PoseTubeExtractor(window_s=1.5, num_frames=8, heatmap_size=32,
                            min_keypoint_score=0.15)
for r in pose_frames(n=20, score=0.9):
    for k in r.keypoints.values():
        k[9, 2] = 0.02          # left wrist not found
    dropped.push(r)
t_drop = dropped.extract("trk-A", dropped.buffer[-1].timestamp_us)
check(
    "an undetected keypoint leaves its channel empty",
    float(t_drop.heatmaps[9].max()) == 0.0
    and float(t_drop.heatmaps[10].max()) > 0.5,
    f"wrist channel peak {t_drop.heatmaps[9].max()}",
)


# Scale invariance: the same action near and far must produce the same volume,
# or the model learns camera distance and fails at the next mounting height.
near = PoseTubeExtractor(window_s=1.5, num_frames=8, heatmap_size=32)
far = PoseTubeExtractor(window_s=1.5, num_frames=8, heatmap_size=32)
for i in range(20):
    img = np.full((480, 640, 3), 40, dtype=np.uint8)
    est = StubPoseEstimator(score=0.9)
    nb = {"t": (100.0, 100.0, 220.0, 460.0)}          # large
    fb = {"t": (300.0, 200.0, 330.0, 290.0)}          # same shape, small
    near.push(PoseFrameRecord(i * (S // 15), est.estimate(img, nb), nb, img))
    far.push(PoseFrameRecord(i * (S // 15), est.estimate(img, fb), fb, img))
tn = near.extract("t", near.buffer[-1].timestamp_us)
tf = far.extract("t", far.buffer[-1].timestamp_us)
check(
    "the same pose at different scales produces near-identical volumes",
    float(np.abs(tn.heatmaps - tf.heatmaps).max()) < 0.05,
    f"max difference {np.abs(tn.heatmaps - tf.heatmaps).max():.4f}; normalising "
    f"to the person's box rather than the frame is what buys this",
)


# One box across the window, not per frame: otherwise a walking person is
# centred in every slice and becomes indistinguishable from a standing one.
moving = PoseTubeExtractor(window_s=1.5, num_frames=12, heatmap_size=32)
for r in pose_frames(n=25, drift=8.0):
    moving.push(r)
still = PoseTubeExtractor(window_s=1.5, num_frames=12, heatmap_size=32)
for r in pose_frames(n=25, drift=0.0):
    still.push(r)
tm = moving.extract("trk-A", moving.buffer[-1].timestamp_us)
ts = still.extract("trk-A", still.buffer[-1].timestamp_us)
spread_moving = float(np.std(tm.heatmaps.sum(axis=(0, 2)), axis=0).mean())
spread_still = float(np.std(ts.heatmaps.sum(axis=(0, 2)), axis=0).mean())
check(
    "movement across the scene survives into the volume",
    spread_moving > spread_still,
    f"moving spread {spread_moving:.3f} vs still {spread_still:.3f}; a per-frame "
    f"box would centre the motion away and make walking look like standing",
)


# Gaps and readiness, same contract as the RGB tube.
gappy_pose = PoseTubeExtractor(window_s=1.5, num_frames=16, heatmap_size=32)
for i, r in enumerate(pose_frames(n=25)):
    if 8 <= i < 13:
        r.keypoints = {}
    gappy_pose.push(r)
tg = gappy_pose.extract("trk-A", gappy_pose.buffer[-1].timestamp_us)
check(
    "a pose track with gaps still yields a full-length volume",
    tg is not None and tg.heatmaps.shape[1] == 16 and tg.coverage < 1.0,
    f"coverage {tg.coverage if tg else None}",
)

sparse_pose = PoseTubeExtractor(window_s=1.5, num_frames=16, heatmap_size=32)
for i, r in enumerate(pose_frames(n=25, boxes_for=("trk-A", "trk-B"))):
    if i % 8 != 0:
        r.keypoints.pop("trk-B", None)
    sparse_pose.push(r)
ready_pose = sparse_pose.tracks_ready(sparse_pose.buffer[-1].timestamp_us)
check(
    "a barely-tracked pose is not offered for classification",
    "trk-A" in ready_pose and "trk-B" not in ready_pose,
    str(ready_pose),
)


# The RGB branch is opt-in, because it costs compute and is only needed for
# object-involving classes.
off = PoseTubeExtractor(window_s=1.5, num_frames=8, heatmap_size=32, keep_rgb=False)
on = PoseTubeExtractor(window_s=1.5, num_frames=8, heatmap_size=32,
                       keep_rgb=True, rgb_size=64)
for r in pose_frames(n=20):
    off.push(r); on.push(r)
t_off = off.extract("trk-A", off.buffer[-1].timestamp_us)
t_on = on.extract("trk-A", on.buffer[-1].timestamp_us)
check(
    "the RGB branch is off by default and carries frames when enabled",
    t_off.frames is None and t_on.frames is not None
    and t_on.frames.shape == (8, 64, 64, 3),
    f"{None if t_off.frames is None else t_off.frames.shape} / "
    f"{None if t_on.frames is None else t_on.frames.shape}",
)


# End to end: pose tube through the same abstention policy as the RGB path.
rec_pose = StubRecognizer(scores={"trk-A": {"walking": 6.0, "standing": 0.3}},
                          label="posec3d-stub")
res_pose = pol.decide(pt.track_id, rec_pose.infer(pt), backend=rec_pose.name)
check(
    "a pose tube runs through the same abstention policy as an RGB tube",
    res_pose.decided and res_pose.label == "walking",
    res_pose.reason,
)

check("COCO keypoint layout is the expected 17", len(COCO_KEYPOINTS) == 17)


# Defaults are read off the published pose_only checkpoint, not chosen. The
# network was fitted to them, so a drift here costs accuracy silently.
defaults = PoseTubeExtractor()
check(
    "tube defaults match the checkpoint's training config",
    defaults.num_frames == 32
    and defaults.heatmap_size == 56
    and defaults.num_keypoints == 17,
    f"clip_len={defaults.num_frames}, scale={defaults.heatmap_size}, "
    f"K={defaults.num_keypoints}; checkpoint says 32 / 56 / 17",
)


# A mismatched volume must fail loudly at the boundary, not deep inside the
# network where the message means nothing.
from posec3d import PoseC3DRecognizer  # noqa: E402


class _Shaped:
    heatmaps = np.zeros((17, 24, 56, 56), dtype=np.float32)
    frames = None


rejected = False
try:
    PoseC3DRecognizer(model_path="none.onnx", classes=["a"]).infer(_Shaped())
except ValueError as exc:
    rejected = "expects" in str(exc)
check(
    "a volume of the wrong shape is rejected at the boundary",
    rejected,
    "a silent shape mismatch either crashes obscurely or runs and returns "
    "nonsense",
)


# ----------------------------------------------------------------------
# End to end through the stub backend
# ----------------------------------------------------------------------

rec = StubRecognizer(scores={"trk-A": {"tightening": 6.0, "standing": 0.4}})
result = pol.decide(pt.track_id, rec.infer(pt), backend=rec.name)
check(
    "tube -> backend -> abstention produces a decided score",
    result.decided and result.label == "tightening",
    result.reason,
)

unknown = StubRecognizer(scores={"trk-A": {"standing": 1.0, "walking": 0.95, "reaching": 0.9}})
result = pol.decide(pt.track_id, unknown.infer(pt), backend=unknown.name)
check(
    "an action outside the vocabulary surfaces as abstention, not as a class",
    result.abstained,
    "this is the whole reason abstention exists: a prohibited-action list "
    "built on a model that cannot say 'unknown' only catches what it was "
    "already taught, and everything else reads as normal",
)


print()
if failures:
    print(f"{len(failures)} action test(s) failed")
    for f in failures:
        print(f"  !! {f}")
    raise SystemExit(1)
print("all action tests passed")
