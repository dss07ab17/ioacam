"""Tests for the PoseC3D backend and the pose -> tube -> action chain.

Run with:  python3 perception/actions/test_posec3d.py

The parts that need no weights run everywhere: the class vocabulary, the volume
contract, and the chain from keypoints through the tube and the abstention
policy. The real network runs only if this machine has the exported graph.

What is worth testing here is not "does it return sixty numbers". It is that
the wiring cannot silently mislabel: a shuffled class list, a volume of the
wrong length, or a tube built from two different people all produce a confident
answer with no error anywhere.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = HERE.parent / "models"
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from base import AbstentionPolicy  # noqa: E402
from posec3d import NTU60_CLASSES, PoseC3DRecognizer  # noqa: E402
from posetube import PoseFrameRecord, PoseTubeExtractor  # noqa: E402

S = 1_000_000
failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"[{'PASS' if condition else 'FAIL'}] {name}")
    if not condition:
        failures.append(f"{name}: {detail}")


# ----------------------------------------------------------------------
# The vocabulary
# ----------------------------------------------------------------------

check(
    "the NTU-60 vocabulary has sixty distinct classes",
    len(NTU60_CLASSES) == 60 and len(set(NTU60_CLASSES)) == 60,
    f"{len(NTU60_CLASSES)} classes, {len(set(NTU60_CLASSES))} distinct",
)

# Order is the only thing mapping a logit index to a meaning. Three anchors
# from the published NTU A1-A60 list; if the tuple is ever re-sorted
# alphabetically or edited in the middle, every prediction becomes a different
# plausible label and nothing raises.
check(
    "the class order matches the NTU index it is decoded against",
    (NTU60_CLASSES[0] == "drink water"
     and NTU60_CLASSES[42] == "falling"
     and NTU60_CLASSES[59] == "walking apart from each other"),
    f"A1={NTU60_CLASSES[0]!r}, A43={NTU60_CLASSES[42]!r}, "
    f"A60={NTU60_CLASSES[59]!r}",
)

check(
    "the tube extractor's defaults still match the checkpoint",
    (PoseTubeExtractor().num_frames == 32
     and PoseTubeExtractor().heatmap_size == 56
     and PoseTubeExtractor().num_keypoints == 17),
    "conv1 is (32, 17, 1, 7, 7) and the model trained at clip_len 32 on "
    "56x56; a volume of another shape runs and returns nonsense because the "
    "head global-pools before the classifier",
)


# ----------------------------------------------------------------------
# The volume contract -- rejected at the boundary, not deep in the network
# ----------------------------------------------------------------------

class _FakeSession:
    """Emits logits peaked at one class, without a runtime."""

    def __init__(self, peak_index: int = 42, peak: float = 8.0, classes: int = 60):
        self.peak_index, self.peak, self.classes = peak_index, peak, classes
        self.calls = 0

    class _Input:
        name = "heatmap_volume"

    def get_inputs(self):
        return [self._Input()]

    def run(self, _outputs, feeds):
        self.calls += 1
        volume = next(iter(feeds.values()))
        assert volume.shape == (1, 17, 32, 56, 56), volume.shape
        logits = np.full((1, self.classes), -1.0, dtype=np.float32)
        logits[0, self.peak_index] = self.peak
        return [logits]


def recognizer_with(session, classes=NTU60_CLASSES):
    rec = PoseC3DRecognizer(model_path="unused.onnx", classes=classes, device="cpu")
    rec._session = session
    return rec


class _Volume:
    def __init__(self, shape):
        self.heatmaps = np.zeros(shape, dtype=np.float32)
        self.frames = None


rejected = False
try:
    recognizer_with(_FakeSession()).infer(_Volume((17, 24, 56, 56)))
except ValueError as exc:
    rejected = "expects" in str(exc)
check(
    "a volume of the wrong clip length is rejected before the model sees it",
    rejected,
    "24 frames instead of 32 pools to the same 512 features and classifies "
    "without complaint -- the error has to come from the boundary or not at all",
)

session = _FakeSession(peak_index=42)
scores = recognizer_with(session).infer(_Volume((17, 32, 56, 56)))
check(
    "a correctly shaped volume reaches the model and returns one score per class",
    session.calls == 1 and len(scores) == 60,
    f"{session.calls} call(s), {len(scores)} scores",
)
check(
    "logit index 42 decodes to the class named at that index",
    max(scores, key=scores.get) == NTU60_CLASSES[42] == "falling",
    f"{max(scores, key=scores.get)!r}",
)


# ----------------------------------------------------------------------
# The chain: keypoints -> tube -> recogniser -> abstention
# ----------------------------------------------------------------------

def run_chain(score=0.9, frames=30, track="trk-A", peak=8.0, policy=None):
    """Everything the live loop does, minus the camera."""
    extractor = PoseTubeExtractor()
    for i in range(frames):
        box = (100.0 + i, 150.0, 160.0 + i, 400.0)
        keypoints = np.zeros((17, 3), dtype=np.float32)
        for k in range(17):
            keypoints[k] = (box[0] + 30, box[1] + 15 * k, score)
        extractor.push(PoseFrameRecord(
            timestamp_us=i * (S // 15),
            keypoints={track: keypoints},
            boxes={track: box},
        ))
    now_us = extractor.buffer[-1].timestamp_us
    ready = extractor.tracks_ready(now_us)
    tube = extractor.extract(track, now_us) if track in ready else None
    if tube is None:
        return ready, None, None
    rec = recognizer_with(_FakeSession(peak_index=42, peak=peak))
    pol = policy or AbstentionPolicy(min_confidence=0.55, min_margin=0.15)
    return ready, tube, pol.decide(track, rec.infer(tube), backend=rec.name)


ready, tube, result = run_chain()
check(
    "a tracked person becomes a volume of the shape the model expects",
    tube is not None and tube.heatmaps.shape == (17, 32, 56, 56),
    f"{None if tube is None else tube.heatmaps.shape}",
)
check(
    "a confident, separated winner is a decision",
    result is not None and result.decided and result.label == "falling",
    f"{result}",
)
check(
    "the tube reports the window it was built over",
    tube.window_end_us > tube.window_start_us and tube.duration_s > 0,
    f"{tube.window_start_us}..{tube.window_end_us} -- an action finding is at "
    f"least one window old by construction, so the window has to travel with "
    f"the verdict or staleness is invisible",
)

# A flat output is the failure mode abstention exists for: the model has no
# class for what it just saw, so it returns its nearest with a mediocre score.
_, _, weak = run_chain(peak=0.2)
check(
    "an unfamiliar action abstains instead of reporting the nearest class",
    weak.abstained and weak.label is None,
    f"{weak.reason} -- on a prohibited-action list this is the difference "
    f"between a review and a silent miss",
)

# A track glimpsed in a few frames of a window is mostly interpolation, and
# classifying it would be classifying the extractor's own fill-in. Note the
# rule is about presence *within* the window, not about a short buffer: a
# track present in every frame of a two-frame buffer is fully covered.
sparse = PoseTubeExtractor()
for i in range(20):
    box = (100.0 + i, 150.0, 160.0 + i, 400.0)
    kp = np.zeros((17, 3), dtype=np.float32)
    for k in range(17):
        kp[k] = (box[0] + 30, box[1] + 15 * k, 0.9)
    # "ghost" appears in only 3 of the 20 frames -- a flicker of false
    # detections, which is exactly what the live run produced.
    people = {"solid": kp}
    if i % 7 == 0:
        people["ghost"] = kp
    sparse.push(PoseFrameRecord(i * (S // 15), people,
                                {t: box for t in people}))
ready_now = sparse.tracks_ready(sparse.buffer[-1].timestamp_us)
check(
    "a track present in only a few frames of the window is not classified",
    "solid" in ready_now and "ghost" not in ready_now,
    f"ready={ready_now} -- a flickering false detection must not get a verdict",
)


# ----------------------------------------------------------------------
# The real network, if this machine has it
# ----------------------------------------------------------------------

onnx_path = MODELS / "posec3d_pose_only.onnx"
try:
    import onnxruntime  # noqa: F401

    have_runtime = True
except ImportError:
    have_runtime = False

if not (onnx_path.exists() and have_runtime):
    print("[SKIP] real weights -- no posec3d_pose_only.onnx with onnxruntime.\n"
          "       Export with: python3 perception/tools/export_posec3d.py --download")
else:
    real = PoseC3DRecognizer(
        model_path=str(onnx_path), classes=NTU60_CLASSES, device="cpu"
    )
    _, real_tube, _ = run_chain()
    real_scores = real.infer(real_tube)

    check(
        "the exported graph returns a score for each of the sixty classes",
        len(real_scores) == 60 and set(real_scores) == set(NTU60_CLASSES),
        f"{len(real_scores)} scores",
    )
    check(
        "the scores are finite logits, not probabilities",
        all(np.isfinite(v) for v in real_scores.values())
        and abs(sum(real_scores.values()) - 1.0) > 1e-3,
        "AbstentionPolicy applies its own softmax with a calibration "
        "temperature; a graph that baked one in would hide the temperature "
        "it needs to divide by",
    )

    policy = AbstentionPolicy(min_confidence=0.55, min_margin=0.15)

    # An empty volume is the one case abstention genuinely covers: no keypoint
    # cleared the score threshold, so there is no evidence, and the logits come
    # out flat. This must stay true -- it is the path a fully occluded person
    # takes.
    empty = real.infer(_Volume((17, 32, 56, 56)))
    check(
        "the real model abstains when the volume carries no evidence at all",
        policy.decide("trk-A", empty, backend=real.name).abstained,
        f"top {max(empty, key=empty.get)!r}",
    )

    # And the limitation that matters, asserted rather than assumed. A closed
    # -set softmax over sixty classes is confidently wrong on input unlike
    # anything it trained on: uniform noise comes back as a named action at
    # ~1.0. Abstention catches an *unsure* model, not an out-of-distribution
    # input -- so it does not, on its own, make a prohibited-action list safe.
    # This check exists so that claim is never quietly assumed; if calibration
    # ever fixes it, this fails and someone updates it deliberately.
    noisy = _Volume((17, 32, 56, 56))
    noisy.heatmaps = np.random.RandomState(0).rand(17, 32, 56, 56).astype(np.float32)
    noise = real.infer(noisy)
    noise_decision = policy.decide("trk-A", noise, backend=real.name)
    check(
        "noise still produces a confident label -- abstention is not an OOD detector",
        noise_decision.decided,
        "if this now abstains, the model or the temperature changed; that is "
        "an improvement, but the README's abstention claim needs rewording "
        "rather than this test being deleted",
    )


print()
if failures:
    print(f"{len(failures)} posec3d test(s) failed")
    for f in failures:
        print(f"  !! {f}")
    raise SystemExit(1)
print("all posec3d tests passed")
