#!/usr/bin/env python3
"""Tests for the PPE detector, its class list, and whose head the hat is on.

    python3 perception/test_ppe.py

No weights, no onnxruntime, no network: the detector takes an injected session
exactly as `Yolo26Detector` does, and the stage takes an injected detector. The
real graph is exercised at run time and its cost reported at shutdown.

Four things are under test, and each is a way to be wrong without an exception:

1. **The class list.** Ten names in training order, and the order IS the class
   id mapping. Index 2 is `NO-Hardhat`. A list that drifted by one would report
   every bare head as a mask and every mask as a bare head, confidently, with
   correct-looking boxes.

2. **The thresholds are asymmetric and stay that way.** The NO-* floors are low
   on purpose -- measured recall 0.55-0.62, and a missed bare head costs more
   than a false alarm. A change that flattened them back to one number would
   pass every other test in this file.

3. **Association, and its refusals.** A hat inside one person's box is theirs.
   A hat inside two indistinguishable boxes is nobody's -- but is still
   emitted, without a track_id, because a bare head we could not attribute is
   not the same fact as no bare head.

4. **Absence is not a violation.** No hat detection over a person emits
   nothing at all. Inferring a bare head from a missing hat would turn every
   person who turned away from the camera into a safety finding.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
# perception/tools is a directory of scripts, not a package, so the validator
# is imported by path rather than as tools.validate_events -- which would
# resolve to the repo-root tools/ package instead and quietly fail.
sys.path.insert(0, str(HERE / "tools"))

import numpy as np  # noqa: E402

import perceive  # noqa: E402
from detectors import BACKENDS, PPE_BACKENDS  # noqa: E402
from detectors.base import Detection  # noqa: E402
from detectors.ppe import (  # noqa: E402
    DEFAULT_PPE_CLASSES, PPE, PPE_CLASSES, PPE_OBSERVATIONS, PpeDetector,
)
from emit import EventEmitter  # noqa: E402
from ppe_stage import PpeStage, attribute, containment  # noqa: E402
from validate_events import validate  # noqa: E402

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"[{'PASS' if condition else 'FAIL'}] {name}")
    if not condition:
        failures.append(f"{name}: {detail}")


def section(title: str) -> None:
    print(f"\n{title}")


FRAME_W, FRAME_H = 1280, 720
PPE_INPUT = 640


def frame() -> np.ndarray:
    """A non-square frame of nothing. The stub session decides what is found.

    Non-square on purpose: it is the only shape in which a letterbox mistake
    has anywhere to show up. On a square frame the pad is zero and a
    centred-vs-corner error is perfectly invisible.
    """
    return np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)


class ScriptedSession:
    """Twelve lines standing in for onnxruntime. Emits the end-to-end layout."""

    def __init__(self, rows, max_det: int = 300) -> None:
        self.input_shape = [1, 3, PPE_INPUT, PPE_INPUT]
        preds = np.zeros((max_det, 6), dtype=np.float32)
        # Confidence-sorted, exactly as the real head emits.
        for i, (class_id, score, box) in enumerate(
            sorted(rows, key=lambda r: -r[1])
        ):
            preds[i] = [*box, score, class_id]
        self.preds = preds[None]

    def run(self, blob):
        return self.preds


def to_model_space(box, detector):
    """Frame pixels -> the letterboxed 640 space the head emits in.

    Built from the detector's own reported mapping rather than recomputed here,
    so this helper cannot drift from the preprocess it is meant to invert.
    """
    _blob, ratio, pad_x, pad_y = detector._preprocess(frame())
    x1, y1, x2, y2 = box
    return (x1 * ratio + pad_x, y1 * ratio + pad_y,
            x2 * ratio + pad_x, y2 * ratio + pad_y)


def ppe_detector(scene=(), **kwargs):
    """A PpeDetector whose head returns `scene`, given in frame pixels."""
    probe = PpeDetector(model_path="unused", session=ScriptedSession([]), **kwargs)
    rows = [
        (PPE.index[label], score, to_model_space(box, probe))
        for label, score, box in scene
    ]
    return PpeDetector(
        model_path="unused", session=ScriptedSession(rows), **kwargs
    )


# ----------------------------------------------------------------------
section("the class list")

# Written out rather than derived, because the point of the test is that the
# tuple in ppe.py matches the model's own `names` metadata. Deriving it from
# the same tuple would test nothing.
check(
    "ten classes, in the checkpoint's training order",
    PPE_CLASSES == (
        "Hardhat", "Mask", "NO-Hardhat", "NO-Mask", "NO-Safety Vest",
        "Person", "Safety Cone", "Safety Vest", "machinery", "vehicle",
    ),
    f"got {PPE_CLASSES}",
)
check(
    "the class ids are the list positions, and NO-Hardhat is 2",
    PPE.index["Hardhat"] == 0 and PPE.index["NO-Hardhat"] == 2
    and PPE.index["Safety Vest"] == 7,
    f"got {PPE.index}",
)
check(
    "only the four classes with a schema observation are on by default",
    set(DEFAULT_PPE_CLASSES) == {
        "Hardhat", "NO-Hardhat", "Safety Vest", "NO-Safety Vest"
    },
    f"got {DEFAULT_PPE_CLASSES}",
)
check(
    "the model's own Person class is NOT enabled by default",
    "Person" not in DEFAULT_PPE_CLASSES,
    "two detectors emitting people would double every body in the frame",
)

# Masks are detectable and have nowhere in the schema to go. Squeezing them
# into ppe_glasses would fire engine rules written about eye protection.
check(
    "Mask and NO-Mask map to no observation rather than the nearest one",
    "Mask" not in PPE_OBSERVATIONS and "NO-Mask" not in PPE_OBSERVATIONS,
    f"got {sorted(PPE_OBSERVATIONS)}",
)
check(
    "the four mapped classes point at the two schema observations",
    PPE_OBSERVATIONS == {
        "Hardhat": ("ppe_helmet", True),
        "NO-Hardhat": ("ppe_helmet", False),
        "Safety Vest": ("ppe_vest", True),
        "NO-Safety Vest": ("ppe_vest", False),
    },
    f"got {PPE_OBSERVATIONS}",
)

schema = json.loads((ROOT / "schema" / "event.schema.json").read_text())
allowed = set(schema["properties"]["observation"]["enum"])
check(
    "every observation this stage can emit is in the schema's closed list",
    {obs for obs, _ in PPE_OBSERVATIONS.values()} <= allowed,
    f"{{obs for obs, _ in PPE_OBSERVATIONS.values()}} - {allowed}",
)

try:
    ppe_detector(classes=["person"])
    check("a COCO name is rejected", False, "it was accepted")
except ValueError as exc:
    check(
        "a COCO name is rejected, and the message lists the real ones",
        "not PPE classes" in str(exc) and "NO-Hardhat" in str(exc),
        str(exc),
    )

# ----------------------------------------------------------------------
section("per-class thresholds")

det = ppe_detector()
check(
    "the NO-* floors default well below the positive ones",
    det.class_thresholds["NO-Hardhat"] < det.class_thresholds["Hardhat"]
    and det.class_thresholds["NO-Safety Vest"] < det.class_thresholds["Safety Vest"],
    f"got {det.class_thresholds}",
)
check(
    "and the NO-* floors are low in absolute terms, not merely lower",
    det.class_thresholds["NO-Hardhat"] <= 0.25
    and det.class_thresholds["NO-Safety Vest"] <= 0.25,
    f"got {det.class_thresholds} -- recall on these classes is 0.55-0.62",
)

# The whole trade, in one pair of assertions: the same score is a detection
# for the class where a miss is expensive and not for the class where it is not.
marginal = ppe_detector(scene=[
    ("NO-Hardhat", 0.25, (300.0, 100.0, 380.0, 160.0)),
    ("Hardhat", 0.25, (700.0, 100.0, 780.0, 160.0)),
])
labels = [d.label for d in marginal.detect(frame())]
check(
    "a 0.25 NO-Hardhat survives -- missing a bare head is the expensive error",
    "NO-Hardhat" in labels,
    f"got {labels}",
)
check(
    "the same 0.25 as a Hardhat does not",
    "Hardhat" not in labels,
    f"got {labels}",
)

override = ppe_detector(class_thresholds={"NO-Hardhat": 0.55})
check(
    "a site can raise a floor back up",
    override.class_thresholds["NO-Hardhat"] == 0.55,
    f"got {override.class_thresholds}",
)
try:
    ppe_detector(class_thresholds={"no-hardhat": 0.5})
    check("a misspelled threshold key is rejected", False, "it was accepted")
except ValueError as exc:
    check(
        "a misspelled threshold key is rejected rather than silently ignored",
        "would never be applied" in str(exc),
        str(exc),
    )

# ----------------------------------------------------------------------
section("the decode is yolo26's, unchanged")

SCENE = [
    ("Hardhat", 0.83, (300.0, 120.0, 372.0, 176.0)),
    ("NO-Safety Vest", 0.31, (880.0, 240.0, 1000.0, 420.0)),
]
decoded = sorted(ppe_detector(scene=SCENE).detect(frame()), key=lambda d: d.x1)
check("both scene boxes come back", len(decoded) == 2, f"got {decoded}")
if len(decoded) == 2:
    hat, vest = decoded
    check(
        "the letterbox is inverted to frame pixels, within a pixel",
        all(abs(a - b) < 1.0 for a, b in zip(
            (hat.x1, hat.y1, hat.x2, hat.y2), (300.0, 120.0, 372.0, 176.0)
        )),
        f"got {hat}",
    )
    check(
        "labels come from the PPE list, not COCO's",
        (hat.label, vest.label) == ("Hardhat", "NO-Safety Vest"),
        f"got {(hat.label, vest.label)}",
    )
    check(
        "the score is the head's single confidence, not multiplied by anything",
        abs(hat.score - 0.83) < 1e-5,
        f"got {hat.score}",
    )

# The 300-row output is padding after the real rows, and padding is removed by
# the confidence floor rather than by a row count.
check(
    "the padded tail of the 300-row output is dropped",
    len(ppe_detector(scene=[("Hardhat", 0.9, (10.0, 10.0, 60.0, 50.0))])
        .detect(frame())) == 1,
    "padding rows leaked through as zero-area boxes",
)

wrong_head = PpeDetector(
    model_path="unused",
    session=type("S", (), {
        "input_shape": [1, 3, 640, 640],
        "run": lambda self, blob: np.zeros((1, 14, 8400), dtype=np.float32),
    })(),
)
try:
    wrong_head.detect(frame())
    check("a one-to-many export is refused", False, "it decoded silently")
except ValueError as exc:
    check(
        "a one-to-many export is named as such, with THIS model's channel count",
        "one-to-many" in str(exc) and "(1, 14, 8400)" in str(exc),
        str(exc),
    )

# ----------------------------------------------------------------------
section("containment, not IoU")

PERSON = (300.0, 100.0, 500.0, 700.0)
HAT = (360.0, 110.0, 440.0, 165.0)

check(
    "a hat wholly inside a person box is fully contained",
    containment(HAT, PERSON) == 1.0,
    f"got {containment(HAT, PERSON)}",
)
# The reason the rule is containment: IoU answers a question nobody asked.
hat_area = (HAT[2] - HAT[0]) * (HAT[3] - HAT[1])
person_area = (PERSON[2] - PERSON[0]) * (PERSON[3] - PERSON[1])
check(
    "...while IoU against the same pair is under 0.05",
    hat_area / person_area < 0.05,
    f"got {hat_area / person_area:.3f} -- an IoU rule would reject every hat",
)
check(
    "a hat outside every person box is contained by none of it",
    containment((900.0, 110.0, 980.0, 165.0), PERSON) == 0.0,
    f"got {containment((900.0, 110.0, 980.0, 165.0), PERSON)}",
)
check(
    "a hat clipped by the person box's top edge still mostly counts",
    containment((360.0, 80.0, 440.0, 160.0), PERSON) > 0.70,
    f"got {containment((360.0, 80.0, 440.0, 160.0), PERSON)}",
)

# ----------------------------------------------------------------------
section("attribution, and its refusals")

near = {"trk-0001": PERSON}
track_id, cover, tied = attribute(HAT, near)
check("a hat in one person's box is that person's", track_id == "trk-0001",
      f"got {track_id}")
check("...reported with its containment", cover == 1.0, f"got {cover}")

track_id, cover, tied = attribute(HAT, {})
check("no people means no attribution", track_id is None, f"got {track_id}")
check("...and that is not an ambiguity", tied is False, f"got {tied}")

# A person standing further back, whose box also swallows the hat. The nearer
# (smaller-boxed) person is not the answer here -- the FAR one is smaller.
two = {
    "trk-near": PERSON,
    "trk-far": (280.0, 90.0, 560.0, 710.0),      # larger box, also contains it
}
track_id, cover, tied = attribute(HAT, two)
check(
    "with two containers, the smaller box wins",
    track_id == "trk-near",
    f"got {track_id}",
)

tied_pair = {
    "trk-a": (300.0, 100.0, 500.0, 700.0),
    "trk-b": (310.0, 102.0, 508.0, 700.0),       # same size, same containment
}
track_id, cover, tied = attribute(HAT, tied_pair)
check(
    "two indistinguishable candidates are a refusal, not a coin toss",
    track_id is None and tied is True,
    f"got track_id={track_id} tied={tied}",
)

# ----------------------------------------------------------------------
section("events")


def hat(box=(360.0, 110.0, 440.0, 165.0), label="Hardhat", score=0.8):
    return Detection(box[0], box[1], box[2], box[3], score, label)


def vest(box=(340.0, 220.0, 470.0, 400.0), label="Safety Vest", score=0.8):
    return Detection(box[0], box[1], box[2], box[3], score, label)


def stage(**cfg):
    """A stage over an injected detector, so nothing loads a graph."""
    return PpeStage(dict(cfg), detector=ppe_detector())


def emitter():
    return EventEmitter("cam-01", stream=io.StringIO())


BOXES = {"trk-0001": PERSON}

st, em = stage(), emitter()
_findings, events = st.process_detections([hat()], BOXES, em)
check("a hat over a person emits one event", len(events) == 1, f"got {events}")
if events:
    event = events[0]
    check("...as ppe_helmet", event["observation"] == "ppe_helmet", str(event))
    check("...with value true", event["value"] is True, str(event))
    check("...attributed to the wearer", event.get("track_id") == "trk-0001",
          str(event))
    check("...about a human", event.get("subject") == {"class": "human"},
          str(event))
    errors = validate(event, schema, None)
    check("...and it validates against the published schema", not errors,
          "; ".join(errors))

st, em = stage(), emitter()
_f, events = st.process_detections([hat(label="NO-Hardhat", score=0.4)], BOXES, em)
check(
    "a bare head emits ppe_helmet false, not silence",
    len(events) == 1 and events[0]["observation"] == "ppe_helmet"
    and events[0]["value"] is False,
    f"got {events}",
)
if events:
    errors = validate(events[0], schema, None)
    check("...and that validates too", not errors, "; ".join(errors))

st, em = stage(), emitter()
_f, events = st.process_detections([vest(label="NO-Safety Vest", score=0.4)],
                                   BOXES, em)
check(
    "a missing vest emits ppe_vest false",
    len(events) == 1 and events[0]["observation"] == "ppe_vest"
    and events[0]["value"] is False,
    f"got {events}",
)

# The one that matters most: nothing detected about a head is not a bare head.
st, em = stage(), emitter()
_f, events = st.process_detections([], BOXES, em)
check(
    "a person with no hat detection at all emits nothing",
    events == [],
    f"got {events} -- absence of evidence is not evidence of a bare head",
)

st, em = stage(loose_frames=1), emitter()
_f, events = st.process_detections([hat(box=(900.0, 110.0, 980.0, 165.0))],
                                   BOXES, em)
check(
    "a hat over nobody still emits",
    len(events) == 1,
    f"got {events} -- dropping it would hide an unattributable bare head",
)
if events:
    check(
        "...with no track_id rather than an invented one",
        "track_id" not in events[0],
        str(events[0]),
    )
    errors = validate(events[0], schema, None)
    check("...and it validates without one", not errors, "; ".join(errors))

# ----------------------------------------------------------------------
section("Hardhat and NO-Hardhat on the same person")

# Each score is compared against its OWN floor, so the low NO-* floor does its
# work in the disagreement as well as in the detection.
st, em = stage(), emitter()
_f, events = st.process_detections(
    [hat(score=0.45), hat(box=(362.0, 112.0, 442.0, 167.0),
                          label="NO-Hardhat", score=0.30)],
    BOXES, em,
)
check(
    "a marginal Hardhat loses to a solid NO-Hardhat",
    len(events) == 1 and events[0]["value"] is False,
    f"got {events} -- 0.45/0.40 = 1.13 against 0.30/0.20 = 1.50",
)

st, em = stage(), emitter()
_f, events = st.process_detections(
    [hat(score=0.90), hat(box=(362.0, 112.0, 442.0, 167.0),
                          label="NO-Hardhat", score=0.21)],
    BOXES, em,
)
check(
    "but a confident Hardhat still beats a barely-passing NO-Hardhat",
    len(events) == 1 and events[0]["value"] is True,
    f"got {events} -- 0.90/0.40 = 2.25 against 0.21/0.20 = 1.05",
)
check(
    "one person and one observation produce one event, never two",
    len(events) == 1,
    f"got {events} -- the classes are mutually exclusive",
)

# ----------------------------------------------------------------------
section("edge triggering")

st, em = stage(), emitter()
_f, first = st.process_detections([hat()], BOXES, em)
_f, second = st.process_detections([hat()], BOXES, em)
_f, third = st.process_detections([hat()], BOXES, em)
check("the first frame of a compliant worker emits", len(first) == 1,
      f"got {first}")
check(
    "and the next two do not -- state, not a per-frame stream",
    second == [] and third == [],
    f"got {second} then {third}",
)

_f, flipped = st.process_detections(
    [hat(label="NO-Hardhat", score=0.5)], BOXES, em
)
check(
    "taking the hat off emits immediately",
    len(flipped) == 1 and flipped[0]["value"] is False,
    f"got {flipped}",
)

# A person the tracker drops for one frame is the same person. Their state has
# to outlast the gap, or every flicker re-emits a verdict that never changed.
st, em = stage(), emitter()
st.process_detections([hat()], BOXES, em)
st.process_detections([], {}, em)                  # one frame of occlusion
_f, back = st.process_detections([hat()], BOXES, em)
check(
    "a one-frame occlusion does not re-emit an unchanged verdict",
    back == [],
    f"got {back}",
)

# But once the person tracker has genuinely given up on the id, the state must
# go: ids get reissued, and the next worker would otherwise be silently assumed
# to be wearing whatever the last one wore.
st, em = stage(forget_after=2, loose_frames=1), emitter()
st.process_detections([hat()], BOXES, em)
for _ in range(4):
    st.process_detections([], {}, em)
_f, returned = st.process_detections([hat()], BOXES, em)
check(
    "a reissued track id does not inherit the last occupant's state",
    len(returned) == 1,
    f"got {returned} -- forget_after was exceeded, so this is a new worker",
)

# The person detector misses frames, and person_boxes_from_tracks only carries
# tracks matched this frame. A hat that never moved must not be re-announced as
# unattributed because its wearer's box blinked. Found by running the live
# demo: one dropped frame produced a duplicate pair of unattributed events.
st, em = stage(), emitter()
_f, first = st.process_detections([hat(label="NO-Hardhat", score=0.6)], BOXES, em)
findings, blink = st.process_detections(
    [hat(label="NO-Hardhat", score=0.6)], {}, em      # wearer's box drops out
)
check(
    "a one-frame dropout of the wearer does not re-announce the hat",
    len(first) == 1 and blink == [],
    f"got {first} then {blink}",
)
check(
    "...and the finding says it was held over rather than freshly seen",
    findings and findings[0].track_id == "trk-0001" and findings[0].held_over,
    f"got {findings}",
)

# But a wearer who is visible and simply does not contain the box is a real
# containment failure, and must be reported as one.
st, em = stage(loose_frames=1), emitter()
st.process_detections([hat(label="NO-Hardhat", score=0.6)], BOXES, em)
_f, moved = st.process_detections(
    [hat(box=(900.0, 110.0, 980.0, 165.0), label="NO-Hardhat", score=0.6)],
    BOXES, em,
)
check(
    "a hat that leaves a visible person's box does come loose",
    len(moved) == 1 and "track_id" not in moved[0],
    f"got {moved} -- holding this over would hide the rule's own error rate",
)

# And once the wearer has been gone longer than the grace window, so has the
# attribution: a hat left on a bench is nobody's.
st, em = stage(forget_after=2), emitter()
st.process_detections([hat(label="NO-Hardhat", score=0.6)], BOXES, em)
late = []
for _ in range(5):
    _f, batch = st.process_detections([hat(label="NO-Hardhat", score=0.6)], {}, em)
    late += batch
check(
    "a wearer gone past the grace window leaves the hat unattributed",
    len(late) == 1 and "track_id" not in late[0],
    f"got {late} -- exactly one, on the frame the hold-over lapsed",
)

# An un-attributable hat is keyed on its own track, or one bare head visible
# for a minute would emit an event on every frame of it.
st, em = stage(loose_frames=1), emitter()
loose = hat(box=(900.0, 110.0, 980.0, 165.0), label="NO-Hardhat", score=0.5)
_f, one = st.process_detections([loose], BOXES, em)
_f, two = st.process_detections([loose], BOXES, em)
check(
    "an unattributed finding emits once, not once per frame",
    len(one) == 1 and two == [],
    f"got {one} then {two}",
)

# A PPE box whose first frame lands on one the person detector missed has
# nothing to hold over from. Announcing it there reports an ownerless hard hat
# one frame before it attaches to the person wearing it. Found by running the
# live demo, then measuring: containment came back bimodal -- 0.98 to 1.00
# wherever a person box exists, 0.0 where none does -- so a lone unattributed
# frame is a detector gap, not a marginal association.
st, em = stage(loose_frames=3), emitter()
_f, gap = st.process_detections([hat(label="NO-Hardhat", score=0.6)], {}, em)
_f, attached = st.process_detections(
    [hat(label="NO-Hardhat", score=0.6)], BOXES, em
)
check(
    "an unattributed finding waits out the debounce before announcing",
    gap == [],
    f"got {gap}",
)
check(
    "...and attaches to its wearer on the next frame instead",
    len(attached) == 1 and attached[0].get("track_id") == "trk-0001",
    f"got {attached}",
)

# Delay, not suppression: a hat genuinely left on a bench is still reported.
st, em = stage(loose_frames=3), emitter()
bench = hat(box=(900.0, 110.0, 980.0, 165.0), label="NO-Hardhat", score=0.6)
announced = []
for _ in range(5):
    _f, batch = st.process_detections([bench], BOXES, em)
    announced += batch
check(
    "a genuinely ownerless hat is still announced, once, after the debounce",
    len(announced) == 1 and "track_id" not in announced[0],
    f"got {announced}",
)

# An attributed finding never waits. Delaying a non-compliance is the one cost
# this stage must not pay, so the debounce must not be able to reach it.
st, em = stage(loose_frames=10), emitter()
_f, immediate = st.process_detections(
    [hat(label="NO-Hardhat", score=0.6)], BOXES, em
)
check(
    "an attributed non-compliance emits on the first frame, debounce or not",
    len(immediate) == 1 and immediate[0]["value"] is False,
    f"got {immediate}",
)

# ----------------------------------------------------------------------
section("zone and confidence")

st, em = stage(), emitter()
_f, events = st.process_detections(
    [hat()], BOXES, em, zone_of=lambda tid: "zone-assembly-4"
)
check(
    "an attributed event carries the wearer's zone",
    events and events[0].get("zone_id") == "zone-assembly-4",
    f"got {events}",
)

st, em = stage(loose_frames=1), emitter()
_f, events = st.process_detections(
    [hat(box=(900.0, 110.0, 980.0, 165.0))], BOXES, em,
    zone_of=lambda tid: "zone-assembly-4",
)
check(
    "an unattributed one carries no zone, because it has no person",
    events and "zone_id" not in events[0],
    f"got {events}",
)

# Confidence is score * persistence, so a flicker is discounted. First frame of
# a new track is 1/1 seen, so drive it down with misses.
st, em = stage(), emitter()
for _ in range(8):
    st.process_detections([], BOXES, em)
    st.process_detections([hat()], BOXES, em)
    st.process_detections([hat(label="NO-Hardhat", score=0.8)], BOXES, em)
_f, events = st.process_detections([hat()], BOXES, em)
check(
    "a flickering detection is discounted below its raw score",
    not events or events[0]["confidence"] < 0.8,
    f"got {events}",
)

# A finding too weak to emit must not record itself as the announced state.
st, em = stage(min_confidence=0.50), emitter()
_f, suppressed = st.process_detections(
    [hat(label="NO-Hardhat", score=0.30)], BOXES, em
)
check(
    "a finding below min_confidence emits nothing",
    suppressed == [],
    f"got {suppressed}",
)
_f, later = st.process_detections([hat(label="NO-Hardhat", score=0.90)], BOXES, em)
check(
    "...and does not swallow the same verdict when it becomes confident",
    len(later) == 1 and later[0]["value"] is False,
    f"got {later} -- the suppressed frame recorded state it never announced",
)

# ----------------------------------------------------------------------
section("wiring")

check(
    "yolo26-ppe is a registered backend",
    "yolo26-ppe" in BACKENDS and "yolo26-ppe" in PPE_BACKENDS,
    f"got {BACKENDS} / {PPE_BACKENDS}",
)
check(
    "the person backends are not PPE backends",
    "yolox-onnx" not in PPE_BACKENDS and "yolo26" not in PPE_BACKENDS,
    f"got {PPE_BACKENDS}",
)
check(
    "ppe is off by default",
    perceive.DEFAULTS["ppe"]["enabled"] is False,
    f"got {perceive.DEFAULTS['ppe']}",
)
check(
    "the shipped defaults keep the NO-* floors low",
    perceive.DEFAULTS["ppe"]["class_thresholds"]["NO-Hardhat"] <= 0.25
    and perceive.DEFAULTS["ppe"]["class_thresholds"]["NO-Safety Vest"] <= 0.25,
    f"got {perceive.DEFAULTS['ppe']['class_thresholds']}",
)


class Args:
    objects = actions = ppe = False


args = Args()
check(
    "every stage stays off with an untouched config",
    perceive.resolve_stages(perceive.DEFAULTS, args) == (False, False, False, False),
    f"got {perceive.resolve_stages(perceive.DEFAULTS, args)}",
)

args.ppe = True
objects_on, pose_on, actions_on, ppe_on = perceive.resolve_stages(
    perceive.DEFAULTS, args
)
check("--ppe turns the stage on", ppe_on, "it did not")
check(
    "--ppe does NOT imply pose -- containment needs no keypoints",
    not pose_on,
    "an RTMPose pass per person would be bought for nothing",
)
check("--ppe leaves objects and actions alone",
      not objects_on and not actions_on, "one of them switched on")

cfg = perceive.merge(perceive.DEFAULTS, {"ppe": {"enabled": True}})
check(
    "config enables it too, not only the flag",
    perceive.resolve_stages(cfg, Args())[3],
    "config was ignored",
)

if failures:
    print(f"\n{len(failures)} FAILED:")
    for failure in failures:
        print(f"  - {failure}")
    sys.exit(1)
print("\nall PPE tests passed")
