"""Tests for object classes, object tracks, and wrist association.

Run with:  python3 perception/test_objects.py

The association rule is an inference, not a measurement: nothing in the frame
says a phone is in a hand. So what is under test is mostly the ways that
inference can be confidently wrong -- an object claimed by an unsure wrist, an
object attributed to a person standing near it, a flicker emitted as an event,
and a class list that silently detects nothing.

The emitted events are validated against `schema/event.schema.json` with the
repo's own validator, because an event the engine rejects is worse than no
event: it fails at the boundary of a different component, at a different time.
"""

from __future__ import annotations

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

from association import associate  # noqa: E402
from detectors.base import Detection  # noqa: E402
from detectors.yolox_onnx import CLASS_INDEX, COCO_CLASSES  # noqa: E402
from emit import EventEmitter  # noqa: E402
from validate_events import validate  # noqa: E402
from tracking import MultiClassTracker  # noqa: E402

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"[{'PASS' if condition else 'FAIL'}] {name}")
    if not condition:
        failures.append(f"{name}: {detail}")


SITE_CLASSES = ("cell phone", "laptop", "bottle", "cup", "backpack",
                "handbag", "book", "scissors", "knife")


def person(x=100.0, y=100.0, wrist_left=(0.0, 0.0, 0.0), wrist_right=(0.0, 0.0, 0.0)):
    """A COCO-17 keypoint array with only the wrists set."""
    kp = np.zeros((17, 3), dtype=np.float32)
    kp[:9] = (x, y, 0.9)          # head/torso, present but irrelevant here
    kp[9] = wrist_left
    kp[10] = wrist_right
    kp[11:] = (x, y + 200, 0.9)
    return kp


def obj(label="cell phone", box=(800.0, 400.0, 860.0, 500.0), score=0.6):
    return Detection(x1=box[0], y1=box[1], x2=box[2], y2=box[3],
                     score=score, label=label)


# ----------------------------------------------------------------------
# The class list is configuration
# ----------------------------------------------------------------------

check(
    "all nine configured classes exist in COCO",
    all(c in CLASS_INDEX for c in SITE_CLASSES),
    f"missing: {[c for c in SITE_CLASSES if c not in CLASS_INDEX]}",
)
check(
    "the COCO table is the model's own order",
    (COCO_CLASSES[0] == "person" and CLASS_INDEX["cell phone"] == 67
     and CLASS_INDEX["knife"] == 43 and len(COCO_CLASSES) == 80),
    f"person={CLASS_INDEX.get('person')}, cell phone="
    f"{CLASS_INDEX.get('cell phone')}, knife={CLASS_INDEX.get('knife')} -- "
    f"a wrong index does not fail, it detects the wrong object",
)

# The config file is where the list lives; a site changing it must not have to
# touch source. Verify the shipped config is coherent with the detector.
webcam = json.loads((ROOT / "perception" / "config" / "webcam.json").read_text())
configured = webcam["detector"]["classes"]
check(
    "the shipped config names the classes, and the code does not",
    "person" in configured and set(SITE_CLASSES) <= set(configured),
    f"{configured}",
)
check(
    "every configured class has a threshold of its own",
    set(webcam["detector"]["class_thresholds"]) >= set(configured),
    "one threshold cannot serve a person filling the frame and a phone 40px "
    "across",
)
check(
    "the sharpest classes are held to the highest bar",
    (webcam["detector"]["class_thresholds"]["knife"]
     > webcam["detector"]["class_thresholds"]["person"]),
    "COCO confuses knives with pens and cutlery; a false knife is the most "
    "expensive false positive in this list",
)


# ----------------------------------------------------------------------
# Objects get their own tracks
# ----------------------------------------------------------------------

tracker = MultiClassTracker()
phone = obj("cell phone", (800.0, 400.0, 860.0, 500.0))
laptop = obj("laptop", (700.0, 350.0, 1000.0, 600.0))

first = {t.detection.label: t.track_id for t in tracker.update([phone, laptop])}
second = {t.detection.label: t.track_id for t in tracker.update([phone, laptop])}
check(
    "an object keeps its id across frames",
    first == second and len(first) == 2,
    f"{first} then {second} -- without this one phone emits a new event every "
    f"frame and the engine cannot tell it is the same phone",
)

# The phone sits almost entirely inside the laptop box. One tracker over all
# classes would hand the phone's id to the laptop the moment the phone drops
# out, and the stream would show a phone becoming a laptop.
check(
    "a phone lying on a laptop does not inherit the laptop's id",
    first["cell phone"] != first["laptop"]
    and first["cell phone"].startswith("obj-cell_phone")
    and first["laptop"].startswith("obj-laptop"),
    f"{first}",
)

gone = MultiClassTracker(max_misses=2)
gone.update([phone])
for _ in range(4):
    gone.update([])            # the phone is put away
gone.update([phone])
check(
    "a tracker told about empty frames ages its tracks out",
    all(t.misses <= 2 for tr in gone.trackers.values() for t in tr.tracks),
    "a tracker never told about the frames where nothing was detected keeps a "
    "stale box alive forever",
)


# ----------------------------------------------------------------------
# Association: whose hand is it in
# ----------------------------------------------------------------------

# A wrist inside the (expanded) phone box.
holder = person(wrist_right=(830.0, 450.0, 0.8))
held = associate([("obj-cell_phone-0001", phone)], {"trk-0001": holder})[0]
check(
    "an object containing a confident wrist is attributed to that person",
    held.attributed and held.held_by == "trk-0001" and held.wrist == "right_wrist",
    f"{held}",
)

# The same person, standing near but not holding: wrists well away from the box.
bystander = person(wrist_right=(200.0, 450.0, 0.9), wrist_left=(180.0, 460.0, 0.9))
unheld = associate([("obj-cell_phone-0001", phone)], {"trk-0001": bystander})[0]
check(
    "an object nobody is holding is returned unattributed, not dropped",
    (not unheld.attributed) and unheld.held_by is None
    and unheld.label == "cell phone",
    f"{unheld} -- a knife on a bench with nobody near it is a real "
    f"observation, and often the more interesting one",
)

# The wrist is in the right place, but the pose model did not believe it.
unsure = person(wrist_right=(830.0, 450.0, 0.10))
low = associate([("obj-cell_phone-0001", phone)], {"trk-0001": unsure},
                min_wrist_score=0.30)[0]
check(
    "an unsure wrist cannot claim an object",
    not low.attributed,
    "an occluded wrist's position is a prediction, and attributing a knife to "
    "a prediction is the confident-wrong-answer failure this system exists to "
    "avoid",
)

# Two people, one holding. The nearer wrist wins.
near = person(wrist_right=(832.0, 452.0, 0.8))
far = person(wrist_right=(870.0, 505.0, 0.8))
contested = associate([("obj-cell_phone-0001", phone)],
                      {"trk-holder": near, "trk-other": far})[0]
check(
    "when two wrists qualify, the nearer one is chosen",
    contested.held_by == "trk-holder",
    f"{contested.held_by} at {contested.distance_px}px -- monocular "
    f"association with no depth; this is evidence, not proof",
)

# A wrist just outside the box still counts: a hand occludes what it holds.
just_outside = person(wrist_right=(795.0, 395.0, 0.8))
margin_hit = associate([("obj-cell_phone-0001", phone)], {"trk-0001": just_outside})[0]
check(
    "a wrist just outside a small object's box still counts as holding it",
    margin_hit.attributed,
    "the wrist sits behind the hand, and the hand occludes the phone -- a "
    "strict containment test fires only on ideal footage",
)

# But not arbitrarily far outside.
way_outside = person(wrist_right=(600.0, 395.0, 0.8))
too_far = associate([("obj-cell_phone-0001", phone)], {"trk-0001": way_outside})[0]
check(
    "the margin does not stretch across the frame",
    not too_far.attributed,
    f"{too_far.distance_px}px away was still attributed",
)

multi = associate(
    [("obj-cell_phone-0001", phone), ("obj-laptop-0001", laptop)],
    {"trk-0001": holder},
)
check(
    "every object is reported once, held or not",
    len(multi) == 2 and sum(h.attributed for h in multi) >= 1,
    f"{[(h.label, h.held_by) for h in multi]}",
)


# ----------------------------------------------------------------------
# The events the engine would receive
# ----------------------------------------------------------------------

schema = json.loads((ROOT / "schema" / "event.schema.json").read_text())
emitter = EventEmitter(sensor_id="cam-01")


def build(held_obj, persistence=1.0):
    return emitter.build(
        observation="object_at_station",
        confidence=round(held_obj.score * persistence, 4),
        value=held_obj.label,
        track_id=held_obj.held_by,
        subject={"class": "object"},
    )


attributed_event = build(held)
unattributed_event = build(unheld)

for name, event in (("attributed", attributed_event),
                    ("unattributed", unattributed_event)):
    errors = validate(event, schema, None)
    check(
        f"the {name} event validates against the published schema",
        not errors,
        f"{errors}",
    )

check(
    "an attributed object carries the holder's track_id",
    attributed_event["track_id"] == "trk-0001"
    and attributed_event["value"] == "cell phone",
    f"{attributed_event}",
)
check(
    "an unattributed object omits track_id rather than inventing one",
    "track_id" not in unattributed_event
    and unattributed_event["value"] == "cell phone",
    f"{unattributed_event} -- the schema says absent means no tracked "
    f"subject; a placeholder id would be a subject the engine could "
    f"accumulate evidence against",
)

# Persistence has to reach the confidence, or a two-frame flicker and a solid
# ten-second observation are indistinguishable in the stream.
flicker = build(held, persistence=2 / 30)
check(
    "a flickering detection produces a low-confidence event",
    flicker["confidence"] < 0.1 < attributed_event["confidence"],
    f"flicker {flicker['confidence']} vs steady {attributed_event['confidence']}",
)


print()
if failures:
    print(f"{len(failures)} object test(s) failed")
    for f in failures:
        print(f"  !! {f}")
    raise SystemExit(1)
print("all object tests passed")
