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

from association import (  # noqa: E402
    AMBIGUOUS, BODY_MARGIN, OUT_OF_REACH, associate, body_scale,
)
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


def obj(label="cell phone", box=(800.0, 400.0, 860.0, 500.0), score=0.6):
    return Detection(x1=box[0], y1=box[1], x2=box[2], y2=box[3],
                     score=score, label=label)


# ----------------------------------------------------------------------
# Real geometry, copied out of the repo's own run logs
# ----------------------------------------------------------------------
#
# Every association case below is built from keypoints and boxes that a camera
# actually produced, not from round numbers chosen to sit either side of a
# threshold. Invented coordinates test the code against the author's mental
# model of a held phone; these test it against a held phone.
#
# Re-derive any of them with:
#     python3 perception/tools/diagnose_association.py logs/run.jsonl --verbose
#
# `assert_matches_log` below checks each fixture against the log it was taken
# from when that log is still present, so a transcription slip is caught rather
# than quietly becoming the new reference.

# logs/run.jsonl frame 180. A phone held up in the left hand. The wrist is
# 70.5 px outside the box -- 1.00 times the object's own longest side, against
# a configured object margin of 0.25 -- which is exactly why this run
# attributed nothing. The extrapolated hand anchor is 24.6 px out.
HELD_PHONE = {
    "log": "logs/run.jsonl", "frame": 180,
    "object": ("cell phone", 0.518, [824.0, 350.1, 881.3, 420.6]),
    "person_box": [303.4, 147.4, 1040.5, 716.9],
    "keypoints": [
        [647.9, 341.0, 0.787], [695.9, 300.2, 0.734], [614.3, 307.4, 0.826],
        [779.9, 326.6, 0.870], [590.3, 329.0, 0.734], [875.9, 525.7, 0.701],
        [535.2, 470.5, 0.719], [1003.1, 722.5, 0.639], [288.0, 566.5, 0.795],
        [931.1, 470.5, 0.640], [472.8, 348.2, 0.796], [799.1, 722.5, 0.175],
        [556.8, 712.9, 0.277], [856.7, 523.3, 0.219], [789.5, 559.3, 0.053],
        [890.3, 475.3, 0.141], [525.6, 249.8, 0.115],
    ],
}

# logs/run.jsonl frame 113. The same gesture, closer in: the extrapolated hand
# lands INSIDE the phone's box while the wrist is still 65.5 px outside it.
HELD_PHONE_CLOSE = {
    "log": "logs/run.jsonl", "frame": 113,
    "object": ("cell phone", 0.645, [756.3, 340.8, 848.6, 433.7]),
    "person_box": [357.2, 153.7, 1064.0, 715.8],
    "keypoints": [
        [568.0, 363.4, 0.669], [611.7, 315.1, 0.630], [542.7, 328.9, 0.782],
        [706.0, 322.0, 0.745], [531.2, 361.1, 0.692], [851.0, 501.5, 0.608],
        [492.1, 547.5, 0.678], [1014.3, 717.7, 0.687], [379.3, 726.9, 0.666],
        [848.7, 499.2, 0.736], [425.3, 554.4, 0.706], [827.9, 733.8, 0.247],
        [551.9, 729.2, 0.218], [825.6, 685.5, 0.074], [772.7, 660.2, 0.055],
        [722.1, 494.6, 0.042], [503.6, 478.5, 0.139],
    ],
}

# logs/objects-run2.jsonl frame 39. A cup standing on the desk in front of a
# seated person -- the negative case, and the one that decides whether the
# reach is honest or merely large. Both wrists are below threshold here, so the
# elbow fallback supplies the only anchors, and the nearest is 148.5 px out.
DESK_CUP = {
    "log": "logs/objects-run2.jsonl", "frame": 39,
    "object": ("cup", 0.541, [609.5, 629.7, 774.3, 716.7]),
    "person_box": [427.4, 256.0, 1000.5, 720.0],
    "keypoints": [
        [678.5, 452.5, 0.884], [715.8, 411.5, 1.000], [635.6, 411.5, 1.000],
        [756.9, 424.6, 0.850], [581.5, 430.2, 0.808], [857.6, 551.4, 0.785],
        [497.6, 590.6, 0.810], [975.2, 738.0, 0.763], [471.5, 771.5, 0.720],
        [1070.3, 726.8, 0.283], [607.7, 769.7, 0.168], [833.4, 963.7, 0.308],
        [581.5, 963.7, 0.176], [1070.3, 691.3, 0.094], [766.2, 682.0, 0.087],
        [725.2, 963.7, 0.080], [594.6, 400.3, 0.152],
    ],
}

# logs/objects-run2.jsonl frame 12. Left wrist 0.244 -- below threshold, so it
# takes no part -- while the left elbow is 0.780. Real evidence of the exact
# situation the elbow fallback exists for.
WEAK_WRIST = {
    "log": "logs/objects-run2.jsonl", "frame": 12,
    "object": ("cell phone", 0.607, [567.2, 553.0, 731.9, 710.3]),
    "person_box": [430.1, 251.9, 1013.3, 720.0],
    "keypoints": [
        [689.4, 430.9, 0.935], [725.5, 392.9, 1.000], [649.5, 391.0, 0.980],
        [767.2, 415.7, 0.874], [590.7, 413.8, 0.859], [867.9, 558.1, 0.747],
        [503.4, 584.7, 0.821], [983.7, 740.3, 0.780], [474.9, 778.3, 0.701],
        [843.2, 723.3, 0.244], [583.1, 738.4, 0.470], [843.2, 970.1, 0.279],
        [600.2, 970.1, 0.180], [626.8, 723.3, 0.069], [640.1, 702.4, 0.099],
        [754.0, 567.6, 0.113], [630.6, 398.6, 0.112],
    ],
}

FIXTURES = (HELD_PHONE, HELD_PHONE_CLOSE, DESK_CUP, WEAK_WRIST)


def keypoints_of(fixture):
    return np.array(fixture["keypoints"], dtype=np.float32)


def scene(fixture, label=None):
    """(objects, poses, person_boxes) for one fixture, ready for associate()."""
    name, score, box = fixture["object"]
    detection = obj(label or name, tuple(box), score)
    return (
        [("obj-0001", detection)],
        {"trk-0001": keypoints_of(fixture)},
        {"trk-0001": tuple(fixture["person_box"])},
    )


def attribute(fixture, label=None, **kwargs):
    objects, poses, boxes = scene(fixture, label)
    return associate(objects, poses, person_boxes=boxes, **kwargs)[0]


def assert_matches_log(fixture) -> str:
    """Confirm a fixture is still a verbatim copy of the frame it names.

    Returns "" on a match, a reason on a mismatch, and "skipped" when the log
    is absent or has been replaced by a later capture -- a fresh recording must
    not fail the suite, but a typo in a transcribed coordinate must.
    """
    path = ROOT / fixture["log"]
    if not path.exists():
        return "skipped"
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("type") != "frame" or record.get("frame") != fixture["frame"]:
            continue
        label, _score, box = fixture["object"]
        logged = [o for o in record.get("objects", []) if o["label"] == label]
        people = record.get("people", [])
        if not logged or not people:
            return "skipped"
        if [round(v, 1) for v in logged[0]["box"]] != [round(v, 1) for v in box]:
            return f"box {logged[0]['box']} != fixture {box}"
        if people[0]["keypoints"] != fixture["keypoints"]:
            return "keypoints differ from the log"
        return ""
    return "skipped"


for fixture in FIXTURES:
    outcome = assert_matches_log(fixture)
    check(
        f"{fixture['log']} frame {fixture['frame']} is copied faithfully"
        + (" (log not present -- skipped)" if outcome == "skipped" else ""),
        outcome in ("", "skipped"),
        f"{outcome} -- a fixture that has drifted from its log is an invented "
        f"number wearing a real number's comment",
    )


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

# --- A phone at the distance actually observed MUST attribute -------------

held = attribute(HELD_PHONE)
check(
    "a phone at the distance actually observed in logs/run.jsonl attributes",
    held.attributed and held.held_by == "trk-0001",
    f"{held.report.outcome}: nearest anchor {held.report.nearest.anchor.name} "
    f"at {held.report.nearest.surface_px:.1f}px against a reach of "
    f"{held.report.nearest.reach_px:.1f}px -- this frame is the whole bug",
)
check(
    "it attributes via the extrapolated hand, not the wrist",
    held.anchor_kind == "hand",
    f"{held.anchor_kind} at {held.distance_px}px -- RTMPose has no finger "
    f"keypoints, so the grip is inferred from the forearm direction; if the "
    f"wrist alone were enough this bug would not have existed",
)

close = attribute(HELD_PHONE_CLOSE)
check(
    "the closer frame of the same gesture attributes too",
    close.attributed and close.distance_px == 0.0,
    f"{close.report.outcome} at {close.distance_px}px -- the extrapolated "
    f"hand lands inside the box here, while the wrist is 65.5px outside it",
)

# The regression guard. This asserts the SHAPE of the failure, so that a future
# change which fixes attribution by widening `margin` instead of by improving
# the geometry fails here: no object-box margin that admits this phone is also
# tight enough to reject the desk cup below.
wrist_candidates = [c for c in held.report.candidates
                    if c.anchor.kind == "wrist"
                    and c.anchor.person_id == held.held_by]
worst_wrist = min(c.surface_over_object for c in wrist_candidates)
check(
    "the observed wrist is further from the box than any sane object margin",
    worst_wrist > 0.9,
    f"the nearest real wrist is {worst_wrist:.2f} object-widths outside the "
    f"box; `objects.margin` was 0.25. Reaching that wrist by raising the "
    f"object margin needs ~4x, which also swallows anything on the desk",
)

# --- An object on a surface, far from any wrist, must NOT attribute -------

cup = attribute(DESK_CUP)
check(
    "a cup standing on the desk is not attributed to the person at it",
    not cup.attributed and cup.report.outcome == OUT_OF_REACH,
    f"{cup.report.outcome}, nearest {cup.report.nearest.anchor.name} at "
    f"{cup.report.nearest.surface_px:.1f}px vs reach "
    f"{cup.report.nearest.reach_px:.1f}px -- a reach loose enough to catch "
    f"this is loose enough to attribute the desk to whoever sits at it",
)
check(
    "an object nobody is holding is returned unattributed, not dropped",
    cup.held_by is None and cup.label == "cup" and cup.box[0] == 609.5,
    f"{cup} -- a knife on a bench with nobody near it is a real observation, "
    f"and often the more interesting one",
)

# The held phone and the desk cup are the two ends of the calibration, and the
# gap between them is what makes the reach a decision rather than a preference.
held_ratio = held.report.nearest.surface_px / held.report.body_scales["trk-0001"]
cup_ratio = cup.report.nearest.surface_px / cup.report.body_scales["trk-0001"]
check(
    "held and not-held are separated by a real margin, not a hair",
    held_ratio < BODY_MARGIN < cup_ratio and (cup_ratio - held_ratio) > 0.2,
    f"held phone at {held_ratio:.2f} x body scale, desk cup at "
    f"{cup_ratio:.2f}, boundary at {BODY_MARGIN} -- if these were adjacent, "
    f"the setting would be a coin toss dressed up as a threshold",
)

# --- Equidistant between two people: attributed to NEITHER ----------------

# The second person is the first reflected about the object's centre line, so
# the two are equidistant by construction while both remain a real pose. Real
# geometry, mirrored, rather than two invented wrists placed at equal offsets.
axis = (HELD_PHONE["object"][2][0] + HELD_PHONE["object"][2][2]) / 2.0


def mirrored(points):
    return np.array([[2 * axis - x, y, s] for x, y, s in points],
                    dtype=np.float32)


px1, py1, px2, py2 = HELD_PHONE["person_box"]
twin_box = (2 * axis - px2, py1, 2 * axis - px1, py2)
objects, poses, boxes = scene(HELD_PHONE)
poses["trk-twin"] = mirrored(HELD_PHONE["keypoints"])
boxes["trk-twin"] = twin_box

tie = associate(objects, poses, person_boxes=boxes)[0]
check(
    "an object equidistant between two people is attributed to neither",
    not tie.attributed and tie.report.outcome == AMBIGUOUS,
    f"{tie.report.outcome} -> {tie.held_by} at "
    f"{tie.report.best.surface_px:.1f}px vs "
    f"{tie.report.runner_up.surface_px:.1f}px -- nearest-anchor between two "
    f"equally close people is a coin toss, and the engine accumulates "
    f"evidence against whichever track_id it lands on",
)

# Reversing the insertion order must not change the answer. A tie broken by
# dict ordering is a bug that only shows up once the tracker renumbers.
reversed_poses = {k: poses[k] for k in reversed(list(poses))}
tie_reversed = associate(objects, reversed_poses, person_boxes=boxes)[0]
check(
    "the tie is decided the same way whichever order the people arrive in",
    tie_reversed.report.outcome == tie.report.outcome
    and tie_reversed.held_by == tie.held_by,
    f"{tie.report.outcome}/{tie.held_by} became "
    f"{tie_reversed.report.outcome}/{tie_reversed.held_by}",
)

# Move the twin clearly further away and the tie resolves to the nearer one.
stepped_back = mirrored(HELD_PHONE["keypoints"])
stepped_back[:, 0] -= 120.0
contested = associate(
    objects,
    {"trk-0001": keypoints_of(HELD_PHONE), "trk-twin": stepped_back},
    person_boxes={"trk-0001": tuple(HELD_PHONE["person_box"]),
                  "trk-twin": (twin_box[0] - 120.0, twin_box[1],
                               twin_box[2] - 120.0, twin_box[3])},
)[0]
check(
    "once one person is clearly nearer, that person gets it",
    contested.attributed and contested.held_by == "trk-0001",
    f"{contested.report.outcome} -> {contested.held_by} -- the tie gate must "
    f"abstain on a genuine tie without abstaining on every contested object",
)

# --- A low-confidence wrist with a confident elbow uses the elbow ---------

# Real keypoints: logs/objects-run2.jsonl frame 12 has left_wrist at 0.244 and
# left_elbow at 0.780. The object box is placed on that elbow, because no log
# in the repo happens to contain an object there -- the pose is observed, the
# object position is the thing under test.
elbow_x, elbow_y, elbow_score = WEAK_WRIST["keypoints"][7]   # left_elbow
wrist_score = WEAK_WRIST["keypoints"][9][2]                   # left_wrist
at_elbow = obj("bottle", (elbow_x - 30, elbow_y - 40, elbow_x + 30, elbow_y + 40))
fallback = associate(
    [("obj-bottle-0001", at_elbow)],
    {"trk-0001": keypoints_of(WEAK_WRIST)},
    person_boxes={"trk-0001": tuple(WEAK_WRIST["person_box"])},
)[0]
check(
    "a low-confidence wrist with a confident elbow falls back to the elbow",
    (wrist_score < 0.30 <= elbow_score and fallback.attributed
     and fallback.wrist == "left_elbow" and fallback.anchor_kind == "elbow"),
    f"wrist={wrist_score} elbow={elbow_score} -> {fallback.report.outcome} "
    f"via {fallback.wrist} -- an occluded wrist should degrade the answer, "
    f"not erase it",
)

# But the fallback is a weaker claim, not a longer arm: the elbow must not be
# able to reach across the frame for the object the wrist could not.
unsure = keypoints_of(HELD_PHONE)
unsure[9][2] = 0.10                       # the wrist that was holding the phone
low = associate(
    [("obj-cell_phone-0001", obj(*HELD_PHONE["object"][:1],
                                 box=tuple(HELD_PHONE["object"][2])))],
    {"trk-0001": unsure},
    person_boxes={"trk-0001": tuple(HELD_PHONE["person_box"])},
)[0]
check(
    "an unsure wrist cannot claim an object, and the elbow cannot rescue it",
    not low.attributed,
    f"{low.report.outcome} -- an occluded wrist's position is a prediction, "
    f"and attributing a knife to a prediction is the confident-wrong-answer "
    f"failure this system exists to avoid; the elbow is a forearm away and "
    f"must only claim what the arm is folded around",
)

# --- Generic across classes, not tuned for phones -------------------------

by_class = {label: attribute(HELD_PHONE, label=label) for label in SITE_CLASSES}
check(
    "the same geometry attributes for all nine configured classes",
    all(h.attributed and h.distance_px == held.distance_px
        for h in by_class.values()),
    f"{ {k: (v.attributed, v.distance_px) for k, v in by_class.items()} } -- "
    f"association reads a box and a label; a rule that works only for phones "
    f"is a rule tuned to one log",
)
by_class_cup = {label: attribute(DESK_CUP, label=label) for label in SITE_CLASSES}
check(
    "and rejects the surface case for all nine, too",
    not any(h.attributed for h in by_class_cup.values()),
    f"{[k for k, v in by_class_cup.items() if v.attributed]}",
)

# --- Body scale: the largest apparent limb, bounded by the person box -----

scale = body_scale(keypoints_of(HELD_PHONE), tuple(HELD_PHONE["person_box"]))
limbs = [
    float(np.hypot(*(np.array(HELD_PHONE["keypoints"][a][:2])
                     - np.array(HELD_PHONE["keypoints"][b][:2]))))
    for a, b in ((5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (5, 9), (6, 10))
]
check(
    "body scale is the longest apparent limb, not the mean",
    abs(scale - max(limbs)) < 0.5 and scale > sum(limbs) / len(limbs),
    f"scale={scale:.1f} limbs={[round(v) for v in limbs]} -- perspective only "
    f"ever shortens a limb, so the longest one is the least foreshortened and "
    f"the best estimate; a mean is dragged down by the arm pointing at the "
    f"camera, which is the arm holding something up to be looked at",
)

wild = keypoints_of(HELD_PHONE)
wild[10] = (9999.0, 9999.0, 0.9)          # one keypoint lands off in the void
clamped = body_scale(wild, tuple(HELD_PHONE["person_box"]))
check(
    "one wild keypoint cannot grant a person a reach across the frame",
    clamped <= np.hypot(px2 - px1, py2 - py1) * 0.5 + 0.5,
    f"{clamped:.1f}px from a person box "
    f"{px2 - px1:.0f}x{py2 - py1:.0f} -- a limb longer than the body's own "
    f"box is a pose failure, not a long arm",
)

# --- Everything is reported, held or not ---------------------------------

# The phone this person is holding, and the cup on the desk in front of them.
# Two real boxes, one of each outcome, so "reported once whatever happened" is
# tested against both branches rather than only the happy one.
multi = associate(
    [("obj-cell_phone-0001", obj("cell phone",
                                 tuple(HELD_PHONE["object"][2]))),
     ("obj-cup-0001", obj("cup", tuple(DESK_CUP["object"][2])))],
    {"trk-0001": keypoints_of(HELD_PHONE)},
    person_boxes={"trk-0001": tuple(HELD_PHONE["person_box"])},
)
check(
    "every object is reported once, held or not",
    len(multi) == 2 and sum(h.attributed for h in multi) == 1,
    f"{[(h.label, h.held_by) for h in multi]}",
)
check(
    "every object carries a report, whatever the outcome",
    all(h.report is not None and h.report.outcome for h in multi),
    "a decision with no recorded reason is what made this bug take a "
    "diagnostic tool to find",
)

unheld = next(h for h in multi if not h.attributed)


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
    and unattributed_event["value"] == "cup",
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
