"""Tests for the opt-in stages wired into the events pipeline.

`perception/perceive.py` is the production path: camera in, JSON-lines events
out for the engine. `perception/tools/preview_pose.py` is the same work with a
window on it. Object detection, pose and action recognition were originally
built only into the demo, so the pipeline that actually feeds the engine could
see people and zones and nothing else.

These cover the wiring that closed that gap. Three properties matter most and
none of them is obvious from reading either script:

  * With every stage off, the pipeline must emit exactly what it emitted
    before. A refactor that quietly changes the default output would break
    every deployment that never asked for the new stages.

  * A stage that is enabled but silently inactive is the precise failure this
    work exists to fix. Enabling actions without the model present must stop
    with a message naming the export command, not run on without them.

  * Identity is additive. With no resolver every subject is the anonymous
    `{"class": "human"}` these events have always carried; with one, a bound
    track carries its enrolled ID and an UNBOUND track carries no identity
    key at all. The last of those is the one worth a test of its own: an
    absent key and a falsy value are not the same thing to the engine's
    role rules, and the cheapest way to break this is to "helpfully" fill in
    None or "unknown".

Run with:  python3 perception/test_stages.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "actions"))
sys.path.insert(0, str(ROOT))

import perceive  # noqa: E402
from actions_stage import (  # noqa: E402
    build_action_recognised_event,
    require_action_model,
    resolve_classes,
)
from association import HeldObject  # noqa: E402
from detectors.base import Detection  # noqa: E402
from emit import EventEmitter  # noqa: E402
from objects import ObjectStage, build_object_at_station_event  # noqa: E402
from pose import require_pose_model  # noqa: E402
from ppe_stage import PpeFinding, PpeStage, build_ppe_event  # noqa: E402

from base import ActionScore  # noqa: E402

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"[{'PASS' if condition else 'FAIL'}] {name}")
    if not condition:
        failures.append(f"{name}: {detail}")


SCHEMA = json.loads((ROOT / "schema" / "event.schema.json").read_text())
ALLOWED_OBS = set(SCHEMA["properties"]["observation"]["enum"])
REQUIRED = set(SCHEMA["required"])


def valid(event: dict) -> bool:
    return (
        REQUIRED.issubset(event.keys())
        and event["observation"] in ALLOWED_OBS
        and 0.0 <= event["confidence"] <= 1.0
    )


def emitter() -> EventEmitter:
    return EventEmitter(sensor_id="cam-01")


# ----------------------------------------------------------------------
# Everything is off unless asked for
# ----------------------------------------------------------------------

defaults = perceive.DEFAULTS
for stage in ("objects", "pose", "actions"):
    check(
        f"{stage} stage is off by default",
        defaults[stage]["enabled"] is False,
        "an unchanged config must behave byte-for-byte as the people-and-zones "
        "path did before these stages existed",
    )

check(
    "the default config declares no object classes",
    "classes" not in defaults.get("detector", {})
    or not defaults["detector"].get("classes"),
    "which objects matter is a site decision, not a constant in the source",
)


# ----------------------------------------------------------------------
# Enabling objects or actions must pull pose in with them
# ----------------------------------------------------------------------


class _Args:
    def __init__(self, **kw):
        self.objects = kw.get("objects", False)
        self.actions = kw.get("actions", False)
        self.action_every = kw.get("action_every", None)


def stages_for(cfg: dict, args: _Args) -> tuple[bool, bool, bool]:
    """Mirror of the resolution in perceive.main, kept in one place."""
    objects_on = bool(cfg.get("objects", {}).get("enabled")) or bool(args.objects)
    actions_on = bool(cfg.get("actions", {}).get("enabled")) or bool(args.actions)
    pose_on = bool(cfg.get("pose", {}).get("enabled")) or objects_on or actions_on
    return objects_on, pose_on, actions_on


empty = {"objects": {}, "pose": {}, "actions": {}}

check(
    "nothing enabled leaves all three stages off",
    stages_for(empty, _Args()) == (False, False, False),
)
check(
    "--objects turns pose on with it",
    stages_for(empty, _Args(objects=True)) == (True, True, False),
    "hand attribution needs arm keypoints; objects without pose would silently "
    "attribute nothing",
)
check(
    "--actions turns pose on with it",
    stages_for(empty, _Args(actions=True)) == (False, True, True),
    "PoseC3D consumes keypoints, not pixels",
)
check(
    "config can enable a stage without a flag",
    stages_for({"objects": {"enabled": True}, "pose": {}, "actions": {}}, _Args())
    == (True, True, False),
)
check(
    "a flag can enable a stage the config left off",
    stages_for({"objects": {"enabled": False}, "pose": {}, "actions": {}},
               _Args(objects=True)) == (True, True, False),
)


# ----------------------------------------------------------------------
# A missing model must stop the run, not disable the stage
# ----------------------------------------------------------------------

for label, fn, hint in (
    ("pose", require_pose_model, "export_rtmpose"),
    ("action", require_action_model, "export_posec3d"),
):
    raised, message = False, ""
    try:
        fn("perception/models/does-not-exist.onnx")
    except FileNotFoundError as exc:
        raised, message = True, str(exc)
    check(
        f"a missing {label} model raises rather than running without the stage",
        raised,
        "silently continuing is exactly the failure this work fixes",
    )
    check(
        f"the {label} error names the export command",
        hint in message,
        f"got: {message[:120]}",
    )


# ----------------------------------------------------------------------
# Object events
# ----------------------------------------------------------------------

held = HeldObject(
    object_track_id="obj-cell_phone-0001",
    label="cell phone",
    score=0.82,
    box=(100.0, 200.0, 160.0, 300.0),
    held_by="trk-0001",
    wrist="right_wrist",
    wrist_score=0.61,
    distance_px=42.0,
)
ev = build_object_at_station_event(emitter(), held, persistence=1.0)
check("an attributed object emits a valid event", valid(ev), json.dumps(ev)[:200])
check("it carries the object class in value", ev["value"] == "cell phone", str(ev.get("value")))
check(
    "it carries the holding person's track_id",
    ev.get("track_id") == "trk-0001",
    "without this the engine cannot attribute the object to anyone",
)

loose = HeldObject(
    object_track_id="obj-bottle-0002",
    label="bottle",
    score=0.77,
    box=(400.0, 500.0, 440.0, 560.0),
)
ev2 = build_object_at_station_event(emitter(), loose, persistence=1.0)
check("an unattributed object still emits", valid(ev2), json.dumps(ev2)[:200])
check(
    "an unattributed object carries no track_id rather than being dropped",
    not ev2.get("track_id"),
    "dropping it would make 'no object' and 'object nobody is holding' the "
    "same stream, and the second is the more interesting one",
)

check(
    "persistence discounts a flickering detection",
    build_object_at_station_event(emitter(), held, persistence=0.4)["confidence"]
    < ev["confidence"],
)

raised = False
try:
    ObjectStage(object_classes=[])
except ValueError:
    raised = True
check(
    "an empty object class list is refused",
    raised,
    "silently detecting nothing is worse than refusing to start",
)


# ----------------------------------------------------------------------
# Action events, including abstentions
# ----------------------------------------------------------------------

decided = ActionScore(
    track_id="trk-0001", label="drink water", confidence=0.733, abstained=False
)
ev3 = build_action_recognised_event(emitter(), decided)
check("a decided action emits a valid event", valid(ev3), json.dumps(ev3)[:200])
check("it carries the class label", ev3["value"] == "drink water", str(ev3.get("value")))

abstained = ActionScore(
    track_id="trk-0001",
    label=None,
    confidence=0.168,
    abstained=True,
    reason="best class 'taking a selfie' at 0.168 below threshold 0.550",
)
ev4 = build_action_recognised_event(emitter(), abstained)
check("an abstention emits rather than being dropped", valid(ev4), json.dumps(ev4)[:200])
check(
    "an abstention is reported as unknown, not as its best guess",
    ev4["value"] == "unknown",
    "the model reaching for 'taking a selfie' at 0.168 is not a recognition; "
    "recording it as one would be the silent-miss failure inverted",
)
check(
    "an abstention keeps the best-class confidence",
    ev4["confidence"] == 0.168,
    "the engine needs to tell a near-miss from noise when it reviews these",
)
check("both stay on the same track", ev3["track_id"] == ev4["track_id"] == "trk-0001")


# ----------------------------------------------------------------------
# Class vocabulary
# ----------------------------------------------------------------------

ntu = resolve_classes("ntu60")
check("the ntu60 vocabulary resolves to 60 classes", len(ntu) == 60, str(len(ntu)))

custom = resolve_classes(["loading fixture", "torque applied"])
check("a site can supply its own class list", custom == ["loading fixture", "torque applied"])

raised = False
try:
    resolve_classes("kinetics400")
except ValueError:
    raised = True
check(
    "an unknown vocabulary is refused",
    raised,
    "a silently wrong class list would mislabel every prediction",
)


# ----------------------------------------------------------------------
# Who the event is about
# ----------------------------------------------------------------------
#
# Two of the three builders live here (action_recognised and the PPE pair);
# action_interval is covered in test_intervals.py, where the builder that
# resolves it at close time lives.
#
# BOUND is a stub resolver, standing in for IdentityStage.subject_for. It is a
# plain callable precisely so these modules never import the identity layer to
# be tested -- if this stub ever has to become the real stage, the dependency
# it was built to avoid has crept back in.


def bound_resolver(track_id: str) -> dict:
    """trk-0001 is enrolled. Everybody else is a stranger.

    Note the stranger branch returns the anonymous subject with NO identity
    key -- not identity=None, not identity="unknown". That mirrors
    `IdentityStage.subject_for`, and it is the behaviour under test.
    """
    if track_id == "trk-0001":
        return {"class": "human", "identity": "EMP-4471", "role": "operator"}
    return {"class": "human"}


action = ActionScore(
    track_id="trk-0001", label="drink water", confidence=0.733, abstained=False
)
stranger_action = ActionScore(
    track_id="trk-0404", label="drink water", confidence=0.733, abstained=False
)

anon = build_action_recognised_event(emitter(), action)
check(
    "with no resolver an action is about an anonymous human, as before",
    anon.get("subject") == {"class": "human"},
    str(anon.get("subject")) + ": identity is additive, and a deployment that "
    "never asked for it must see byte-identical events",
)

named = build_action_recognised_event(emitter(), action, subject_for=bound_resolver)
check(
    "a bound track's action carries the enrolled identity",
    named["subject"]["identity"] == "EMP-4471",
    str(named.get("subject")),
)
check(
    "and the role that came with the binding",
    named["subject"].get("role") == "operator",
    str(named.get("subject")),
)
check(
    "the track_id is still there beside it, not replaced by the identity",
    named.get("track_id") == "trk-0001",
    "the track id is what the camera saw and the identity is an inference "
    "about who that was; dropping the first makes the second unquestionable",
)
check("the identified action still validates", valid(named), json.dumps(named)[:200])

unbound = build_action_recognised_event(
    emitter(), stranger_action, subject_for=bound_resolver
)
check(
    "an unbound track emits NO identity key, rather than an empty one",
    "identity" not in unbound["subject"],
    f"got {unbound['subject']!r}: the engine's wrong_role rules read an "
    "absent key and a falsy value differently, so None or 'unknown' here "
    "would be a confident claim about somebody nobody recognised",
)
check(
    "nor a role key",
    "role" not in unbound["subject"],
    str(unbound.get("subject")),
)
check(
    "but it keeps its track_id, so the observation is still attributable",
    unbound.get("track_id") == "trk-0404",
    str(unbound.get("track_id")),
)

# An abstention goes through the same path: the subject is a fact about the
# person, not about whether the classifier managed to decide anything.
abstained_named = build_action_recognised_event(
    emitter(),
    ActionScore(track_id="trk-0001", label=None, confidence=0.168, abstained=True),
    subject_for=bound_resolver,
)
check(
    "an abstention is still attributed to whoever it is about",
    abstained_named["subject"]["identity"] == "EMP-4471"
    and abstained_named["value"] == "unknown",
    json.dumps(abstained_named)[:200],
)


# PPE, where "whose head" is the first question anyone asks about a finding.

def finding(track_id):
    return PpeFinding(
        observation="ppe_helmet",
        value=False,
        label="NO-Hardhat",
        score=0.31,
        confidence=0.28,
        ppe_track_id="ppe-NO-Hardhat-0001",
        box=(100.0, 40.0, 150.0, 90.0),
        track_id=track_id,
        containment=0.95,
    )


ppe_anon = build_ppe_event(emitter(), finding("trk-0001"))
check(
    "with no resolver a PPE finding is about an anonymous human, as before",
    ppe_anon.get("subject") == {"class": "human"},
    str(ppe_anon.get("subject")),
)

ppe_named = build_ppe_event(
    emitter(), finding("trk-0001"), subject_for=bound_resolver
)
check(
    "a bound track's PPE finding names the person",
    ppe_named["subject"]["identity"] == "EMP-4471",
    str(ppe_named.get("subject")),
)
check(
    "and keeps the track it was attributed to",
    ppe_named.get("track_id") == "trk-0001",
    str(ppe_named.get("track_id")),
)
check("the identified PPE event still validates", valid(ppe_named),
      json.dumps(ppe_named)[:200])

ppe_unbound = build_ppe_event(
    emitter(), finding("trk-0404"), subject_for=bound_resolver
)
check(
    "an unbound wearer gets no identity key on the PPE event either",
    "identity" not in ppe_unbound["subject"],
    f"got {ppe_unbound['subject']!r}",
)

# An unattributed box has no person at all, so there is nobody to resolve. The
# resolver must never be called with None, and the subject stays anonymous.
called = []
ppe_loose = build_ppe_event(
    emitter(),
    finding(None),
    subject_for=lambda t: called.append(t) or {"class": "human", "identity": "X"},
)
check(
    "an unattributed PPE box resolves to nobody rather than to anybody",
    ppe_loose["subject"] == {"class": "human"} and not called,
    f"subject={ppe_loose['subject']!r} resolver_calls={called!r}: a hat that "
    "could not be pinned to one person must not inherit an identity",
)
check(
    "and it still carries no track_id, as before",
    not ppe_loose.get("track_id"),
    str(ppe_loose.get("track_id")),
)

# The whole stage, not just the builder. A builder that accepts a resolver
# nobody passes it is the same silent-no-op this file exists to catch, so the
# resolver is followed through the call perceive.py actually makes.


class _ScriptedPpeDetector:
    """Just enough detector for process_detections: no graph, no frame."""

    name = "scripted-ppe"
    licence = "n/a (test)"
    classes = ["NO-Hardhat"]
    class_thresholds = {"NO-Hardhat": 0.20}


_stage = PpeStage({"loose_frames": 1}, detector=_ScriptedPpeDetector())
_person = {"trk-0001": (300.0, 100.0, 500.0, 700.0)}
_hat = Detection(360.0, 110.0, 440.0, 165.0, 0.45, "NO-Hardhat")
_findings, _events = _stage.process_detections(
    [_hat], _person, emitter(), subject_for=bound_resolver
)
check(
    "PpeStage.process_detections carries the resolver through to the event",
    len(_events) == 1
    and _events[0]["subject"].get("identity") == "EMP-4471",
    f"got {[e.get('subject') for e in _events]}: a resolver the stage accepts "
    "but never forwards is exactly the silently-inactive failure this suite "
    "was written for",
)

_stage = PpeStage({"loose_frames": 1}, detector=_ScriptedPpeDetector())
_findings, _events = _stage.process_detections([_hat], _person, emitter())
check(
    "and without one the stage emits the anonymous subject it always did",
    len(_events) == 1 and _events[0]["subject"] == {"class": "human"},
    f"got {[e.get('subject') for e in _events]}",
)


print()
if failures:
    print(f"{len(failures)} stage test(s) failed")
    for f in failures:
        print(f"  !! {f}")
    raise SystemExit(1)
print("all stage tests passed")
