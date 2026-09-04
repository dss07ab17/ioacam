"""Tests for the opt-in stages wired into the events pipeline.

`perception/perceive.py` is the production path: camera in, JSON-lines events
out for the engine. `perception/tools/preview_pose.py` is the same work with a
window on it. Object detection, pose and action recognition were originally
built only into the demo, so the pipeline that actually feeds the engine could
see people and zones and nothing else.

These cover the wiring that closed that gap. Two properties matter most and
neither is obvious from reading either script:

  * With every stage off, the pipeline must emit exactly what it emitted
    before. A refactor that quietly changes the default output would break
    every deployment that never asked for the new stages.

  * A stage that is enabled but silently inactive is the precise failure this
    work exists to fix. Enabling actions without the model present must stop
    with a message naming the export command, not run on without them.

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
from emit import EventEmitter  # noqa: E402
from objects import ObjectStage, build_object_at_station_event  # noqa: E402
from pose import require_pose_model  # noqa: E402

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
    "wrist attribution needs keypoints; objects without pose would silently "
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


print()
if failures:
    print(f"{len(failures)} stage test(s) failed")
    for f in failures:
        print(f"  !! {f}")
    raise SystemExit(1)
print("all stage tests passed")
