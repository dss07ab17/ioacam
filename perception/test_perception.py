#!/usr/bin/env python3
"""Offline tests for the perception layer. No camera, no weights, no network.

    python perception/test_perception.py

Runs the real CLI end to end against a synthetic video with the stub detector,
so what is exercised is the shipped code path, not a reimplementation of it.
The stub is what makes that possible: geometry, hysteresis, calibration and
emission are all independent of which model produced the boxes.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import confidence as conf  # noqa: E402
from detectors.base import Detection  # noqa: E402
from tracking import IouTracker  # noqa: E402
from zones import Membership, Zone, load_zones  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
PERCEPTION = REPO_ROOT / "perception"

FRAME_W, FRAME_H = 640, 480
# A square occupying the middle of the frame, in normalised coordinates.
ZONE = {
    "zone_id": "zone-assembly-4",
    "coordinates": "normalized",
    "polygon": [[0.35, 0.30], [0.65, 0.30], [0.65, 0.95], [0.35, 0.95]],
}

failures: list[str] = []


def check(condition: bool, label: str, detail: str = "") -> None:
    if condition:
        print(f"  pass  {label}")
    else:
        failures.append(label)
        print(f"  FAIL  {label}" + (f"\n          {detail}" if detail else ""))


def box(cx: float, bottom: float, height: float = 220.0, score: float = 0.90) -> dict:
    """A person-shaped box whose ground point is (cx, bottom)."""
    width = height * 0.4
    return {
        "x1": cx - width / 2, "y1": bottom - height,
        "x2": cx + width / 2, "y2": bottom, "score": score,
    }


# ---------------------------------------------------------------------------
# unit level


def test_polygon() -> None:
    print("\npoint in polygon")
    zone = load_zones([ZONE])[0]
    check(zone.contains((320, 300), FRAME_W, FRAME_H), "centre is inside")
    check(not zone.contains((60, 300), FRAME_W, FRAME_H), "far left is outside")
    check(not zone.contains((320, 60), FRAME_W, FRAME_H), "above the zone is outside")

    # Membership is tested at the feet, not the centroid. A tall subject
    # standing outside but leaning over the boundary must read as outside.
    leaning = Detection(**box(cx=200, bottom=300))
    check(
        not zone.contains(leaning.ground_point, FRAME_W, FRAME_H),
        "membership uses the ground point, not the box centre",
    )

    concave = Zone("z", [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 0.5), (0.0, 1.0)])
    check(concave.contains((320, 100), FRAME_W, FRAME_H), "concave polygon: inside the arms")
    check(not concave.contains((320, 460), FRAME_W, FRAME_H), "concave polygon: inside the notch")

    try:
        load_zones([{"zone_id": "bad", "polygon": [[0.1, 0.1], [0.9, 0.9]]}])
        check(False, "a 2-point polygon is rejected")
    except ValueError:
        check(True, "a 2-point polygon is rejected")

    try:
        load_zones([{"zone_id": "bad", "polygon": [[0, 0], [640, 0], [640, 480]]}])
        check(False, "pixel coordinates declared as normalised are rejected")
    except ValueError:
        check(True, "pixel coordinates declared as normalised are rejected")


def test_hysteresis() -> None:
    print("\nhysteresis")
    membership = Membership()
    transitions = [membership.update(True, 5, 10) for _ in range(5)]
    check(transitions.count("entered") == 1, "entry fires once after enter_frames")
    check(transitions[-1] == "entered", "entry fires on the 5th frame, not the 1st")

    quiet = [membership.update(True, 5, 10) for _ in range(50)]
    check(all(t is None for t in quiet), "standing still emits nothing for 50 frames")

    # A single dropped detection is the common case and must not read as an exit.
    flicker = [membership.update(False, 5, 10) for _ in range(9)]
    check(all(t is None for t in flicker), "9 frames outside is below exit_frames, no event")
    check(membership.update(False, 5, 10) == "left", "exit fires on the 10th frame")


def test_confidence() -> None:
    print("\nconfidence")
    raw = 0.95
    calibrated = conf.temperature_scale(raw, 1.8)
    check(calibrated < raw, "T > 1 softens an overconfident score",
          f"{raw} -> {calibrated:.4f}")
    check(abs(conf.temperature_scale(raw, 1.0) - raw) < 1e-6, "T = 1 is the identity")
    check(abs(conf.temperature_scale(0.5, 3.7) - 0.5) < 1e-6, "0.5 is the fixed point")

    big = Detection(**box(cx=320, bottom=400, height=220))
    small = Detection(**box(cx=320, bottom=400, height=45))
    check(
        conf.quality(big, FRAME_W, FRAME_H) > conf.quality(small, FRAME_W, FRAME_H),
        "a larger subject scores higher quality",
    )
    tiny = Detection(**box(cx=320, bottom=400, height=30))
    check(conf.quality(tiny, FRAME_W, FRAME_H) == 0.0,
          "a subject below min_height_px scores zero quality")

    occluded = [Detection(**box(cx=320, bottom=400)), Detection(**box(cx=330, bottom=400))]
    check(
        conf.quality(occluded[0], FRAME_W, FRAME_H, others=occluded)
        < conf.quality(occluded[0], FRAME_W, FRAME_H),
        "overlapping subjects are discounted for occlusion",
    )

    value, components = conf.compose(0.95, 1.8, 0.8, 1.0, 30)
    check(0.0 <= value <= 1.0, "confidence stays in 0..1")
    check(
        set(components) == {"raw_score", "calibrated_score", "quality",
                            "persistence", "agreement", "frames_observed"},
        "components block carries every field the schema names",
    )
    check(components["raw_score"] == 0.95, "raw score is retained, not overwritten")
    check(conf.compose(0.95, 1.8, 0.8, 1.0, 30, agreement=9.0)[1]["agreement"] == 1.2,
          "agreement is clamped to the schema's ceiling")


def test_tracker() -> None:
    print("\ntracking")
    tracker = IouTracker()
    first = tracker.update([Detection(**box(cx=300, bottom=400))])
    second = tracker.update([Detection(**box(cx=308, bottom=402))])
    check(first[0].track_id == second[0].track_id, "a slowly moving subject keeps its id")

    third = tracker.update([Detection(**box(cx=50, bottom=400))])
    check(third[0].track_id != second[0].track_id, "a disjoint box starts a new track")

    tracker = IouTracker(max_misses=3)
    tracker.update([Detection(**box(cx=300, bottom=400))])
    for _ in range(4):
        tracker.update([])
    check(tracker.tracks == [], "an unmatched track is aged out after max_misses")

    # Persistence measures detection stability, not how long the subject has
    # been in a zone. Defining it the other way scores every entry event near
    # zero at the instant of entry, which is when it matters most.
    steady = IouTracker()
    for i in range(30):
        steady.update([Detection(**box(cx=300 + i, bottom=400))])
    check(steady.tracks[0].persistence() > 0.95, "a steadily detected track scores high")

    intermittent = IouTracker(max_misses=30)
    for i in range(30):
        intermittent.update([Detection(**box(cx=300, bottom=400))] if i % 3 else [])
    check(intermittent.tracks[0].persistence() < 0.75,
          "an intermittently detected track scores low",
          "got {:.3f}".format(intermittent.tracks[0].persistence()))

    fresh = IouTracker()
    fresh.update([Detection(**box(cx=300, bottom=400))])
    check(fresh.tracks[0].persistence() == 1.0,
          "a brand new but cleanly detected track is not penalised for being new")


def test_no_engine_import() -> None:
    """The separation the README asks for, enforced rather than asserted.

    Parsed, not grepped: a substring search matches its own source text and
    reports a false positive, which is worse than no check at all.
    """
    print("\nprocess separation")
    import ast

    sources = sorted(PERCEPTION.rglob("*.py"))
    offenders = []
    for source in sources:
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if name == "engine" or name.startswith("engine."):
                    offenders.append(f"{source.relative_to(REPO_ROOT)}:{node.lineno}")

    check(not offenders, "no module in perception/ imports from engine/", str(offenders))
    check(len(sources) > 5, "the scan actually found the source files",
          f"scanned {len(sources)}")


# ---------------------------------------------------------------------------
# end to end, through the real CLI


def write_video(path: Path, frames: int) -> None:
    """Synthetic footage. Content is irrelevant: the stub supplies the boxes.

    Textured rather than flat so the blur and luminance quality terms land in a
    realistic range instead of on their floors.
    """
    rng = np.random.default_rng(7)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 15.0, (FRAME_W, FRAME_H)
    )
    if not writer.isOpened():
        # OpenCV returns a writer object even when it cannot encode, and then
        # silently drops every frame -- the test would fail later with a
        # baffling "no events" instead of naming the real cause.
        raise RuntimeError(
            "This OpenCV build cannot write MJPG/AVI, so the end-to-end tests "
            "cannot generate their synthetic footage. The perception layer "
            "itself is unaffected; only these two tests need it."
        )
    for _ in range(frames):
        frame = rng.integers(96, 160, (FRAME_H, FRAME_W, 3), dtype=np.uint8)
        cv2.rectangle(frame, (40, 40), (600, 440), (200, 200, 200), 3)
        writer.write(frame)
    writer.release()


def build_walk() -> list[list[dict]]:
    """A subject walks left to right, crosses the zone, then leaves frame."""
    script = []
    for x in range(60, 600, 12):          # approach, cross, exit the zone
        script.append([box(cx=float(x), bottom=400.0)])
    script.extend([[] for _ in range(25)])  # then out of the camera's view
    return script


def test_end_to_end() -> None:
    print("\nend to end (real CLI, stub detector, synthetic video)")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        script = build_walk()
        video = tmp_path / "walk.avi"
        write_video(video, len(script) + 5)

        config = {
            "sensor_id": "cam-01",
            "detector": {"backend": "stub", "script": script},
            "calibration": {"temperature": 1.8},
            "zones": [ZONE],
            "emission": {
                "enter_frames": 3, "exit_frames": 5, "persistence_window": 30,
                "min_confidence": 0.02, "emit_person_count": True,
                "count_debounce_frames": 3,
            },
            "integrity": {"enabled": True},
        }
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")

        result = subprocess.run(
            [sys.executable, str(PERCEPTION / "perceive.py"),
             "--config", str(config_path), "--source", str(video)],
            capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=180,
        )
        check(result.returncode == 0, "the CLI exits 0", result.stderr[-800:])

        events = []
        malformed = []
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                malformed.append(line)

        check(not malformed, "stdout is JSON lines and nothing else", str(malformed[:2]))
        check(bool(events), "events were emitted", result.stderr[-800:])
        if not events:
            return

        observations = [e["observation"] for e in events]
        print(f"        emitted: {observations}")

        check(observations.count("person_in_zone") == 1,
              "one crossing produces exactly one entry", str(observations))
        check(observations.count("person_left_zone") == 1,
              "one crossing produces exactly one exit", str(observations))
        check(observations.index("person_in_zone") < observations.index("person_left_zone"),
              "the entry precedes the exit")

        entry = events[observations.index("person_in_zone")]
        check(entry["zone_id"] == ZONE["zone_id"], "the entry names the zone")
        check(entry["value"] is True, "person_in_zone carries value true")
        check(entry["subject"] == {"class": "human"},
              "identity and role are omitted, not guessed", str(entry.get("subject")))
        check("track_id" in entry, "the entry carries a track id")
        check(0.0 < entry["confidence"] < 1.0,
              "a camera event carries genuine uncertainty", str(entry["confidence"]))
        check(entry["confidence_components"]["raw_score"] > entry["confidence"],
              "calibration and quality discount the raw score",
              json.dumps(entry["confidence_components"]))
        check(entry["confidence"] > 0.3,
              "an entry seen clearly is not crushed to near-zero confidence",
              json.dumps(entry["confidence_components"]))

        # The invariant that caught the unpaired-exit bug: whatever the
        # geometry knows, the EMITTED stream must balance per track, or the
        # engine sees people leaving zones it never saw them enter.
        open_tracks = set()
        unpaired = []
        for event in events:
            if event["observation"] == "person_in_zone":
                open_tracks.add((event["track_id"], event["zone_id"]))
            elif event["observation"] == "person_left_zone":
                key = (event["track_id"], event["zone_id"])
                if key not in open_tracks:
                    unpaired.append(key)
                open_tracks.discard(key)
        check(not unpaired, "every exit is preceded by an entry for the same track",
              str(unpaired))

        health = [e for e in events if e["observation"] == "sensor_health"]
        check(len(health) >= 2, "coverage is announced at start and withdrawn at stop")
        check(all(e["source"] == "timer" and e["confidence"] == 1.0 for e in health),
              "sensor_health is a fact: source timer, confidence exactly 1.0")
        check(health[0]["value"] == "healthy" and health[-1]["value"] == "unhealthy",
              "a clean shutdown still reports coverage lost")

        counts = [e for e in events if e["observation"] == "person_count"]
        check(bool(counts), "occupancy changes are reported")
        check(all(isinstance(e["value"], int) for e in counts), "person_count carries an integer")
        check(max((e["value"] for e in counts), default=0) == 1,
              "occupancy peaks at one for one subject")

        timestamps = [e["timestamp_us"] for e in events]
        check(timestamps == sorted(timestamps), "timestamps are non-decreasing")
        check(len({e["event_id"] for e in events}) == len(events), "event ids are unique")

        seqs = [e["integrity"]["seq"] for e in events]
        check(seqs == list(range(len(events))), "the integrity counter has no gaps")

        validation = subprocess.run(
            [sys.executable, str(PERCEPTION / "tools" / "validate_events.py")],
            input=result.stdout, capture_output=True, text=True,
            cwd=str(REPO_ROOT), timeout=120,
        )
        check(validation.returncode == 0,
              "every event validates against schema/event.schema.json",
              validation.stderr[-1500:])
        print(f"        {validation.stderr.strip().splitlines()[-1]}")


def test_flicker_is_not_traffic() -> None:
    print("\nflicker suppression (real CLI)")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        # A subject loitering exactly on the boundary, detected intermittently.
        # This is the case that floods an engine if the debounce is missing.
        script = []
        for i in range(120):
            on_edge = 0.35 * FRAME_W + (4 if i % 2 else -4)
            script.append([box(cx=on_edge, bottom=400.0)] if i % 3 else [])

        video = tmp_path / "flicker.avi"
        write_video(video, len(script) + 5)
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({
            "sensor_id": "cam-01",
            "detector": {"backend": "stub", "script": script},
            "zones": [ZONE],
            "emission": {"enter_frames": 5, "exit_frames": 10, "persistence_window": 30,
                         "min_confidence": 0.02, "emit_person_count": False,
                         "count_debounce_frames": 8},
        }), encoding="utf-8")

        result = subprocess.run(
            [sys.executable, str(PERCEPTION / "perceive.py"),
             "--config", str(config_path), "--source", str(video)],
            capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=180,
        )
        zone_events = [
            json.loads(line) for line in result.stdout.splitlines() if line.strip()
        ]
        crossings = [
            e for e in zone_events
            if e["observation"] in ("person_in_zone", "person_left_zone")
        ]
        check(len(crossings) <= 4,
              "120 flickering frames on the boundary produce at most 4 events",
              f"got {len(crossings)}")


def _parse_events(stdout: str) -> list[dict]:
    events = []
    for line in stdout.splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


def test_stages_off_are_people_and_zones_only() -> None:
    """With objects/pose/actions disabled, stdout is the classic four observations."""
    print("\nstages disabled (production path unchanged)")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        script = build_walk()
        video = tmp_path / "walk.avi"
        write_video(video, len(script) + 5)
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({
            "sensor_id": "cam-01",
            "detector": {"backend": "stub", "script": script},
            "zones": [ZONE],
            "emission": {
                "enter_frames": 3, "exit_frames": 5, "persistence_window": 30,
                "min_confidence": 0.02, "emit_person_count": True,
                "count_debounce_frames": 3,
            },
            "objects": {"enabled": False},
            "pose": {"enabled": False},
            "actions": {"enabled": False},
        }), encoding="utf-8")

        result = subprocess.run(
            [sys.executable, str(PERCEPTION / "perceive.py"),
             "--config", str(config_path), "--source", str(video)],
            capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=180,
        )
        check(result.returncode == 0, "stages-off CLI exits 0", result.stderr[-500:])
        events = _parse_events(result.stdout)
        observations = {e["observation"] for e in events}
        check(
            observations <= {"person_in_zone", "person_left_zone",
                             "person_count", "sensor_health"},
            "no object/action observations when stages are off",
            str(observations),
        )
        check("object_at_station" not in observations, "objects stay silent when disabled")
        check("action_recognised" not in observations, "actions stay silent when disabled")


def test_objects_attributed_and_unattributed() -> None:
    """Objects stage emits object_at_station with and without track_id."""
    print("\nobjects stage (attributed + unattributed)")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        # StubPoseEstimator places the right wrist near mid-box.
        # Person box: cx=300, bottom=400, h=220 → wrist ≈ (300, 317.5).
        person = {**box(cx=300, bottom=400.0), "label": "person"}
        held_phone = {
            "x1": 285.0, "y1": 300.0, "x2": 315.0, "y2": 335.0,
            "score": 0.72, "label": "cell phone",
        }
        free_phone = {
            "x1": 20.0, "y1": 20.0, "x2": 60.0, "y2": 60.0,
            "score": 0.68, "label": "cell phone",
        }
        script = [[person, held_phone, free_phone] for _ in range(40)]
        video = tmp_path / "objects.avi"
        write_video(video, len(script) + 2)
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({
            "sensor_id": "cam-01",
            "detector": {
                "backend": "stub",
                "script": script,
                "classes": ["person", "cell phone"],
            },
            "zones": [ZONE],
            "emission": {
                "enter_frames": 3, "exit_frames": 5, "persistence_window": 30,
                "min_confidence": 0.02, "emit_person_count": False,
                "count_debounce_frames": 3,
            },
            "objects": {"enabled": True, "margin": 0.25, "min_wrist_score": 0.30},
            "pose": {"enabled": True, "backend": "stub", "score": 0.9},
            "actions": {"enabled": False},
        }), encoding="utf-8")

        result = subprocess.run(
            [sys.executable, str(PERCEPTION / "perceive.py"),
             "--config", str(config_path), "--source", str(video)],
            capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=180,
        )
        check(result.returncode == 0, "objects CLI exits 0", result.stderr[-800:])
        events = _parse_events(result.stdout)
        objects = [e for e in events if e["observation"] == "object_at_station"]
        check(len(objects) >= 2, "at least two object_at_station events",
              f"got {len(objects)}: {objects!r}")

        attributed = [e for e in objects if "track_id" in e]
        unattributed = [e for e in objects if "track_id" not in e]
        check(bool(attributed), "wrist-overlapping object carries track_id",
              str(objects))
        check(bool(unattributed), "unheld object omits track_id rather than dropping",
              str(objects))
        check(all(e["value"] == "cell phone" for e in objects),
              "value carries the object class")
        check(all(e.get("subject") == {"class": "object"} for e in objects),
              "subject class is object")

        validation = subprocess.run(
            [sys.executable, str(PERCEPTION / "tools" / "validate_events.py")],
            input=result.stdout, capture_output=True, text=True,
            cwd=str(REPO_ROOT), timeout=120,
        )
        check(validation.returncode == 0,
              "object events validate against schema/event.schema.json",
              validation.stderr[-1500:])


def test_actions_missing_model_fails_loud() -> None:
    """Enabled actions with a missing ONNX must exit, not run silently."""
    print("\nactions missing model (fail loud)")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        missing = tmp_path / "no-such-posec3d.onnx"
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({
            "sensor_id": "cam-01",
            "detector": {"backend": "stub", "script": [[]]},
            "zones": [],
            "actions": {
                "enabled": True,
                "backend": "posec3d",
                "model_path": str(missing),
            },
            "pose": {"enabled": True, "backend": "stub"},
        }), encoding="utf-8")

        result = subprocess.run(
            [sys.executable, str(PERCEPTION / "perceive.py"),
             "--config", str(config_path), "--source", str(tmp_path / "no.avi")],
            capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=60,
        )
        check(result.returncode != 0, "missing action model exits non-zero",
              result.stderr[-500:])
        err = result.stderr + result.stdout
        check("export_posec3d" in err or "PoseC3D" in err,
              "error names the export command",
              err[-800:])
        check("object_at_station" not in result.stdout,
              "no events stream when the stage failed to start")


def test_pose_missing_model_fails_loud() -> None:
    print("\npose missing model (fail loud)")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        missing = tmp_path / "no-such-rtmpose.onnx"
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({
            "sensor_id": "cam-01",
            "detector": {"backend": "stub", "script": [[]]},
            "zones": [],
            "objects": {"enabled": True},
            "pose": {
                "enabled": True,
                "backend": "rtmpose",
                "model_path": str(missing),
            },
            "detector_classes_note": "stub",
        }), encoding="utf-8")
        # Force objects so pose is required; stub detector needs object classes
        # listed so ObjectStage constructs after pose — pose must fail first.
        cfg = json.loads(config_path.read_text())
        cfg["detector"]["classes"] = ["person", "cell phone"]
        config_path.write_text(json.dumps(cfg), encoding="utf-8")

        result = subprocess.run(
            [sys.executable, str(PERCEPTION / "perceive.py"),
             "--config", str(config_path), "--max-frames", "1"],
            capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=60,
        )
        check(result.returncode != 0, "missing pose model exits non-zero",
              result.stderr[-500:])
        check("export_rtmpose" in (result.stderr + result.stdout),
              "error names export_rtmpose.py",
              result.stderr[-800:])


def test_action_recognised_events_validate() -> None:
    """Decided and abstained action events both validate against the schema."""
    print("\naction_recognised event shapes")
    sys.path.insert(0, str(PERCEPTION))
    sys.path.insert(0, str(PERCEPTION / "tools"))
    sys.path.insert(0, str(PERCEPTION / "actions"))
    from actions_stage import build_action_recognised_event  # noqa: WPS433
    from base import ActionScore  # noqa: WPS433
    from emit import EventEmitter  # noqa: WPS433
    from validate_events import validate  # noqa: WPS433

    schema = json.loads((REPO_ROOT / "schema" / "event.schema.json").read_text())
    import io
    emitter = EventEmitter(sensor_id="cam-01", stream=io.StringIO())

    decided = ActionScore(
        track_id="trk-0001", label="reaching", confidence=0.81, abstained=False
    )
    abstained = ActionScore(
        track_id="trk-0001", label=None, confidence=0.22, abstained=True,
        reason="best class below threshold",
    )
    for name, score in (("decided", decided), ("abstained", abstained)):
        event = build_action_recognised_event(emitter, score)
        errors = validate(event, schema, None)
        check(not errors, f"{name} action_recognised validates", str(errors))
    check(
        build_action_recognised_event(emitter, abstained)["value"] == "unknown",
        "abstention is emitted as value unknown, not dropped",
    )


def main() -> int:
    print("perception layer tests")
    test_polygon()
    test_hysteresis()
    test_confidence()
    test_tracker()
    test_no_engine_import()
    test_end_to_end()
    test_flicker_is_not_traffic()
    test_stages_off_are_people_and_zones_only()
    test_objects_attributed_and_unattributed()
    test_actions_missing_model_fails_loud()
    test_pose_missing_model_fails_loud()
    test_action_recognised_events_validate()

    print()
    if failures:
        print(f"{len(failures)} FAILED:")
        for name in failures:
            print(f"  - {name}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
