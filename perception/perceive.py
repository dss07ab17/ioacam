#!/usr/bin/env python3
"""iOACAM perception layer: webcam -> detection -> zones -> events.

A SEPARATE PROCESS from the workflow engine. It imports nothing from engine/
and never will. The only thing that crosses between them is a stream of JSON
lines matching schema/event.schema.json, which is what makes the board port a
swap of this process alone.

This is the **production events pipeline**. Object detection, pose and action
recognition are opt-in stages (config or --objects / --actions); with all
stages off the behaviour matches the people-and-zones path alone.

    python perception/perceive.py --config perception/config/zones.example.json
    python perception/perceive.py --preview
    python perception/perceive.py --objects --actions
    python perception/perceive.py | python your_consumer.py

stdout is events. stderr is everything else.

The live demo with a preview window and JSONL logging is
`perception/tools/preview_pose.py`. It uses the same stage modules so the
demo and this path cannot silently diverge.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2  # noqa: E402

import confidence as conf  # noqa: E402
from actions_stage import ActionsStage  # noqa: E402
from detectors import build_detector  # noqa: E402
from emit import EventEmitter, log, monotonic_us  # noqa: E402
from objects import ObjectStage  # noqa: E402
from pose import PoseStage, person_boxes_from_tracks  # noqa: E402
from tracking import IouTracker  # noqa: E402
from zones import Membership, load_zones  # noqa: E402

DEFAULTS = {
    "sensor_id": "cam-01",
    "camera": {"index": 0, "width": 1280, "height": 720, "fps": 15},
    "detector": {"backend": "yolox-onnx",
                 "model_path": "perception/models/yolox_tiny.onnx",
                 "input_size": [416, 416]},
    "calibration": {"temperature": 1.8},
    "tracking": {"iou_threshold": 0.30, "max_misses": 15},
    "emission": {
        "enter_frames": 5,
        "exit_frames": 10,
        "persistence_window": 30,
        "min_confidence": 0.05,
        "emit_person_count": True,
        "count_debounce_frames": 8,
    },
    "quality": {"reference_height_px": 220, "min_height_px": 40},
    "integrity": {"enabled": False},
    "zones": [],
    # Opt-in stages. Off by default so an unchanged config is byte-compatible
    # with the people-and-zones path.
    "objects": {
        "enabled": False,
        "margin": 0.25,
        "min_wrist_score": 0.30,
        "persistence_window": 30,
    },
    "pose": {
        "enabled": False,
        "backend": "rtmpose",
        "model_path": "perception/models/rtmpose_t.onnx",
        "runtime": "auto",
        "device": "cpu",
    },
    "actions": {
        "enabled": False,
        "backend": "posec3d",
        "model_path": "perception/models/posec3d_pose_only.onnx",
        "every_s": 1.0,
        "device": "cpu",
        "classes": "ntu60",
        "min_confidence": 0.55,
        "min_margin": 0.15,
    },
}


def merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return merge(DEFAULTS, json.load(fh))


def check_preview_available() -> None:
    """Fail early and clearly if this OpenCV build has no GUI.

    `opencv-python-headless` has no highgui at all, and it is what a great many
    environments install by default. Without this probe the first imshow call
    raises cv2.error deep inside the capture loop, after the camera is open and
    events are already flowing. Checking up front costs one hidden window.
    """
    try:
        import numpy as np

        cv2.namedWindow("__probe__", cv2.WINDOW_NORMAL)
        cv2.imshow("__probe__", np.zeros((1, 1, 3), dtype=np.uint8))
        cv2.waitKey(1)
        cv2.destroyWindow("__probe__")
    except cv2.error as exc:
        raise RuntimeError(
            """--preview needs an OpenCV build with GUI support, and this one has none.
  Most likely you have opencv-python-headless installed:
    pip uninstall opencv-python-headless && pip install opencv-python
  On Linux over SSH you also need a display (X11 forwarding or a desktop session).
Everything except --preview works headless, so you can drop the flag and read
the events instead.
  original error: {}""".format(str(exc).strip().splitlines()[-1])
        ) from exc


def open_capture(camera_cfg: dict, source_override):
    """Open a webcam index or a video file.

    The file path is not a convenience. Replaying recorded site footage is the
    only honest way to measure a false-alarm rate, and it has to run through
    exactly this code path to mean anything.
    """
    source = source_override if source_override is not None else camera_cfg.get("index", 0)
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    if isinstance(source, int) and sys.platform == "win32":
        # MSMF takes several seconds to open a laptop webcam on Windows and
        # frequently reports the wrong resolution. DSHOW is the reliable path.
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(source)

    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_cfg.get("width", 1280))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_cfg.get("height", 720))
        cap.set(cv2.CAP_PROP_FPS, camera_cfg.get("fps", 15))

    if not cap.isOpened():
        hints = {
            "win32": "Check no other application holds the camera, and that "
                     "Settings > Privacy > Camera allows desktop apps.",
            "darwin": "macOS gates the camera per application: the terminal or "
                      "IDE running this needs Camera permission in "
                      "System Settings > Privacy & Security > Camera. Until it "
                      "is granted, capture fails or returns black frames.",
        }.get(sys.platform,
              "On Linux check the device exists (ls /dev/video*) and that your "
              "user is in the 'video' group.")
        raise RuntimeError(
            """Could not open video source {!r}.
  {}
  If the built-in camera is not index 0, try --source 1, 2, ...""".format(
                source, hints)
        )
    return cap, source


def draw_preview(frame, zones, tracks, memberships, frame_w, frame_h):
    import numpy as np

    overlay = frame.copy()
    for zone in zones:
        pts = np.array(zone.pixel_polygon(frame_w, frame_h), dtype=np.int32)
        cv2.fillPoly(overlay, [pts], (0, 140, 255))
    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)

    for zone in zones:
        pts = np.array(zone.pixel_polygon(frame_w, frame_h), dtype=np.int32)
        cv2.polylines(frame, [pts], True, (0, 140, 255), 2)
        cv2.putText(frame, zone.zone_id, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 140, 255), 1, cv2.LINE_AA)

    for track in tracks:
        det = track.detection
        inside = any(
            m.inside for (tid, _zid), m in memberships.items() if tid == track.track_id
        )
        colour = (0, 220, 0) if inside else (200, 200, 200)
        cv2.rectangle(frame, (int(det.x1), int(det.y1)), (int(det.x2), int(det.y2)), colour, 2)
        cv2.circle(frame, (int(det.ground_point[0]), int(det.ground_point[1])), 4, colour, -1)
        label = "{} {:.2f}".format(track.track_id, det.score)
        cv2.putText(frame, label, (int(det.x1), int(det.y1) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
    return frame


def resolve_stages(cfg: dict, args) -> tuple[bool, bool, bool]:
    """CLI overrides config. Objects/actions imply pose (wrists / tubes)."""
    objects_on = bool(cfg.get("objects", {}).get("enabled")) or bool(
        getattr(args, "objects", False)
    )
    actions_on = bool(cfg.get("actions", {}).get("enabled")) or bool(
        getattr(args, "actions", False)
    )
    pose_on = bool(cfg.get("pose", {}).get("enabled")) or objects_on or actions_on
    return objects_on, pose_on, actions_on


def run(cfg: dict, args) -> int:
    objects_on, pose_on, actions_on = resolve_stages(cfg, args)

    # Person-only when objects are off: keeps post-NMS cost down and stops
    # phones inheriting person track ids. Forward-pass cost is unchanged.
    detector_cfg = dict(cfg["detector"])
    if not objects_on:
        detector_cfg["classes"] = ["person"]

    em = cfg["emission"]
    qcfg = cfg["quality"]
    temperature = float(cfg["calibration"]["temperature"])

    # Build optional stages before opening the camera (and before zone load)
    # so a missing model fails immediately with the export command.
    pose_stage = None
    if pose_on:
        pose_stage = PoseStage(cfg.get("pose") or {})
        log("[perception] pose={}".format(pose_stage.name))

    object_stage = None
    if objects_on:
        configured = list(
            cfg["detector"].get("classes")
            or detector_cfg.get("classes")
            or []
        )
        object_classes = [c for c in configured if c != "person"]
        if not object_classes:
            if detector_cfg.get("backend") == "stub":
                object_classes = ["cell phone"]
            else:
                raise ValueError(
                    "objects enabled but detector.classes has no object classes. "
                    "Add COCO names alongside 'person' in the config."
                )
        ocfg = cfg.get("objects") or {}
        object_stage = ObjectStage(
            object_classes=object_classes,
            margin=float(ocfg.get("margin", 0.25)),
            min_wrist_score=float(ocfg.get("min_wrist_score", 0.30)),
            persistence_window=int(
                ocfg.get("persistence_window", em["persistence_window"])
            ),
            mode="edge",
        )
        log("[perception] objects={}".format(", ".join(object_stage.object_classes)))

    actions_stage = None
    if actions_on:
        acfg = dict(cfg.get("actions") or {})
        if getattr(args, "action_every", None) is not None:
            acfg["every_s"] = args.action_every
        actions_stage = ActionsStage(acfg)
        log(
            "[perception] actions={} every {:.1f}s".format(
                actions_stage.name, actions_stage.every_s
            )
        )

    zones = load_zones(cfg["zones"])
    detector = build_detector(detector_cfg)
    tracker = IouTracker(
        iou_threshold=cfg["tracking"]["iou_threshold"],
        max_misses=cfg["tracking"]["max_misses"],
    )
    emitter = EventEmitter(
        sensor_id=cfg["sensor_id"], integrity=cfg["integrity"].get("enabled", False)
    )

    if args.preview:
        check_preview_available()

    cap, source = open_capture(cfg["camera"], args.source)
    log("[perception] sensor={} source={} detector={} licence={}".format(
        cfg["sensor_id"], source, detector.name, detector.licence))
    log("[perception] zones={} temperature={}".format(
        [z.zone_id for z in zones], temperature))

    def person_confidence(track, detections, blur_score, luminance_score, frame_w, frame_h):
        quality_score = conf.quality(
            track.detection, frame_w, frame_h,
            others=detections,
            blur_score=blur_score,
            luminance_score=luminance_score,
            reference_height_px=qcfg["reference_height_px"],
            min_height_px=qcfg["min_height_px"],
        )
        window = em["persistence_window"]
        return conf.compose(
            raw_score=track.detection.score,
            temperature=temperature,
            quality_score=quality_score,
            persistence=track.persistence(window),
            frames_observed=min(track.hits, window),
            agreement=1.0,  # one camera: no independent corroboration to claim
        )

    # Coverage is a fact about the sensor, not an inference, so it is emitted as
    # source 'timer' with confidence 1.0 -- the same shape scenario 12 uses.
    emitter.emit(emitter.build("sensor_health", 1.0, source="timer", value="healthy"))

    memberships = {}
    zone_counts = {z.zone_id: 0 for z in zones}
    count_candidate = {}
    consecutive_read_failures = 0
    healthy = True
    frames = 0
    exit_code = 0
    started = time.monotonic()

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                consecutive_read_failures += 1
                if isinstance(source, str) and consecutive_read_failures > 2:
                    log("[perception] end of video file")
                    break
                if healthy and consecutive_read_failures >= 10:
                    emitter.emit(
                        emitter.build("sensor_health", 1.0, source="timer", value="unhealthy")
                    )
                    healthy = False
                    log("[perception] frame grabs failing -- reported coverage lost")
                if consecutive_read_failures > 300:
                    log("[perception] camera did not recover, stopping")
                    exit_code = 1
                    break
                continue

            if not healthy:
                emitter.emit(emitter.build("sensor_health", 1.0, source="timer", value="healthy"))
                healthy = True
                log("[perception] camera recovered")
            consecutive_read_failures = 0

            frames += 1
            frame_h, frame_w = frame.shape[:2]
            blur_score, luminance_score = conf.blur_and_luminance(frame)
            now_us = monotonic_us()

            detections = detector.detect(frame)
            person_dets = [d for d in detections if d.label == "person"]
            object_dets = [d for d in detections if d.label != "person"]
            # Zone / occupancy logic must only see people. An object box must
            # never mint a person track_id.
            tracks = tracker.update(person_dets)
            live_ids = set(t.track_id for t in tracks)

            for track in tracks:
                for zone in zones:
                    key = (track.track_id, zone.zone_id)
                    membership = memberships.setdefault(key, Membership())
                    raw_inside = zone.contains(track.detection.ground_point, frame_w, frame_h)
                    transition = membership.update(
                        raw_inside, em["enter_frames"], em["exit_frames"]
                    )
                    if transition is None:
                        continue

                    value, components = person_confidence(
                        track, person_dets, blur_score, luminance_score, frame_w, frame_h
                    )
                    membership.last_confidence = value

                    if value < em["min_confidence"]:
                        # Below this the event carries no information the engine
                        # can act on, and floods the log it would be found in.
                        continue
                    if transition == "left" and not membership.announced:
                        # The matching entry was suppressed, so this exit would
                        # arrive unpaired. Dropping it keeps the stream balanced.
                        continue
                    membership.announced = transition == "entered"

                    emitter.emit(emitter.build(
                        observation=(
                            "person_in_zone" if transition == "entered" else "person_left_zone"
                        ),
                        confidence=value,
                        zone_id=zone.zone_id,
                        track_id=track.track_id,
                        value=True,
                        # Identity and role are unknown: this layer runs no face
                        # recognition and reads no badges. The schema is explicit
                        # that unknown fields are omitted rather than guessed, and
                        # the engine's wrong_role rules depend on that honesty.
                        subject={"class": "human"},
                        confidence_components=components,
                    ))

            # A track that walks out of frame while inside a zone never produces
            # a geometric 'left'. Close it out explicitly, or the engine believes
            # a person entered and stayed forever.
            for key in list(memberships.keys()):
                track_id, zone_id = key
                membership = memberships[key]
                if track_id in live_ids:
                    continue
                if membership.inside:
                    membership.inside = False
                    if membership.announced:
                        membership.announced = False
                        emitter.emit(emitter.build(
                            observation="person_left_zone",
                            # Inferred from the absence of a detection rather
                            # than observed, so it is discounted against the
                            # confidence of the entry it closes out.
                            confidence=max(
                                em["min_confidence"], membership.last_confidence * 0.8
                            ),
                            zone_id=zone_id,
                            track_id=track_id,
                            value=True,
                            subject={"class": "human"},
                        ))
                if not any(t.track_id == track_id for t in tracker.tracks):
                    del memberships[key]

            if em["emit_person_count"]:
                for zone in zones:
                    occupants = [
                        t for t in tracks
                        if memberships.get((t.track_id, zone.zone_id), Membership()).inside
                    ]
                    observed = len(occupants)
                    if observed == zone_counts[zone.zone_id]:
                        count_candidate.pop(zone.zone_id, None)
                        continue
                    pending, streak = count_candidate.get(zone.zone_id, (observed, 0))
                    streak = streak + 1 if pending == observed else 1
                    count_candidate[zone.zone_id] = (observed, streak)
                    if streak < em["count_debounce_frames"]:
                        continue
                    zone_counts[zone.zone_id] = observed
                    count_candidate.pop(zone.zone_id, None)
                    # Weakest link, not the mean: one clear detection must not
                    # paper over a marginal one in the same count. Same rule the
                    # engine applies across a step's evidence.
                    if occupants:
                        occupancy_conf = min(
                            person_confidence(
                                t, person_dets, blur_score, luminance_score, frame_w, frame_h
                            )[0]
                            for t in occupants
                        )
                    else:
                        # "Nobody is in the zone" rests on the same detections
                        # that would have found somebody, so it is no more
                        # certain than the frame is usable.
                        occupancy_conf = min(1.0, blur_score * luminance_score)
                    emitter.emit(emitter.build(
                        observation="person_count",
                        confidence=occupancy_conf,
                        zone_id=zone.zone_id,
                        value=observed,
                        unit="persons",
                    ))

            poses: dict = {}
            boxes = person_boxes_from_tracks(tracks)
            if pose_stage is not None:
                poses = pose_stage.estimate(frame, boxes)

            if object_stage is not None:
                _held, object_events = object_stage.process(
                    object_dets, poses, emitter, timestamp_us=now_us
                )
                for event in object_events:
                    emitter.emit(event)

            if actions_stage is not None:
                actions_stage.push(now_us, poses, boxes)
                results = actions_stage.maybe_infer(now_us)
                for event in actions_stage.build_events(
                    emitter, results, timestamp_us=now_us
                ):
                    emitter.emit(event)

            if args.preview:
                cv2.imshow(
                    "iOACAM perception (q to quit)",
                    draw_preview(frame, zones, tracks, memberships, frame_w, frame_h),
                )
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break

            if args.max_frames and frames >= args.max_frames:
                break

    except KeyboardInterrupt:
        log("[perception] interrupted")
    finally:
        # A clean shutdown is still a loss of coverage. The engine must stop
        # treating these zones as observed, or an unwatched zone keeps reporting
        # conformant -- the exact failure scenario 12 exists to prevent.
        if healthy:
            emitter.emit(emitter.build("sensor_health", 1.0, source="timer", value="unhealthy"))
        cap.release()
        if args.preview:
            cv2.destroyAllWindows()
        elapsed = max(1e-6, time.monotonic() - started)
        log("[perception] {} frames in {:.1f}s ({:.1f} fps), {} events emitted".format(
            frames, elapsed, frames / elapsed, emitter.count))
        if frames > 30:
            fps = frames / elapsed
            log("[perception] at {:.1f} fps, enter_frames={} is {:.1f}s and "
                "exit_frames={} is {:.1f}s".format(
                    fps, em["enter_frames"], em["enter_frames"] / fps,
                    em["exit_frames"], em["exit_frames"] / fps))

    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="perception/config/zones.example.json")
    parser.add_argument("--source", default=None,
                        help="Override the camera index, or give a video file path")
    parser.add_argument("--backend", default=None,
                        choices=["yolox-onnx", "ultralytics", "stub"])
    parser.add_argument("--sensor-id", default=None)
    parser.add_argument("--preview", action="store_true",
                        help="Show the annotated frame. Use it to tune the polygon.")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--objects", action="store_true",
                        help="Enable object detection and wrist attribution "
                             "(overrides config objects.enabled)")
    parser.add_argument("--actions", action="store_true",
                        help="Enable PoseC3D action recognition "
                             "(overrides config actions.enabled; implies pose)")
    parser.add_argument("--action-every", type=float, default=None,
                        help="Seconds between action inferences per run "
                             "(PoseC3D is ~130-185 ms; not per-frame)")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        log("[perception] config not found: {}".format(args.config))
        return 2

    cfg = load_config(args.config)
    if args.backend:
        cfg["detector"]["backend"] = args.backend
    if args.sensor_id:
        cfg["sensor_id"] = args.sensor_id

    try:
        return run(cfg, args)
    except (FileNotFoundError, PermissionError, RuntimeError, ValueError,
            cv2.error) as exc:
        log("[perception] {}: {}".format(type(exc).__name__, exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
