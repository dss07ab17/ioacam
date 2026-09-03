#!/usr/bin/env python3
"""Watch RTMPose run on a live camera. The eyeball test the unit tests cannot do.

    python3 perception/tools/preview_pose.py                    # live window
    python3 perception/tools/preview_pose.py --source 1         # second camera
    python3 perception/tools/preview_pose.py --detector         # real person boxes
    python3 perception/tools/preview_pose.py --max-frames 40 --save shot.png
    python3 perception/tools/preview_pose.py --log              # record the run

## The log

`--log` writes JSON lines to `logs/pose-<timestamp>.jsonl` (or a path you give
it). One header line naming the model, runtime, detector and threshold; one
line per frame with the box, all 17 keypoints as x/y/score, and that frame's
detect and pose milliseconds; one summary line with the per-joint means. The
summary lives in the same file as the frames on purpose -- totals kept beside a
log get separated from it the first time anyone copies one of the two.

Roughly 600 bytes a frame, so about 5 KB/s at 9 fps. Fine for a drill, not for
a shift. `logs/` is already gitignored.

Note what this file is: keypoints of an identifiable person, which is
biometric-adjacent personal data. On your own machine debugging your own
camera that is unremarkable. On a deployment it is a retention question with a
legal answer, and this tool is not the place that answer gets decided by
default -- which is why logging is opt-in and off unless you ask for it.

A synthetic figure proves the decode is wired up correctly. It cannot tell you
whether the model is any good on a real body in real light, at the angle and
distance your camera actually sits at -- and that is the question the board
measurement eventually turns on. Ten seconds in front of a webcam answers it.

## Where the boxes come from

RTMPose is top-down: it needs a person box before it can do anything. Two ways
to get one here:

**Whole frame (default).** The centre 3:4 of the frame is treated as one
person. Right for a laptop webcam, where there is one subject and they are in
the middle, and it needs nothing downloaded. It is not the production path:
with no detector there is no second person, no tracking, and no box that
follows anyone.

**--detector.** The configured YOLOX detector, exactly as the pipeline uses it,
with a track id per box. Needs the weights:

    python3 perception/tools/fetch_model.py --model yolox_tiny

That is the honest end-to-end test, and the one that shows what pose costs on
top of detection rather than on its own.

## What to look for

Joint colour is the model's confidence, which is the number `posetube.py` turns
into blob amplitude. Watch what happens when you put a hand behind your back:
the wrist should go dim rather than disappearing or, worse, staying bright in
the wrong place. That fading is the property the whole heatmap design rests on,
and this is the only place you can see it happen.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent          # perception/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "actions"))

from association import associate  # noqa: E402
from base import AbstentionPolicy  # noqa: E402
from emit import EventEmitter, monotonic_us, wall_time_now  # noqa: E402
from perceive import check_preview_available, open_capture  # noqa: E402
from posec3d import NTU60_CLASSES, PoseC3DRecognizer  # noqa: E402
from posetube import COCO_KEYPOINTS, PoseFrameRecord, PoseTubeExtractor  # noqa: E402
from rtmpose import RTMPoseEstimator  # noqa: E402
from tracking import IouTracker, MultiClassTracker  # noqa: E402

DEFAULT_MODEL = str(ROOT / "models" / "rtmpose_t.onnx")
DEFAULT_ACTION_MODEL = str(ROOT / "models" / "posec3d_pose_only.onnx")

# COCO-17 skeleton. Legs, arms, torso, face -- in that order, so the drawing
# order puts the face on top where it is smallest.
EDGES = (
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
    (0, 1), (0, 2), (1, 3), (2, 4),
)

# Left limbs warm, right limbs cool, so a left/right swap is visible instantly
# rather than looking like a plausible skeleton.
LEFT = {1, 3, 5, 7, 9, 11, 13, 15}


def confidence_colour(score: float) -> tuple[int, int, int]:
    """Red at 0, green at 1. The point is that low confidence looks wrong."""
    return (0, int(255 * min(1.0, max(0.0, score))), int(255 * (1 - min(1.0, max(0.0, score)))))


def draw_pose(frame, keypoints, box, min_score: float) -> None:
    x1, y1, x2, y2 = (int(v) for v in box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (90, 90, 90), 1)

    for a, b in EDGES:
        if keypoints[a, 2] < min_score or keypoints[b, 2] < min_score:
            continue
        colour = (80, 180, 255) if a in LEFT or b in LEFT else (255, 180, 80)
        cv2.line(
            frame,
            (int(keypoints[a, 0]), int(keypoints[a, 1])),
            (int(keypoints[b, 0]), int(keypoints[b, 1])),
            colour, 2, cv2.LINE_AA,
        )

    for k in range(keypoints.shape[0]):
        x, y, score = keypoints[k]
        if score < min_score:
            continue
        # Radius as well as colour: a dim small dot reads as uncertainty at a
        # glance, which is exactly how the tube treats it.
        radius = 2 + int(3 * score)
        cv2.circle(frame, (int(x), int(y)), radius, confidence_colour(score), -1, cv2.LINE_AA)


def contact_sheet(samples, columns: int = 3, cell=(426, 240)):
    """Tile sampled frames into one image, so a whole drill can be read at once.

    A live window tells you the model works *now*. It cannot tell you which of
    the poses you just cycled through it handled, because you were busy holding
    them. Sampling on a timer and tiling the result is how one 30-second run
    becomes a reviewable answer to "does it cope with X".
    """
    if not samples:
        return None
    cw, ch = cell
    rows = (len(samples) + columns - 1) // columns
    sheet = np.zeros((rows * ch, columns * cw, 3), dtype=np.uint8)
    for i, (image, label) in enumerate(samples):
        cell_img = cv2.resize(image, (cw, ch), interpolation=cv2.INTER_AREA)
        cv2.putText(cell_img, label, (8, ch - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 2, cv2.LINE_AA)
        r, c = divmod(i, columns)
        sheet[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw] = cell_img
    return sheet


def whole_frame_box(width: int, height: int) -> tuple[float, float, float, float]:
    """The centre 3:4 of the frame, as one person-shaped box."""
    side_h = float(height)
    side_w = side_h * 0.75
    cx = width / 2.0
    return (cx - side_w / 2, 0.0, cx + side_w / 2, side_h)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--source", default=None,
                        help="camera index or a video file; default is camera 0")
    parser.add_argument("--runtime", default="auto",
                        choices=("auto", "onnxruntime", "cv2", "torch"))
    parser.add_argument("--detector", action="store_true",
                        help="use the configured YOLOX detector for boxes")
    parser.add_argument("--config", default=str(ROOT / "config" / "webcam.json"))
    parser.add_argument("--min-score", type=float, default=0.30)
    parser.add_argument("--max-frames", type=int, default=0,
                        help="stop after N frames; 0 runs until q or Esc")
    parser.add_argument("--save", default=None,
                        help="write the last annotated frame here")
    parser.add_argument("--no-window", action="store_true",
                        help="run headless, for a smoke test over --save")
    parser.add_argument("--objects", action="store_true",
                        help="also detect the configured object classes and "
                             "attribute them to the wrist holding them")
    parser.add_argument("--object-margin", type=float, default=0.25,
                        help="how far outside an object's box a wrist may be "
                             "and still count as holding it")
    parser.add_argument("--actions", action="store_true",
                        help="run the full chain: pose -> tube -> PoseC3D -> "
                             "abstention (implies --detector)")
    parser.add_argument("--action-model", default=DEFAULT_ACTION_MODEL)
    parser.add_argument("--action-every", type=float, default=1.0,
                        help="seconds between action inferences per frame; "
                             "PoseC3D costs ~130 ms, so this is not per-frame")
    parser.add_argument("--min-confidence", type=float, default=0.55)
    parser.add_argument("--min-margin", type=float, default=0.15)
    parser.add_argument("--log", nargs="?", const="", default=None,
                        metavar="PATH",
                        help="write a JSON-lines record of the run; bare --log "
                             "defaults to logs/pose-<timestamp>.jsonl")
    parser.add_argument("--sheet", default=None,
                        help="write a contact sheet of sampled frames here")
    parser.add_argument("--sheet-every", type=float, default=3.0,
                        help="seconds between contact-sheet samples")
    args = parser.parse_args()

    show = not args.no_window
    if show:
        check_preview_available()

    estimator = RTMPoseEstimator(model_path=args.model, runtime=args.runtime)

    detector = None
    tracker = None
    if args.detector or args.actions or args.objects:
        from detectors import build_detector
        from perceive import load_config

        cfg = load_config(args.config)
        detector_cfg = dict(cfg["detector"])
        if not args.objects:
            # Person only unless objects were asked for, so the default path
            # keeps exactly the cost it had.
            detector_cfg["classes"] = ["person"]
        detector = build_detector(detector_cfg)
        # A tube accumulates 1.5 seconds of one person. Keyed on a raw
        # detection index that reshuffles between frames, it would splice two
        # people into one volume and classify the splice -- so identity comes
        # from the tracker, which is the whole reason the pose stage is
        # top-down in the first place.
        tracker = IouTracker()
        print(f"boxes from {detector.name}, ids from the IoU tracker",
              file=sys.stderr)

    object_tracker = emitter = None
    object_classes: list[str] = []
    if args.objects:
        object_classes = [c for c in detector.classes if c != "person"]
        if not object_classes:
            raise SystemExit(
                "no object classes configured. Add a 'classes' list to the "
                "detector block of " + args.config + " -- the point of this "
                "flag is that the list is a site decision, not a constant in "
                "the source."
            )
        # One detector pass for people and objects together, split by label
        # afterwards. A second instance would double the most expensive stage
        # in the pipeline to re-run the same convolutions over the same frame;
        # the per-class thresholds that motivated a second instance are
        # already per class inside one.
        object_tracker = MultiClassTracker()
        emitter = EventEmitter(sensor_id=cfg.get("sensor_id", "cam-01"))
        print(
            "objects: " + ", ".join(
                f"{c} @ {detector.class_thresholds[c]:.2f}" for c in object_classes
            ),
            file=sys.stderr,
        )
    else:
        print(
            "boxes from the whole frame -- one subject, centred. "
            "Add --detector once yolox_tiny.onnx is fetched for the real path.",
            file=sys.stderr,
        )

    extractor = recognizer = policy = None
    if args.actions:
        # Defaults, deliberately: 32 frames of 56x56 over 17 keypoints are what
        # the checkpoint was trained on. Passing anything else here would
        # produce a volume the network accepts and misreads, since the head
        # global-pools before the classifier and never objects to the shape.
        extractor = PoseTubeExtractor()
        recognizer = PoseC3DRecognizer(
            model_path=args.action_model,
            classes=NTU60_CLASSES,
            num_frames=extractor.num_frames,
            heatmap_size=extractor.heatmap_size,
            num_keypoints=extractor.num_keypoints,
            device="cpu",
        )
        policy = AbstentionPolicy(
            min_confidence=args.min_confidence, min_margin=args.min_margin
        )
        print(
            f"actions from {recognizer.name} "
            f"({extractor.num_frames} frames, {extractor.heatmap_size}px, "
            f"{len(NTU60_CLASSES)} NTU classes), "
            f"every {args.action_every:.1f}s per track",
            file=sys.stderr,
        )
        print(
            "NOTE: NTU-60 weights are research-only, and its vocabulary is "
            "daily-living actions recorded in a lab -- see "
            "perception/actions/LICENCE-NOTES.md.",
            file=sys.stderr,
        )

    log_file = None
    if args.log is not None:
        log_path = Path(args.log) if args.log else (
            ROOT.parent / "logs" /
            f"pose-{datetime.now().strftime('%Y%m%d-%H%M%S')}.jsonl"
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("w", encoding="utf-8")
        print(f"logging to {log_path}", file=sys.stderr)

    cap, source = open_capture({"index": 0, "width": 1280, "height": 720, "fps": 15},
                               args.source)
    print(f"camera {source} open; q or Esc to stop", file=sys.stderr)

    def write(record: dict) -> None:
        if log_file is not None:
            log_file.write(json.dumps(record) + "\n")

    # A run that cannot say what produced it is not evidence of anything. Two
    # numbers from different models, runtimes or thresholds look identical in a
    # bare log, so the header carries everything needed to tell them apart.
    write({
        "type": "run",
        "started": wall_time_now(),
        "model": args.model,
        "runtime": args.runtime,
        "detector": detector.name if detector is not None else "whole-frame",
        "source": str(source),
        "min_score": args.min_score,
        "keypoints": list(COCO_KEYPOINTS),
    })

    frames = 0
    people = 0
    pose_ms = 0.0
    detect_ms = 0.0
    scores: list[float] = []
    per_joint: list[np.ndarray] = []
    actions: dict = {}
    last_action_line: dict = {}
    last_action = -1e9
    action_ms = 0.0
    object_ms = 0.0
    seen_objects = collections.Counter()
    attributed_objects = collections.Counter()
    tube_ms = 0.0
    infer_ms = 0.0
    action_runs = 0
    action_tubes = 0
    samples: list[tuple[np.ndarray, str]] = []
    next_sample = args.sheet_every
    started = time.perf_counter()
    annotated = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames += 1
            height, width = frame.shape[:2]

            det_frame_ms = 0.0
            if detector is not None:
                t_det = time.perf_counter()
                detections = detector.detect(frame)
                det_frame_ms = (time.perf_counter() - t_det) * 1000
                detect_ms += det_frame_ms
                object_detections = [d for d in detections if d.label != "person"]
                detections = [d for d in detections if d.label == "person"]
                boxes = {
                    t.track_id: (t.detection.x1, t.detection.y1,
                                 t.detection.x2, t.detection.y2)
                    for t in tracker.update(detections)
                }
            else:
                boxes = {"whole-frame": whole_frame_box(width, height)}

            now_us_frame = monotonic_us()
            t0 = time.perf_counter()
            poses = estimator.estimate(frame, boxes)
            pose_frame_ms = (time.perf_counter() - t0) * 1000
            pose_ms += pose_frame_ms
            people += len(poses)

            for track_id, keypoints in poses.items():
                draw_pose(frame, keypoints, boxes[track_id], args.min_score)
                scores.append(float(keypoints[:, 2].mean()))
                per_joint.append(keypoints[:, 2].copy())

            held_objects = []
            object_events = []
            object_frame_ms = 0.0
            if object_tracker is not None:
                t_obj = time.perf_counter()
                object_tracks = object_tracker.update(object_detections)
                held_objects = associate(
                    [(t.track_id, t.detection) for t in object_tracks],
                    poses,
                    margin=args.object_margin,
                    min_wrist_score=args.min_score,
                )
                object_frame_ms = (time.perf_counter() - t_obj) * 1000
                object_ms += object_frame_ms

                by_id = {t.track_id: t for t in object_tracks}
                for held in held_objects:
                    seen_objects[held.label] += 1
                    if held.attributed:
                        attributed_objects[held.label] += 1

                    # Persistence over the track's own history, exactly as the
                    # person path does it. A phone seen in 2 frames of 30 is a
                    # flicker, and the confidence has to say so rather than the
                    # event simply not existing.
                    track = by_id[held.object_track_id]
                    persistence = track.persistence(30)
                    event = emitter.build(
                        observation="object_at_station",
                        # Raw detector score times persistence. NOT the
                        # calibrated composition confidence.py builds for
                        # people: quality is computed against person box height
                        # and a temperature for objects has never been fitted.
                        # Claiming a calibrated number here would be inventing
                        # one -- see the note in the summary.
                        confidence=round(held.score * persistence, 4),
                        value=held.label,
                        # The whole point: attributed to the person holding it,
                        # or emitted with no track_id at all. An object nobody
                        # is holding is a real observation, and dropping it
                        # would make "no object" and "unattributed object"
                        # indistinguishable downstream.
                        track_id=held.held_by,
                        subject={"class": "object"},
                        timestamp_us=now_us_frame,
                    )
                    object_events.append(event)

            action_frame_ms = 0.0
            if extractor is not None:
                now_us = now_us_frame
                extractor.push(PoseFrameRecord(
                    timestamp_us=now_us, keypoints=poses, boxes=boxes,
                ))

                # Not every frame. One tube costs ~130 ms against a ~110 ms
                # frame budget, so classifying continuously would halve the
                # capture rate to re-answer a question whose evidence window is
                # 1.5 seconds long and barely moves between frames.
                if time.perf_counter() - last_action >= args.action_every:
                    last_action = time.perf_counter()
                    t_act = time.perf_counter()
                    ready = extractor.tracks_ready(now_us)
                    for track_id in ready:
                        t_tube = time.perf_counter()
                        tube = extractor.extract(track_id, now_us)
                        if tube is None:
                            continue
                        tube_ms += (time.perf_counter() - t_tube) * 1000
                        t_inf = time.perf_counter()
                        raw = recognizer.infer(tube)
                        infer_ms += (time.perf_counter() - t_inf) * 1000
                        result = policy.decide(
                            track_id, raw, backend=recognizer.name,
                        )
                        # Carry the tube's window onto the decision. Without it
                        # a stale verdict is indistinguishable from a fresh
                        # one, and every action finding is at least a window
                        # old by construction.
                        result.window_start_us = tube.window_start_us
                        result.window_end_us = tube.window_end_us
                        actions[track_id] = (result, tube.coverage)

                        line = (
                            f"{track_id}  "
                            + (f"{result.label} {result.confidence:.2f}"
                               if result.decided
                               else f"UNKNOWN ({result.reason})")
                            + f"  [coverage {tube.coverage:.2f}]"
                        )
                        if line != last_action_line.get(track_id):
                            print(f"  action  {line}", file=sys.stderr)
                            last_action_line[track_id] = line

                    # Tracks the tracker dropped must not keep a verdict on
                    # screen; a label outliving its subject is a lie.
                    for gone in set(actions) - set(ready):
                        actions.pop(gone, None)
                        last_action_line.pop(gone, None)
                    action_frame_ms = (time.perf_counter() - t_act) * 1000
                    action_ms += action_frame_ms
                    action_runs += 1
                    action_tubes += len(ready)

            for held in held_objects:
                ox1, oy1, ox2, oy2 = (int(v) for v in held.box)
                # Attributed objects are drawn joined to the wrist that claimed
                # them. The line is the inference, and it is the thing to check
                # by eye -- a line to the wrong person is the failure this
                # whole file exists to make visible.
                colour = (60, 220, 200) if held.attributed else (150, 150, 150)
                cv2.rectangle(frame, (ox1, oy1), (ox2, oy2), colour, 2)
                caption = held.label + (f" -> {held.held_by}" if held.attributed
                                        else " (unheld)")
                cv2.putText(frame, f"{caption} {held.score:.2f}",
                            (ox1, max(16, oy1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, colour, 2, cv2.LINE_AA)
                if held.attributed and held.held_by in poses:
                    index = 9 if held.wrist == "left_wrist" else 10
                    wx, wy = poses[held.held_by][index][:2]
                    cv2.line(frame, (int((ox1 + ox2) / 2), int((oy1 + oy2) / 2)),
                             (int(wx), int(wy)), colour, 1, cv2.LINE_AA)

            for track_id, (result, coverage) in actions.items():
                if track_id not in boxes:
                    continue
                x1, y1 = int(boxes[track_id][0]), int(boxes[track_id][1])
                label = (f"{result.label} {result.confidence:.2f}"
                         if result.decided else "unknown")
                colour = (120, 230, 120) if result.decided else (120, 200, 255)
                cv2.putText(frame, f"{track_id}: {label}", (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA)

            if log_file is not None:
                write({
                    "type": "frame",
                    "frame": frames,
                    "t_us": monotonic_us(),
                    "detect_ms": round(det_frame_ms, 2),
                    "pose_ms": round(pose_frame_ms, 2),
                    "people": [
                        {
                            "track_id": tid,
                            "box": [round(float(v), 1) for v in boxes[tid]],
                            # x, y, score per joint, in COCO order. Rounded:
                            # sub-pixel precision beyond a decimal is noise at
                            # this resolution, and it halves the file.
                            "keypoints": [
                                [round(float(x), 1), round(float(y), 1), round(float(sc), 3)]
                                for x, y, sc in kp
                            ],
                        }
                        for tid, kp in poses.items()
                    ],
                    # Every frame carries the current verdict, including the
                    # frames between inferences. The window bounds are what
                    # say how old it is, so a reader can tell a fresh decision
                    # from one being held over without guessing the cadence.
                    "actions": {
                        tid: {
                            "label": result.label,
                            "confidence": result.confidence,
                            "abstained": result.abstained,
                            "reason": result.reason,
                            "coverage": coverage,
                            "window_start_us": result.window_start_us,
                            "window_end_us": result.window_end_us,
                        }
                        for tid, (result, coverage) in actions.items()
                    },
                    "action_ms": round(action_frame_ms, 2),
                    "object_ms": round(object_frame_ms, 2),
                    # Logged the same shape as people: one entry per tracked
                    # object per frame, with what claimed it. This is the
                    # record a false-positive rate gets counted from, so an
                    # unattributed object is present here, not filtered out.
                    "objects": [
                        {
                            "track_id": held.object_track_id,
                            "label": held.label,
                            "score": round(held.score, 3),
                            "box": [round(v, 1) for v in held.box],
                            "held_by": held.held_by,
                            "wrist": held.wrist,
                            "wrist_score": held.wrist_score,
                            "distance_px": held.distance_px,
                        }
                        for held in held_objects
                    ],
                    # The events as the engine would receive them, schema-shaped
                    # and built by the real EventEmitter -- but written here
                    # rather than to stdout, because nothing has measured the
                    # false-positive rate yet and stdout is the engine's feed.
                    "events": object_events,
                })

            elapsed = time.perf_counter() - started
            readout = f"{frames / max(elapsed, 1e-6):4.1f} fps   "
            if detector is not None:
                readout += f"detect {detect_ms / max(frames, 1):5.1f} ms   "
            readout += (
                f"pose {pose_ms / max(frames, 1):5.1f} ms   {len(poses)} person(s)"
            )
            cv2.putText(
                frame,
                readout,
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA,
            )
            annotated = frame

            if args.sheet and elapsed >= next_sample:
                mean = float(np.mean([kp[:, 2].mean() for kp in poses.values()])) if poses else 0.0
                samples.append((frame.copy(), f"t={elapsed:4.1f}s  score {mean:.2f}"))
                next_sample += args.sheet_every

            if show:
                cv2.imshow("rtmpose preview", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            if args.max_frames and frames >= args.max_frames:
                break
    finally:
        cap.release()
        if show:
            cv2.destroyAllWindows()

    if log_file is not None:
        # The summary goes in the same file as the frames, not beside it. A
        # log whose totals live somewhere else gets separated from them the
        # first time anyone copies one of the two.
        elapsed = time.perf_counter() - started
        summary = {
            "type": "summary",
            "ended": wall_time_now(),
            "frames": frames,
            "fps": round(frames / max(elapsed, 1e-6), 2),
            "detect_ms_mean": round(detect_ms / max(frames, 1), 2),
            "pose_ms_mean": round(pose_ms / max(frames, 1), 2),
            "person_inferences": people,
        }
        if seen_objects:
            summary["objects_seen"] = dict(seen_objects)
            summary["objects_attributed"] = dict(attributed_objects)
            summary["object_ms_mean"] = round(object_ms / max(frames, 1), 2)
        if action_runs:
            summary["action_runs"] = action_runs
            summary["action_ms_mean"] = round(action_ms / action_runs, 2)
            summary["tube_ms_mean"] = round(tube_ms / max(action_tubes, 1), 2)
            summary["posec3d_ms_mean"] = round(infer_ms / max(action_tubes, 1), 2)
        if per_joint:
            stack = np.stack(per_joint)
            summary["mean_keypoint_score"] = round(float(np.mean(scores)), 4)
            summary["per_joint"] = {
                joint: {
                    "mean": round(float(stack[:, k].mean()), 4),
                    "above_threshold": round(
                        float((stack[:, k] >= args.min_score).mean()), 4
                    ),
                }
                for k, joint in enumerate(COCO_KEYPOINTS)
            }
        write(summary)
        log_file.close()

    if args.sheet and samples:
        sheet = contact_sheet(samples)
        cv2.imwrite(args.sheet, sheet)
        print(f"saved {args.sheet} ({len(samples)} samples)", file=sys.stderr)

    if args.save and annotated is not None:
        cv2.imwrite(args.save, annotated)
        print(f"saved {args.save}", file=sys.stderr)

    elapsed = time.perf_counter() - started
    print()
    print(f"frames          {frames}")
    print(f"capture         {frames / max(elapsed, 1e-6):.1f} fps overall")
    if detector is not None:
        print(f"detect          {detect_ms / max(frames, 1):.1f} ms per frame")
    print(f"pose            {pose_ms / max(frames, 1):.1f} ms per frame, "
          f"{people} person-inferences")
    if action_runs:
        # Split, because the two halves have different fixes. The volume is
        # numpy on the CPU and stays there; the network is the part that would
        # move to an NPU. Reporting one number hides which one to attack.
        print(f"action          {action_ms / action_runs:.1f} ms per run "
              f"({action_runs} runs at {args.action_every:.1f}s, "
              f"{action_tubes} tube(s))")
        if action_tubes:
            print(f"  tube build    {tube_ms / action_tubes:.1f} ms per tube "
                  f"(numpy heatmap volume)")
            print(f"  posec3d       {infer_ms / action_tubes:.1f} ms per tube "
                  f"(onnxruntime)")
    if detector is not None:
        # The number the board question actually turns on: pose is a *second*
        # model, and what matters is the two of them together against the frame
        # interval, not either one on its own.
        total = (detect_ms + pose_ms) / max(frames, 1)
        print(f"both models     {total:.1f} ms per frame "
              f"-- {total / 1000 * 15 * 100:.0f}% of a 15 fps frame budget")
    if seen_objects:
        print()
        print(f"objects         {object_ms / max(frames, 1):.1f} ms per frame "
              f"(tracking + wrist association; detection is in the line above)")
        print("  class            detections   attributed to a wrist")
        for label, count in seen_objects.most_common():
            held = attributed_objects[label]
            print(f"  {label:<16} {count:6d}       {held:5d}  "
                  f"({100 * held / count:4.1f}%)")
        print()
        print("Detections here are per frame per track, not distinct objects. "
              "Read the columns together: a class with many detections and "
              "none ever attributed is either furniture or a false positive, "
              "and the log has the boxes to tell which.")

    if scores:
        print(f"mean keypoint   {float(np.mean(scores)):.3f}  "
              f"(worst frame {float(np.min(scores)):.3f})")
        print()
        # Per joint, not just the mean. A mediocre average can mean "the whole
        # skeleton is shaky" or "the visible half is solid and the occluded
        # half is honestly unsure" -- opposite situations, and only this tells
        # them apart. It is also how you see which body parts a given camera
        # angle simply never gets.
        stack = np.stack(per_joint)
        print("per joint (mean score, and how often it clears the threshold):")
        for k, joint in enumerate(COCO_KEYPOINTS):
            column = stack[:, k]
            print(f"  {joint:<16} {column.mean():.3f}   "
                  f"{100 * float((column >= args.min_score).mean()):5.1f}% of frames")
        print()
        print("A mean below ~0.3 on a person filling the frame means something is "
              "wrong upstream -- wrong box, wrong colour order, or the wrong graph.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
