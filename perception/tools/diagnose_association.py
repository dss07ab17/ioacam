#!/usr/bin/env python3
"""Why did that object not attribute to anybody?

`objects_seen: {"cell phone": 8}` next to `objects_attributed: {}` is a
symptom, not a diagnosis. It is consistent with the detector being wrong, the
pose being wrong, the threshold being wrong, the margin being wrong, and the
two stages running on different frames -- and picking one to "loosen" without
looking is how a system ends up attributing a bottle on the desk to whoever
walks past.

This replays a `preview_pose --log` / `perceive` JSONL run through the REAL
`associate()` and prints, for every object detection, the numbers behind the
decision: the object box and centre, every hand anchor with its score, the
distance from the object to each anchor in pixels and as a fraction of the
object box and of body scale, and which gate rejected it.

It imports the production association code rather than reimplementing the
geometry, so what it prints is what `perceive.py` decided, not a second
opinion that can drift away from it.

    python3 perception/tools/diagnose_association.py logs/run.jsonl
    python3 perception/tools/diagnose_association.py logs/run.jsonl --verbose
    python3 perception/tools/diagnose_association.py logs/run.jsonl \
        --body-margin 0.40 --min-wrist-score 0.25

`--compare` runs the log twice, once with the object-box-only reach that was
in place before this rule changed, and prints both attribution rates. That is
the before/after number, measured on the same footage, rather than a claim.

The log must carry per-frame `people[].keypoints` and `objects[].box`, which
`preview_pose.py --objects --log` and `perceive.py --log` both write.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PERCEPTION = HERE.parent
if str(PERCEPTION) not in sys.path:
    sys.path.insert(0, str(PERCEPTION))

from association import (  # noqa: E402
    AMBIGUOUS, ATTRIBUTED, BODY_MARGIN, HAND_EXTENSION, MIN_KEYPOINT_SCORE,
    NO_PERSON, NO_USABLE_ANCHOR, OBJECT_MARGIN, OUT_OF_REACH, TIE_RATIO,
    associate,
)
from detectors.base import Detection  # noqa: E402

# Reading a log means reconstructing the two inputs association takes. A frame
# record carries everything needed for both, which is the only reason this can
# be answered offline at all.
KEYPOINT_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)

OUTCOME_BLURB = {
    NO_PERSON: "no tracked person in the frame at all",
    NO_USABLE_ANCHOR: "people present, but no wrist or elbow above min_wrist_score",
    OUT_OF_REACH: "nearest hand anchor is further from the box than reach allows",
    AMBIGUOUS: "two people are equally close; attributed to neither on purpose",
    ATTRIBUTED: "held",
}


def read_frames(path: Path):
    """Yield the frame records of a run log, skipping header and summary."""
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("type") == "frame":
                yield record


def frame_inputs(record):
    """(objects, poses, person_boxes) for one logged frame.

    The object detections are rebuilt as `Detection` so the same `associate()`
    the pipeline calls can be called on them unchanged. The logged `held_by` is
    deliberately ignored -- it is the answer under test, not an input.
    """
    objects = [
        (
            obj["track_id"],
            Detection(
                x1=obj["box"][0], y1=obj["box"][1],
                x2=obj["box"][2], y2=obj["box"][3],
                score=float(obj.get("score", 0.0)),
                label=obj["label"],
            ),
        )
        for obj in record.get("objects", [])
    ]
    poses = {p["track_id"]: p["keypoints"] for p in record.get("people", [])}
    boxes = {
        p["track_id"]: tuple(p["box"])
        for p in record.get("people", [])
        if p.get("box")
    }
    return objects, poses, boxes


def describe(report, verbose: bool) -> list[str]:
    """The numbers behind one object's decision, as printable lines."""
    x1, y1, x2, y2 = report.box
    lines = [
        "  object {} {!r} box=({:.0f},{:.0f})-({:.0f},{:.0f}) "
        "{:.0f}x{:.0f}px centre=({:.0f},{:.0f})".format(
            report.object_track_id, report.label, x1, y1, x2, y2,
            x2 - x1, y2 - y1, report.centre[0], report.centre[1],
        ),
        "    outcome: {}  -- {}".format(
            report.outcome, OUTCOME_BLURB.get(report.outcome, "")
        ),
    ]
    if report.people_seen:
        lines.append("    people={}  body_scale={}  reach={}".format(
            report.people_seen, report.body_scales, report.reaches
        ))
    if report.rejected_low_score:
        lines.append("    below min_wrist_score: " + ", ".join(
            "{}.{}={:.2f}".format(pid, name, score)
            for pid, name, score in report.rejected_low_score
        ))

    shown = report.candidates if verbose else report.candidates[:4]
    for cand in shown:
        anchor = cand.anchor
        scale = report.body_scales.get(anchor.person_id) or 0.0
        lines.append(
            "    {:<8} {:<12} at ({:7.1f},{:7.1f}) score={:.2f}  "
            "surface={:6.1f}px centre={:6.1f}px  "
            "= {:5.2f} x object  = {:5.2f} x body  reach={:6.1f}px  {}".format(
                anchor.person_id, anchor.name, anchor.x, anchor.y, anchor.score,
                cand.surface_px, cand.centre_px,
                cand.surface_over_object,
                (cand.surface_px / scale) if scale else math.inf,
                cand.reach_px,
                "WITHIN" if cand.within_reach else "beyond",
            )
        )
    if not verbose and len(report.candidates) > 4:
        lines.append("    ... {} further anchors (--verbose to show)".format(
            len(report.candidates) - 4
        ))
    if not report.candidates and report.people_seen:
        lines.append("    no anchors at all: every wrist AND elbow was below "
                     "min_wrist_score, so nothing could claim this object")
    return lines


def run(path: Path, opts, quiet: bool = False):
    """Replay a log and return (seen, attributed, outcomes) counters."""
    seen = collections.Counter()
    attributed = collections.Counter()
    outcomes = collections.Counter()
    distances = []

    for record in read_frames(path):
        objects, poses, boxes = frame_inputs(record)
        if not objects:
            continue
        held_objects = associate(
            objects, poses,
            margin=opts.margin,
            min_wrist_score=opts.min_wrist_score,
            person_boxes=None if opts.ignore_person_boxes else boxes,
            body_margin=opts.body_margin,
            hand_extension=opts.hand_extension,
            tie_ratio=opts.tie_ratio,
        )
        for held in held_objects:
            report = held.report
            seen[held.label] += 1
            outcomes[report.outcome] += 1
            if held.attributed:
                attributed[held.label] += 1
            if report.nearest is not None:
                distances.append(report.nearest.surface_px)
            if not quiet:
                print("frame {}".format(record.get("frame")))
                print("\n".join(describe(report, opts.verbose)))
    return seen, attributed, outcomes, distances


def rate(seen, attributed) -> float:
    total = sum(seen.values())
    return (sum(attributed.values()) / total) if total else 0.0


def summarise(title, seen, attributed, outcomes, distances):
    total = sum(seen.values())
    print()
    print("== {} ==".format(title))
    print("objects_seen:       {}".format(dict(seen)))
    print("objects_attributed: {}".format(dict(attributed)))
    print("attribution rate:   {}/{} = {:.0%}".format(
        sum(attributed.values()), total, rate(seen, attributed)))
    print("outcomes:           {}".format(dict(outcomes)))
    if distances:
        ordered = sorted(distances)
        print("nearest-anchor surface distance: min={:.0f} median={:.0f} "
              "max={:.0f} px".format(
                  ordered[0], ordered[len(ordered) // 2], ordered[-1]))


class _Opts:
    """A settings bundle, so `run` takes one object in both modes."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def previous_rule(objects, poses, margin: float, min_wrist_score: float):
    """The association rule exactly as it stood before this change.

    A frozen copy, kept here and nowhere else, so "before and after" is a
    measurement on the same footage rather than a recollection. Reproducing it
    by zeroing the new parameters would not be faithful: the elbow fallback and
    the tie gate have no setting that turns them off, and should not gain one
    just to make this comparison convenient.

    The rule: grow the object box by a fraction of its own size (floored at
    12 px), and attribute to the nearest wrist keypoint that lands inside.
    """
    left_wrist, right_wrist = 9, 10
    results = []
    for track_id, det in objects:
        cx, cy = (det.x1 + det.x2) / 2.0, (det.y1 + det.y2) / 2.0
        grow_x = max((det.x2 - det.x1) * margin, 12.0)
        grow_y = max((det.y2 - det.y1) * margin, 12.0)
        ex1, ey1 = det.x1 - grow_x, det.y1 - grow_y
        ex2, ey2 = det.x2 + grow_x, det.y2 + grow_y

        best = None
        for person_id, keypoints in poses.items():
            for index in (left_wrist, right_wrist):
                if index >= len(keypoints):
                    continue
                wx, wy, ws = (float(v) for v in keypoints[index])
                if ws < min_wrist_score:
                    continue
                if not (ex1 <= wx <= ex2 and ey1 <= wy <= ey2):
                    continue
                distance = math.hypot(wx - cx, wy - cy)
                if best is None or distance < best[1]:
                    best = (person_id, distance)
        results.append((det.label, best[0] if best else None))
    return results


def run_previous(path: Path, margin: float, min_wrist_score: float):
    seen = collections.Counter()
    attributed = collections.Counter()
    for record in read_frames(path):
        objects, poses, _boxes = frame_inputs(record)
        if not objects:
            continue
        for label, held_by in previous_rule(objects, poses, margin,
                                            min_wrist_score):
            seen[label] += 1
            if held_by is not None:
                attributed[label] += 1
    return seen, attributed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("log", type=Path,
                        help="a run JSONL with per-frame people and objects")
    parser.add_argument("--verbose", action="store_true",
                        help="print every anchor, not just the nearest four")
    parser.add_argument("--quiet", action="store_true",
                        help="totals only, no per-object detail")
    parser.add_argument("--compare", action="store_true",
                        help="also replay with the previous object-box-only "
                             "reach, and print both rates")
    parser.add_argument("--margin", type=float, default=OBJECT_MARGIN,
                        help="reach as a fraction of the object box "
                             "(default %(default)s)")
    parser.add_argument("--body-margin", type=float, default=BODY_MARGIN,
                        help="reach as a fraction of body scale "
                             "(default %(default)s)")
    parser.add_argument("--hand-extension", type=float, default=HAND_EXTENSION,
                        help="wrist-to-grip as a fraction of the forearm "
                             "(default %(default)s)")
    parser.add_argument("--min-wrist-score", type=float,
                        default=MIN_KEYPOINT_SCORE,
                        help="keypoints below this do not participate "
                             "(default %(default)s)")
    parser.add_argument("--tie-ratio", type=float, default=TIE_RATIO,
                        help="two people within this fraction of reach are "
                             "treated as tied (default %(default)s)")
    parser.add_argument("--ignore-person-boxes", action="store_true",
                        help="derive body scale from keypoints alone, as if "
                             "the caller passed no person boxes")
    args = parser.parse_args(argv)

    if not args.log.exists():
        print("no such log: {}".format(args.log), file=sys.stderr)
        return 2

    opts = _Opts(
        margin=args.margin, body_margin=args.body_margin,
        hand_extension=args.hand_extension,
        min_wrist_score=args.min_wrist_score, tie_ratio=args.tie_ratio,
        ignore_person_boxes=args.ignore_person_boxes, verbose=args.verbose,
    )
    seen, attributed, outcomes, distances = run(args.log, opts, args.quiet)
    if sum(seen.values()) == 0:
        print("no object detections in {} -- nothing to diagnose. Capture one "
              "with:\n  python3 perception/tools/preview_pose.py --objects "
              "--log logs/run.jsonl".format(args.log), file=sys.stderr)
        return 1
    summarise("current rule", seen, attributed, outcomes, distances)

    if args.compare:
        b_seen, b_attr = run_previous(args.log, args.margin,
                                      args.min_wrist_score)
        summarise("previous rule (object-box reach, wrist only)",
                  b_seen, b_attr, {}, [])
        print()
        print("before -> after: {}/{} = {:.0%}  ->  {}/{} = {:.0%}".format(
            sum(b_attr.values()), sum(b_seen.values()), rate(b_seen, b_attr),
            sum(attributed.values()), sum(seen.values()),
            rate(seen, attributed)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
