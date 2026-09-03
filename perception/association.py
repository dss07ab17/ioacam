"""Deciding whose hand an object is in.

A detector says "there is a phone at (820, 460)". A pose estimator says "this
person's right wrist is at (835, 470)". Neither says the person is holding the
phone -- that is an inference, and this file is where it is made, on purpose,
in one place, with the rule written down rather than spread through a loop.

The rule: an object is held by a person if one of that person's wrist keypoints
falls inside the object's box, grown by a margin. Nearest wrist wins when more
than one qualifies.

Three things that rule gets right, and one it does not.

**A wrist, not a box overlap.** A phone on a desk sits inside its owner's
bounding box whenever they lean over it, so box-to-box overlap would attribute
every object on the bench to whoever is standing near it. The wrist is the only
landmark that distinguishes holding from standing beside.

**A margin, because the box is not the grip.** A hand holding a phone occludes
part of it, and the wrist is behind the hand rather than in it, so the wrist
frequently lands just outside the phone's box. The margin is what makes the
rule fire on real footage instead of only on ideal footage.

**Confidence gates the wrist, not the object.** A wrist the pose model was
unsure about must not be allowed to claim an object: an occluded wrist's
predicted position is a guess, and attributing a knife to a guess is exactly
the kind of confident wrong answer this system is built to avoid. Below
`min_wrist_score` the wrist does not participate at all.

**What it gets wrong:** two people standing shoulder to shoulder, one holding
the object, will sometimes attribute it to the wrong one, because the tracker
already swaps ids when people cross and because the nearest wrist is not always
the holding one. This is single-camera monocular association with no depth --
it is evidence, not proof, and the events it produces carry a confidence for
that reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# COCO-17 wrist indices. Named rather than inlined: 9 and 10 are meaningless at
# the call site and wrong by one is a silent, plausible bug.
LEFT_WRIST = 9
RIGHT_WRIST = 10
WRIST_INDICES = (LEFT_WRIST, RIGHT_WRIST)


@dataclass
class HeldObject:
    """One object, and the person holding it if anyone is."""

    object_track_id: str
    label: str
    score: float
    box: tuple[float, float, float, float]
    held_by: Optional[str] = None          # a person's track_id, or None
    wrist: Optional[str] = None            # which wrist claimed it
    wrist_score: float = 0.0
    distance_px: float = 0.0

    @property
    def attributed(self) -> bool:
        return self.held_by is not None


def _expanded(box, margin: float) -> tuple[float, float, float, float]:
    """Grow a box by a fraction of its own size, floored for tiny boxes.

    Proportional alone fails on the objects that matter most: a phone 30 px
    across grows by 6 px at margin 0.2, which is less than the offset between a
    wrist keypoint and the phone in the hand holding it. The floor is what
    makes small objects associable at all.
    """
    x1, y1, x2, y2 = box
    grow_x = max((x2 - x1) * margin, 12.0)
    grow_y = max((y2 - y1) * margin, 12.0)
    return (x1 - grow_x, y1 - grow_y, x2 + grow_x, y2 + grow_y)


def associate(
    objects,
    poses: dict,
    margin: float = 0.25,
    min_wrist_score: float = 0.30,
) -> list[HeldObject]:
    """Attribute each object to a person's wrist, or to nobody.

    `objects` are (track_id, Detection)-shaped: anything with `.label`,
    `.score` and the four box coordinates. `poses` is {track_id: (K,3)} exactly
    as `PoseEstimator.estimate` returns it.

    An object nobody is holding is returned with `held_by=None` rather than
    dropped. That is deliberate: a knife on a bench with no one near it is a
    real observation and often a more interesting one than a knife in a hand.
    Dropping it would make "no object" and "unattributed object" the same event
    stream, and they are not the same thing.
    """
    out: list[HeldObject] = []

    for track_id, det in objects:
        box = (det.x1, det.y1, det.x2, det.y2)
        ex1, ey1, ex2, ey2 = _expanded(box, margin)
        centre_x, centre_y = (det.x1 + det.x2) / 2.0, (det.y1 + det.y2) / 2.0

        best = None
        for person_id, keypoints in poses.items():
            for index in WRIST_INDICES:
                if index >= len(keypoints):
                    continue
                wx, wy, wscore = (float(v) for v in keypoints[index])
                if wscore < min_wrist_score:
                    # An unsure wrist is a predicted position, not an observed
                    # one, and must not be allowed to claim an object.
                    continue
                if not (ex1 <= wx <= ex2 and ey1 <= wy <= ey2):
                    continue
                distance = ((wx - centre_x) ** 2 + (wy - centre_y) ** 2) ** 0.5
                if best is None or distance < best[3]:
                    best = (person_id, index, wscore, distance)

        held = HeldObject(
            object_track_id=track_id,
            label=det.label,
            score=float(det.score),
            box=box,
        )
        if best is not None:
            held.held_by = best[0]
            held.wrist = "left_wrist" if best[1] == LEFT_WRIST else "right_wrist"
            held.wrist_score = round(best[2], 4)
            held.distance_px = round(best[3], 1)
        out.append(held)

    return out
