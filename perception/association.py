"""Deciding whose hand an object is in.

A detector says "there is a phone at (820, 460)". A pose estimator says "this
person's left wrist is at (931, 470)". Neither says the person is holding the
phone -- that is an inference, and this file is where it is made, on purpose,
in one place, with the rule written down rather than spread through a loop.

The rule, in three parts:

**Anchor.** For each arm, the best available estimate of where the *hand* is.
COCO-17 has no finger keypoints, so the wrist is the closest landmark the pose
model offers -- and it is the wrong one. A phone is gripped several centimetres
beyond the wrist, in the direction the forearm points. So when the elbow and
the wrist are both confident, the anchor is the wrist pushed forward along the
elbow-to-wrist vector by a fraction of the forearm's length. The plain wrist is
kept as a second anchor because extrapolation overshoots when the arm is
foreshortened, and the elbow serves as a third when the wrist is not confident
enough to be believed at all.

**Reach.** How far from that anchor an object may sit and still be held. This
used to be a fraction of the OBJECT's box, which is exactly backwards: a phone
60 px across allowed a 15 px reach, which is smaller than the wrist-to-grip
offset the anchor exists to model, while a laptop 400 px across allowed 100 px
for no reason but its own size. Reach is a property of the arm, not of the
thing in it, so it is now a fraction of BODY SCALE -- the longest apparent limb
on that person -- with the old object-box fraction kept as a floor so a large
object is still associable when it is genuinely under the hand.

Apparent limb length is used because perspective can only ever *shorten* a limb
in the image, never lengthen it, so the longest limb measured is the least
foreshortened one and the best estimate of that person's scale in pixels. It
also does the right thing automatically as somebody walks towards the camera.

**Doubt.** Two gates, both of which decline to answer rather than guess. A
keypoint below `min_wrist_score` is a predicted position, not an observed one,
and attributing a knife to a prediction is precisely the confident wrong answer
this system exists to avoid. And when two different people's anchors are within
a hair of each other, the object is attributed to NEITHER: at that point the
nearest-anchor rule is a coin toss, and a coin toss recorded as evidence is
worse than a gap in the record.

**What it still gets wrong:** this is single-camera monocular association with
no depth. An object directly behind a hand looks exactly like an object in it.
The tracker also swaps ids when people cross. The events this produces carry a
confidence, and the anchor kind records which of the three anchors claimed the
object, so a downstream consumer can tell a wrist-in-the-box attribution from
an extrapolated-hand one.

Every decision above is recorded on `HeldObject.report`, whether it succeeded
or failed, so `perception/tools/diagnose_association.py` can print the numbers
rather than leaving "it did not attribute" as the only observation available.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

# COCO-17 indices. Named rather than inlined: 9 and 10 are meaningless at the
# call site and wrong by one is a silent, plausible bug.
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_ELBOW, RIGHT_ELBOW = 7, 8
LEFT_WRIST, RIGHT_WRIST = 9, 10
WRIST_INDICES = (LEFT_WRIST, RIGHT_WRIST)

# (side, shoulder, elbow, wrist) -- one entry per arm, so the anchor builder is
# a loop over arms rather than the same eight lines written twice.
ARMS = (
    ("left", LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
    ("right", RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST),
)

# Pairs of keypoints whose apparent separation estimates body scale. Limbs
# only: head keypoints sit close together and would drag the maximum down for
# nobody's benefit, and the hips and legs are frequently out of frame on the
# torso-filling views this runs on.
SCALE_PAIRS = (
    (LEFT_SHOULDER, RIGHT_SHOULDER),      # shoulder width
    (LEFT_SHOULDER, LEFT_ELBOW),          # upper arms
    (RIGHT_SHOULDER, RIGHT_ELBOW),
    (LEFT_ELBOW, LEFT_WRIST),             # forearms
    (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_SHOULDER, LEFT_WRIST),          # whole arm, when it is extended
    (RIGHT_SHOULDER, RIGHT_WRIST),
)

# ----------------------------------------------------------------------
# Defaults. Each is a claim about anatomy or about the failure it prevents,
# so each is named and justified rather than appearing as a bare number in a
# signature. A site overrides them in the `objects` config block.
# ----------------------------------------------------------------------

# Wrist-to-grip, as a fraction of forearm length. A forearm is about 26 cm and
# the centre of a gripping hand sits about 9 cm beyond the wrist crease, so
# roughly a third. This is the whole reason a phone in the fingers used to miss.
HAND_EXTENSION = 0.35

# Reach, as a fraction of body scale. Body scale is the longest apparent limb,
# which on a torso-filling view is roughly a whole arm; a third of that is about
# a hand's width plus the depth error of a monocular guess. Loose enough for a
# phone in the fingers, tight enough that a bottle on the far side of the desk
# stays unattributed -- which is the balance being struck, and the reason this
# is not simply set to whatever makes the current run attribute.
BODY_MARGIN = 0.33

# Reach floor, as a fraction of the object's own longest side. Keeps a large
# object associable when a hand is genuinely on it but the person's limbs are
# short in the image. This was the ONLY term, and is now the smaller of the two
# on exactly the small objects it was failing.
OBJECT_MARGIN = 0.25

# Absolute floor in pixels, for the case where pose gives no usable scale at
# all and the object is tiny. Small on purpose: a backstop, not a policy.
MIN_REACH_PX = 12.0

# A keypoint below this is a prediction rather than an observation.
MIN_KEYPOINT_SCORE = 0.30

# Two people count as tied when their best anchors differ by less than this
# fraction of the reach. Relative to reach so it scales with the subject's
# distance from the camera, instead of being a pixel count that means different
# things at different depths.
TIE_RATIO = 0.10

# A limb longer than half the person's own box diagonal is a pose failure, not
# a long arm. Clamping stops one wild keypoint from granting that person a
# reach across the whole frame.
MAX_SCALE_FRACTION_OF_BOX = 0.5

# Rejection reasons, so callers compare against a name instead of a string
# literal that can be misspelled without failing.
NO_PERSON = "no_person"
NO_USABLE_ANCHOR = "no_usable_anchor"
OUT_OF_REACH = "out_of_reach"
AMBIGUOUS = "ambiguous"
ATTRIBUTED = "attributed"


@dataclass(frozen=True)
class Anchor:
    """One estimate of where a person's hand is, and how it was arrived at."""

    person_id: str
    side: str                  # "left" or "right"
    kind: str                  # "hand" | "wrist" | "elbow"
    x: float
    y: float
    score: float               # the weakest keypoint the estimate rests on

    @property
    def name(self) -> str:
        return f"{self.side}_{self.kind}"


@dataclass
class Candidate:
    """One anchor measured against one object."""

    anchor: Anchor
    surface_px: float          # distance to the object box, 0 when inside it
    centre_px: float           # distance to the object's centre
    reach_px: float
    object_extent: float

    @property
    def within_reach(self) -> bool:
        return self.surface_px <= self.reach_px

    @property
    def surface_over_object(self) -> float:
        """Surface distance as a multiple of the object's longest side.

        The number that makes the old rule's failure legible: this is what
        `objects.margin` had to exceed, and on a phone held in the fingers it
        is close to 1 -- against a configured 0.25.
        """
        return self.surface_px / self.object_extent if self.object_extent else math.inf

    def sort_key(self) -> tuple:
        # Surface distance first, centre distance to separate two anchors that
        # are both inside the box, then the names so the result never depends
        # on dict ordering.
        return (self.surface_px, self.centre_px, self.anchor.person_id,
                self.anchor.name)


@dataclass
class AssociationReport:
    """Why one object was or was not attributed. Always populated.

    This exists because `objects_attributed: {}` is not a diagnosis. Every
    number a person would ask for on being shown that summary -- how far, how
    far relative to what, which gate said no -- is recorded here at the moment
    the decision is made, rather than reconstructed afterwards from a log that
    did not save it.
    """

    object_track_id: str
    label: str
    box: tuple[float, float, float, float]
    centre: tuple[float, float]
    object_extent: float
    outcome: str = NO_PERSON
    people_seen: int = 0
    body_scales: dict = field(default_factory=dict)     # person_id -> px
    reaches: dict = field(default_factory=dict)         # person_id -> px
    candidates: list = field(default_factory=list)      # all Candidates, sorted
    rejected_low_score: list = field(default_factory=list)  # (pid, name, score)
    best: Optional[Candidate] = None
    runner_up: Optional[Candidate] = None

    @property
    def nearest(self) -> Optional[Candidate]:
        """The closest anchor of any kind, whether or not it was in reach."""
        return self.candidates[0] if self.candidates else None


@dataclass
class HeldObject:
    """One object, and the person holding it if anyone is."""

    object_track_id: str
    label: str
    score: float
    box: tuple[float, float, float, float]
    held_by: Optional[str] = None          # a person's track_id, or None
    wrist: Optional[str] = None            # which anchor claimed it
    wrist_score: float = 0.0
    distance_px: float = 0.0
    anchor_kind: Optional[str] = None      # "hand" | "wrist" | "elbow"
    anchor_point: Optional[tuple[float, float]] = None
    reach_px: float = 0.0
    report: Optional[AssociationReport] = None

    @property
    def attributed(self) -> bool:
        return self.held_by is not None


def distance_to_box(x: float, y: float, box) -> float:
    """Distance from a point to the nearest point of a box; 0 inside it.

    Surface distance rather than centre distance, because the question is "is
    the hand on the object", and a hand on the far corner of a laptop is on the
    laptop. Centre distance would make big objects harder to hold than small
    ones, which is the wrong way round.
    """
    x1, y1, x2, y2 = box
    dx = max(x1 - x, 0.0, x - x2)
    dy = max(y1 - y, 0.0, y - y2)
    return math.hypot(dx, dy)


def _confident(keypoints, index: int, min_score: float):
    """(x, y, score) when the keypoint exists and is believed, else None."""
    if index >= len(keypoints):
        return None
    x, y, score = (float(v) for v in keypoints[index])
    return (x, y, score) if score >= min_score else None


def body_scale(keypoints, person_box=None,
               min_score: float = MIN_KEYPOINT_SCORE) -> float:
    """This person's size in pixels, from the longest limb we can measure.

    Maximum rather than mean: perspective foreshortens limbs but never
    lengthens them, so every measurement is a lower bound on the true scale and
    the largest one is the best estimate. A mean would be dragged down by
    whichever arm happens to point at the camera -- which is exactly the arm
    holding something up to be looked at.

    Falls back to a fraction of the person box when no limb has both ends
    confident, and is clamped by the box either way, because a limb longer than
    the body's own bounding box is a pose failure and must not be allowed to
    grant a reach across the frame.
    """
    lengths = []
    for a, b in SCALE_PAIRS:
        pa = _confident(keypoints, a, min_score)
        pb = _confident(keypoints, b, min_score)
        if pa and pb:
            lengths.append(math.hypot(pa[0] - pb[0], pa[1] - pb[1]))
    scale = max(lengths) if lengths else 0.0

    if person_box is not None:
        x1, y1, x2, y2 = person_box
        diagonal = math.hypot(x2 - x1, y2 - y1)
        # A shoulder width is roughly a fifth of a person's box diagonal; used
        # only when the limbs gave us nothing at all.
        if not lengths:
            scale = diagonal * 0.20
        scale = min(scale, diagonal * MAX_SCALE_FRACTION_OF_BOX)
    return scale


def hand_anchors(person_id: str, keypoints,
                 min_score: float = MIN_KEYPOINT_SCORE,
                 hand_extension: float = HAND_EXTENSION) -> list[Anchor]:
    """Every usable estimate of where this person's hands are.

    Up to three per arm, in decreasing order of what they are worth:

    * ``hand`` -- wrist and elbow both confident, so the forearm direction is
      known and the wrist can be pushed forward to the grip point. This is the
      anchor that matters for held objects and the one COCO does not give us.
    * ``wrist`` -- the wrist keypoint itself, kept alongside the extrapolated
      hand because extrapolation overshoots on a foreshortened arm, and the
      nearest anchor wins.
    * ``elbow`` -- only when the wrist is NOT confident. A weaker claim, and
      labelled as one: the elbow is a forearm away from the hand, so it comes
      within reach only of an object the arm is genuinely folded around. It
      exists so an occluded wrist degrades the answer instead of erasing it.

    Keypoints below `min_score` never participate, in any role.
    """
    anchors: list[Anchor] = []
    for side, _shoulder_i, elbow_i, wrist_i in ARMS:
        wrist = _confident(keypoints, wrist_i, min_score)
        elbow = _confident(keypoints, elbow_i, min_score)

        if wrist is not None:
            wx, wy, ws = wrist
            if elbow is not None:
                ex, ey, es = elbow
                dx, dy = wx - ex, wy - ey
                if math.hypot(dx, dy) > 0:
                    anchors.append(Anchor(
                        person_id=person_id, side=side, kind="hand",
                        x=wx + dx * hand_extension,
                        y=wy + dy * hand_extension,
                        # The extrapolation is only as good as the weaker of the
                        # two keypoints that define its direction.
                        score=min(ws, es),
                    ))
            anchors.append(Anchor(person_id=person_id, side=side, kind="wrist",
                                  x=wx, y=wy, score=ws))
        elif elbow is not None:
            ex, ey, es = elbow
            anchors.append(Anchor(person_id=person_id, side=side, kind="elbow",
                                  x=ex, y=ey, score=es))
    return anchors


def reach(body_scale_px: float, object_extent: float,
          body_margin: float = BODY_MARGIN,
          object_margin: float = OBJECT_MARGIN,
          min_reach_px: float = MIN_REACH_PX) -> float:
    """How far from a hand anchor an object may sit and still be held.

    The maximum of an arm-sized term and an object-sized term. The arm term is
    what fixes small objects: a phone's box says nothing about how far a grip
    is from a wrist keypoint, and the person's arm says everything. The object
    term is kept so a hand on the corner of a large object still counts.
    """
    return max(body_scale_px * body_margin,
               object_extent * object_margin,
               min_reach_px)


def associate(
    objects,
    poses: dict,
    margin: float = OBJECT_MARGIN,
    min_wrist_score: float = MIN_KEYPOINT_SCORE,
    person_boxes: Optional[dict] = None,
    body_margin: float = BODY_MARGIN,
    hand_extension: float = HAND_EXTENSION,
    tie_ratio: float = TIE_RATIO,
) -> list[HeldObject]:
    """Attribute each object to a person's hand, or to nobody.

    `objects` are (track_id, Detection)-shaped: anything with `.label`,
    `.score` and the four box coordinates. `poses` is {track_id: (K,3)} exactly
    as `PoseEstimator.estimate` returns it, and `person_boxes` is the optional
    {track_id: (x1,y1,x2,y2)} that was fed to the pose estimator -- used only to
    bound body scale when the keypoints are unreliable.

    An object nobody is holding is returned with `held_by=None` rather than
    dropped. That is deliberate: a knife on a bench with no one near it is a
    real observation and often a more interesting one than a knife in a hand.
    Dropping it would make "no object" and "unattributed object" the same event
    stream, and they are not the same thing.

    Every returned object carries a populated `.report` regardless of outcome.
    """
    person_boxes = person_boxes or {}

    # Anchors and scale are per person, not per object, so they are computed
    # once here rather than once per (object, person) pair.
    anchors_by_person: dict[str, list[Anchor]] = {}
    scales: dict[str, float] = {}
    weak: dict[str, list] = {}
    for person_id, keypoints in poses.items():
        anchors_by_person[person_id] = hand_anchors(
            person_id, keypoints, min_wrist_score, hand_extension
        )
        scales[person_id] = body_scale(
            keypoints, person_boxes.get(person_id), min_wrist_score
        )
        weak[person_id] = [
            (person_id, name, round(float(keypoints[i][2]), 4))
            for name, i in (("left_wrist", LEFT_WRIST),
                            ("right_wrist", RIGHT_WRIST),
                            ("left_elbow", LEFT_ELBOW),
                            ("right_elbow", RIGHT_ELBOW))
            if i < len(keypoints) and float(keypoints[i][2]) < min_wrist_score
        ]

    out: list[HeldObject] = []
    for track_id, det in objects:
        box = (det.x1, det.y1, det.x2, det.y2)
        extent = max(det.x2 - det.x1, det.y2 - det.y1)
        centre = ((det.x1 + det.x2) / 2.0, (det.y1 + det.y2) / 2.0)

        report = AssociationReport(
            object_track_id=track_id, label=det.label, box=box, centre=centre,
            object_extent=extent, people_seen=len(poses),
            body_scales={p: round(s, 1) for p, s in scales.items()},
        )
        for person_id in poses:
            report.rejected_low_score.extend(weak[person_id])

        for person_id, anchors in anchors_by_person.items():
            person_reach = reach(scales[person_id], extent, body_margin, margin)
            report.reaches[person_id] = round(person_reach, 1)
            for anchor in anchors:
                report.candidates.append(Candidate(
                    anchor=anchor,
                    surface_px=distance_to_box(anchor.x, anchor.y, box),
                    centre_px=math.hypot(anchor.x - centre[0], anchor.y - centre[1]),
                    reach_px=person_reach,
                    object_extent=extent,
                ))
        report.candidates.sort(key=Candidate.sort_key)

        held = HeldObject(object_track_id=track_id, label=det.label,
                          score=float(det.score), box=box, report=report)
        report.outcome = _decide(report, held, tie_ratio)
        out.append(held)

    return out


def _decide(report: AssociationReport, held: HeldObject, tie_ratio: float) -> str:
    """Apply the gates in order and record which one spoke. Mutates `held`."""
    if not report.people_seen:
        return NO_PERSON
    if not report.candidates:
        # People are in frame, but every wrist and elbow was below threshold.
        return NO_USABLE_ANCHOR

    in_reach = [c for c in report.candidates if c.within_reach]
    if not in_reach:
        report.best = report.candidates[0]
        return OUT_OF_REACH

    best = in_reach[0]
    report.best = best

    # The runner-up that matters belongs to a DIFFERENT person. Two anchors on
    # one person are not a contest: both arms of one holder give the same
    # answer to the question actually being asked.
    other = next((c for c in in_reach
                  if c.anchor.person_id != best.anchor.person_id), None)
    report.runner_up = other
    if other is not None and other.surface_px - best.surface_px <= tie_ratio * best.reach_px:
        # A coin toss recorded as evidence is worse than a gap in the record:
        # the engine accumulates against a track_id, and half of those
        # accumulations would be against the wrong person.
        return AMBIGUOUS

    held.held_by = best.anchor.person_id
    held.wrist = best.anchor.name
    held.wrist_score = round(best.anchor.score, 4)
    held.distance_px = round(best.surface_px, 1)
    held.anchor_kind = best.anchor.kind
    held.anchor_point = (round(best.anchor.x, 1), round(best.anchor.y, 1))
    held.reach_px = round(best.reach_px, 1)
    return ATTRIBUTED
