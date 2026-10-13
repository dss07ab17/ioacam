"""PPE detections -> the person wearing them -> ppe_helmet / ppe_vest events.

A SECOND detector, running alongside the person one on the same frame. It is
not a second class list on the first: hard hats and high-vis vests are not COCO
classes, so they need a different checkpoint and therefore a different forward
pass. `perception/detectors/ppe.py` holds the model, its ten classes and its
per-class floors; this file holds what happens to the boxes afterwards.

Off by default (`ppe.enabled`, or `--ppe`) because that second forward pass is
not free -- see the cost note in the config README.

## What this stage does NOT do

It never mints a track id and never touches tracking, zones, occupancy or
pose. Those all belong to the person detector, which runs unchanged whether
this stage is on or off. The PPE model has a `Person` class of its own and it
is off by default for exactly this reason: two detectors both emitting people
would put two boxes on every body with nothing able to tell they are the same
one.

## Association is containment, and it is the weakest part

A hard hat is on a head, a vest is on a torso, and both are inside that
person's box. So the rule is containment: the fraction of the PPE box that
falls inside a person box. Not IoU -- a hard hat is a tiny fraction of a
standing person's area and would score an IoU of about 0.02 against the person
genuinely wearing it.

Two people standing close together overlap, and a hat sitting inside both boxes
is a real and common situation. When the two best candidates are
indistinguishable -- similar containment, similar box size -- the hat is
attributed to NEITHER, on the same reasoning `association.py` gives for objects
in hands: a coin toss recorded as evidence against a named track is worse than
a gap in the record.

But it is not dropped. An unattributed `NO-Hardhat` is still a bare head on the
floor, and silently discarding it would make "everyone is compliant" and "we
could not tell whose head that was" the same stream. It is emitted with no
`track_id`, which is exactly what `object_at_station` does with an unheld
object and what the schema's optional `track_id` is for.

There is no depth here and no vertical prior. A hard hat held at waist height,
or one worn by somebody standing directly behind the person the box is matched
to, looks the same as one being worn. That is a monocular limitation, not a
tuning problem.

## Absence of a detection is not a violation

If a person has neither `Hardhat` nor `NO-Hardhat` associated with them this
frame, this stage emits nothing about their helmet. It does not infer a bare
head from a missing hat. The schema is explicit about the same distinction for
identity -- `identity_unverified` is a finding in its own right and must not be
inferable from the absence of `person_identified` -- and the reasoning carries:
a person facing away, half out of frame, or 20 pixels tall produces no hat
detection for reasons that have nothing to do with whether they are wearing
one.

## Which way a disagreement is resolved

The model can return both `Hardhat` and `NO-Hardhat` over the same person in
one frame. They are mutually exclusive, so one has to win, and the comparison
is `score / that class's own threshold` rather than raw score.

Using raw score would undo the asymmetry the thresholds exist to create: the
NO-* floors are set low (0.20 against 0.40) because recall on those classes is
0.55-0.62 and a miss matters more than a false alarm, and then a raw-score
comparison would let a barely-passing positive beat a solidly-passing negative
anyway. Comparing each score against its own bar keeps one decision --
"how much do we trust this class" -- expressed in one number, used in both
places it is needed.

## Emission is edge-triggered, and keyed differently for the two cases

An attributed finding is keyed on `(person track, observation)`: what the
engine wants is one worker's compliance state, and it should change when the
worker's state changes, not when the detector's box flickers into a new PPE
track id.

An unattributed one has no person to key on, so it is keyed on the PPE track
instead. Without a key of some kind, one un-attributable bare head would emit
an event on every single frame it is visible.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from detectors import build_detector
from detectors.ppe import PPE_OBSERVATIONS
from emit import EventEmitter, subject_of
from tracking import MultiClassTracker, Track

# Fraction of the PPE box that must fall inside a person box before that person
# is a candidate to be wearing it. Not 1.0: a hard hat on someone at the edge of
# frame is clipped by the person box as often as not, and a vest on a person
# leaning forward pushes past their own box's top edge.
MIN_CONTAINMENT = 0.70

# Two candidate people whose containment differs by less than this, and whose
# boxes are within this fraction of each other in area, are treated as tied --
# the PPE box is attributed to neither. See the module note.
TIE_RATIO = 0.10


@dataclass(frozen=True)
class PpeFinding:
    """One resolved PPE observation, before it becomes an event."""

    observation: str            # "ppe_helmet" or "ppe_vest"
    value: bool                 # True = wearing it, False = not
    label: str                  # the model class that decided it
    score: float                # raw detector score
    confidence: float           # score * track persistence
    ppe_track_id: str
    # Frame pixels, so a caller drawing the overlay does not have to reach
    # back into the tracker to find the box this finding came from.
    box: tuple = (0.0, 0.0, 0.0, 0.0)
    track_id: Optional[str] = None      # person track, None when unattributed
    containment: float = 0.0
    ambiguous: bool = False     # unattributed because two people tied, not
                                # because nobody contained it
    held_over: bool = False     # attributed from memory, because the wearer's
                                # own detection dropped for a frame


def containment(box, person_box) -> float:
    """Fraction of `box`'s area that falls inside `person_box`.

    Deliberately asymmetric, and deliberately not IoU: what is being asked is
    "is this hat inside that person", and a hat is 1-2% of a standing person's
    area. IoU answers a question nobody asked and answers it 0.02.
    """
    bx1, by1, bx2, by2 = box
    px1, py1, px2, py2 = person_box
    area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    if area <= 0:
        return 0.0
    ix = max(0.0, min(bx2, px2) - max(bx1, px1))
    iy = max(0.0, min(by2, py2) - max(by1, py1))
    return (ix * iy) / area


def _area(box) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def attribute(
    box,
    person_boxes: dict,
    min_containment: float = MIN_CONTAINMENT,
    tie_ratio: float = TIE_RATIO,
) -> tuple[Optional[str], float, bool]:
    """Which person's box contains this one. Returns (track_id, containment, tied).

    Best containment wins; the smaller person box breaks a containment tie,
    because a hat inside both a near person and a far one behind them is on the
    near one far more often than not.

    `tied` distinguishes "nobody contained it" from "two people did and we
    cannot say which". Both return a `None` track id -- the caller emits either
    way -- but only the second is a limitation of this rule rather than of the
    frame.
    """
    covers = [
        (containment(box, pbox), -_area(pbox), track_id)
        for track_id, pbox in person_boxes.items()
    ]
    candidates = sorted(
        (c for c in covers if c[0] >= min_containment), reverse=True
    )
    if not candidates:
        # The best containment achieved is still worth reporting: it is the
        # difference between "no people in frame" and "just under the bar".
        return None, max((c[0] for c in covers), default=0.0), False

    best_cover, best_neg_area, best_id = candidates[0]
    if len(candidates) > 1:
        next_cover, next_neg_area, _ = candidates[1]
        best_area, next_area = -best_neg_area, -next_neg_area
        larger = max(best_area, next_area)
        similar_size = larger > 0 and abs(best_area - next_area) / larger < tie_ratio
        if abs(best_cover - next_cover) < tie_ratio and similar_size:
            return None, best_cover, True

    return best_id, best_cover, False


class PpeStage:
    """Second detector, containment association, edge-triggered PPE events."""

    #: Config keys forwarded to `build_detector`. Everything else in the `ppe`
    #: block is association or emission policy and means nothing to a detector.
    DETECTOR_KEYS = (
        "backend", "model_path", "input_size", "score_threshold",
        "classes", "class_thresholds", "device",
    )

    def __init__(self, cfg: dict, detector=None) -> None:
        if detector is None:
            # Only keys the site actually set. `build_detector` reads with
            # `.get(key, default)`, so forwarding a key that is present and
            # None would replace the default with None rather than keep it.
            detector_cfg = {
                key: cfg[key]
                for key in self.DETECTOR_KEYS
                if cfg.get(key) is not None
            }
            detector_cfg.setdefault("backend", "yolo26-ppe")
            detector = build_detector(detector_cfg)
        self.detector = detector

        self.min_containment = float(cfg.get("min_containment", MIN_CONTAINMENT))
        self.tie_ratio = float(cfg.get("tie_ratio", TIE_RATIO))
        self.persistence_window = int(cfg.get("persistence_window", 30))
        self.min_confidence = float(cfg.get("min_confidence", 0.05))
        # How many frames a person track may be missing before their
        # compliance state is dropped. It must be at least the person
        # tracker's `max_misses`, or a worker occluded for one frame loses
        # their state and re-emits the same verdict when they come back --
        # `perceive.py` passes that value straight through.
        self.forget_after = int(cfg.get("forget_after", 15))
        # Consecutive frames a finding must stay unattributed before it is
        # announced as such. The same debounce `emission.enter_frames` applies
        # to zone entry, for the same reason: a PPE box whose first frame lands
        # on one the person detector happened to miss has nothing to hold over
        # from, and would otherwise be announced as an ownerless hard hat one
        # frame before it attaches to the person wearing it.
        #
        # Measured on a live webcam run, containment is bimodal -- 0.98 to 1.00
        # on every frame a person box exists, 0.0 on the frames none does, with
        # nothing in between. So a lone unattributed frame is a detector gap
        # and not a marginal association, and waiting a few frames costs
        # nothing real.
        #
        # Only unattributed findings wait. An attributed one emits on the frame
        # it is seen: delaying a non-compliance is the one cost this stage must
        # not pay.
        self.loose_frames = int(cfg.get("loose_frames", 3))
        self.tracker = MultiClassTracker(
            max_misses=int(cfg.get("max_misses", 5)),
            history_window=self.persistence_window,
            prefix="ppe",
        )

        # (person track_id, observation) -> last emitted boolean.
        self._person_state: dict[tuple[str, str], bool] = {}
        # person track_id -> the frame it was last seen in, for the grace
        # window above. Frames counted here, not wall time, because that is
        # the unit `max_misses` is in.
        self._person_seen: dict[str, int] = {}
        self._frame = 0
        # ppe track_id -> last emitted (observation, value), for the ones that
        # could not be attributed to anybody.
        self._loose_state: dict[str, tuple[str, bool]] = {}
        # ppe track_id -> consecutive frames it has been unattributed for.
        self._loose_streak: dict[str, int] = {}
        # ppe track_id -> (person track_id, the frame it was attributed on).
        # The person detector misses a frame here and there, and
        # `person_boxes_from_tracks` only carries tracks matched THIS frame --
        # so without this, one dropped frame turns a hat that never moved into
        # an unattributed finding and re-announces it. See `_resolve`.
        self._ppe_attribution: dict[str, tuple[str, int]] = {}

        # Forward-pass cost, so "what did enabling this add" is answerable from
        # a run rather than from a benchmark nobody re-runs on the target board.
        self.detect_calls = 0
        self.detect_seconds = 0.0

    @property
    def name(self) -> str:
        return self.detector.name

    @property
    def mean_detect_ms(self) -> float:
        if not self.detect_calls:
            return 0.0
        return 1000.0 * self.detect_seconds / self.detect_calls

    def detect(self, frame) -> list:
        """The second forward pass, timed. Separate so tests can skip it."""
        started = time.perf_counter()
        detections = self.detector.detect(frame)
        self.detect_seconds += time.perf_counter() - started
        self.detect_calls += 1
        return detections

    def process(
        self,
        frame,
        person_boxes: dict,
        emitter: EventEmitter,
        timestamp_us: Optional[int] = None,
        zone_of: Optional[Callable[[str], Optional[str]]] = None,
        subject_for: Optional[Callable[[str], dict]] = None,
    ) -> tuple[list[PpeFinding], list[dict]]:
        """Detect, associate, and return (findings, events) for one frame."""
        return self.process_detections(
            self.detect(frame), person_boxes, emitter, timestamp_us, zone_of,
            subject_for,
        )

    def process_detections(
        self,
        detections: Sequence,
        person_boxes: dict,
        emitter: EventEmitter,
        timestamp_us: Optional[int] = None,
        zone_of: Optional[Callable[[str], Optional[str]]] = None,
        subject_for: Optional[Callable[[str], dict]] = None,
    ) -> tuple[list[PpeFinding], list[dict]]:
        """Association and emission, with the forward pass already done.

        `subject_for` resolves a person track to a subject block, and is a
        callable for the same reason `zone_of` is: this stage is handed the
        answers the loop already knows rather than reaching into the identity
        or zone layers itself. Omit it and every subject stays anonymous.

        `person_boxes` must be THIS frame's, from the same frame the detections
        came from. A hat matched against last frame's person boxes is matched
        against boxes that have moved by exactly the distance a person moves in
        a frame, which on a close view is a large fraction of a head.
        """
        self._frame += 1
        for track_id in person_boxes:
            self._person_seen[track_id] = self._frame

        tracks = self.tracker.update(
            [d for d in detections if d.label in PPE_OBSERVATIONS]
        )
        findings = self._resolve(tracks, person_boxes)

        events = [
            build_ppe_event(emitter, finding, timestamp_us, zone_of, subject_for)
            for finding in self._changed(findings)
        ]
        return findings, events

    # -- association and resolution ---------------------------------------

    def _resolve(self, tracks: list[Track], person_boxes: dict) -> list[PpeFinding]:
        """Per person and observation, one winner. Unattributed boxes all pass."""
        thresholds = self.detector.class_thresholds
        loose: list[PpeFinding] = []
        # (person, observation) -> the best finding so far for that pair.
        best: dict[tuple[str, str], PpeFinding] = {}

        for track in tracks:
            det = track.detection
            mapped = PPE_OBSERVATIONS.get(det.label)
            if mapped is None:                  # Mask, machinery, ... no schema slot
                continue
            observation, value = mapped
            track_id, cover, tied = attribute(
                (det.x1, det.y1, det.x2, det.y2),
                person_boxes,
                self.min_containment,
                self.tie_ratio,
            )
            held_over = False
            if track_id is None and not tied:
                remembered = self._ppe_attribution.get(track.track_id)
                if remembered is not None:
                    person, when = remembered
                    # Only when the wearer is missing from THIS frame. A person
                    # who IS visible and whose box does not contain the hat is
                    # a real containment failure and must be reported as one --
                    # papering over that would hide the rule's own error rate.
                    if (person not in person_boxes
                            and self._frame - when <= self.forget_after):
                        track_id = person
                        held_over = True

            finding = PpeFinding(
                observation=observation,
                value=value,
                label=det.label,
                score=float(det.score),
                confidence=round(
                    det.score * track.persistence(self.persistence_window), 4
                ),
                ppe_track_id=track.track_id,
                box=(det.x1, det.y1, det.x2, det.y2),
                track_id=track_id,
                containment=round(cover, 4),
                ambiguous=tied,
                held_over=held_over,
            )
            if track_id is not None and not held_over:
                # Refreshed only on a real attribution. Refreshing it on a
                # hold-over would make the grace window slide forward for as
                # long as the wearer stayed missing, and the hat would never
                # come loose at all.
                self._ppe_attribution[track.track_id] = (track_id, self._frame)
            if track_id is None:
                loose.append(finding)
                continue

            key = (track_id, observation)
            incumbent = best.get(key)
            if incumbent is None or _margin(finding, thresholds) > _margin(
                incumbent, thresholds
            ):
                best[key] = finding

        return list(best.values()) + loose

    # -- emission ----------------------------------------------------------

    def _changed(self, findings: list[PpeFinding]) -> list[PpeFinding]:
        """Edge trigger, plus state cleanup for tracks that have gone away."""
        out: list[PpeFinding] = []
        for finding in findings:
            if finding.confidence < self.min_confidence:
                # Gated BEFORE the state is written, not after. A finding too
                # weak to announce must not record itself as the announced
                # state -- it would swallow the transition, and the next frame
                # that agreed with it would look like no change at all.
                continue
            if finding.track_id is not None:
                key = (finding.track_id, finding.observation)
                if self._person_state.get(key, _MISSING) != finding.value:
                    self._person_state[key] = finding.value
                    out.append(finding)
            else:
                key = finding.ppe_track_id
                streak = self._loose_streak.get(key, 0) + 1
                self._loose_streak[key] = streak
                if streak < self.loose_frames:
                    continue
                current = (finding.observation, finding.value)
                if self._loose_state.get(key, _MISSING) != current:
                    self._loose_state[key] = current
                    out.append(finding)

        # A person who leaves and whose id is later reissued must not inherit
        # the previous occupant's compliance state -- the next worker would be
        # silently assumed to be wearing whatever the last one wore. But the
        # grace window has to outlast a one-frame occlusion, or every flicker
        # in the person tracker re-emits a verdict that never changed.
        stale = [
            person for person, seen in self._person_seen.items()
            if self._frame - seen > self.forget_after
        ]
        for person in stale:
            del self._person_seen[person]
            for key in [k for k in self._person_state if k[0] == person]:
                del self._person_state[key]

        # Loose state is dropped the moment a box IS attributed, so that going
        # loose again later always announces. The flicker that used to cause
        # is prevented by the hold-over in `_resolve`, not by keeping stale
        # state here -- a hat its wearer really did walk away from has to be
        # reportable as unattributed even if it was announced that way before.
        loose = {f.ppe_track_id for f in findings if f.track_id is None}
        for gone in list(self._loose_state):
            if gone not in loose:
                del self._loose_state[gone]
        for gone in list(self._loose_streak):
            if gone not in loose:
                del self._loose_streak[gone]

        tracked = {f.ppe_track_id for f in findings}
        for gone in list(self._ppe_attribution):
            if gone not in tracked:
                del self._ppe_attribution[gone]

        return out


def _margin(finding: PpeFinding, thresholds: dict) -> float:
    """How far past its own floor this detection scored. See the module note."""
    floor = thresholds.get(finding.label) or 1e-6
    return finding.score / floor


def build_ppe_event(
    emitter: EventEmitter,
    finding: PpeFinding,
    timestamp_us: Optional[int] = None,
    zone_of: Optional[Callable[[str], Optional[str]]] = None,
    subject_for: Optional[Callable[[str], dict]] = None,
) -> dict:
    """Schema-shaped ppe_helmet / ppe_vest event.

    `value` is boolean, which is what the schema asks of a PPE observation, and
    `false` is a positive assertion of non-compliance rather than a missing
    reading -- nothing is emitted at all when the model said nothing about that
    person's head.

    Confidence is the raw detector score times track persistence, matching
    `object_at_station`. It is NOT the calibrated composition in
    `confidence.py`: no temperature has been fitted for this model, and the
    quality terms there are written against person-box height.

    `zone_id` is filled from the person's zone where one is known, because
    almost every PPE rule is zone-scoped ("hard hats in the cell"). An
    unattributed finding has no person and so no zone.

    `subject` is filled from `subject_for` on the same terms, and for the same
    reason this matters most here of all three observations: a non-compliance
    is the finding somebody acts on, and "whose head" is the first question
    asked about it. An unattributed finding has no track and therefore no
    identity either -- it stays the anonymous human it has always been, which
    is not a downgrade but the honest answer. So does an attributed finding on
    a track the cascade never bound: the identity key is omitted, never set to
    a falsy stand-in, because the engine's wrong_role rules read an absent key
    and an empty one differently.
    """
    zone_id = None
    if zone_of is not None and finding.track_id is not None:
        zone_id = zone_of(finding.track_id)
    return emitter.build(
        observation=finding.observation,
        confidence=finding.confidence,
        value=finding.value,
        # Omitted when the box could not be pinned to one person. The engine
        # can still count it; it cannot attribute it, and neither can we.
        track_id=finding.track_id,
        zone_id=zone_id,
        subject=subject_of(subject_for, finding.track_id),
        timestamp_us=timestamp_us,
    )


_MISSING = object()
