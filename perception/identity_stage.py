"""Bind person tracks to enrolled identities, inside the perception loop.

Until this existed, `perceive.py` and `identity/live_recognize.py` were two
programs that opened two cameras and shared nothing: perception produced
events with `subject: {"class": "human"}` and no name, while the recogniser
drew names on a window nobody downstream could see. This is the join.

WHAT IT DOES PER FRAME

  1. Detect faces (RFB-320).
  2. Attribute each face to a person track, by containment.
  3. For tracks not yet bound, align the face and run the identity cascade.
  4. Emit the cascade's own events, and expose the binding so that every
     other event the loop emits for that track can carry the name.

WHAT IT DELIBERATELY DOES NOT DO

**It does not re-match a bound track.** `IdentityResolver` binds once per
track and that is the right call: the answer cannot change while the track
persists, and re-running it every frame would spend an ArcFace inference to
confirm what is already known. The cost is that a tracker id swap carries the
wrong name until the track dies. That is a real error and it belongs in the
open list, not hidden behind a per-frame re-check that would mask it.

**It does not invent presence.** Where there is no access control system,
the roster is empty, and an empty roster means the cascade has nobody to
compare against. `access_control=False` seeds the roster from the enrolled
templates instead -- see `_seed_from_templates` for what that costs.

**It does not slow the loop down when there is nothing to do.** Face
detection runs only when at least one track is unbound. A frame in which
everyone visible is already identified costs nothing.

**It does not put a name in an event.** The employee directory is loaded here
and read by exactly one method, `display_label`, which feeds a screen. Every
event this stage emits carries the person ID, built by `subject_for` from the
resolution alone -- see `identity/directory.py` for why that boundary is the
point rather than a detail.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from identity.alignment import (  # noqa: E402
    AlignmentError,
    align_face,
    landmarks_from_box,
)
from identity.directory import build_directory  # noqa: E402
from identity.matcher import ThresholdPolicy, build_matcher  # noqa: E402
from identity.resolver import IdentityResolver, Resolution  # noqa: E402
from identity.roster import Roster  # noqa: E402
from perception.face import build_face_detector  # noqa: E402
from emit import wall_time_at  # noqa: E402


class IdentityStage:
    """Face recognition attached to person tracks."""

    def __init__(self, cfg: dict, sensor_id: str, log=None) -> None:
        self.log = log or (lambda msg: None)

        face_cfg = dict(cfg.get("face") or {})

        self.detector = build_face_detector(
            {
                "backend": face_cfg.get("detector_backend", "rfb-320-onnx"),
                "model_path": face_cfg.get(
                    "detector_model_path",
                    "perception/models/version-RFB-320-int8.onnx",
                ),
            }
        )

        # Landmarks are needed for alignment and RFB-320 does not produce
        # them. Built lazily-ish here so a missing cascade fails at startup
        # rather than on the first face.
        self.landmark_source = None
        lm_cfg = face_cfg.get("landmarks") or {}
        if lm_cfg.get("backend", "cascade") != "none":
            from identity.landmarks import build_landmark_source

            self.landmark_source = build_landmark_source(lm_cfg)

        self.matcher = build_matcher(face_cfg)
        self.policy = ThresholdPolicy.from_config(cfg["threshold_policy"])

        roster_cfg = cfg.get("roster") or {}
        self.roster = Roster(
            presence_ttl_s=float(roster_cfg.get("presence_ttl_s", 43200)),
            has_exit_readers=bool(roster_cfg.get("has_exit_readers", False)),
        )

        # ID -> name, for the screen. Display-only: nothing this stage puts
        # into an event ever reads from it (see `subject_for`, which builds
        # the subject block from the resolution alone). An absent file is the
        # normal case and costs nothing -- every label falls back to the ID.
        self.directory = build_directory(cfg.get("directory"), log=self.log)

        self.resolver = IdentityResolver(
            roster=self.roster,
            matcher=self.matcher,
            policy=self.policy,
            sensor_id=sensor_id,
            directory=self.directory,
        )

        # Is there an access control system feeding this roster?
        self.access_control = bool(cfg.get("access_control", False))
        if not self.access_control:
            self._seed_from_templates(face_cfg)

        # Minimum fraction of the person box height a face must occupy before
        # it is worth trying. Below this the crop is too small for the eye
        # cascade and the alignment will be dominated by landmark error.
        self.min_face_fraction = float(cfg.get("min_face_fraction", 0.04))

        self._unbindable: set[str] = set()
        self.attempts = 0
        self.bindings = 0
        # Matches against IDs the directory marks as departed. Collected
        # rather than counted, because what an operator needs is which ID and
        # at what score, not how many. Drained by the caller that routes them;
        # see `drain_inactive_matches`.
        self.inactive_matches: list = []
        # So a leaver standing in frame for ten minutes produces one line and
        # not fifteen thousand. The per-track binding already runs once, but a
        # tracker id swap would otherwise re-report the same person.
        self._inactive_reported: set[str] = set()

    # ------------------------------------------------------------------

    def _seed_from_templates(self, face_cfg: dict) -> None:
        """Put every enrolled person on the roster, with no zone.

        The cascade compares a face against people the site believes are
        present, and that belief normally comes from badge readers. With no
        readers there is no such belief, so the fallback is to treat everyone
        enrolled as potentially present.

        Two things this costs, both worth stating out loud:

        The zone tier stops existing. Tier 1 matches against people badged
        into THIS zone and tier 2 against everyone on site; with no zone
        information every match is a tier 2 match. So `presence_unbadged` is
        suppressed (see `events_for` handling in `process`) -- with no readers
        the claim "they never authenticated into this zone" is not false, it
        is meaningless, and emitting it for every person would bury the log.

        The candidate set stops shrinking. The whole roster is compared every
        time, so the threshold is derived from the full enrolment count rather
        than the handful of people actually in the room. That is the
        conservative direction -- stricter, not looser -- but it is also why
        `ThresholdPolicy.max_candidates()` matters: past that count nothing
        can be identified at all.

        None of this is a substitute for access control. It is what makes a
        demo possible on a laptop.
        """
        from identity.enrollment import load_templates

        templates = load_templates(
            face_cfg.get("templates_path", "identity/config/templates.json"),
            id_pattern=face_cfg.get("person_id_pattern"),
            # stderr is already where this process puts everything that is
            # not an event, and a warning about an unresolvable subject
            # identity belongs in the same stream as the rest of the startup
            # banner rather than in the event stream on stdout.
            warn=self.log,
        )
        for identity in templates:
            self.roster.badge_in(identity, None, None, 0)

        ceiling = self.policy.max_candidates()
        self.log(
            "[identity] no access control: seeded {} enrolled identities, "
            "zone tier disabled (model ceiling {} candidates)".format(
                len(templates), ceiling
            )
        )
        if len(templates) > ceiling:
            self.log(
                "[identity] WARNING enrolment ({}) exceeds the ceiling ({}); "
                "no threshold is strict enough and nothing will bind".format(
                    len(templates), ceiling
                )
            )

    # ------------------------------------------------------------------

    @staticmethod
    def _face_to_track(face, tracks):
        """Which tracked person does this face belong to?

        Containment of the face centre, then smallest containing box. Two
        people at different depths can overlap in image space, and the nearer
        one's box is the larger; picking the smallest containing box gives the
        face to the person it is actually on rather than to whoever happens to
        be standing behind them.

        Returns None when the face sits in no person box at all -- a
        reflection, a poster, or a detection the person detector missed.
        Attributing it to the closest track anyway is how a face gets stapled
        to the wrong body.
        """
        cx = (face.x1 + face.x2) / 2.0
        cy = (face.y1 + face.y2) / 2.0

        best = None
        best_area = None
        for track in tracks:
            det = track.detection
            if not (det.x1 <= cx <= det.x2 and det.y1 <= cy <= det.y2):
                continue
            area = max(1.0, (det.x2 - det.x1) * (det.y2 - det.y1))
            if best_area is None or area < best_area:
                best, best_area = track, area
        return best

    # ------------------------------------------------------------------

    def bound(self, track_id: str) -> Optional[Resolution]:
        """The binding for a track, if it has one."""
        return self.resolver._bound.get(track_id)

    def subject_for(self, track_id: str) -> dict:
        """The `subject` block to attach to this track's events.

        Unknown fields are omitted rather than guessed. The schema requires
        that, and the engine's wrong_role rules depend on it: a guessed role
        would produce a confident wrong_role finding against a name nobody
        verified.

        Note what is NOT here: the employee directory. This block goes into
        the event store, and the event store holds IDs. A `name` key added
        here would not even be schema-valid, and that is deliberate.
        """
        resolution = self.bound(track_id)
        if resolution is None or not resolution.bound:
            return {"class": "human"}
        subject = {"class": "human", "identity": resolution.identity}
        if resolution.role:
            subject["role"] = resolution.role
        return subject

    def display_label(self, track_id: str) -> str:
        """What to draw next to this track's box. Never written anywhere.

        The employee directory's only job, and the only method on this class
        that reads it. Falls back through name -> ID -> nothing, so a site
        with no directory file gets badge numbers on screen and everything
        else behaves identically.

        A leaver is labelled as one. The box must not look like a normal
        recognition when the person it names left the company in March.
        """
        resolution = self.bound(track_id)
        if resolution is None or not resolution.bound:
            return ""
        label = self.directory.display_name(resolution.identity)
        if resolution.inactive is not None:
            label = f"{label} [INACTIVE]"
        return label

    def drain_inactive_matches(self) -> list:
        """Take the inactive-match reports collected since the last call.

        Returned rather than printed so the caller decides where they go: a
        log line on a demo, an operator queue on a site. They are not events
        -- the observation enum is closed and extending it is a commissioning
        decision, not a side effect of adding a CSV -- but they are also not
        nothing, and swallowing them here would make this module the place
        where a revoked enrolment stops being visible.
        """
        drained, self.inactive_matches = self.inactive_matches, []
        return drained

    def forget(self, track_id: str) -> None:
        """Drop a binding when its track ends.

        A tracker reissues ids, so without this a new person inherits the
        previous occupant's name.
        """
        self.resolver.forget(track_id)
        self._unbindable.discard(track_id)

    # ------------------------------------------------------------------

    def process(self, frame, tracks, zone_of, now_us: int) -> list[dict]:
        """Try to bind unbound tracks. Returns events to emit.

        `zone_of` maps a track_id to its current zone_id or None.
        """
        pending = [
            t
            for t in tracks
            if self.bound(t.track_id) is None and t.track_id not in self._unbindable
        ]
        if not pending:
            # Everyone visible is already resolved. Skip the face detector
            # entirely rather than pay for it to find faces we would discard.
            return []

        faces = self.detector.detect(frame)
        if not faces:
            return []

        pending_ids = {t.track_id for t in pending}
        events: list[dict] = []

        for face in faces:
            track = self._face_to_track(face, tracks)
            if track is None or track.track_id not in pending_ids:
                continue

            det = track.detection
            person_h = max(1.0, det.y2 - det.y1)
            if face.height / person_h < self.min_face_fraction:
                # Too small to align reliably. Leaving the track unbound and
                # trying again next frame is right: the person may walk
                # closer, and a bad binding made now would never be revisited.
                continue

            marks = landmarks_from_box(face)
            if marks is None and self.landmark_source is not None:
                marks = self.landmark_source.find(frame, face)
            if marks is None:
                continue

            try:
                aligned, _method = align_face(frame, marks)
            except AlignmentError:
                continue

            self.attempts += 1
            zone_id = zone_of(track.track_id)
            resolution = self.resolver.resolve(
                track_id=track.track_id,
                zone_id=zone_id,
                probe=aligned,
                at_us=now_us,
            )

            if not resolution.bound:
                # An honest failure, but repeating it every frame for a track
                # that is simply not enrolled would spend an ArcFace inference
                # per frame per stranger. Bind the refusal instead: the
                # resolver already caches it, and `forget` clears it when the
                # track ends.
                self._unbindable.add(track.track_id)
            else:
                self.bindings += 1
                self._report_if_inactive(resolution)

            # Same wall clock every other camera event carries, converted
            # from the frame's monotonic time rather than read now.
            for event in self.resolver.events_for(
                resolution, now_us, wall_time=wall_time_at(now_us)
            ):
                if (
                    not self.access_control
                    and event["observation"] == "presence_unbadged"
                ):
                    # Meaningless without readers -- see _seed_from_templates.
                    continue
                events.append(event)

        return events

    # ------------------------------------------------------------------

    def _report_if_inactive(self, resolution: Resolution) -> None:
        """Surface a match against a departed employee, once per ID.

        Not suppressed and not demoted to unverified: the attribution is
        probably right, and which leaver it is happens to be the single most
        useful fact available. What is wrong is that a template exists for
        somebody who has gone, which is either a revocation that never ran or
        somebody else matching against their face.
        """
        flag = resolution.inactive
        if flag is None:
            return
        self.inactive_matches.append(flag)
        if flag.employee_id in self._inactive_reported:
            return
        self._inactive_reported.add(flag.employee_id)
        self.log(f"[identity] INACTIVE MATCH {flag.detail}")
