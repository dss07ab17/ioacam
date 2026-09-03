"""The identity cascade: zone roster first, then the whole site, then nothing.

Emits events, not findings. The engine judges; this layer observes. What it
observes is which of three situations holds:

  tier 1  matched against people badged into THIS zone
          -> expected person in an expected place. Bind and carry on.

  tier 2  matched only against everyone on site
          -> a known person in a zone they never badged into. Identification
             succeeded, but the result is itself a security finding: either
             they entered without authenticating or the zone's reader missed
             them.

  none    no confident match in either set
          -> someone is present whom the site cannot account for. Tailgating,
             an unlogged visitor, or an intruder. A door-only product is
             structurally blind to this: it knows who opened the door and has
             no idea how many people went through.

The tier that fails is more informative than the tier that succeeds.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Optional

from matcher import FaceMatcher, ThresholdPolicy
from roster import Roster

US_PER_S = 1_000_000


@dataclass
class Resolution:
    """What the cascade concluded about one track."""

    track_id: str
    zone_id: Optional[str]
    identity: Optional[str]
    role: Optional[str]
    tier: str                    # "zone" | "site" | "none"
    confidence: float
    reason: str
    candidates_considered: int
    best_score: float = 0.0
    runner_up: float = 0.0

    @property
    def bound(self) -> bool:
        return self.identity is not None


class IdentityResolver:
    """Binds track ids to enrolled identities, once per track.

    Matching runs at track creation, not per frame. One comparison per track is
    cheap enough to run alongside detection, and re-running it every frame would
    buy nothing: the answer cannot change while the track persists. It also
    means a tracker id swap triggers a fresh match rather than silently
    carrying the wrong name forward.
    """

    def __init__(
        self,
        roster: Roster,
        matcher: FaceMatcher,
        policy: ThresholdPolicy,
        sensor_id: str = "cam-01",
    ) -> None:
        self.roster = roster
        self.matcher = matcher
        self.policy = policy
        self.sensor_id = sensor_id
        self._bound: dict[str, Resolution] = {}

    # ------------------------------------------------------------------

    def resolve(
        self, track_id: str, zone_id: Optional[str], probe: Any, at_us: int
    ) -> Resolution:
        """Run the cascade for one newly seen track."""
        if track_id in self._bound:
            return self._bound[track_id]

        # --- tier 1: this zone ---------------------------------------
        if zone_id:
            candidates = self.roster.in_zone(zone_id)
            result = self._try(track_id, zone_id, probe, candidates, "zone")
            if result is not None:
                self.roster.touch(result.identity, at_us)
                self._bound[track_id] = result
                return result

        # --- tier 2: everyone still on site --------------------------
        candidates = self.roster.in_building()
        if zone_id:
            # Anyone already ruled out at tier 1 stays ruled out; re-testing
            # them only widens N and tightens the threshold for no benefit.
            in_zone_ids = {p.identity for p in self.roster.in_zone(zone_id)}
            candidates = [p for p in candidates if p.identity not in in_zone_ids]

        result = self._try(track_id, zone_id, probe, candidates, "site")
        if result is not None:
            self.roster.touch(result.identity, at_us)
            self._bound[track_id] = result
            return result

        # --- no match -------------------------------------------------
        unverified = Resolution(
            track_id=track_id,
            zone_id=zone_id,
            identity=None,
            role=None,
            tier="none",
            confidence=0.0,
            reason="no confident match against zone roster or site roster",
            candidates_considered=len(self.roster),
        )
        self._bound[track_id] = unverified
        return unverified

    def forget(self, track_id: str) -> None:
        """Drop a binding when its track ends, so a reissued id re-matches."""
        self._bound.pop(track_id, None)

    # ------------------------------------------------------------------

    def _try(
        self, track_id: str, zone_id: Optional[str], probe, candidates, tier: str
    ) -> Optional[Resolution]:
        if not candidates:
            return None

        threshold = self.policy.threshold_for(len(candidates))
        if threshold is None:
            # No operating point on the measured ROC is strict enough for this
            # many candidates. Saying so is the correct answer.
            return None

        ids = [p.identity for p in candidates]
        scores = self.matcher.compare(probe, ids)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

        best_id, best = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0

        if best < threshold:
            return None
        if best - runner_up < self.policy.min_margin:
            # Two candidates are indistinguishable. Picking the higher one
            # would attach a name to noise, and that name may later appear on
            # a violation.
            return None

        person = next(p for p in candidates if p.identity == best_id)
        return Resolution(
            track_id=track_id,
            zone_id=zone_id,
            identity=best_id,
            role=person.role,
            tier=tier,
            # The door match was an authentication; this is an attribution.
            # It carries the match score as its confidence precisely so that
            # downstream rules can refuse to act on a weak one.
            confidence=round(min(1.0, best), 4),
            reason=(
                f"matched {best_id} at {best:.3f} against {len(candidates)} "
                f"{tier} candidate(s), threshold {threshold:.3f}, "
                f"margin {best - runner_up:.3f}"
            ),
            candidates_considered=len(candidates),
            best_score=round(best, 4),
            runner_up=round(runner_up, 4),
        )

    # ------------------------------------------------------------------
    # Event emission
    # ------------------------------------------------------------------

    def events_for(self, r: Resolution, at_us: int, wall_time: str = "") -> list[dict]:
        """Turn a resolution into schema-conforming observation events.

        Tier 1 emits identification alone. Tiers 2 and 3 also emit the
        observation that carries the security meaning, because "we identified
        them" and "they should not be here" are different facts and the second
        must not be inferable only from the absence of the first.
        """
        base = {
            "timestamp_us": at_us,
            "source": "camera",
            "sensor_id": self.sensor_id,
            "track_id": r.track_id,
            "zone_id": r.zone_id,
        }
        if wall_time:
            base["wall_time"] = wall_time

        out: list[dict] = []

        if r.bound:
            out.append({
                **base,
                "event_id": str(uuid.uuid4()),
                "observation": "person_identified",
                "value": r.identity,
                "confidence": r.confidence,
                "subject": {
                    "class": "human",
                    "identity": r.identity,
                    **({"role": r.role} if r.role else {}),
                },
            })

        if r.bound and r.tier == "site":
            out.append({
                **base,
                "event_id": str(uuid.uuid4()),
                "observation": "presence_unbadged",
                "value": r.identity,
                "confidence": r.confidence,
                "subject": {
                    "class": "human",
                    "identity": r.identity,
                    **({"role": r.role} if r.role else {}),
                },
            })

        if not r.bound:
            out.append({
                **base,
                "event_id": str(uuid.uuid4()),
                "observation": "identity_unverified",
                "value": r.reason,
                # The failure to identify is itself certain: we did the
                # comparison and nothing cleared the bar.
                "confidence": 1.0,
                "subject": {"class": "human"},
            })

        return out

    @staticmethod
    def badge_event_to_roster(event: dict, roster: Roster) -> bool:
        """Apply a credential_presented event to the roster.

        Only granted transactions add presence. A denied attempt is a security
        event in its own right but it does not put anyone in a room.
        """
        if event.get("observation") != "credential_presented":
            return False
        subject = event.get("subject") or {}
        identity = subject.get("identity")
        if not identity:
            return False

        value = event.get("value")
        at_us = int(event["timestamp_us"])

        if value in ("granted", "entry", True):
            roster.badge_in(identity, event.get("zone_id"), subject.get("role"), at_us)
            return True
        if value in ("exit", "badge_out"):
            roster.badge_out(identity, at_us)
            return True
        return False
