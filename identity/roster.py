"""Who is where, according to the doors.

The roster is built from access control transactions, not from cameras. That
matters: a badge-in is a fact with confidence 1.0, verified at 30cm on a
cooperative subject in controlled light. Re-identifying the same person at 8
metres, oblique, in a helmet, is a much harder problem that we are trying to
avoid solving. Do the hard identification once, where it is easy, and carry the
answer forward.

The roster's job is to shrink the candidate set. Matching a face against the
four people known to be in this zone is a different statistical problem from
matching against five hundred enrolled employees, and the difference is what
makes workspace identification viable at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

US_PER_S = 1_000_000


@dataclass
class Presence:
    """One person, believed present, according to the last door they used."""

    identity: str
    zone_id: Optional[str]
    role: Optional[str]
    entered_at_us: int
    last_seen_us: int

    def age_s(self, now_us: int) -> float:
        return (now_us - self.last_seen_us) / US_PER_S


class Roster:
    """Occupancy derived from access control events.

    Deliberately conservative in one direction: a person stays on the roster
    until they badge out or expire. An over-long roster costs accuracy (a wider
    candidate set) but never correctness. Dropping someone too early is worse,
    because they then appear as an unaccounted presence and generate a security
    finding for a person who did nothing wrong.
    """

    def __init__(
        self,
        presence_ttl_s: Optional[float] = 12 * 3600,
        has_exit_readers: bool = False,
    ) -> None:
        # Most sites have entry readers and no exit readers, so presence has to
        # expire on a timer. The TTL should exceed the longest plausible shift;
        # too short and people vanish from the roster mid-shift.
        self.presence_ttl_s = presence_ttl_s
        self.has_exit_readers = has_exit_readers
        self._people: dict[str, Presence] = {}

    # ------------------------------------------------------------------

    def badge_in(
        self,
        identity: str,
        zone_id: Optional[str],
        role: Optional[str],
        at_us: int,
    ) -> None:
        """Record an authenticated entry.

        A badge-in at zone B also means the person is no longer at zone A, so
        this replaces rather than accumulates. Without that, someone who moved
        between three areas would appear to be in all three, and every zone's
        candidate set would grow until identification stopped working.
        """
        existing = self._people.get(identity)
        entered = existing.entered_at_us if existing else at_us
        self._people[identity] = Presence(
            identity=identity,
            zone_id=zone_id,
            role=role or (existing.role if existing else None),
            entered_at_us=entered,
            last_seen_us=at_us,
        )

    def badge_out(self, identity: str, at_us: int) -> None:
        self._people.pop(identity, None)

    def touch(self, identity: str, at_us: int) -> None:
        """Refresh presence after a confirmed sighting.

        A person seen by a camera is demonstrably still here, so the TTL should
        restart. Without this, someone working a long shift silently expires off
        the roster and then reads as an unaccounted presence.
        """
        p = self._people.get(identity)
        if p:
            p.last_seen_us = at_us

    def expire(self, now_us: int) -> list[str]:
        """Drop stale presences. Returns who was dropped, for logging."""
        if self.presence_ttl_s is None:
            return []
        dropped = [
            ident
            for ident, p in self._people.items()
            if p.age_s(now_us) > self.presence_ttl_s
        ]
        for ident in dropped:
            del self._people[ident]
        return dropped

    def reset(self) -> None:
        """End of day. Everyone is assumed to have left."""
        self._people.clear()

    # ------------------------------------------------------------------

    def in_zone(self, zone_id: str) -> list[Presence]:
        """Tier 1 candidates: badged into this specific zone."""
        return [p for p in self._people.values() if p.zone_id == zone_id]

    def in_building(self) -> list[Presence]:
        """Tier 2 candidates: everyone still believed on site."""
        return list(self._people.values())

    def get(self, identity: str) -> Optional[Presence]:
        return self._people.get(identity)

    def __len__(self) -> int:
        return len(self._people)
