"""The employee directory: IDs to names, for the screen and nothing else.

Everything else in this codebase is keyed on a person ID -- ``EMP-4471`` --
and that is deliberate (see `enrollment.py` for the three reasons). The cost
of that choice is that an operator watching a screen sees badge numbers. This
file is the one place that turns a badge number back into a person, and it is
a lookup for DISPLAY, not a second source of truth.

WHAT CROSSES THE BOUNDARY, AND WHAT DOES NOT

  name         never. Not into an event, not into a log line, not into the
               event store. The ID is what is recorded, always.
  role         only when asked, and only to the roster. See `supply_role`.
  active       as a flag on a recognition, and as a refusal at enrolment.
  department   display only, and in the inactive-match report, because
               "whose departed badge is this" is the first question asked.

That asymmetry is the whole point of keying on an ID. The stored data holds
no personal names on its face, and the mapping to a person lives in one
small file that can be access-controlled, audited and deleted separately
from years of event history. If this module ever starts writing names into
events, the architecture has lost the property it was built for.

WHY A CSV

It needs no dependency, `csv` is in the standard library, and it still opens
in Excel -- which matters, because the person who maintains it is in HR, not
engineering. An `.xlsx` would mean adding openpyxl to the device image to
read five columns.

OPERATIONAL PROPERTIES

  Loaded once.      At startup, into a dict. Never per frame and never per
                    recognition; a recogniser that stats a file for every
                    face is a recogniser that stops when the file is on a
                    network share.
  Absence is fine.  No file means no names. Every lookup falls back to the
                    ID and the system runs exactly as it did before this
                    module existed. This is the normal state on a demo
                    laptop and it must never be an error.
  Misses are quiet. An ID that is not in the file logs once, naming the ID,
                    and then never again. At 25 fps a per-frame warning
                    would produce 90,000 identical lines an hour and bury
                    the one line that mattered.
  Reloadable.       `reload()` re-reads on demand, and with `auto_reload`
                    the file's mtime is checked at most every few seconds.
                    Staff change weekly; restarting a device to add one
                    person is not an acceptable procedure.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional

DEFAULT_DIRECTORY_PATH = "identity/data/employees.csv"

# The columns this module reads. Extra columns in the file are ignored rather
# than rejected, because a directory exported from an HR system carries
# whatever that system carries, and failing on an unexpected column would
# make the file unusable for the one job it has.
COLUMNS = ("employee_id", "name", "role", "department", "active")

# Status strings returned by `status()`. Four, not two, because "not in the
# file" and "in the file and marked gone" are different facts with different
# consequences, and so is "in the file and the column was not usable".
STATUS_ACTIVE = "active"
STATUS_INACTIVE = "inactive"
STATUS_UNSTATED = "unstated"
STATUS_UNKNOWN = "unknown"

_TRUE = {"true", "t", "yes", "y", "1", "active"}
_FALSE = {"false", "f", "no", "n", "0", "inactive", "left", "terminated"}


def parse_active(raw: Optional[str]) -> Optional[bool]:
    """Read the `active` column. Returns None when the file did not say.

    Three outcomes rather than two. A blank cell, a missing column or a value
    nobody anticipated is not evidence that somebody has left the company,
    and treating it as such would refuse to enrol a current employee over a
    typo. It is also not evidence that they are still here. None means
    unstated, and the caller decides -- which in practice means enrolment is
    allowed and the inactive flag does not fire, with a warning naming the
    row so the file gets fixed.
    """
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if not text:
        return None
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    return None


@dataclass(frozen=True)
class DirectoryEntry:
    """One row. `employee_id` is the key; everything else is about a person."""

    employee_id: str
    name: str = ""
    role: str = ""
    department: str = ""
    # None means the file did not say. See parse_active.
    active: Optional[bool] = None

    @property
    def display_name(self) -> str:
        """The name, or the ID when there is none.

        The ID is always a sufficient label. A row with a blank name is a
        row that has not been filled in yet, not an error.
        """
        return self.name or self.employee_id


@dataclass(frozen=True)
class InactiveMatch:
    """A recognition that matched somebody the directory says has left.

    Deliberately a distinct object rather than a boolean, because the three
    questions asked next are always the same: who, which department, and how
    confident was the match. A bare flag answers none of them.

    What it means is one of two things, and the system cannot tell which:

      * the enrolment was never revoked when they left -- a process failure,
        and the finding is that the revocation step does not work; or
      * somebody else is matching against their template -- which is either
        a false match or a real one, and both are worth a human look.

    Neither is a normal recognition, and neither should be drawn as a green
    box with a name in it.
    """

    employee_id: str
    department: str = ""
    confidence: float = 0.0
    detail: str = ""


class EmployeeDirectory:
    """A cached, reloadable map from employee ID to a person.

    Construct once at startup and keep it. Every lookup is a dict hit.
    """

    def __init__(
        self,
        path: str | Path = DEFAULT_DIRECTORY_PATH,
        auto_reload: bool = True,
        stat_interval_s: float = 5.0,
        supply_roles: bool = False,
        log: Optional[Callable[[str], None]] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.path = Path(path)
        self.auto_reload = auto_reload
        # How often the file may be stat'ed, at most. The lookup itself is a
        # dict hit and happens per drawn frame; a stat per frame on an NFS
        # mount is a different proposition, so the check is rate-limited
        # rather than run every time.
        self.stat_interval_s = float(stat_interval_s)
        # Rule 4, and off by default. The directory may fill in a role the
        # badge reader did not send, but only for a caller that asked for it
        # in so many words -- the engine uses role for zone rules and for a
        # step's actor_role, so a role quietly invented here would produce a
        # confident wrong_role finding against a name nobody verified.
        self.supply_roles = bool(supply_roles)
        self.log = log or (lambda msg: None)
        self._clock = clock

        self._entries: dict[str, DirectoryEntry] = {}
        # (mtime_ns, size). Compared as a pair because mtime granularity on
        # some filesystems is coarse enough that a file rewritten within the
        # same tick keeps its timestamp, and an edit that changes the length
        # is then still caught.
        self._stamp: Optional[tuple[int, int]] = None
        self._checked_at: float = -1e9
        # IDs already reported missing, so the miss is logged once per ID
        # rather than once per frame.
        self._missed: set[str] = set()
        # Problems found in the file itself, kept for a startup banner and
        # for the tests. Malformed rows are skipped, never fatal.
        self.load_errors: list[str] = []

        self.reload(announce=True)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def reload(self, announce: bool = False) -> bool:
        """Re-read the file. Returns True when the contents changed.

        Safe to call at any time, including on a file that has just been
        deleted: the directory then goes empty and every lookup falls back
        to the ID, which is the same state as never having had a file.
        """
        before = self._entries
        self._entries, self.load_errors = self._read()
        self._stamp = self._current_stamp()
        self._checked_at = self._clock()

        changed = self._entries != before

        if changed:
            # An ID that was missing may now be present. Forgetting the
            # misses lets the next lookup report an ID that is still missing
            # after a reload, which is the one case worth saying twice.
            self._missed.clear()

        for problem in self.load_errors:
            self.log(f"[directory] {problem}")

        if announce:
            if not self.path.exists():
                self.log(
                    f"[directory] no employee directory at {self.path}; "
                    f"IDs will be shown as themselves"
                )
            else:
                self.log(
                    f"[directory] loaded {len(self._entries)} employee(s) "
                    f"from {self.path} "
                    f"({self.inactive_count()} marked inactive)"
                )
        elif changed:
            self.log(
                f"[directory] reloaded {self.path}: "
                f"{len(self._entries)} employee(s)"
            )

        return changed

    def maybe_reload(self) -> bool:
        """Reload if the file changed since last time. Returns True if so.

        Rate-limited by `stat_interval_s`, and a no-op when `auto_reload` is
        off. Callers that want a deterministic reload call `reload()`.
        """
        if not self.auto_reload:
            return False
        now = self._clock()
        if now - self._checked_at < self.stat_interval_s:
            return False
        self._checked_at = now
        if self._current_stamp() == self._stamp:
            return False
        return self.reload()

    def _current_stamp(self) -> Optional[tuple[int, int]]:
        try:
            st = self.path.stat()
        except OSError:
            # Absent, or unreadable. Both mean "no directory", and neither is
            # an error -- see the module docstring.
            return None
        return (st.st_mtime_ns, st.st_size)

    def _read(self) -> tuple[dict[str, DirectoryEntry], list[str]]:
        """Parse the CSV. Never raises; problems come back as text."""
        if not self.path.exists():
            return {}, []

        problems: list[str] = []
        entries: dict[str, DirectoryEntry] = {}

        try:
            # utf-8-sig because Excel writes a BOM, and a BOM on the first
            # header turns `employee_id` into `﻿employee_id`, which then
            # silently matches nothing. newline="" is what csv wants.
            with self.path.open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                reader = csv.DictReader(handle)
                fields = {
                    (name or "").strip().lower() for name in (reader.fieldnames or [])
                }
                if "employee_id" not in fields:
                    return {}, [
                        f"{self.path} has no employee_id column "
                        f"(found {sorted(fields) or 'nothing'}); "
                        f"expected {', '.join(COLUMNS)}. Ignoring the file."
                    ]

                for row in reader:
                    # reader.line_num, not a counter: DictReader skips blank
                    # lines without yielding them, so a counter drifts by one
                    # per blank line and every problem after the first blank
                    # row is then reported against the wrong line. The point
                    # of printing a line number is that somebody can go and
                    # look at it.
                    line_no = reader.line_num
                    cells = {
                        (key or "").strip().lower(): (value or "").strip()
                        for key, value in row.items()
                        if key is not None
                    }

                    employee_id = cells.get("employee_id", "")
                    if not employee_id:
                        # A trailing blank line is normal and not worth
                        # mentioning; a row with other data but no ID is a
                        # row somebody meant to fill in.
                        if any(cells.values()):
                            problems.append(
                                f"{self.path}:{line_no} has no employee_id; "
                                f"row skipped"
                            )
                        continue

                    if employee_id in entries:
                        # First wins, deterministically, and say so. Last
                        # would mean the answer depends on row order, and two
                        # rows for one ID is exactly the case where somebody
                        # has pasted a leaver and a rejoiner into one file.
                        problems.append(
                            f"{self.path}:{line_no} repeats employee_id "
                            f"{employee_id!r}; keeping the first row"
                        )
                        continue

                    raw_active = cells.get("active")
                    active = parse_active(raw_active)
                    if raw_active and active is None:
                        problems.append(
                            f"{self.path}:{line_no} active={raw_active!r} for "
                            f"{employee_id} is not yes/no; treated as unstated"
                        )

                    entries[employee_id] = DirectoryEntry(
                        employee_id=employee_id,
                        name=cells.get("name", ""),
                        role=cells.get("role", ""),
                        department=cells.get("department", ""),
                        active=active,
                    )
        except OSError as exc:
            return {}, [f"cannot read {self.path}: {exc}. Ignoring the file."]
        except csv.Error as exc:
            return {}, [f"{self.path} is not readable as CSV: {exc}. Ignoring it."]

        return entries, problems

    # ------------------------------------------------------------------
    # Display -- the reason this module exists
    # ------------------------------------------------------------------

    def display_name(self, employee_id: str) -> str:
        """What to show a human for this ID. Never written to an event.

        Falls back to the ID itself, which is always a sufficient label, and
        logs the miss once per ID.
        """
        entry = self.get(employee_id)
        if entry is None:
            self._note_miss(employee_id)
            return employee_id
        return entry.display_name

    def label(self, employee_id: str) -> str:
        """`EMP-4471 (Rajesh Kumar)`, or just the ID when unknown.

        For an operator display where the ID is itself worth seeing -- a
        startup banner, an enrolment summary -- because the ID is what every
        log line and every event will carry, and hiding it behind a name
        makes those logs harder to read, not easier.
        """
        shown = self.display_name(employee_id)
        return employee_id if shown == employee_id else f"{employee_id} ({shown})"

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def get(self, employee_id: str) -> Optional[DirectoryEntry]:
        """The row for this ID, or None. Checks for a file change first."""
        self.maybe_reload()
        return self._entries.get(str(employee_id))

    def status(self, employee_id: str) -> str:
        """One of active / inactive / unstated / unknown."""
        entry = self.get(employee_id)
        if entry is None:
            return STATUS_UNKNOWN
        if entry.active is True:
            return STATUS_ACTIVE
        if entry.active is False:
            return STATUS_INACTIVE
        return STATUS_UNSTATED

    def is_inactive(self, employee_id: str) -> bool:
        """True only for an ID the file explicitly marks as gone.

        Not the negation of `is_active`: an unknown ID is neither. The two
        are asked in different places and conflating them would make every
        contractor read as a leaver.
        """
        return self.status(employee_id) == STATUS_INACTIVE

    def is_active(self, employee_id: str) -> bool:
        return self.status(employee_id) == STATUS_ACTIVE

    def role_for(self, employee_id: str) -> Optional[str]:
        """The role on the row, or None. See `supply_role` before using it."""
        entry = self.get(employee_id)
        if entry is None or not entry.role:
            return None
        return entry.role

    def department_for(self, employee_id: str) -> str:
        entry = self.get(employee_id)
        return entry.department if entry else ""

    @property
    def available(self) -> bool:
        """Is there a directory at all?

        False on a demo laptop with no file, and the system is expected to
        run in that state.
        """
        return bool(self._entries)

    def inactive_count(self) -> int:
        return sum(1 for e in self._entries.values() if e.active is False)

    def entries(self) -> Iterator[DirectoryEntry]:
        self.maybe_reload()
        return iter(sorted(self._entries.values(), key=lambda e: e.employee_id))

    def __contains__(self, employee_id: object) -> bool:
        return str(employee_id) in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    # ------------------------------------------------------------------
    # Rule 4: the directory may fill in a role, if asked
    # ------------------------------------------------------------------

    def supply_role(
        self, employee_id: str, known_role: Optional[str] = None
    ) -> Optional[str]:
        """The role to use, given what the badge event already said.

        A role that arrived with the credential always wins: it came from the
        access control system, which is the system of record for who is
        allowed where. The directory only fills a gap, and only when
        `supply_roles` is on.

        Off by default, and explicitly so. The engine uses role for zone
        rules and for a step's `actor_role`, so a role invented here does not
        stay cosmetic -- it decides whether somebody is reported for being in
        the wrong place. A site that wants that behaviour should have decided
        that its directory is accurate enough to carry it.
        """
        if known_role:
            return known_role
        if not self.supply_roles:
            return None
        role = self.role_for(employee_id)
        if role:
            self.log(
                f"[directory] role {role!r} for {employee_id} supplied by the "
                f"directory; the badge event carried none"
            )
        return role

    # ------------------------------------------------------------------
    # Rule 5: a leaver is not a normal recognition
    # ------------------------------------------------------------------

    def is_enrollable(self, employee_id: str) -> bool:
        """May this ID be enrolled?

        Everything except an explicit inactive. An ID absent from the
        directory is enrollable: contractors, new starters and sites with no
        directory at all are all that case, and refusing them would make
        this module a prerequisite for enrolment rather than a convenience
        for the screen.
        """
        return not self.is_inactive(employee_id)

    def enrolment_refusal(self, employee_id: str) -> Optional[str]:
        """Why enrolment was refused, in words an operator can act on."""
        if not self.is_inactive(employee_id):
            return None
        entry = self.get(employee_id)
        who = entry.display_name if entry else employee_id
        dept = f", {entry.department}" if entry and entry.department else ""
        return (
            f"{employee_id} is marked inactive in {self.path} ({who}{dept}). "
            f"Someone who has left must not be enrolled: their template would "
            f"keep matching and every match would read as a current employee. "
            f"If they have returned, set active to yes in the directory first."
        )

    def inactive_match(
        self, employee_id: str, confidence: float = 0.0
    ) -> Optional[InactiveMatch]:
        """Report a match against an inactive ID, or None if there is nothing
        to report.

        Returns a structured report rather than raising or suppressing. The
        match itself is real and the attribution is probably correct; what is
        wrong is that this person should not have a live template. Callers
        route the report -- a warning line, a distinct colour on screen, an
        operator queue -- and must not draw it as an ordinary recognition.

        Note what this does NOT do: it does not refuse the binding. Turning a
        leaver into `identity_unverified` would throw away the one piece of
        information worth having, which is which leaver.
        """
        if not self.is_inactive(employee_id):
            return None
        entry = self.get(employee_id)
        dept = entry.department if entry else ""
        return InactiveMatch(
            employee_id=employee_id,
            department=dept,
            confidence=round(float(confidence), 4),
            detail=(
                f"{employee_id} matched at {confidence:.3f} but is marked "
                f"inactive in the employee directory"
                + (f" ({dept})" if dept else "")
                + ". Either the enrolment was never revoked when they left, "
                "or somebody else is matching against their template."
            ),
        )

    # ------------------------------------------------------------------

    def _note_miss(self, employee_id: str) -> None:
        if employee_id in self._missed:
            return
        self._missed.add(employee_id)
        if not self.path.exists():
            # No file is the expected state, not a miss worth a line per ID.
            return
        self.log(
            f"[directory] {employee_id} is not in {self.path}; showing the ID. "
            f"Logged once per ID, not per frame."
        )


def build_directory(cfg: Optional[dict] = None, log=None) -> EmployeeDirectory:
    """Build a directory from an identity config's `directory` block.

    Same shape as `matcher.build_matcher` and
    `perception.detectors.build_detector`: the config decides, and nothing
    else needs to know. An absent block builds a directory on the default
    path, which on a machine with no such file is an empty one -- the point
    being that wiring this in can never be what breaks a running system.
    """
    cfg = cfg or {}
    return EmployeeDirectory(
        path=cfg.get("path", DEFAULT_DIRECTORY_PATH),
        auto_reload=bool(cfg.get("auto_reload", True)),
        stat_interval_s=float(cfg.get("stat_interval_s", 5.0)),
        supply_roles=bool(cfg.get("supply_roles", False)),
        log=log,
    )
