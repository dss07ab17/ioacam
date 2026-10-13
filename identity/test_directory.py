"""Tests for the employee directory lookup.

Run with:  python3 identity/test_directory.py

No camera, no model, no enrolment data. Every test writes its own CSV into a
temporary directory, so what is under test is the lookup, the fallbacks and
the reload -- which is where this module can go wrong in ways that matter.

The one property worth stating up front: the interesting cases here are all
ABSENCE. No file, no row, no name, no active column. A directory lookup that
only works when the file is perfect is a lookup that takes the whole system
down the first time somebody saves the CSV in the wrong place.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
# Only the project root goes on the path, for the same reason as every other
# suite here: adding identity/ as well imports the same module under two
# names and gives two copies of every class.
sys.path.insert(0, str(ROOT))

from identity.directory import (  # noqa: E402
    STATUS_ACTIVE,
    STATUS_INACTIVE,
    STATUS_UNKNOWN,
    STATUS_UNSTATED,
    EmployeeDirectory,
    build_directory,
    parse_active,
)
from identity.matcher import RocPoint, StubMatcher, ThresholdPolicy  # noqa: E402
from identity.resolver import IdentityResolver  # noqa: E402
from identity.roster import Roster  # noqa: E402

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"[{'PASS' if condition else 'FAIL'}] {name}")
    if not condition:
        failures.append(f"{name}: {detail}")


SAMPLE = """employee_id,name,role,department,active
EMP-4471,Rajesh Kumar,operator,Assembly,yes
EMP-5522,Priya Nair,maintenance_engineer,Maintenance,yes
EMP-9012,Daniel Okafor,operator,Assembly,no
"""


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def directory(tmp: Path, text: str | None = SAMPLE, **kw) -> EmployeeDirectory:
    """A directory over a fresh CSV, or over no file at all when text is None."""
    path = tmp / "employees.csv"
    if text is not None:
        write(path, text)
    kw.setdefault("auto_reload", False)
    return EmployeeDirectory(path, **kw)


# ----------------------------------------------------------------------
# A known ID returns a name and a role
# ----------------------------------------------------------------------

with tempfile.TemporaryDirectory() as tmpdir:
    tmp = Path(tmpdir)
    d = directory(tmp)

    check(
        "a known ID returns its name",
        d.display_name("EMP-4471") == "Rajesh Kumar",
        d.display_name("EMP-4471"),
    )
    check(
        "a known ID returns its role",
        d.role_for("EMP-5522") == "maintenance_engineer",
        str(d.role_for("EMP-5522")),
    )
    check(
        "the whole row is available, keyed on the ID",
        (lambda e: e.department == "Assembly" and e.active is True)(d.get("EMP-4471")),
        str(d.get("EMP-4471")),
    )
    check(
        "label() keeps the ID visible beside the name",
        d.label("EMP-4471") == "EMP-4471 (Rajesh Kumar)",
        d.label("EMP-4471"),
    )
    check(
        "three rows loaded, one of them inactive",
        len(d) == 3 and d.inactive_count() == 1,
        f"{len(d)} rows, {d.inactive_count()} inactive",
    )
    check(
        "a clean file produces no load errors",
        not d.load_errors,
        str(d.load_errors),
    )

    # The cache is the point of rule 2: the file is read once, not per
    # recognition. With auto_reload off, nothing may re-read it even if it
    # changes underneath.
    write(tmp / "employees.csv", SAMPLE.replace("Rajesh Kumar", "SOMEONE ELSE"))
    check(
        "lookups are served from cache, not re-read per call",
        d.display_name("EMP-4471") == "Rajesh Kumar",
        d.display_name("EMP-4471"),
    )


# ----------------------------------------------------------------------
# An unknown ID falls back to the ID, and logs once
# ----------------------------------------------------------------------

with tempfile.TemporaryDirectory() as tmpdir:
    tmp = Path(tmpdir)
    lines: list[str] = []
    d = directory(tmp, log=lines.append)

    try:
        shown = d.display_name("EMP-0000")
        raised = ""
    except Exception as exc:          # noqa: BLE001 - the test is that it cannot
        shown, raised = "", repr(exc)

    check(
        "an unknown ID falls back to the ID itself",
        shown == "EMP-0000",
        shown,
    )
    check(
        "an unknown ID does not raise",
        not raised,
        raised,
    )
    check(
        "an unknown ID has no role to offer, rather than a guessed one",
        d.role_for("EMP-0000") is None,
        str(d.role_for("EMP-0000")),
    )
    check(
        "an unknown ID reads as unknown, not as inactive",
        d.status("EMP-0000") == STATUS_UNKNOWN and not d.is_inactive("EMP-0000"),
        d.status("EMP-0000"),
    )

    # Rule 3. At 25 fps a per-frame warning is ninety thousand identical
    # lines an hour, and the one line that mattered is somewhere in them.
    misses_before = [ln for ln in lines if "EMP-0000" in ln]
    for _ in range(500):
        d.display_name("EMP-0000")
    misses_after = [ln for ln in lines if "EMP-0000" in ln]
    check(
        "the miss is logged once per ID, not once per lookup",
        len(misses_before) == 1 and len(misses_after) == 1,
        f"{len(misses_before)} then {len(misses_after)} line(s)",
    )
    check(
        "the miss line names the ID and says it is shown as itself",
        "EMP-0000" in misses_after[0] and "showing the ID" in misses_after[0],
        misses_after[0] if misses_after else "(nothing logged)",
    )

    # A second unknown ID is its own miss. Deduplicating by anything coarser
    # would hide the second person.
    d.display_name("EMP-1111")
    check(
        "a different unknown ID gets its own line",
        any("EMP-1111" in ln for ln in lines),
        str(lines),
    )


# ----------------------------------------------------------------------
# An inactive ID is reported as such
# ----------------------------------------------------------------------

with tempfile.TemporaryDirectory() as tmpdir:
    tmp = Path(tmpdir)
    d = directory(tmp)

    check(
        "an inactive ID reads as inactive",
        d.status("EMP-9012") == STATUS_INACTIVE and d.is_inactive("EMP-9012"),
        d.status("EMP-9012"),
    )
    check(
        "an active ID reads as active",
        d.status("EMP-4471") == STATUS_ACTIVE and d.is_active("EMP-4471"),
        d.status("EMP-4471"),
    )
    check(
        "an inactive ID still resolves to a name, for the screen",
        d.display_name("EMP-9012") == "Daniel Okafor",
        d.display_name("EMP-9012"),
    )

    # Rule 5, first half: not enrollable.
    check(
        "an inactive ID is not enrollable",
        not d.is_enrollable("EMP-9012"),
        str(d.is_enrollable("EMP-9012")),
    )
    refusal = d.enrolment_refusal("EMP-9012") or ""
    check(
        "the refusal names the person and says what to do about it",
        "EMP-9012" in refusal and "Daniel Okafor" in refusal and "active" in refusal,
        refusal,
    )
    check(
        "an active ID is enrollable and has no refusal",
        d.is_enrollable("EMP-4471") and d.enrolment_refusal("EMP-4471") is None,
        str(d.enrolment_refusal("EMP-4471")),
    )
    check(
        "an unknown ID is enrollable -- contractors and new starters",
        d.is_enrollable("EMP-0000"),
        "refusing unknown IDs would make the directory a prerequisite",
    )

    # Rule 5, second half: a match is a report, not a normal recognition.
    flag = d.inactive_match("EMP-9012", confidence=0.91)
    check(
        "a match against an inactive ID produces a report",
        flag is not None
        and flag.employee_id == "EMP-9012"
        and flag.department == "Assembly"
        and flag.confidence == 0.91,
        str(flag),
    )
    check(
        "the report explains both ways it can have happened",
        flag is not None
        and "never revoked" in flag.detail
        and "somebody else" in flag.detail,
        flag.detail if flag else "",
    )
    check(
        "the report carries the ID, not the name",
        flag is not None and "Daniel Okafor" not in flag.detail,
        flag.detail if flag else "",
    )
    check(
        "an active match produces no report at all",
        d.inactive_match("EMP-4471", confidence=0.91) is None,
        str(d.inactive_match("EMP-4471", confidence=0.91)),
    )
    check(
        "an unknown ID produces no report -- absent is not departed",
        d.inactive_match("EMP-0000", confidence=0.91) is None,
        str(d.inactive_match("EMP-0000", confidence=0.91)),
    )


# ----------------------------------------------------------------------
# The cascade marks an inactive match without refusing it
# ----------------------------------------------------------------------

ROC = [
    RocPoint(0.40, 1e-2),
    RocPoint(0.55, 1e-3),
    RocPoint(0.70, 1e-4),
    RocPoint(0.82, 1e-5),
    RocPoint(0.90, 1e-6),
]

with tempfile.TemporaryDirectory() as tmpdir:
    tmp = Path(tmpdir)
    d = directory(tmp)

    roster = Roster(presence_ttl_s=12 * 3600)
    roster.badge_in("EMP-9012", "zone-assembly-4", "operator", 0)

    resolver = IdentityResolver(
        roster=roster,
        matcher=StubMatcher({"probe-leaver": {"EMP-9012": 0.95}}),
        policy=ThresholdPolicy(ROC, target_false_match_rate=1e-3),
        directory=d,
    )
    r = resolver.resolve("t-1", "zone-assembly-4", "probe-leaver", 0)

    check(
        "an inactive match still binds -- which leaver it is, is the point",
        r.bound and r.identity == "EMP-9012",
        f"{r.tier} / {r.identity}",
    )
    check(
        "and it is marked, so it is not a normal recognition",
        r.inactive is not None and r.inactive.employee_id == "EMP-9012",
        str(r.inactive),
    )

    # The event stream is unchanged: the directory supplies no part of it.
    events = resolver.events_for(r, 0)
    blob = str(events)
    check(
        "the events carry the ID and no name from the directory",
        "EMP-9012" in blob and "Daniel Okafor" not in blob and "Assembly" not in blob,
        blob,
    )

    # A resolver with no directory behaves exactly as it did before this
    # module existed. That is what makes the whole thing optional.
    plain = IdentityResolver(
        roster=roster,
        matcher=StubMatcher({"probe-leaver": {"EMP-9012": 0.95}}),
        policy=ThresholdPolicy(ROC, target_false_match_rate=1e-3),
    )
    check(
        "with no directory there is no mark and no error",
        plain.resolve("t-2", "zone-assembly-4", "probe-leaver", 0).inactive is None,
        str(plain.resolve("t-2", "zone-assembly-4", "probe-leaver", 0)),
    )


# ----------------------------------------------------------------------
# Rule 4: the directory supplies a role only when asked
# ----------------------------------------------------------------------

with tempfile.TemporaryDirectory() as tmpdir:
    tmp = Path(tmpdir)

    badge = {
        "observation": "credential_presented",
        "timestamp_us": 0,
        "value": "granted",
        "zone_id": "zone-assembly-4",
        "subject": {"identity": "EMP-5522"},      # no role from the reader
    }

    silent = directory(tmp)
    roster = Roster()
    IdentityResolver.badge_event_to_roster(dict(badge), roster, directory=silent)
    check(
        "a directory without supply_roles supplies no role",
        roster.get("EMP-5522").role is None,
        str(roster.get("EMP-5522")),
    )

    asked = directory(tmp, supply_roles=True)
    roster = Roster()
    IdentityResolver.badge_event_to_roster(dict(badge), roster, directory=asked)
    check(
        "with supply_roles on, the directory fills the gap",
        roster.get("EMP-5522").role == "maintenance_engineer",
        str(roster.get("EMP-5522")),
    )

    # The access control system is the system of record for who is allowed
    # where. A role that arrived with the credential must win.
    with_role = dict(badge)
    with_role["subject"] = {"identity": "EMP-5522", "role": "visitor"}
    roster = Roster()
    IdentityResolver.badge_event_to_roster(with_role, roster, directory=asked)
    check(
        "a role on the badge event wins over the directory",
        roster.get("EMP-5522").role == "visitor",
        str(roster.get("EMP-5522")),
    )

    roster = Roster()
    IdentityResolver.badge_event_to_roster(dict(badge), roster)
    check(
        "passing no directory leaves badge_event_to_roster as it was",
        roster.get("EMP-5522").role is None,
        str(roster.get("EMP-5522")),
    )


# ----------------------------------------------------------------------
# No file at all: the system runs with IDs only
# ----------------------------------------------------------------------

with tempfile.TemporaryDirectory() as tmpdir:
    tmp = Path(tmpdir)
    lines: list[str] = []

    try:
        d = directory(tmp, text=None, log=lines.append)
        raised = ""
    except Exception as exc:          # noqa: BLE001 - the test is that it cannot
        d, raised = None, repr(exc)

    check(
        "an absent file does not raise at construction",
        not raised,
        raised,
    )
    check(
        "an absent file is an empty, unavailable directory",
        d is not None and len(d) == 0 and not d.available,
        "" if d is None else f"{len(d)} rows, available={d.available}",
    )
    check(
        "every lookup falls back to the ID",
        d.display_name("EMP-4471") == "EMP-4471"
        and d.label("EMP-4471") == "EMP-4471",
        d.label("EMP-4471"),
    )
    check(
        "nothing is inactive, so nothing is refused enrolment",
        not d.is_inactive("EMP-4471")
        and d.is_enrollable("EMP-4471")
        and d.inactive_match("EMP-4471") is None,
        d.status("EMP-4471"),
    )
    check(
        "absence is announced once, not reported as a miss per ID",
        sum(1 for ln in lines if "no employee directory" in ln) == 1
        and not any("not in" in ln for ln in lines),
        str(lines),
    )

    # An absent file that appears later must be picked up, because that is
    # the order things happen in: the device ships, then HR sends the CSV.
    write(tmp / "employees.csv", SAMPLE)
    d.reload()
    check(
        "a file that appears after startup is picked up by a reload",
        d.available and d.display_name("EMP-4471") == "Rajesh Kumar",
        d.display_name("EMP-4471"),
    )

    # And a file that disappears must not take the system with it.
    (tmp / "employees.csv").unlink()
    d.reload()
    check(
        "a file that is deleted leaves the system running on IDs",
        not d.available and d.display_name("EMP-4471") == "EMP-4471",
        d.display_name("EMP-4471"),
    )


# ----------------------------------------------------------------------
# Rule 6: a reload picks up a newly added row
# ----------------------------------------------------------------------

with tempfile.TemporaryDirectory() as tmpdir:
    tmp = Path(tmpdir)
    path = tmp / "employees.csv"
    d = directory(tmp)

    check(
        "the new starter is not there yet",
        d.display_name("EMP-7777") == "EMP-7777",
        d.display_name("EMP-7777"),
    )

    write(path, SAMPLE + "EMP-7777,Mei Tanaka,operator,Assembly,yes\n")
    changed = d.reload()

    check(
        "an explicit reload reports that the contents changed",
        changed,
        str(changed),
    )
    check(
        "a reload picks up a newly added row",
        d.display_name("EMP-7777") == "Mei Tanaka"
        and d.role_for("EMP-7777") == "operator",
        f"{d.display_name('EMP-7777')} / {d.role_for('EMP-7777')}",
    )
    check(
        "rows that did not change are still there",
        len(d) == 4 and d.display_name("EMP-4471") == "Rajesh Kumar",
        f"{len(d)} rows",
    )
    check(
        "a reload with no change says so, so a caller can stay quiet",
        not d.reload(),
        "reload() returned True for an unchanged file",
    )

    # A reload clears the remembered misses, so an ID that is STILL missing
    # after somebody edited the file gets said again -- that is the one case
    # worth repeating, because it means the edit did not do what they thought.
    lines: list[str] = []
    d.log = lines.append
    d.display_name("EMP-8888")
    write(
        path,
        SAMPLE
        + "EMP-7777,Mei Tanaka,operator,Assembly,yes\n"
        + "EMP-6001,Ada Byron,admin,Office,yes\n",
    )
    d.reload()
    d.display_name("EMP-8888")
    check(
        "a still-missing ID is reported again after a reload",
        sum(1 for ln in lines if "EMP-8888" in ln) == 2,
        str(lines),
    )

    # Rule 6, the other half: mtime watching, so nobody has to call reload.
    watched = EmployeeDirectory(path, auto_reload=True, stat_interval_s=0.0)
    check(
        "the watched directory starts from what is on disk",
        watched.display_name("EMP-6001") == "Ada Byron",
        watched.display_name("EMP-6001"),
    )
    write(
        path,
        SAMPLE
        + "EMP-6001,Ada Byron,admin,Office,yes\n"
        + "EMP-6002,Alan T,admin,Office,yes\n",
    )
    # Windows and some Linux filesystems have coarse mtime granularity, so a
    # file rewritten within one tick can keep its timestamp. The stamp is
    # (mtime, size) for exactly that reason, and this write changes the size;
    # bump the mtime too so the test does not depend on which it noticed.
    os.utime(path, (0, 0))
    check(
        "a changed file is picked up without anyone calling reload",
        watched.display_name("EMP-6002") == "Alan T",
        watched.display_name("EMP-6002"),
    )

    # Rate limiting is what keeps rule 2 true: the lookup is a dict hit, and
    # the file is stat'ed at most every stat_interval_s, never per frame.
    slow = EmployeeDirectory(path, auto_reload=True, stat_interval_s=3600.0)
    write(path, SAMPLE + "EMP-6003,Grace H,admin,Office,yes\n")
    os.utime(path, (0, 0))
    check(
        "stat_interval_s stops the file being checked per lookup",
        slow.display_name("EMP-6003") == "EMP-6003",
        slow.display_name("EMP-6003"),
    )
    check(
        "and an explicit reload still works regardless of the interval",
        slow.reload() and slow.display_name("EMP-6003") == "Grace H",
        slow.display_name("EMP-6003"),
    )


# ----------------------------------------------------------------------
# A file maintained by hand, in Excel, by somebody in HR
# ----------------------------------------------------------------------

check(
    "active parses the spellings a human will actually type",
    [parse_active(v) for v in ("yes", "Y", "TRUE", "1", "active")] == [True] * 5
    and [parse_active(v) for v in ("no", "N", "FALSE", "0", "left")] == [False] * 5,
    str([parse_active(v) for v in ("yes", "no", "TRUE", "left")]),
)
check(
    "a blank or unparseable active is unstated, not a leaver",
    parse_active("") is None
    and parse_active(None) is None
    and parse_active("maybe") is None,
    str(parse_active("maybe")),
)

with tempfile.TemporaryDirectory() as tmpdir:
    tmp = Path(tmpdir)
    lines: list[str] = []

    # Everything wrong with a hand-maintained file, at once: a BOM from
    # Excel, padded cells, a header in the wrong case, an extra column from
    # an HR export, a blank line, a row with no ID, a duplicate, a garbage
    # active value, and a row with no name.
    messy = (
        "﻿Employee_ID, Name ,role,department,active,cost_centre\n"
        "EMP-4471, Rajesh Kumar ,operator,Assembly,yes,CC-1\n"
        "\n"
        ",Nobody,operator,Assembly,yes,CC-1\n"
        "EMP-4471,Someone Else,admin,Office,no,CC-9\n"
        "EMP-5522,Priya Nair,maintenance_engineer,Maintenance,maybe,CC-2\n"
        "EMP-9012,,operator,Assembly,no,CC-3\n"
    )
    write(tmp / "employees.csv", messy)
    d = EmployeeDirectory(tmp / "employees.csv", auto_reload=False, log=lines.append)

    check(
        "a BOM, padding and header case do not hide a row",
        d.display_name("EMP-4471") == "Rajesh Kumar",
        d.display_name("EMP-4471"),
    )
    check(
        "an extra column from an HR export is ignored, not rejected",
        len(d) == 3,
        f"{len(d)} rows: {[e.employee_id for e in d.entries()]}",
    )
    check(
        "a duplicate ID keeps the first row and says so",
        d.display_name("EMP-4471") == "Rajesh Kumar"
        and any("repeats employee_id" in ln for ln in lines),
        str(lines),
    )
    check(
        "a row with no employee_id is skipped and named by line number",
        any("no employee_id" in ln and ":4" in ln for ln in lines),
        str(lines),
    )
    check(
        "an unparseable active is unstated, and the row is named",
        d.status("EMP-5522") == STATUS_UNSTATED
        and any("maybe" in ln and "EMP-5522" in ln for ln in lines),
        f"{d.status('EMP-5522')} / {lines}",
    )
    check(
        "an unstated active is still enrollable -- a typo is not a departure",
        d.is_enrollable("EMP-5522") and d.inactive_match("EMP-5522") is None,
        d.status("EMP-5522"),
    )
    check(
        "a row with no name shows its ID",
        d.display_name("EMP-9012") == "EMP-9012",
        d.display_name("EMP-9012"),
    )
    check(
        "a blank line is not reported as a problem",
        not any(":3" in ln for ln in lines),
        str(lines),
    )

with tempfile.TemporaryDirectory() as tmpdir:
    tmp = Path(tmpdir)
    lines = []
    write(tmp / "employees.csv", "name,role\nRajesh Kumar,operator\n")
    d = EmployeeDirectory(tmp / "employees.csv", auto_reload=False, log=lines.append)
    check(
        "a file with no employee_id column is ignored, loudly, not guessed at",
        len(d) == 0 and any("no employee_id column" in ln for ln in lines),
        str(lines),
    )
    check(
        "and the system carries on with IDs",
        d.display_name("EMP-4471") == "EMP-4471",
        d.display_name("EMP-4471"),
    )


# ----------------------------------------------------------------------
# The shipped example, and the config seam
# ----------------------------------------------------------------------

example = ROOT / "identity" / "data" / "employees.example.csv"
check(
    "the example directory is committed where the docs say it is",
    example.exists(),
    str(example),
)
if example.exists():
    d = EmployeeDirectory(example, auto_reload=False)
    check(
        "the example parses, with rows and at least one leaver to demonstrate",
        len(d) >= 2 and d.inactive_count() >= 1 and not d.load_errors,
        f"{len(d)} rows, {d.inactive_count()} inactive, {d.load_errors}",
    )

built = build_directory({"path": str(example), "supply_roles": True})
check(
    "build_directory reads the config block, like every other backend seam",
    built.supply_roles and len(built) == len(d),
    f"supply_roles={built.supply_roles}, {len(built)} rows",
)
check(
    "an absent config block builds an empty directory, not an exception",
    len(build_directory({"path": "identity/data/does-not-exist.csv"})) == 0,
    "a missing default path must never be what breaks startup",
)
check(
    "supply_roles is off unless the config says otherwise",
    not build_directory({"path": str(example)}).supply_roles,
    "role feeds the engine's zone rules; it must not default on",
)


print()
if failures:
    print(f"{len(failures)} directory test(s) failed")
    for f in failures:
        print(f"  !! {f}")
    raise SystemExit(1)
print("all directory tests passed")
