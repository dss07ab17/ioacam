"""Tests for the identity layer.

Run with:  python3 identity/test_identity.py

No camera, no face model, no enrolment data. The matcher is a stub, so what is
under test is the cascade logic and the threshold arithmetic -- which is where
the mistakes that matter live.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
# Only the project root goes on the path. Adding identity/ as well would let
# the same module be imported under two names (matcher and identity.matcher),
# which gives two copies of every class and makes isinstance checks fail in
# ways that take an afternoon to find.
sys.path.insert(0, str(ROOT))

from identity.enrollment import (  # noqa: E402
    EnrolledTemplate,
    display_name_for,
    is_person_id,
    load_templates,
    save_templates,
)
from identity.matcher import RocPoint, StubMatcher, ThresholdPolicy  # noqa: E402
from identity.resolver import IdentityResolver  # noqa: E402
from identity.roster import Roster  # noqa: E402

S = 1_000_000
failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"[{'PASS' if condition else 'FAIL'}] {name}")
    if not condition:
        failures.append(f"{name}: {detail}")


# A plausible measured ROC: stricter thresholds buy lower false match rates.
ROC = [
    RocPoint(0.40, 1e-2),
    RocPoint(0.55, 1e-3),
    RocPoint(0.70, 1e-4),
    RocPoint(0.82, 1e-5),
    RocPoint(0.90, 1e-6),
]


def policy(target=1e-4, margin=0.05):
    return ThresholdPolicy(ROC, target_false_match_rate=target, min_margin=margin)


def fresh_roster():
    r = Roster(presence_ttl_s=12 * 3600)
    r.badge_in("EMP-4471", "zone-assembly-4", "operator", 0)
    r.badge_in("EMP-5522", "zone-assembly-4", "operator", 0)
    r.badge_in("EMP-9012", "zone-robot-cell-3", "maintenance_engineer", 0)
    r.badge_in("EMP-3001", "zone-office", "admin", 0)
    return r


# ----------------------------------------------------------------------
# Threshold arithmetic
# ----------------------------------------------------------------------

pol = policy()

t_small = pol.threshold_for(4)
t_large = pol.threshold_for(50)
check(
    "threshold tightens as the candidate set grows",
    t_small is not None and t_large is not None and t_large > t_small,
    f"4 candidates -> {t_small}, 50 -> {t_large}",
)

check(
    "threshold is derived from target/N, not fixed",
    # 4 candidates need per-comparison FMR <= 2.5e-5, met by the 1e-5 point
    # at 0.82. 50 need <= 2e-6, so only the 1e-6 point at 0.90 will do.
    t_small == 0.82 and t_large == 0.90,
    f"got {t_small}, {t_large}",
)

check(
    "a candidate set beyond the model's reach returns no threshold",
    pol.threshold_for(200) is None,
    "200 candidates need per-comparison FMR <= 5e-7 and the strongest measured "
    "point is 1e-6; the honest answer is 'cannot identify', not a guess",
)

check(
    "the ceiling on candidate count is computable",
    pol.max_candidates() == 100,
    f"got {pol.max_candidates()}; this is the number that decides whether "
    f"whole-site matching is viable, and it must be known at commissioning",
)


# ----------------------------------------------------------------------
# Tier 1: the expected case
# ----------------------------------------------------------------------

roster = fresh_roster()
matcher = StubMatcher({"probe-A": {"EMP-4471": 0.91, "EMP-5522": 0.20}})
r = IdentityResolver(roster, matcher, policy()).resolve(
    "trk-A", "zone-assembly-4", "probe-A", 10 * S
)
check("tier 1 binds against the zone roster", r.tier == "zone" and r.identity == "EMP-4471", r.reason)
check("tier 1 carries the enrolled role", r.role == "operator", str(r.role))
check("tier 1 confidence is the match score", r.confidence == 0.91, str(r.confidence))


# ----------------------------------------------------------------------
# Tier 2: identified, but in the wrong place
# ----------------------------------------------------------------------

roster = fresh_roster()
# EMP-9012 badged into the robot cell, and is now seen at the assembly station.
matcher = StubMatcher({
    "probe-B": {"EMP-4471": 0.15, "EMP-5522": 0.12, "EMP-9012": 0.93, "EMP-3001": 0.10}
})
resolver = IdentityResolver(roster, matcher, policy())
r = resolver.resolve("trk-B", "zone-assembly-4", "probe-B", 20 * S)
check("tier 2 falls through to the site roster", r.tier == "site", r.reason)
check("tier 2 still identifies the person", r.identity == "EMP-9012", str(r.identity))

events = resolver.events_for(r, 20 * S)
obs = [e["observation"] for e in events]
check(
    "tier 2 emits both identification and the security observation",
    obs == ["person_identified", "presence_unbadged"],
    str(obs),
)


# ----------------------------------------------------------------------
# Tier 3: nobody we know
# ----------------------------------------------------------------------

roster = fresh_roster()
matcher = StubMatcher({"probe-X": {}}, default=0.11)
resolver = IdentityResolver(roster, matcher, policy())
r = resolver.resolve("trk-X", "zone-assembly-4", "probe-X", 30 * S)
check("no confident match leaves the track unbound", not r.bound and r.tier == "none", r.reason)

events = resolver.events_for(r, 30 * S)
check(
    "unaccounted presence is emitted, and with confidence 1.0",
    len(events) == 1
    and events[0]["observation"] == "identity_unverified"
    and events[0]["confidence"] == 1.0,
    "the comparison definitely ran and definitely cleared nothing; what is "
    "uncertain is who they are, which is the finding itself",
)


# ----------------------------------------------------------------------
# Candidates with no enrolment
# ----------------------------------------------------------------------
# The stub scores every identity it is asked about. The real ArcFace matcher
# cannot: it has nothing to compare against for someone who has badged in but
# never been enrolled, which on any real site is most of the roster on the
# first day. A matcher that returns a partial result -- or no result at all --
# must produce an unverified track, not an exception.


class SparseMatcher:
    """Scores only identities that have enrolled templates."""

    licence = "n/a (test stub)"

    def __init__(self, enrolled: dict) -> None:
        self.enrolled = enrolled

    def compare(self, probe, identities):
        return {i: self.enrolled[i] for i in identities if i in self.enrolled}


roster = fresh_roster()
# Nobody in zone-assembly-4 is enrolled; EMP-9012 elsewhere on site is.
resolver = IdentityResolver(roster, SparseMatcher({"EMP-9012": 0.95}), policy())
r = resolver.resolve("trk-U", "zone-assembly-4", "probe-U", 50 * S)
check(
    "an unenrolled zone roster falls through instead of raising",
    r.tier == "site" and r.identity == "EMP-9012",
    r.reason,
)

roster = fresh_roster()
resolver = IdentityResolver(roster, SparseMatcher({}), policy())
r = resolver.resolve("trk-V", "zone-assembly-4", "probe-V", 51 * S)
check(
    "a roster with nobody enrolled at all reports unverified",
    not r.bound and r.tier == "none",
    "no scores to rank is the same outcome as no score clearing the bar; "
    "the resolver used to raise IndexError here",
)


# ----------------------------------------------------------------------
# The margin rule
# ----------------------------------------------------------------------

roster = fresh_roster()
# Two candidates essentially tied. Picking the higher one attaches a name to
# noise, and that name may later appear on a violation.
matcher = StubMatcher({"probe-T": {"EMP-4471": 0.87, "EMP-5522": 0.855}})
r = IdentityResolver(roster, matcher, policy(margin=0.05)).resolve(
    "trk-T", "zone-assembly-4", "probe-T", 40 * S
)
check("a tie between candidates refuses to bind", not r.bound, r.reason)

matcher = StubMatcher({"probe-T2": {"EMP-4471": 0.91, "EMP-5522": 0.30}})
r = IdentityResolver(roster, matcher, policy(margin=0.05)).resolve(
    "trk-T2", "zone-assembly-4", "probe-T2", 40 * S
)
check("a clear winner does bind", r.bound and r.identity == "EMP-4471", r.reason)


# ----------------------------------------------------------------------
# Roster behaviour
# ----------------------------------------------------------------------

r2 = Roster(presence_ttl_s=8 * 3600)
r2.badge_in("EMP-1", "zone-a", "operator", 0)
r2.badge_in("EMP-1", "zone-b", "operator", 60 * S)
check(
    "badging into a new zone moves the person rather than duplicating them",
    len(r2.in_zone("zone-a")) == 0 and len(r2.in_zone("zone-b")) == 1,
    "otherwise every zone's candidate set grows until matching stops working",
)

r2.badge_in("EMP-2", "zone-b", "operator", 0)
r2.touch("EMP-1", 8 * 3600 * S)  # EMP-1 was seen an hour ago; EMP-2 was not
dropped = r2.expire(9 * 3600 * S)
check("stale presence expires without exit readers", dropped == ["EMP-2"], str(dropped))
check("recently seen presence survives expiry", r2.get("EMP-1") is not None)

r3 = Roster(presence_ttl_s=100)
r3.badge_in("EMP-9", "zone-a", "operator", 0)
r3.touch("EMP-9", 90 * S)
check(
    "a camera sighting refreshes presence",
    r3.expire(150 * S) == [],
    "without this, someone working a long shift expires off the roster and "
    "then reads as an unaccounted presence",
)


# ----------------------------------------------------------------------
# Binding is per track, and once
# ----------------------------------------------------------------------

roster = fresh_roster()
matcher = StubMatcher({"probe-A": {"EMP-4471": 0.91, "EMP-5522": 0.20}})
resolver = IdentityResolver(roster, matcher, policy())
first = resolver.resolve("trk-C", "zone-assembly-4", "probe-A", 10 * S)
second = resolver.resolve("trk-C", "zone-assembly-4", "probe-nonsense", 11 * S)
check("a bound track is not re-matched", second is first, "matching runs once per track")

resolver.forget("trk-C")
third = resolver.resolve("trk-C", "zone-assembly-4", "probe-A", 12 * S)
check("forgetting a track allows a reissued id to re-match", third is not first)


# ----------------------------------------------------------------------
# Badge events drive the roster
# ----------------------------------------------------------------------

r4 = Roster()
granted = {
    "observation": "credential_presented",
    "timestamp_us": 5 * S,
    "value": "granted",
    "zone_id": "zone-assembly-4",
    "subject": {"class": "human", "identity": "EMP-7", "role": "operator"},
}
IdentityResolver.badge_event_to_roster(granted, r4)
check("a granted badge-in adds presence", len(r4.in_zone("zone-assembly-4")) == 1)

denied = dict(granted, value="denied", subject={"identity": "EMP-8", "role": "operator"})
IdentityResolver.badge_event_to_roster(denied, r4)
check(
    "a denied attempt puts nobody in the room",
    r4.get("EMP-8") is None,
    "a refused credential is a security event, not an occupancy fact",
)


# ----------------------------------------------------------------------
# Emitted events must satisfy the contract
# ----------------------------------------------------------------------

schema = json.loads((ROOT / "schema" / "event.schema.json").read_text())
allowed = set(schema["properties"]["observation"]["enum"])
required = set(schema["required"])

roster = fresh_roster()
resolver = IdentityResolver(
    roster, StubMatcher({"p": {"EMP-4471": 0.95, "EMP-5522": 0.1}}), policy()
)
all_events = []
all_events += resolver.events_for(resolver.resolve("t1", "zone-assembly-4", "p", S), S)
all_events += resolver.events_for(
    resolver.resolve("t2", "zone-assembly-4", "unknown-probe", S), S
)
# Live, identification usually lands before zone entry has debounced, so the
# track is in no zone. That path must not emit zone_id: null.
all_events += resolver.events_for(resolver.resolve("t3", None, "p", S), S)
all_events += resolver.events_for(resolver.resolve("t4", None, "unknown-probe", S), S)

check(
    "every emitted observation is in the closed enum",
    all(e["observation"] in allowed for e in all_events),
    str([e["observation"] for e in all_events]),
)
check(
    "every emitted event carries the required fields",
    all(required.issubset(e.keys()) for e in all_events),
    str(sorted(required)),
)
check(
    "confidences are in range",
    all(0.0 <= e["confidence"] <= 1.0 for e in all_events),
)
check(
    "no field is emitted as null (the schema types them as strings)",
    all(v is not None for e in all_events for v in e.values()),
    str([sorted(k for k, v in e.items() if v is None) for e in all_events]),
)

# ----------------------------------------------------------------------
# The enrolment key is a person ID, not a name
# ----------------------------------------------------------------------
# Names are not unique, names change, and the ID is already what the badge
# reader emits -- so roster, door and workflow speak one language with no
# translation step between them. A translation step is exactly where the
# wrong person gets named.

check(
    "a person ID is accepted as an enrolment key",
    is_person_id("EMP-4471") and is_person_id("CONTR-100234"),
    "EMP-4471 is the badge number; it is what every other layer already uses",
)

check(
    "a bare name is refused as an enrolment key",
    not is_person_id("Rajesh") and not is_person_id("Rajesh Kumar"),
    "two people can share a name, and then they share a face template",
)

check(
    "an ID with a name stuck on the end is refused",
    not is_person_id("EMP-4471 Rajesh"),
    "the pattern is anchored; an unanchored one would let this through and "
    "the key would drift back to being free text",
)

check(
    "the pattern is configurable, not hardcoded",
    is_person_id("b12345", r"[a-z][0-9]{5}")
    and not is_person_id("EMP-4471", r"[a-z][0-9]{5}"),
    "site ID schemes differ; what must not differ is that the key is an ID",
)


# ----------------------------------------------------------------------
# display_name travels beside the key, never as it
# ----------------------------------------------------------------------

EMB = np.zeros(512, dtype=np.float32)
EMB[0] = 1.0


def template(identity: str, **kw) -> EnrolledTemplate:
    return EnrolledTemplate(
        identity=identity,
        embedding=EMB.copy(),
        model_sha256="sha-test",
        alignment="5point",
        **kw,
    )


named = template("EMP-4471", display_name="Rajesh Kumar")
anonymous = template("EMP-5522")

round_tripped = EnrolledTemplate.from_json("EMP-4471", named.to_json())
check(
    "display_name round-trips through to_json/from_json",
    round_tripped.display_name == "Rajesh Kumar",
    str(round_tripped.display_name),
)

check(
    "an unsupplied display_name is absent from the JSON, not empty in it",
    "display_name" not in anonymous.to_json(),
    "absence should be visible in the file rather than encoded as an empty "
    "string, so reading templates.json shows who has a name recorded",
)

check(
    "a template with no display_name loads with an empty one",
    EnrolledTemplate.from_json("EMP-5522", anonymous.to_json()).display_name == "",
    "every template written before display_name existed takes this path",
)

check(
    "the key is the ID, and from_json does not take it from display_name",
    round_tripped.identity == "EMP-4471",
    str(round_tripped.identity),
)


# ----------------------------------------------------------------------
# Matching keys on identity, and display_name does not touch it
# ----------------------------------------------------------------------


def matching_fields(t: EnrolledTemplate) -> dict:
    """Everything a comparison actually reads."""
    return {k: v for k, v in t.to_json().items() if k != "display_name"}


check(
    "display_name changes nothing a comparison reads",
    matching_fields(named) == matching_fields(template("EMP-4471")),
    "embedding, model_sha256 and alignment are what a match is computed "
    "from; a name must not be able to move a score",
)

store = {"EMP-4471": [named], "EMP-5522": [anonymous]}
check(
    "the store is keyed on the ID even when a name is present",
    sorted(store) == ["EMP-4471", "EMP-5522"] and "Rajesh Kumar" not in store,
    str(sorted(store)),
)

roster = fresh_roster()
matcher = StubMatcher({"probe-N": {"EMP-4471": 0.91, "EMP-5522": 0.20}})
resolver = IdentityResolver(roster, matcher, policy())
r = resolver.resolve("trk-N", "zone-assembly-4", "probe-N", 60 * S)
check(
    "resolution binds the ID, and the emitted event carries the ID",
    r.identity == "EMP-4471"
    and all(
        e["subject"]["identity"] == "EMP-4471"
        for e in resolver.events_for(r, 60 * S)
    ),
    "the event store holds no personal names on its face, which is most of "
    "why the key is an ID",
)

check(
    "display_name_for shows the name when there is one",
    display_name_for(store, "EMP-4471") == "Rajesh Kumar",
    "this is the only thing a screen should print; events use the ID",
)
check(
    "display_name_for falls back to the ID when there is not",
    display_name_for(store, "EMP-5522") == "EMP-5522"
    and display_name_for(store, "EMP-0000") == "EMP-0000",
    "an unnamed enrolment still has to label its own box",
)


# ----------------------------------------------------------------------
# A legacy key loads and warns; it is neither rewritten nor dropped
# ----------------------------------------------------------------------
# The shipped templates.json was enrolled under "user". Rewriting it would
# mean inventing a badge number, and dropping it would silently stop
# identifying a real person. So: warn, and keep working.

with tempfile.TemporaryDirectory() as tmp:
    legacy_path = Path(tmp) / "templates.json"
    legacy_path.write_text(
        json.dumps(
            {
                "format_version": 3,
                "templates": [
                    {
                        "identity": "user",
                        "samples": [template("user").to_json()],
                    },
                    {"identity": "EMP-4471", "samples": [named.to_json()]},
                ],
            }
        )
    )

    warnings: list[str] = []
    loaded = load_templates(str(legacy_path), warn=warnings.append)

    check(
        "a legacy non-conforming key still loads",
        sorted(loaded) == ["EMP-4471", "user"] and len(loaded["user"]) == 1,
        str(sorted(loaded)),
    )
    check(
        "loading a legacy key warns, naming the key and how to re-enrol",
        len(warnings) == 1
        and "'user'" in warnings[0]
        and "enroll.py" in warnings[0],
        str(warnings),
    )
    check(
        "a conforming key alongside it warns about nothing",
        all("EMP-4471" not in w for w in warnings),
        str(warnings),
    )

    # The default sink is stderr, so an operator sees it without anyone
    # having to pass a callback, and it stays out of the event stream.
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        load_templates(str(legacy_path))
    check(
        "the default warning goes to stderr, where stdout is the event stream",
        "user" in err.getvalue(),
        repr(err.getvalue()),
    )

    # Round-trip through save/load: the key stays the ID, the name rides
    # along beside it.
    out_path = Path(tmp) / "saved.json"
    save_templates(str(out_path), {"EMP-4471": [named], "EMP-5522": [anonymous]})
    reloaded = load_templates(str(out_path), warn=warnings.append)
    check(
        "display_name survives a save/load round-trip, keyed on the ID",
        display_name_for(reloaded, "EMP-4471") == "Rajesh Kumar"
        and reloaded["EMP-5522"][0].display_name == ""
        and sorted(reloaded) == ["EMP-4471", "EMP-5522"],
        str({k: [s.display_name for s in v] for k, v in reloaded.items()}),
    )

print()
if failures:
    print(f"{len(failures)} identity test(s) failed")
    for f in failures:
        print(f"  !! {f}")
    raise SystemExit(1)
print("all identity tests passed")