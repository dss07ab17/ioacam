#!/usr/bin/env python3
"""Tests for joining per-window action scores into intervals. No model.

    python3 perception/test_intervals.py

Every case is built from synthetic `ActionScore`s on a fixed one-second
cadence, because what is under test is the join rule, not the recogniser.
Feeding real inference in here would make the suite slow, non-deterministic,
and unable to construct the one case that matters most: a clean run of a
label with a single abstention dropped into the middle of it.

The cases are the ways this join goes wrong in the field:

  * flicker splitting one action into two intervals with a gap between them,
    which reads as the person stopping and starting again,
  * an interval whose end is the moment the change was CONFIRMED rather than
    the last window that actually showed the old label, which overstates
    every duration by however long the confirmation takes,
  * a track that walks out of frame leaving its interval open forever,
  * one twitchy window emitted as an interval in its own right,
  * two people's windows leaking into one another's intervals,
  * an interval attributed to the wrong person, or to nobody, because the
    identity was read when it opened rather than when it closed -- a track is
    routinely unbound for its first window and binds partway through,
  * a wall clock read when the interval was WRITTEN rather than derived from
    when it ran, which dates every span to a few seconds after the fact.
"""

from __future__ import annotations

import io
import json
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "actions"))
sys.path.insert(0, str(HERE / "tools"))
sys.path.insert(0, str(ROOT))

from action_intervals import (  # noqa: E402
    ActionInterval,
    IntervalBuilder,
    build_action_interval_event,
)
from emit import EventEmitter, wall_time_at  # noqa: E402
from validate_events import validate  # noqa: E402

from base import ActionScore  # noqa: E402

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"[{'PASS' if condition else 'FAIL'}] {name}")
    if not condition:
        failures.append(f"{name}: {detail}")


# ----------------------------------------------------------------------
# Synthetic windows
# ----------------------------------------------------------------------
#
# One window per second, 1.5s long, overlapping the way real tubes do: window
# n covers [n, n + 1.5). Written as a helper rather than as literals so that a
# case reads as the sequence of labels it is testing.

WINDOW_S = 1.5
CADENCE_S = 1.0
US = 1_000_000


def window(n: int, label, track_id: str = "trk-0001", confidence: float = 0.8):
    """Window `n` of the cadence. `label=None` means the policy abstained."""
    return ActionScore(
        track_id=track_id,
        label=label,
        confidence=confidence,
        abstained=label is None,
        reason="" if label else "synthetic abstention",
        window_start_us=int(n * CADENCE_S * US),
        window_end_us=int((n * CADENCE_S + WINDOW_S) * US),
    )


def feed(builder: IntervalBuilder, labels, track_id: str = "trk-0001",
         start: int = 0, confidences=None) -> list[ActionInterval]:
    """Push a sequence of labels through, returning what closed."""
    closed: list[ActionInterval] = []
    for i, label in enumerate(labels):
        conf = 0.8 if confidences is None else confidences[i]
        closed.extend(
            builder.update(window(start + i, label, track_id, conf))
        )
    return closed


def end_of(n: int) -> int:
    return int((n * CADENCE_S + WINDOW_S) * US)


def start_of(n: int) -> int:
    return int(n * CADENCE_S * US)


# ----------------------------------------------------------------------
# Three identical labels are one interval
# ----------------------------------------------------------------------

b = IntervalBuilder()
closed = feed(b, ["reading", "reading", "reading"])
check(
    "a run of one label closes nothing while it continues",
    closed == [],
    f"got {closed}",
)
check("it is held open on the track", b.open_label("trk-0001") == "reading")

final = b.close_all()
check("closing the builder emits the one interval", len(final) == 1, str(final))
if final:
    iv = final[0]
    check("it merged all three windows", iv.windows == 3, str(iv.windows))
    check("it carries the label", iv.label == "reading", iv.label)
    check(
        "it starts at the leading edge of the first window",
        iv.start_us == start_of(0),
        f"{iv.start_us} != {start_of(0)}",
    )
    check(
        "it ends at the trailing edge of the last",
        iv.end_us == end_of(2),
        f"{iv.end_us} != {end_of(2)}",
    )
    check(
        "duration spans the windows, not the cadence ticks",
        abs(iv.duration_s - 3.5) < 1e-6,
        f"{iv.duration_s}",
    )

# Confidence is the mean over the windows the interval covered, so one weak
# window in a strong run shows up rather than being masked by the first
# window's score or overwritten by the last one's.
b = IntervalBuilder()
feed(b, ["reading"] * 3, confidences=[0.9, 0.6, 0.9])
mean = b.close_all()[0].confidence
check(
    "confidence is the mean of the windows it covered",
    abs(mean - 0.8) < 1e-6,
    f"{mean}",
)


# ----------------------------------------------------------------------
# One abstention in the middle must not split it
# ----------------------------------------------------------------------
#
# This is the case the whole delayed close exists for. The recogniser
# abstaining for one window mid-page-turn is not the person stopping reading.

b = IntervalBuilder()
closed = feed(b, ["reading", "reading", None, "reading", "reading"])
check(
    "a single abstention closes nothing",
    closed == [],
    f"one flickering window must not split an action: got {closed}",
)

final = b.close_all()
check("the run is still one interval", len(final) == 1, str(final))
if final:
    iv = final[0]
    check(
        "it spans straight through the abstention",
        (iv.start_us, iv.end_us) == (start_of(0), end_of(4)),
        f"{iv.start_us}..{iv.end_us}",
    )
    check(
        "the abstained window is not counted as evidence for the label",
        iv.windows == 4,
        f"{iv.windows}: the interval covers 5 windows but only 4 voted for it",
    )

# The same holds for a different label appearing once, which is the other
# half of flicker: one window of 'writing' inside a run of 'reading'.
b = IntervalBuilder()
closed = feed(b, ["reading", "reading", "writing", "reading", "reading"])
check(
    "a single off-label window does not split it either",
    closed == [] and len(b.close_all()) == 1,
    "close_after=2 means one window is never enough to close",
)

# Abstentions count towards a close, but two of them in a row still cannot
# leave an 'unknown' interval behind: only the reading interval closes.
b = IntervalBuilder()
closed = feed(b, ["reading", "reading", None, None, None])
check(
    "two consecutive abstentions do close the interval",
    len(closed) == 1 and closed[0].label == "reading",
    f"got {closed}",
)
check(
    "abstentions never open an interval of their own",
    b.open_label("trk-0001") is None and b.close_all() == [],
    "an interval of 'unknown' is not something anyone can act on",
)


# ----------------------------------------------------------------------
# A genuine change closes the first and opens the second
# ----------------------------------------------------------------------

b = IntervalBuilder()
closed = feed(b, ["reading", "reading", "reading", "writing", "writing"])
check(
    "the change closes exactly one interval",
    len(closed) == 1,
    f"got {closed}",
)
if closed:
    check("the closed one is the old label", closed[0].label == "reading")
check(
    "the new label is left open, not emitted yet",
    b.open_label("trk-0001") == "writing",
    "the second interval is still running; emitting it now would fix an end "
    "time the recogniser has not yet observed",
)

second = b.close_all()
check("the second interval closes in its turn", len(second) == 1, str(second))
if second:
    check(
        "it starts at the first window that carried the new label",
        second[0].start_us == start_of(3),
        f"{second[0].start_us} != {start_of(3)}: the interval must not start "
        "at the window that CONFIRMED the change",
    )


# ----------------------------------------------------------------------
# End time is the last window with the old label
# ----------------------------------------------------------------------
#
# Windows 0-2 are reading, 3-4 are writing. The close is confirmed at window
# 4, and the naive implementation ends the reading interval there -- crediting
# reading with two windows of time the recogniser said was writing.

b = IntervalBuilder()
closed = feed(b, ["reading", "reading", "reading", "writing", "writing"])
check(
    "the interval ends at the last window that carried its label",
    bool(closed) and closed[0].end_us == end_of(2),
    f"{closed[0].end_us if closed else None} != {end_of(2)}",
)
check(
    "not at the window that confirmed the change",
    bool(closed) and closed[0].end_us != end_of(4),
    "ending on the confirmation would inflate every interval by close_after "
    "windows",
)

# Same rule with an abstention as the confirming evidence: reading ends at
# window 1, not at either of the abstentions that proved it had stopped.
b = IntervalBuilder()
closed = feed(b, ["reading", "reading", None, None])
check(
    "an abstention-confirmed close ends on the last real window too",
    bool(closed) and closed[0].end_us == end_of(1),
    f"{closed[0].end_us if closed else None} != {end_of(1)}",
)

# close_after is configurable, and raising it delays the close without moving
# the end time.
b = IntervalBuilder(close_after=3)
closed = feed(b, ["reading", "reading", "writing", "writing"])
check(
    "close_after=3 holds the interval open through two disagreements",
    closed == [],
    f"got {closed}",
)
closed = feed(b, ["writing"], start=4)
check(
    "and closes on the third",
    len(closed) == 1 and closed[0].end_us == end_of(1),
    f"got {closed}",
)


# ----------------------------------------------------------------------
# A track going silent closes its interval
# ----------------------------------------------------------------------
#
# Nothing else ever would: a person who leaves frame stops producing windows
# entirely, so there is no later label to disagree with the open one.

b = IntervalBuilder(track_timeout_s=3.0)
feed(b, ["reading", "reading", "reading"])
last_window_end = end_of(2)

check(
    "a track still within the timeout stays open",
    b.expire(last_window_end + int(2.0 * US)) == [],
    "closing at 2s with a 3s timeout would end intervals during a normal "
    "gap in detections",
)

closed = b.expire(last_window_end + int(3.5 * US))
check(
    "a track silent past the timeout is closed",
    len(closed) == 1 and closed[0].label == "reading",
    f"got {closed}",
)
check(
    "it ends at its last window, not at the moment it was noticed missing",
    bool(closed) and closed[0].end_us == last_window_end,
    f"{closed[0].end_us if closed else None} != {last_window_end}",
)
check(
    "the track is forgotten, so the timeout does not fire twice",
    b.expire(last_window_end + int(9.0 * US)) == [],
)


# ----------------------------------------------------------------------
# Too short to mean anything
# ----------------------------------------------------------------------
#
# One window at the wrong label is the classifier twitching, not an action.
# min_duration_s has to be read against the tube length: with 1.5s windows a
# single window is already a 1.5s interval, so a site that wants single
# windows gone sets the floor above the window, which is what this checks.

b = IntervalBuilder(min_duration_s=2.0)
closed = feed(b, ["reading", "writing", "writing", "writing"])
check(
    "a one-window interval below the floor is dropped, not emitted",
    closed == [],
    f"got {closed}: a 1.5s interval must not survive a 2.0s floor",
)
check(
    "dropping it does not stop the next one opening",
    b.open_label("trk-0001") == "writing",
    "the short interval is discarded; the run that displaced it is not",
)
check(
    "and a long enough interval still comes through",
    len(b.close_all()) == 1,
)

# The floor is on the interval, not on the window count: three windows that
# happen to span less than the floor are dropped just the same.
b = IntervalBuilder(min_duration_s=10.0)
feed(b, ["reading"] * 3)
check(
    "a multi-window interval under the floor is dropped too",
    b.close_all() == [],
    "min_duration_s is a duration test, not a window-count test",
)


# ----------------------------------------------------------------------
# Two tracks are kept completely separate
# ----------------------------------------------------------------------
#
# Interleaved, because that is how they arrive: one cadence tick produces one
# window per visible track, and a builder keyed on anything but track_id
# would let one person's label close the other's interval.

b = IntervalBuilder()
closed: list[ActionInterval] = []
for i, (a_label, c_label) in enumerate(
    [("reading", "walking"), ("reading", "walking"), ("reading", "walking"),
     ("reading", "sitting"), ("reading", "sitting")]
):
    closed.extend(b.update(window(i, a_label, "trk-0001")))
    closed.extend(b.update(window(i, c_label, "trk-0002")))

check(
    "one track changing does not close the other's interval",
    len(closed) == 1 and closed[0].track_id == "trk-0002",
    f"got {closed}",
)
check(
    "the first track is untouched and still reading",
    b.open_label("trk-0001") == "reading",
)
check(
    "the second has opened its new interval",
    b.open_label("trk-0002") == "sitting",
)

remaining = b.close_all()
check("both tracks close at the end", len(remaining) == 2, str(remaining))
check(
    "each interval stays on its own track",
    {iv.track_id for iv in remaining} == {"trk-0001", "trk-0002"},
    str([iv.track_id for iv in remaining]),
)
if len(remaining) == 2:
    by_track = {iv.track_id: iv for iv in remaining}
    check(
        "the first track's interval covers all five of its windows",
        by_track["trk-0001"].windows == 5,
        str(by_track["trk-0001"].windows),
    )
    check(
        "the second track's only covers the two after its change",
        by_track["trk-0002"].windows == 2,
        str(by_track["trk-0002"].windows),
    )

# A timeout on one track leaves the other alone, which is the same separation
# from the other direction: two people rarely leave frame together.
b = IntervalBuilder(track_timeout_s=3.0)
feed(b, ["reading"] * 3, track_id="trk-0001", start=0)
feed(b, ["walking"] * 3, track_id="trk-0002", start=5)
closed = b.expire(end_of(7) + int(0.5 * US))
check(
    "only the track that went silent is expired",
    len(closed) == 1 and closed[0].track_id == "trk-0001",
    f"got {[iv.track_id for iv in closed]}",
)
check("the track still producing windows stays open",
      b.open_label("trk-0002") == "walking")


# ----------------------------------------------------------------------
# The event the engine sees
# ----------------------------------------------------------------------

SCHEMA = json.loads((ROOT / "schema" / "event.schema.json").read_text())

check(
    "action_interval is in the schema's observation enum",
    "action_interval" in SCHEMA["properties"]["observation"]["enum"],
    "the engine rejects any observation outside the closed list",
)

emitter = EventEmitter(sensor_id="cam-01", stream=io.StringIO())
interval = ActionInterval(
    track_id="trk-0001",
    label="reading",
    start_us=start_of(0),
    end_us=end_of(3),
    confidence=0.78,
    windows=4,
)
event = build_action_interval_event(emitter, interval, timestamp_us=end_of(5))

errors = validate(event, SCHEMA, None)
check("the interval event validates against the schema", not errors, str(errors))
check("value is the label", event["value"] == "reading", str(event.get("value")))
check("it carries the track", event["track_id"] == "trk-0001")
check("it carries the mean confidence", event["confidence"] == 0.78)
check(
    "it carries the span",
    (event["start_us"], event["end_us"]) == (interval.start_us, interval.end_us),
    json.dumps({k: event.get(k) for k in ("start_us", "end_us")}),
)
check(
    "timestamp_us is emission time, at or after the span it describes",
    event["timestamp_us"] >= event["end_us"],
    "the stream is ordered by emission; an event stamped with its own start "
    "would arrive out of order behind the windows that produced it",
)

# The per-window events are untouched by any of this. The interval is an
# additional view, and a deployment reading action_recognised must see exactly
# what it saw before.
from actions_stage import build_action_recognised_event  # noqa: E402

window_event = build_action_recognised_event(
    emitter, window(0, "reading")
)
check(
    "per-window action_recognised events still emit unchanged",
    window_event["observation"] == "action_recognised"
    and window_event["value"] == "reading"
    and "start_us" not in window_event,
    json.dumps(window_event)[:200],
)


# ----------------------------------------------------------------------
# Who the interval is about
# ----------------------------------------------------------------------
#
# The resolver is a plain callable, which is the point: this module must stay
# testable, and runnable, with no identity layer anywhere near it. A stub of
# two lines is the whole dependency.

bindings = {"trk-0001": "EMP-4471"}


def resolver(track_id: str) -> dict:
    """Mirror of IdentityStage.subject_for, over a dict we can mutate.

    An unbound track gets the anonymous subject with NO identity key, which is
    what the real one does and what the assertions below are about.
    """
    employee = bindings.get(track_id)
    if employee is None:
        return {"class": "human"}
    return {"class": "human", "identity": employee}


# No resolver: byte-for-byte what this builder produced before identity.
b = IntervalBuilder()
feed(b, ["reading"] * 3)
plain = b.close_all()[0]
check(
    "with no resolver the interval carries no subject of its own",
    plain.subject is None,
    f"got {plain.subject!r}",
)
plain_event = build_action_interval_event(
    EventEmitter(sensor_id="cam-01", stream=io.StringIO()),
    plain,
    timestamp_us=end_of(5),
)
check(
    "and its event is about an anonymous human, exactly as before",
    plain_event["subject"] == {"class": "human"},
    str(plain_event.get("subject")),
)

# A resolver, and a track it knows.
b = IntervalBuilder(subject_for=resolver)
feed(b, ["reading"] * 3)
named = b.close_all()[0]
check(
    "a bound track's interval carries the identity",
    named.subject == {"class": "human", "identity": "EMP-4471"},
    f"got {named.subject!r}",
)
check(
    "the track_id survives beside it rather than being replaced",
    named.track_id == "trk-0001",
    "the track id is what the camera tracked; the identity is an inference "
    "about who that was, and an attribution has to stay questionable",
)

emit_stream = EventEmitter(sensor_id="cam-01", stream=io.StringIO())
named_event = build_action_interval_event(emit_stream, named, timestamp_us=end_of(5))
check(
    "the event carries subject.identity",
    named_event["subject"].get("identity") == "EMP-4471",
    str(named_event.get("subject")),
)
check(
    "and still validates against the published schema",
    not validate(named_event, SCHEMA, None),
    str(validate(named_event, SCHEMA, None)),
)

# A track the resolver does not know. The assertion is on the KEY, not on its
# value: identity=None or identity="" would satisfy a falsiness check and
# would still be a claim the engine's wrong_role rules would read as an answer.
b = IntervalBuilder(subject_for=resolver)
feed(b, ["reading"] * 3, track_id="trk-0404")
stranger = b.close_all()[0]
check(
    "an unbound track's interval has no identity KEY at all",
    "identity" not in stranger.subject,
    f"got {stranger.subject!r}",
)
stranger_event = build_action_interval_event(
    EventEmitter(sensor_id="cam-01", stream=io.StringIO()),
    stranger,
    timestamp_us=end_of(5),
)
check(
    "nor does its event, which is not the same as carrying an empty one",
    "identity" not in stranger_event["subject"]
    and stranger_event["subject"] == {"class": "human"},
    f"got {stranger_event['subject']!r}",
)
check(
    "the unidentified interval still carries its track_id",
    stranger_event.get("track_id") == "trk-0404",
    str(stranger_event.get("track_id")),
)

# The case the close-time rule exists for: the face is not matchable until the
# person turns towards the camera, so the track is unbound when the interval
# opens and bound by the time it closes. The identity known at close is the
# answer for the whole span.
bindings.pop("trk-0002", None)
b = IntervalBuilder(subject_for=resolver)
feed(b, ["reading", "reading"], track_id="trk-0002")
check(
    "the track really is unbound while the interval is open",
    resolver("trk-0002") == {"class": "human"},
    str(resolver("trk-0002")),
)
bindings["trk-0002"] = "EMP-5522"          # the cascade binds, mid-interval
feed(b, ["reading"], track_id="trk-0002", start=2)
midway = b.close_all()[0]
check(
    "an interval that binds midway carries the identity it had at close",
    midway.subject.get("identity") == "EMP-5522",
    f"got {midway.subject!r}: resolving at open would attribute a named "
    "person's whole span to nobody",
)
check(
    "and it covers the windows from before the binding too",
    midway.windows == 3 and midway.start_us == start_of(0),
    f"{midway.windows} window(s) from {midway.start_us}",
)

# Expiry closes an interval as well, and it must resolve on the same terms.
bindings["trk-0003"] = "EMP-9012"
b = IntervalBuilder(subject_for=resolver, track_timeout_s=3.0)
feed(b, ["writing"] * 2, track_id="trk-0003")
expired = b.expire(end_of(1) + int(3.5 * US))
check(
    "a timed-out interval is attributed too, not just a confirmed close",
    len(expired) == 1 and expired[0].subject.get("identity") == "EMP-9012",
    f"got {[iv.subject for iv in expired]}",
)


# ----------------------------------------------------------------------
# When it happened, on both clocks
# ----------------------------------------------------------------------
#
# The monotonic pair orders and subtracts; the wall-clock pair is readable.
# Neither replaces the other, and the event must carry all four.

wall_event = build_action_interval_event(
    EventEmitter(sensor_id="cam-01", stream=io.StringIO()),
    ActionInterval(
        track_id="trk-0001",
        label="reading",
        start_us=start_of(0),
        end_us=end_of(3),
        confidence=0.78,
        windows=4,
    ),
    timestamp_us=end_of(5),
)

check(
    "the interval event carries a wall clock for both ends of the span",
    "wall_time_start" in wall_event and "wall_time_end" in wall_event,
    str(sorted(wall_event)),
)
check(
    "the monotonic fields are untouched by that",
    (wall_event["start_us"], wall_event["end_us"]) == (start_of(0), end_of(3)),
    "the wall clock is additive: every duration rule in the engine reads the "
    "monotonic pair, which an NTP correction cannot reorder",
)
check(
    "and timestamp_us still means emission time",
    wall_event["timestamp_us"] == end_of(5),
    str(wall_event.get("timestamp_us")),
)

# ISO 8601, UTC, to the millisecond -- the same shape emit.wall_time_now
# writes, because a consumer parsing one must be able to parse the others.
ISO = "%Y-%m-%dT%H:%M:%S.%fZ"
parsed = {}
for key in ("wall_time", "wall_time_start", "wall_time_end"):
    try:
        parsed[key] = datetime.strptime(wall_event[key], ISO)
    except ValueError as exc:
        parsed[key] = None
        check(f"{key} is ISO-formatted", False, f"{wall_event[key]!r}: {exc}")
check(
    "all three wall-clock fields parse in the one format this layer writes",
    all(parsed.values()),
    str({k: wall_event[k] for k in parsed}),
)

if all(parsed.values()):
    span_s = (parsed["wall_time_end"] - parsed["wall_time_start"]).total_seconds()
    # Both ends convert through the same startup offset, so the only error is
    # the millisecond rounding at each end.
    check(
        "wall_time_end minus wall_time_start equals the interval duration",
        abs(span_s - 4.5) <= 0.002,
        f"{span_s}s against a {4.5}s span: if these disagree the two ends are "
        "being converted through different offsets, or one of them is being "
        "read from the clock at write time",
    )
    check(
        "...which is the duration the monotonic fields give as well",
        abs(span_s - (wall_event["end_us"] - wall_event["start_us"]) / 1e6)
        <= 0.002,
        f"{span_s}s wall against "
        f"{(wall_event['end_us'] - wall_event['start_us']) / 1e6}s monotonic",
    )
    check(
        "the span is dated before the event that reports it",
        parsed["wall_time_end"] <= parsed["wall_time"],
        "an interval is emitted only once later windows confirmed it ended, "
        "so its end cannot be after its own emission",
    )

# The conversion is a pure function of the offset, not of when it is called:
# the same monotonic reading must give the same wall time however much later
# it is converted. That is what stops a span being dated to its write time.
check(
    "converting the same instant twice gives the same wall time",
    wall_time_at(start_of(0)) == wall_event["wall_time_start"],
    f"{wall_time_at(start_of(0))} != {wall_event['wall_time_start']}",
)

# Two tracks, one of them identified, closed from the same builder: the
# subject must follow the track rather than the close order.
bindings["trk-0001"] = "EMP-4471"
b = IntervalBuilder(subject_for=resolver)
feed(b, ["reading"] * 3, track_id="trk-0001")
feed(b, ["reading"] * 3, track_id="trk-0909")
both = {iv.track_id: iv for iv in b.close_all()}
check(
    "each track's interval gets its own subject, not the other's",
    both["trk-0001"].subject.get("identity") == "EMP-4471"
    and "identity" not in both["trk-0909"].subject,
    str({k: v.subject for k, v in both.items()}),
)


print()
if failures:
    print(f"{len(failures)} interval test(s) failed")
    for f in failures:
        print(f"  !! {f}")
    raise SystemExit(1)
print("all interval tests passed")
