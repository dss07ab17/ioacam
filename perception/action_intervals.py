"""Per-window action scores -> `action_interval` events.

`ActionsStage` answers "what is this person doing right now", once per
cadence tick, for as long as the track lives. That is the right shape for a
classifier and the wrong shape for the question anyone actually asks. Nobody
wants fourteen `action_recognised` events saying `reading`; they want one line
saying trk-0001 was reading from 14:02:10 to 14:02:14.

This file does that join and nothing else. It consumes `ActionScore`s and
produces closed intervals. The per-window events keep flowing unchanged: an
interval is an additional, coarser view of the same stream, not a replacement
for it, and an engine rule that wants the raw windows still has them.

## Why closing is delayed

The classifier flickers. A person reading for four seconds produces
`reading, reading, unknown, reading` about as often as four clean `reading`s --
one window catches them mid-page-turn, the margin test fires, and the policy
abstains. Closing an interval the moment the label changed would split that
into two intervals with a gap between them, which reads as "stopped reading,
started reading again" and is simply false.

So a disagreeing window opens a pending run instead of closing anything. The
interval closes only once `close_after` consecutive windows have failed to
match it. Abstentions count towards that run -- a person who genuinely stopped
doing something usually produces abstentions rather than a crisp new label --
but an abstention can never OPEN an interval. An interval of `unknown` says
nothing anybody can act on.

When a close is confirmed, the interval ends at the last window that actually
carried its label, NOT at the window that confirmed the change. Those are
`close_after` windows apart, and using the confirmation point would stretch
every interval by a second or two of time the action was provably not
happening.

## Precision

START AND END ARE ACCURATE TO ABOUT ONE WINDOW, which is the tube length
(1.5 s by default), not to the microsecond. The bounds are the edges of the
windows that voted for the label, and the classifier cannot see inside a
window: an action that began halfway through one is recorded as beginning at
that window's start. Tubes overlap, so two consecutive intervals can overlap
each other by up to a window as well -- the second one's start can precede the
first one's end. That is the same imprecision seen from the other side, not a
bug to clamp away: clamping would invent a boundary the recogniser never
observed. Treat these as "about 14:02:10", never as a measurement,
and do not render them to a precision the number does not have. The
microseconds are there because the rest of the schema is in microseconds, not
because they mean anything.

## Who, and when

An interval carries `track_id` and, where an identity resolver is supplied, a
`subject` naming the person. Both, always: see `_finish` for why the identity
never replaces the track id, and why it is resolved when the interval CLOSES
rather than when it opened.

The event also carries a wall clock for the span (`wall_time_start` /
`wall_time_end`) beside the monotonic `start_us` / `end_us`. Additive, not a
replacement -- every duration rule in the engine keys on the monotonic pair,
and the wall clock exists so that a human reading a log can tell what time of
day 338483682608 was. Both are derived from one offset captured at process
start (`emit.wall_time_at`), never from the clock at close time, which would
date the span to the moment it was written rather than the moment it ran.

## Overlapping intervals for one identity are NOT merged here

Once identities are bound, two track ids can resolve to the same person -- a
tracker that lost somebody and reissued an id produces exactly that, and their
intervals then overlap in the log. `logs/interval-demo2.jsonl` has the case:
trk-0001 and trk-0003 each open an interval at the same microsecond,
338473276005, and trk-0003's "cross hands in front (say stop)" span then
covers one of trk-0001's intervals of the same label end to end.

This file does not merge or dedupe them, deliberately. Two reasons:

The overlap IS the evidence that the tracker dropped someone. Merging it away
here would leave a clean-looking record of one continuous person and destroy
the only in-band signal that re-identification failed -- a signal worth more
than the tidier log, because it is a measurable defect rate for the tracker,
and because the known id-swap limitation is the reason identity exists as a
correlation key at all.

And merging needs a judgement this layer is not allowed to make. Deciding that
two overlapping spans are one activity means trusting both attributions enough
to combine their evidence, which is exactly the "perception never judges" line:
the identity on an interval is an inference with a confidence, and two weak
attributions to the same ID do not become one strong fact by being adjacent.

So the builder keeps one interval per TRACK, as it always has, and reconciling
tracks that turn out to be the same person belongs to the analytics layer,
which can see the whole run at once, can weigh the match confidences, and can
present the merge as a derived view beside the raw intervals rather than
instead of them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional

from emit import EventEmitter, subject_of, wall_time_at

DEFAULT_CLOSE_AFTER = 2
DEFAULT_MIN_DURATION_S = 1.0
DEFAULT_TRACK_TIMEOUT_S = 3.0


@dataclass
class ActionInterval:
    """One track doing one thing, over a span of windows.

    `start_us` and `end_us` are window edges and carry window-level precision
    only -- see the note at the top of this file.
    """

    track_id: str
    label: str
    start_us: int
    end_us: int
    confidence: float          # mean over the windows that carried the label
    windows: int               # how many windows voted for it
    # Who the builder believed this track was AT CLOSE TIME, or None when no
    # identity resolver was supplied. Never replaces `track_id`.
    subject: Optional[dict] = None

    @property
    def duration_s(self) -> float:
        return max(0.0, (self.end_us - self.start_us) / 1_000_000)

    def describe(self) -> str:
        """Human line for a console or a log. Tenths of a second, because that
        is already finer than the measurement justifies."""
        return (
            f"{self.track_id}  {self.label}  "
            f"{self.start_us / 1_000_000:.1f}s -> {self.end_us / 1_000_000:.1f}s "
            f"(~{self.duration_s:.1f}s, {self.windows} window(s), "
            f"conf {self.confidence:.2f})"
        )


@dataclass
class _Open:
    """An interval still accumulating. Not part of the public surface."""

    track_id: str
    label: str
    start_us: int
    end_us: int
    confidence_sum: float
    windows: int

    def extend(self, result) -> None:
        # end_us only ever moves to a window that matched the label, which is
        # what makes a confirmed close land on the right edge.
        self.end_us = max(self.end_us, int(result.window_end_us))
        self.confidence_sum += float(result.confidence)
        self.windows += 1

    def finish(self) -> ActionInterval:
        return ActionInterval(
            track_id=self.track_id,
            label=self.label,
            start_us=self.start_us,
            end_us=self.end_us,
            confidence=round(self.confidence_sum / max(1, self.windows), 4),
            windows=self.windows,
        )


def label_of(result) -> Optional[str]:
    """The label a window votes with, or None for an abstention.

    `ActionScore.decided` is the authority here: a score carrying a label but
    flagged abstained is an abstention, and reading `.label` directly would
    silently resurrect the best guess the policy already rejected.
    """
    return result.label if result.decided else None


class IntervalBuilder:
    """Keeps one open interval per track and closes them on the rules above.

    Every method returns the intervals that CLOSED during it, already filtered
    by `min_duration_s`. Usually that list is empty.

    `min_duration_s` drops intervals too short to mean anything -- normally a
    single misclassified window. Note how it interacts with the tube length:
    one window of a 1.5 s tube is already a 1.5 s interval, so the 1.0 s
    default drops nothing on its own. Setting it above the tube length is what
    actually discards single-window intervals, and that is a site decision
    about how twitchy the floor's classifier turns out to be.
    """

    def __init__(
        self,
        close_after: int = DEFAULT_CLOSE_AFTER,
        min_duration_s: float = DEFAULT_MIN_DURATION_S,
        track_timeout_s: float = DEFAULT_TRACK_TIMEOUT_S,
        subject_for: Optional[Callable[[str], dict]] = None,
    ) -> None:
        self.close_after = max(1, int(close_after))
        self.min_duration_s = float(min_duration_s)
        self.track_timeout_s = float(track_timeout_s)
        # A callable, not the identity stage itself: this module has no
        # business importing the identity layer, and a callable is what makes
        # the close-time resolution testable with a two-line stub.
        self._subject_for = subject_for

        self._open: dict[str, _Open] = {}
        # Windows seen since the open interval last matched. Buffered rather
        # than counted, so that a confirmed close can replay them and start the
        # next interval at the first window that carried the new label.
        self._pending: dict[str, list] = {}
        self._last_seen_us: dict[str, int] = {}

    # ------------------------------------------------------------------

    @staticmethod
    def from_config(
        d: dict, subject_for: Optional[Callable[[str], dict]] = None
    ) -> "IntervalBuilder":
        # `subject_for` is an argument rather than a config key on purpose: it
        # is a live callable the caller already holds, and nothing about which
        # identities exist belongs in the actions block of a camera config.
        return IntervalBuilder(
            close_after=int(d.get("close_after", DEFAULT_CLOSE_AFTER)),
            min_duration_s=float(d.get("min_duration_s", DEFAULT_MIN_DURATION_S)),
            track_timeout_s=float(
                d.get("track_timeout_s", DEFAULT_TRACK_TIMEOUT_S)
            ),
            subject_for=subject_for,
        )

    @property
    def open_tracks(self) -> list[str]:
        return sorted(self._open)

    def open_label(self, track_id: str) -> Optional[str]:
        """What this track is currently believed to be doing, if anything."""
        held = self._open.get(track_id)
        return held.label if held else None

    # ------------------------------------------------------------------

    def update(self, result) -> list[ActionInterval]:
        """Feed one window's verdict for one track."""
        track_id = result.track_id
        self._last_seen_us[track_id] = max(
            self._last_seen_us.get(track_id, 0), int(result.window_end_us)
        )
        label = label_of(result)
        held = self._open.get(track_id)

        if held is None:
            if label is not None:
                self._open[track_id] = _Open(
                    track_id=track_id,
                    label=label,
                    start_us=int(result.window_start_us),
                    end_us=int(result.window_end_us),
                    confidence_sum=float(result.confidence),
                    windows=1,
                )
            # An abstention with nothing open opens nothing, by design.
            return []

        if label == held.label:
            held.extend(result)
            # The run has to be CONSECUTIVE, so one matching window clears it.
            # This is the line that stops a mid-action abstention from ever
            # accumulating towards a close.
            self._pending.pop(track_id, None)
            return []

        pending = self._pending.setdefault(track_id, [])
        pending.append(result)
        if len(pending) < self.close_after:
            return []

        # Confirmed: the disagreement held for close_after consecutive windows.
        del self._open[track_id]
        del self._pending[track_id]
        closed = self._emit(self._finish(held))

        # Replay the buffered windows into the now-empty slot. They are the
        # evidence that closed the old interval and they are also the start of
        # whatever comes next, so the new interval begins at the first of them
        # that carried a label rather than at the next cadence tick. This
        # cannot recurse more than one level deep: opening consumes at least
        # one buffered window, leaving fewer than close_after behind it.
        for buffered in pending:
            closed.extend(self.update(buffered))
        return closed

    def expire(self, now_us: int) -> list[ActionInterval]:
        """Close intervals for tracks that have gone quiet.

        A track that leaves frame stops producing windows entirely, and
        nothing else in this file would ever close its interval. Without this,
        a person who walked out at 14:02 is still reading at 18:00.
        """
        cutoff = int(now_us) - int(self.track_timeout_s * 1_000_000)
        closed: list[ActionInterval] = []
        for track_id, last_seen in sorted(self._last_seen_us.items()):
            if last_seen > cutoff:
                continue
            closed.extend(self._close_track(track_id))
            self._last_seen_us.pop(track_id, None)
        return closed

    def process(
        self, results: Iterable, now_us: Optional[int] = None
    ) -> list[ActionInterval]:
        """One cadence tick: every result, then the timeout sweep."""
        closed: list[ActionInterval] = []
        for result in results:
            closed.extend(self.update(result))
        if now_us is not None:
            closed.extend(self.expire(now_us))
        return closed

    def close_all(self) -> list[ActionInterval]:
        """Shutdown. An interval still open at exit is an interval nobody ever
        sees, so the run ends by closing them all."""
        closed: list[ActionInterval] = []
        for track_id in sorted(self._open):
            closed.extend(self._close_track(track_id))
        self._last_seen_us.clear()
        return closed

    # ------------------------------------------------------------------

    def _close_track(self, track_id: str) -> list[ActionInterval]:
        held = self._open.pop(track_id, None)
        # Pending windows are dropped rather than opened: they never reached
        # close_after, so nothing confirms they were an action rather than the
        # classifier wobbling on the way out of frame.
        self._pending.pop(track_id, None)
        return self._emit(self._finish(held)) if held is not None else []

    def _finish(self, held: _Open) -> ActionInterval:
        """Seal one open interval, resolving who it was at CLOSE time.

        Close time, not open time, and the difference is not cosmetic. A track
        is frequently unbound when its first window lands -- the person has to
        face the camera before a face can be detected and matched at all --
        and binds partway through. The identity known at close is therefore
        the better answer for the whole span; resolving at open would
        attribute a named person's four seconds of work to nobody.

        It remains an inference about the span rather than a measurement of
        it: a binding made at second three is being applied back to second
        one. That is sound only because `IdentityResolver` binds once per
        track and holds, so the identity is a property of the track rather
        than of the moment -- and because `track_id` stays on the interval
        either way, so the attribution can be re-examined against the raw
        tracks later. An identity is never written in place of the track id:
        the track records what the camera saw, the identity is a claim about
        who that was, and collapsing the two would make the claim
        unquestionable.

        With no resolver, `subject` stays None and the event builder falls
        back to the anonymous subject -- byte-identical to the behaviour from
        before identity existed.
        """
        interval = held.finish()
        if self._subject_for is not None:
            interval.subject = self._subject_for(held.track_id)
        return interval

    def _emit(self, interval: ActionInterval) -> list[ActionInterval]:
        if interval.duration_s < self.min_duration_s:
            return []
        return [interval]


def build_action_interval_event(
    emitter: EventEmitter,
    interval: ActionInterval,
    timestamp_us: Optional[int] = None,
    subject_for: Optional[Callable[[str], dict]] = None,
) -> dict:
    """Schema-shaped `action_interval` event for one closed interval.

    `timestamp_us` is when the interval CLOSED, not when it ran: the event
    stream is ordered by emission time and the engine's duration rules depend
    on that staying monotonic. The span the event is about is carried in
    start_us / end_us, which is why those two fields exist.

    The subject comes from the interval itself, resolved by the builder at
    close time (`IntervalBuilder._finish`) -- which is the right moment, and
    the reason the resolver is handed to the builder rather than to this
    function. `subject_for` here is the fallback for an `ActionInterval` built
    outside a builder (a test, a replay tool) and is ignored when the interval
    already carries a subject. With neither, the subject is the anonymous
    human this event has always carried.
    """
    event = emitter.build(
        observation="action_interval",
        confidence=interval.confidence,
        value=interval.label,
        track_id=interval.track_id,
        subject=interval.subject or subject_of(subject_for, interval.track_id),
        timestamp_us=timestamp_us,
    )
    # Set after build() rather than through it: these four belong to this one
    # observation, and emit.py stays the generic builder it is.
    event["start_us"] = interval.start_us
    event["end_us"] = interval.end_us
    # The same two instants on the wall clock. KEPT ALONGSIDE the monotonic
    # pair, never instead of it: every rule that measures a duration reads
    # start_us / end_us, which an NTP correction cannot reorder, and these two
    # are for the human who has to say which shift this happened on.
    # Converted through the offset captured at startup, so they name the
    # moments the recogniser observed and not the moment this line ran.
    event["wall_time_start"] = wall_time_at(interval.start_us)
    event["wall_time_end"] = wall_time_at(interval.end_us)
    return event
