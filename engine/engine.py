"""The workflow engine.

Consumes Events, compares them against the declared Policy, emits Findings.

It does exactly two things with time:

  ingest(event)  - advances the clock to the event, then judges the event
  tick(now_us)   - advances the clock only, firing any expired deadlines

Deadlines are what detect omitted steps, so tick() must be called even when no
events are arriving. On the device this is a periodic timer; in the harness it
is called explicitly at the end of a scenario.
"""

from __future__ import annotations

import itertools
from typing import Iterable, Optional

from .instance import StepRun, StepState, WorkflowInstance
from .model import (
    US_PER_S,
    Deviation,
    Event,
    Finding,
    Response,
    Route,
    Severity,
    Subject,
    Verdict,
)
from .policy import Policy, Step, Zone


class WorkflowEngine:
    def __init__(self, policy: Policy) -> None:
        self.policy = policy
        self.instances: list[WorkflowInstance] = []
        self.now_us: int = 0
        self._ids = itertools.count(1)
        # sensor_id -> healthy. Absent means assumed healthy until told
        # otherwise.
        self.sensor_health: dict[str, bool] = {}
        # zone_id -> current occupancy, from person_count events.
        self.occupancy: dict[str, int] = {}
        # Zones already reported as uncovered, so the finding fires on the
        # transition rather than on every subsequent event.
        self._degraded_zones: set[str] = set()
        # Workflows currently at their concurrent-instance cap, so the warning
        # fires on the transition rather than on every further trigger.
        self._at_capacity: set[str] = set()
        # (workflow, zone) pairs already reported as ambiguous for correlation.
        self._ambiguous: set[tuple] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(self, event: Event) -> list[Finding]:
        """Judge one event. Returns every finding it produced."""
        findings: list[Finding] = []

        # Deadlines that fell due before this event must fire first, otherwise
        # a late event could satisfy a step whose deadline had already passed.
        findings += self.tick(event.timestamp_us)

        if event.observation == "sensor_health":
            findings += self._handle_sensor_health(event)
            return findings

        if event.observation == "person_count" and event.zone_id:
            self.occupancy[event.zone_id] = int(event.value or 0)

        if event.observation == "person_left_zone" and event.track_id:
            findings += self._lose_correlation_for_track(event)

        consumed = False

        findings_zone = self._check_zone_rules(event)
        findings += findings_zone
        if findings_zone:
            consumed = True

        # Feed open instances before opening new ones, so a trigger
        # observation that is also step evidence advances the existing
        # instance rather than spawning a duplicate.
        step_findings, matched = self._apply_to_instances(event)
        findings += step_findings
        consumed = consumed or matched

        trigger_findings, triggered = self._check_triggers(event)
        findings += trigger_findings
        consumed = consumed or triggered

        if not consumed:
            # Observed something the declaration does not cover. This is never
            # treated as conformant: a system that silently accepts what it
            # does not recognise is blind to exactly the behaviour it exists
            # to catch.
            findings.append(
                self._make_finding(
                    event.timestamp_us,
                    Verdict.UNKNOWN,
                    severity=None,
                    confidence=event.confidence,
                    zone_id=event.zone_id,
                    subject=event.subject,
                    detail=f"undeclared observation '{event.observation}'={event.value!r}",
                    triggering_event_id=event.event_id,
                )
            )

        return findings

    def tick(self, now_us: int) -> list[Finding]:
        """Advance the clock, firing expired step deadlines and instance
        timeouts. Safe to call repeatedly; each deadline fires once."""
        if now_us < self.now_us:
            # Events must arrive in monotonic order. Out-of-order arrival would
            # corrupt duration measurement silently, so hold the clock.
            now_us = self.now_us
        self.now_us = now_us

        findings: list[Finding] = []
        progressed = True
        # Firing one deadline can promote successors whose own deadlines may
        # already be due, so iterate to a fixed point. Earliest deadline first,
        # and successors are promoted at the moment the predecessor SETTLED,
        # not at the current tick. On the device tick() runs every few hundred
        # milliseconds and the difference is invisible; in a test that jumps
        # sixty seconds it decides whether the next step's clock is right.
        while progressed:
            progressed = False
            for inst in list(self.instances):
                if inst.closed:
                    continue
                # Checked before step deadlines: if we have lost the subject,
                # their pending steps must not fire as violations.
                ct = inst.workflow.correlation_timeout_s
                if ct is not None and not inst.workflow.singleton:
                    silent_since = inst.last_event_us + int(ct * US_PER_S)
                    if now_us >= silent_since and not inst.all_settled():
                        findings += self._close_correlation_lost(
                            inst,
                            silent_since,
                            f"no correlated event for {ct}s",
                        )
                        progressed = True
                        continue
                due = sorted(
                    (
                        (run.deadline_us(), run)
                        for run in inst.open_steps()
                        if run.deadline_us() is not None
                        and now_us >= run.deadline_us()
                    ),
                    key=lambda pair: pair[0],
                )
                for dl, run in due:
                    findings.append(self._fail_step(inst, run, dl))
                    inst.refresh_eligibility(dl)
                    progressed = True
                    break  # re-derive the due list; promotion may have added one
                findings += self._maybe_close(inst, now_us)
        return findings

    def flush(self, now_us: Optional[int] = None) -> list[Finding]:
        """Close out everything still open. Used at end of scenario or shift."""
        if now_us is None:
            now_us = self.now_us
        findings = self.tick(now_us)
        for inst in list(self.instances):
            if not inst.closed:
                findings += self._close_incomplete(inst, now_us)
        return findings

    # ------------------------------------------------------------------
    # Zone rules -- certain, no perception uncertainty in the rule itself
    # ------------------------------------------------------------------

    def _check_zone_rules(self, event: Event) -> list[Finding]:
        findings: list[Finding] = []
        zone = self.policy.zones.get(event.zone_id) if event.zone_id else None
        if zone is None:
            return findings

        if event.observation in ("person_in_zone", "person_present") and event.value:
            day, hhmm = _wall_parts(event.wall_time)

            if not zone.role_permitted(event.subject.role):
                findings.append(
                    self._make_finding(
                        event.timestamp_us,
                        Verdict.VIOLATION,
                        severity=_zone_severity(zone),
                        deviation=Deviation.WRONG_ZONE,
                        # The rule is certain; the uncertainty is only in
                        # whether we correctly saw who this was.
                        confidence=event.confidence,
                        zone_id=zone.zone_id,
                        subject=event.subject,
                        detail=(
                            f"role {event.subject.role!r} not permitted in "
                            f"{zone.zone_id} (allowed: {zone.allowed_roles})"
                        ),
                        triggering_event_id=event.event_id,
                    )
                )
            elif not zone.time_permitted(day, hhmm):
                findings.append(
                    self._make_finding(
                        event.timestamp_us,
                        Verdict.VIOLATION,
                        severity=_zone_severity(zone),
                        deviation=Deviation.WRONG_ZONE,
                        confidence=event.confidence,
                        zone_id=zone.zone_id,
                        subject=event.subject,
                        detail=f"presence outside permitted window ({day} {hhmm})",
                        triggering_event_id=event.event_id,
                    )
                )

        if event.observation == "identity_unverified":
            # Someone is present whom the site cannot account for. This is the
            # case a door-only product cannot see at all: it knows who opened
            # the door and not how many people walked through.
            findings.append(
                self._make_finding(
                    event.timestamp_us,
                    Verdict.VIOLATION,
                    severity=_zone_severity(zone),
                    deviation=Deviation.WRONG_ZONE,
                    # The failed match is itself certain -- the comparison ran
                    # and nothing cleared the bar. What is uncertain is who
                    # this person is, which is precisely the finding.
                    confidence=event.confidence,
                    zone_id=zone.zone_id,
                    subject=event.subject,
                    detail=(
                        f"unaccounted presence in {zone.zone_id}: {event.value}"
                    ),
                    triggering_event_id=event.event_id,
                )
            )

        if event.observation == "presence_unbadged":
            # A known person in a zone they never authenticated into. Either
            # they entered without badging or the zone's reader missed them.
            # Both are worth knowing; neither is an unaccounted presence.
            findings.append(
                self._make_finding(
                    event.timestamp_us,
                    Verdict.VIOLATION,
                    severity=_zone_severity(zone),
                    deviation=Deviation.WRONG_ZONE,
                    confidence=event.confidence,
                    zone_id=zone.zone_id,
                    subject=event.subject,
                    detail=(
                        f"{event.subject.identity or event.value} is in "
                        f"{zone.zone_id} without a badge-in for this zone"
                    ),
                    triggering_event_id=event.event_id,
                )
            )

        if (
            event.observation == "person_count"
            and zone.max_occupancy is not None
            and isinstance(event.value, int)
            and event.value > zone.max_occupancy
        ):
            findings.append(
                self._make_finding(
                    event.timestamp_us,
                    Verdict.VIOLATION,
                    severity=_zone_severity(zone),
                    confidence=event.confidence,
                    zone_id=zone.zone_id,
                    subject=event.subject,
                    detail=(
                        f"occupancy {event.value} exceeds max {zone.max_occupancy}"
                    ),
                    triggering_event_id=event.event_id,
                )
            )
        return findings

    def _handle_sensor_health(self, event: Event) -> list[Finding]:
        """Track sensor liveness and degrade any zone that loses all coverage.

        A zone nobody can see must report UNKNOWN, not CONFORMANT. Reporting
        'nothing wrong' from a blind zone is the most dangerous failure mode
        this system has.
        """
        findings: list[Finding] = []
        healthy = event.value not in ("unhealthy", "lost", "failed", False)
        if event.sensor_id:
            self.sensor_health[event.sensor_id] = healthy

        for zone in self.policy.zones.values():
            if not zone.covered_by:
                continue
            covered = any(
                self.sensor_health.get(s, True) for s in zone.covered_by
            )
            if not covered and zone.zone_id not in self._degraded_zones:
                self._degraded_zones.add(zone.zone_id)
                findings.append(
                    self._make_finding(
                        event.timestamp_us,
                        Verdict.UNKNOWN,
                        severity=Severity.CRITICAL,
                        confidence=1.0,
                        zone_id=zone.zone_id,
                        route_override=Route.MAINTENANCE,
                        detail=(
                            f"zone {zone.zone_id} has no healthy sensor "
                            f"({zone.covered_by}); coverage lost"
                        ),
                        triggering_event_id=event.event_id,
                    )
                )
                # Any instance depending on this zone can no longer be
                # observed. Failing its steps on deadline would report
                # violations that the blindness caused, not the actor.
                for inst in list(self.instances):
                    if inst.closed or inst.all_settled():
                        continue
                    if zone.zone_id in inst.zones():
                        findings += self._close_correlation_lost(
                            inst,
                            event.timestamp_us,
                            f"zone {zone.zone_id} lost sensor coverage",
                            event.event_id,
                        )
            elif covered and zone.zone_id in self._degraded_zones:
                self._degraded_zones.discard(zone.zone_id)
        return findings

    # ------------------------------------------------------------------
    # Triggers and instances
    # ------------------------------------------------------------------

    def _check_triggers(self, event: Event) -> tuple[list[Finding], bool]:
        findings: list[Finding] = []
        triggered = False
        for wf in self.policy.workflows:
            if not wf.trigger.matches(event):
                continue
            triggered = True

            attr, value = _correlation_of(event, wf.correlation)

            if wf.singleton:
                # One instance at a time. Still correct for workflows that
                # genuinely cannot overlap, and remains the default.
                if any(
                    i.workflow.workflow_id == wf.workflow_id and not i.closed
                    for i in self.instances
                ):
                    continue
            elif attr is None:
                # The workflow wants instances told apart, but the triggering
                # event carries none of the declared keys. Opening an
                # uncorrelated instance would silently absorb events belonging
                # to every other actor, so refuse and say so.
                findings.append(
                    self._make_finding(
                        event.timestamp_us,
                        Verdict.UNKNOWN,
                        severity=Severity.WARNING,
                        confidence=1.0,
                        zone_id=event.zone_id,
                        subject=event.subject,
                        route_override=Route.MAINTENANCE,
                        detail=(
                            f"{wf.workflow_id}: trigger event carries none of the "
                            f"declared correlation attributes {wf.correlation}; "
                            f"instance not opened"
                        ),
                        triggering_event_id=event.event_id,
                    )
                )
                continue
            else:
                key = f"{attr}={value}"
                if any(
                    i.workflow.workflow_id == wf.workflow_id
                    and not i.closed
                    and i.correlation_key == key
                    for i in self.instances
                ):
                    # This actor already has an instance running. A second
                    # trigger from the same actor is not a new instance.
                    continue

            open_count = sum(
                1
                for i in self.instances
                if i.workflow.workflow_id == wf.workflow_id and not i.closed
            )
            if not wf.singleton and open_count >= wf.max_concurrent_instances:
                # Hitting the cap is itself diagnostic. In a correctly
                # configured site it means the tracker is churning ids, which
                # is a perception problem, not a workflow deviation.
                if wf.workflow_id not in self._at_capacity:
                    self._at_capacity.add(wf.workflow_id)
                    findings.append(
                        self._make_finding(
                            event.timestamp_us,
                            Verdict.UNKNOWN,
                            severity=Severity.WARNING,
                            confidence=1.0,
                            zone_id=event.zone_id,
                            route_override=Route.MAINTENANCE,
                            detail=(
                                f"{wf.workflow_id}: at capacity "
                                f"({wf.max_concurrent_instances} concurrent "
                                f"instances); further triggers ignored. Usually "
                                f"means tracker id churn rather than real load."
                            ),
                            triggering_event_id=event.event_id,
                        )
                    )
                continue

            inst = WorkflowInstance(
                instance_id=f"{wf.workflow_id}#{next(self._ids)}",
                workflow=wf,
                started_at_us=event.timestamp_us,
                trigger_zone=event.zone_id,
                correlation_attr=attr if not wf.singleton else None,
                correlation_value=value if not wf.singleton else None,
                last_event_us=event.timestamp_us,
            )
            self.instances.append(inst)
            # The triggering event may itself be evidence for the first step.
            step_findings, _ = self._apply_to_instance(inst, event)
            findings += step_findings
        return findings, triggered

    def _apply_to_instances(self, event: Event) -> tuple[list[Finding], bool]:
        """Route the event to the instances it belongs to.

        Key matches win outright. Zone fallback exists because certain event
        sources carry no per-actor identity at all -- a PLC reporting a torque
        cycle knows the station, not the operator -- and refusing to correlate
        those would leave every bus-evidenced step permanently unprovable.
        """
        findings: list[Finding] = []
        matched = False

        by_key: list[WorkflowInstance] = []
        by_zone: list[WorkflowInstance] = []
        for inst in self.instances:
            if inst.closed:
                continue
            how = self._instance_match(inst, event)
            if how == "key":
                by_key.append(inst)
            elif how == "zone":
                by_zone.append(inst)

        if by_key:
            targets = by_key
        elif by_zone:
            # Prefer the instance that has been waiting longest for evidence.
            # Deterministic, and the better guess: the station that started
            # first is the one whose cycle should finish first.
            targets = [min(by_zone, key=lambda i: i.oldest_open_eligible_us())]
            if len(by_zone) > 1:
                findings += self._note_ambiguity(event, by_zone, targets[0])
        else:
            targets = []

        for inst in targets:
            f, m = self._apply_to_instance(inst, event)
            findings += f
            if m:
                inst.last_event_us = max(inst.last_event_us, event.timestamp_us)
            matched = matched or m

        return findings, matched

    def _instance_match(
        self, inst: WorkflowInstance, event: Event
    ) -> Optional[str]:
        """How, if at all, this event belongs to this instance."""
        if inst.workflow.singleton:
            return "key"

        if inst.correlation_attr is not None:
            value = _attr_of(event, inst.correlation_attr)
            if value is not None:
                # The event carries this instance's key attribute, so it is
                # decisive either way: a mismatch means it belongs to a sibling.
                return "key" if value == inst.correlation_value else None

        # No correlation value on the event. Fall back to geography.
        if event.zone_id and event.zone_id in inst.zones():
            return "zone"
        return None

    def _note_ambiguity(
        self,
        event: Event,
        candidates: list[WorkflowInstance],
        chosen: WorkflowInstance,
    ) -> list[Finding]:
        """Report, once per workflow and zone, that events cannot be attributed.

        Emitted once rather than per event: a station with two concurrent
        instances and a shared PLC will produce this on every bus message, and
        the operator needs to know the configuration is ambiguous, not to be
        buried in it. The usual fix is for the perception layer to attach a
        correlation attribute to bus events, or to key the workflow on zone.
        """
        wf_id = candidates[0].workflow.workflow_id
        tag = (wf_id, event.zone_id)
        if tag in self._ambiguous:
            return []
        self._ambiguous.add(tag)
        return [
            self._make_finding(
                event.timestamp_us,
                Verdict.UNKNOWN,
                severity=Severity.WARNING,
                confidence=1.0,
                zone_id=event.zone_id,
                route_override=Route.MAINTENANCE,
                detail=(
                    f"{wf_id}: '{event.observation}' in {event.zone_id} could belong "
                    f"to {len(candidates)} concurrent instances and carries no "
                    f"correlation attribute; assigned to {chosen.instance_id} "
                    f"(longest waiting). Attribution is a guess until bus events "
                    f"carry a correlation key."
                ),
                triggering_event_id=event.event_id,
            )
        ]

    def _apply_to_instance(
        self, inst: WorkflowInstance, event: Event
    ) -> tuple[list[Finding], bool]:
        """Judge one event against one instance.

        Open steps are considered before pending and completed ones, and if an
        open step engages with the event the others are not offered it. Real
        policies reuse observations across steps -- `person_in_zone` may prove
        both step 1 and step 4 -- and without this precedence a perfectly
        normal event would simultaneously complete a future step out of order
        and flag an earlier step as repeated. The currently expected step gets
        first claim on the evidence.
        """
        findings: list[Finding] = []

        open_runs = [r for r in inst.runs.values() if r.is_open()]
        other_runs = [r for r in inst.runs.values() if not r.is_open()]

        f, engaged = self._offer(inst, event, open_runs)
        findings += f
        if engaged:
            return findings, True

        f, engaged = self._offer(inst, event, other_runs)
        return findings + f, engaged

    def _offer(
        self, inst: WorkflowInstance, event: Event, runs: list[StepRun]
    ) -> tuple[list[Finding], bool]:
        findings: list[Finding] = []
        matched = False

        for run in list(runs):
            rel = [
                i
                for i, req in enumerate(run.step.evidence)
                if req.matches_observation(event)
            ]
            if not rel:
                continue

            # Zone mismatch: the event is about this step but happened
            # somewhere else. Recognised, flagged, and not accepted as proof.
            if run.step.zone_id and event.zone_id and event.zone_id != run.step.zone_id:
                matched = True
                findings.append(
                    self._step_finding(
                        inst,
                        run,
                        event.timestamp_us,
                        Verdict.VIOLATION,
                        Deviation.WRONG_ZONE,
                        event.confidence,
                        event.subject,
                        f"step evidence seen in {event.zone_id}, expected "
                        f"{run.step.zone_id}",
                        event.event_id,
                    )
                )
                continue

            if run.state is StepState.COMPLETE:
                matched = True
                run.completions += 1
                findings.append(
                    self._step_finding(
                        inst,
                        run,
                        event.timestamp_us,
                        Verdict.VIOLATION,
                        Deviation.REPEATED,
                        event.confidence,
                        event.subject,
                        f"step already completed, seen again "
                        f"(x{run.completions + 1})",
                        event.event_id,
                    )
                )
                continue

            if run.state is StepState.FAILED:
                # Already reported as skipped or incomplete. Late arrival is
                # noted but does not resurrect the step.
                matched = True
                continue

            # Wrong role: performed correctly, by the wrong person. Still a
            # violation, and the evidence does not count.
            if (
                run.step.actor_role
                and event.subject.role
                and event.subject.role != run.step.actor_role
            ):
                matched = True
                findings.append(
                    self._step_finding(
                        inst,
                        run,
                        event.timestamp_us,
                        Verdict.VIOLATION,
                        Deviation.WRONG_ROLE,
                        event.confidence,
                        event.subject,
                        f"role {event.subject.role!r} performed step reserved "
                        f"for {run.step.actor_role!r}",
                        event.event_id,
                    )
                )
                continue

            # Confidence gate. Below the declared threshold the observation is
            # logged but does not count as proof.
            accepted = [
                i for i in rel if run.step.evidence[i].satisfied_by(event)
            ]
            if not accepted:
                matched = True
                continue

            matched = True
            if run.state is StepState.ELIGIBLE:
                run.state = StepState.PARTIAL
                run.first_evidence_us = event.timestamp_us
            for i in accepted:
                if i not in run.satisfied:
                    run.satisfied.add(i)
                    run.evidence_confidence.append(event.confidence)

            if len(run.satisfied) == len(run.step.evidence):
                findings += self._complete_step(inst, run, event)

        return findings, matched

    # ------------------------------------------------------------------
    # Step outcomes
    # ------------------------------------------------------------------

    def _complete_step(
        self, inst: WorkflowInstance, run: StepRun, event: Event
    ) -> list[Finding]:
        findings: list[Finding] = []
        run.state = StepState.COMPLETE
        run.completed_at_us = event.timestamp_us
        step = run.step

        out_of_order = inst.unmet_predecessors(run)
        if out_of_order:
            findings.append(
                self._step_finding(
                    inst,
                    run,
                    event.timestamp_us,
                    Verdict.VIOLATION,
                    Deviation.OUT_OF_ORDER,
                    run.confidence(),
                    event.subject,
                    f"completed before predecessor(s) {out_of_order}",
                    event.event_id,
                )
            )
        else:
            verdict, deviation, detail = _judge_duration(step, run.duration_s())
            findings.append(
                self._step_finding(
                    inst,
                    run,
                    event.timestamp_us,
                    verdict,
                    deviation,
                    run.confidence(),
                    event.subject,
                    detail,
                    event.event_id,
                )
            )

        inst.refresh_eligibility(event.timestamp_us)
        findings += self._maybe_close(inst, event.timestamp_us)
        return findings

    def _fail_step(
        self, inst: WorkflowInstance, run: StepRun, at_us: int
    ) -> Finding:
        """Deadline expired. Distinguish never-started from started-not-finished."""
        run.state = StepState.FAILED
        step = run.step

        if run.satisfied:
            deviation = Deviation.INCOMPLETE
            detail = (
                f"deadline expired with {len(run.satisfied)}/"
                f"{len(step.evidence)} evidence items"
            )
            confidence = run.confidence()
        else:
            deviation = Deviation.SKIPPED
            detail = (
                f"no evidence within {step.max_duration_s}s "
                f"(+{step.overrun_tol}s overrun tolerance)"
            )
            # A deadline expiry is a computed fact, not an inference. There is
            # nothing uncertain about a timer.
            confidence = 1.0

        verdict = Verdict.TOLERATED if step.optional else Verdict.VIOLATION
        if step.optional:
            detail += " (step declared optional)"

        return self._step_finding(
            inst, run, at_us, verdict, deviation, confidence, Subject(), detail, None
        )

    def _lose_correlation_for_track(self, event: Event) -> list[Finding]:
        """The subject an instance was following has left. Close, do not accuse.

        A departed or lost subject is not a worker who skipped a step. If the
        remaining steps were failed normally, a tracker limitation would be
        reported as a critical violation against a named person -- a false
        accusation with a technical cause, and the fastest way to lose a
        customer's trust in the system.
        """
        findings: list[Finding] = []
        for inst in list(self.instances):
            if inst.closed or inst.workflow.singleton:
                continue
            # Match on whatever key this instance actually uses, not just
            # track_id: an instance keyed on an enrolled identity must close
            # on departure too. Zone fallback deliberately does not count --
            # one person leaving must not close everyone else's instance.
            if self._instance_match(inst, event) != "key":
                continue
            if inst.all_settled():
                continue
            findings += self._close_correlation_lost(
                inst,
                event.timestamp_us,
                f"subject {event.track_id} left {event.zone_id}",
                event.event_id,
            )
        return findings

    def _close_correlation_lost(
        self,
        inst: WorkflowInstance,
        at_us: int,
        reason: str,
        event_id: Optional[str] = None,
    ) -> list[Finding]:
        """Close an instance whose subject we can no longer follow.

        One finding for the instance, not one per unfinished step, and UNKNOWN
        rather than VIOLATION. The distinction that matters operationally: the
        actor is present but not progressing (violation) versus we have lost
        the actor (unknown, and someone should look at why).
        """
        unfinished = [r.step_id for r in inst.runs.values() if not r.is_settled()]
        for run in inst.runs.values():
            if not run.is_settled():
                run.state = StepState.FAILED
        inst.correlation_lost = True
        inst.closed = True
        return [
            self._make_finding(
                at_us,
                Verdict.UNKNOWN,
                severity=Severity.WARNING,
                confidence=1.0,
                zone_id=inst.trigger_zone,
                workflow_id=inst.workflow.workflow_id,
                instance_id=inst.instance_id,
                route_override=Route.REVIEW_QUEUE,
                detail=(
                    f"correlation lost ({reason}); {len(unfinished)} step(s) "
                    f"{unfinished} unresolved. Not reported as deviations: the "
                    f"subject was not observed to the end."
                ),
                triggering_event_id=event_id,
            )
        ]

    def _maybe_close(self, inst: WorkflowInstance, now_us: int) -> list[Finding]:
        if inst.closed:
            return []
        if inst.all_settled():
            inst.closed = True
            return []
        t = inst.timeout_us()
        if t is not None and now_us >= t:
            return self._close_incomplete(inst, t)
        return []

    def _close_incomplete(
        self, inst: WorkflowInstance, at_us: int
    ) -> list[Finding]:
        """Instance ran out of time with steps still open or pending.

        Prevents orphaned instances accumulating when an actor simply walks
        away, and turns the abandonment itself into a reportable finding.
        """
        findings: list[Finding] = []
        for run in inst.runs.values():
            if run.is_settled():
                continue
            run.state = StepState.FAILED
            deviation = (
                Deviation.INCOMPLETE if run.satisfied else Deviation.SKIPPED
            )
            confidence = run.confidence() if run.satisfied else 1.0
            verdict = (
                Verdict.TOLERATED if run.step.optional else Verdict.VIOLATION
            )
            findings.append(
                self._step_finding(
                    inst,
                    run,
                    at_us,
                    verdict,
                    deviation,
                    confidence,
                    Subject(),
                    "instance timed out before step settled",
                    None,
                )
            )
        inst.closed = True
        return findings

    # ------------------------------------------------------------------
    # Finding construction
    # ------------------------------------------------------------------

    def _step_finding(
        self,
        inst: WorkflowInstance,
        run: StepRun,
        at_us: int,
        verdict: Verdict,
        deviation: Optional[Deviation],
        confidence: float,
        subject: Subject,
        detail: str,
        event_id: Optional[str],
    ) -> Finding:
        severity = (
            run.step.severity_for(deviation)
            if deviation and verdict is not Verdict.CONFORMANT
            else None
        )
        return self._make_finding(
            at_us,
            verdict,
            severity=severity,
            deviation=deviation,
            confidence=confidence,
            zone_id=run.step.zone_id or inst.trigger_zone,
            subject=subject,
            detail=f"{run.step.name}: {detail}",
            workflow_id=inst.workflow.workflow_id,
            instance_id=inst.instance_id,
            step_id=run.step_id,
            triggering_event_id=event_id,
        )

    def _make_finding(
        self,
        at_us: int,
        verdict: Verdict,
        *,
        severity: Optional[Severity],
        confidence: float,
        deviation: Optional[Deviation] = None,
        zone_id: Optional[str] = None,
        subject: Optional[Subject] = None,
        detail: str = "",
        workflow_id: Optional[str] = None,
        instance_id: Optional[str] = None,
        step_id: Optional[str] = None,
        route_override: Optional[Route] = None,
        triggering_event_id: Optional[str] = None,
    ) -> Finding:
        rule = self.policy.resolve(verdict, severity, confidence)
        return Finding(
            timestamp_us=at_us,
            verdict=verdict,
            severity=severity,
            deviation=deviation,
            confidence=confidence,
            response=rule.response,
            route=route_override or rule.route,
            retain_evidence=rule.retain_evidence,
            zone_id=zone_id,
            subject=subject or Subject(),
            detail=detail,
            workflow_id=workflow_id,
            instance_id=instance_id,
            step_id=step_id,
            # In shadow mode the response is computed and logged in full, but
            # nothing is emitted. This is what lets a site discover that its
            # first-draft declarations are wrong without anyone being paged.
            suppressed=self.policy.shadow,
            policy_version=self.policy.version,
            triggering_event_id=triggering_event_id,
        )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _attr_of(event: Event, attr: str) -> Optional[str]:
    """Read a correlation attribute off an event, or None if absent."""
    if attr == "track_id":
        return event.track_id
    if attr == "subject.identity":
        return event.subject.identity
    if attr == "subject.asset_id":
        return event.subject.asset_id
    if attr == "zone_id":
        return event.zone_id
    return None


def _correlation_of(
    event: Event, attrs: list[str]
) -> tuple[Optional[str], Optional[str]]:
    """First declared attribute the event actually carries.

    Ordered rather than best-match, so the policy author controls precedence:
    an enrolled identity is a better instance key than a tracker id, because it
    survives the subject leaving the frame and coming back.
    """
    for attr in attrs:
        value = _attr_of(event, attr)
        if value is not None:
            return attr, value
    return None, None


def _judge_duration(
    step: Step, duration_s: Optional[float]
) -> tuple[Verdict, Optional[Deviation], str]:
    """Compare a completed step's duration against its declared band.

    The tolerance band is what most reduces false alarms during commissioning:
    it is the difference between 'slightly slow' and 'raise an alarm'.
    """
    if duration_s is None:
        return Verdict.CONFORMANT, None, "completed"

    d = f"completed in {duration_s:.1f}s"

    if step.min_duration_s is not None and duration_s < step.min_duration_s:
        if duration_s >= step.min_duration_s - step.underrun_tol:
            return Verdict.TOLERATED, Deviation.UNDERRUN, f"{d} (min {step.min_duration_s}s, within tolerance)"
        return (
            Verdict.VIOLATION,
            Deviation.UNDERRUN,
            f"{d}, below min {step.min_duration_s}s "
            f"(-{step.underrun_tol}s underrun tolerance)",
        )

    if step.max_duration_s is not None and duration_s > step.max_duration_s:
        # Beyond max + tolerance the deadline would already have fired, so
        # anything reaching here is inside the grace band.
        return Verdict.TOLERATED, Deviation.OVERRUN, f"{d} (max {step.max_duration_s}s, within tolerance)"

    return Verdict.CONFORMANT, None, d


def _zone_severity(zone: Zone) -> Severity:
    return {
        "public": Severity.INFORMATIONAL,
        "restricted": Severity.WARNING,
        "critical": Severity.CRITICAL,
        "safety_critical": Severity.SAFETY_RELEVANT,
    }.get(zone.criticality, Severity.WARNING)


def _wall_parts(wall_time: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """Extract weekday and HH:MM from an ISO timestamp, if one was supplied."""
    if not wall_time:
        return None, None
    try:
        from datetime import datetime

        dt = datetime.fromisoformat(wall_time.replace("Z", "+00:00"))
        return ["mon", "tue", "wed", "thu", "fri", "sat", "sun"][dt.weekday()], dt.strftime("%H:%M")
    except (ValueError, IndexError):
        return None, None
