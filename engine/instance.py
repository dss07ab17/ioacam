"""One running instance of a declared workflow.

Step lifecycle:

    PENDING  -- predecessors not yet complete
       |
       v  (all predecessors complete; clock starts here)
    ELIGIBLE
       |  \\
       |   \\ (some but not all evidence seen)
       |    -> PARTIAL
       |        |
       v        v
    COMPLETE  or  FAILED (deadline expired)

A FAILED step still releases its successors. Otherwise one skipped step stalls
the whole instance and every later step reports as skipped too, which buries
the real finding under noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .model import US_PER_S
from .policy import Step, Workflow


class StepState(str, Enum):
    PENDING = "pending"
    ELIGIBLE = "eligible"
    PARTIAL = "partial"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class StepRun:
    """Runtime state of one step within one instance."""

    step: Step
    state: StepState = StepState.PENDING
    eligible_at_us: Optional[int] = None
    first_evidence_us: Optional[int] = None
    completed_at_us: Optional[int] = None

    # Which evidence requirements have been met, by index into step.evidence.
    satisfied: set[int] = field(default_factory=set)
    # Confidence of each satisfying event, so the step's confidence can be the
    # weakest link rather than the most flattering one.
    evidence_confidence: list[float] = field(default_factory=list)
    # Set once so a repeat completion is reported as REPEATED, not as a second
    # normal completion.
    completions: int = 0

    @property
    def step_id(self) -> str:
        return self.step.step_id

    def confidence(self) -> float:
        """Weakest link across the evidence that proved this step.

        A step is only as well evidenced as its least certain component. Taking
        the mean would let one certain bus reading paper over a marginal visual
        detection.
        """
        if not self.evidence_confidence:
            return 1.0
        return min(self.evidence_confidence)

    def duration_s(self) -> Optional[float]:
        if self.eligible_at_us is None or self.completed_at_us is None:
            return None
        return (self.completed_at_us - self.eligible_at_us) / US_PER_S

    def deadline_us(self) -> Optional[int]:
        if self.eligible_at_us is None:
            return None
        return self.step.deadline_us(self.eligible_at_us)

    def is_open(self) -> bool:
        return self.state in (StepState.ELIGIBLE, StepState.PARTIAL)

    def is_settled(self) -> bool:
        return self.state in (StepState.COMPLETE, StepState.FAILED)


@dataclass
class WorkflowInstance:
    instance_id: str
    workflow: Workflow
    started_at_us: int
    runs: dict[str, StepRun] = field(default_factory=dict)
    closed: bool = False
    # Subject that triggered the instance, used to attribute findings when a
    # later event carries no identity of its own.
    trigger_zone: Optional[str] = None

    # Which event attribute distinguishes this instance from its siblings, and
    # the value it was opened with. None for singleton workflows.
    correlation_attr: Optional[str] = None
    correlation_value: Optional[str] = None

    # Last time any event was correlated to this instance. Used to detect that
    # the subject has been lost rather than that they stopped working.
    last_event_us: int = 0

    # Set when the instance was closed because correlation was lost. Such an
    # instance must NOT report its remaining steps as violations: the cause is
    # a tracking or sensor failure on our side, not a deviation by the actor.
    correlation_lost: bool = False

    @property
    def correlation_key(self) -> Optional[str]:
        if self.correlation_attr is None:
            return None
        return f"{self.correlation_attr}={self.correlation_value}"

    def zones(self) -> set[str]:
        """Every zone this instance could legitimately receive events from.

        Used as the fallback when an event carries no correlation value of its
        own -- a PLC reporting a torque cycle has no track_id, but it does have
        a zone.
        """
        z = self.workflow.zones()
        if self.trigger_zone:
            z = z | {self.trigger_zone}
        return z

    def __post_init__(self) -> None:
        if not self.runs:
            self.runs = {s.step_id: StepRun(step=s) for s in self.workflow.steps}
        if not self.last_event_us:
            self.last_event_us = self.started_at_us
        self.refresh_eligibility(self.started_at_us)

    def refresh_eligibility(self, now_us: int) -> list[StepRun]:
        """Promote PENDING steps whose predecessors have all settled.

        Returns the steps newly promoted, so the caller can start their clocks.
        A predecessor that FAILED still counts as settled: the instance carries
        on so later deviations are still detected.
        """
        promoted: list[StepRun] = []
        for run in self.runs.values():
            if run.state is not StepState.PENDING:
                continue
            preds = run.step.predecessors
            if all(self.runs[p].is_settled() for p in preds):
                run.state = StepState.ELIGIBLE
                run.eligible_at_us = now_us
                promoted.append(run)
        return promoted

    def open_steps(self) -> list[StepRun]:
        return [r for r in self.runs.values() if r.is_open()]

    def pending_steps(self) -> list[StepRun]:
        return [r for r in self.runs.values() if r.state is StepState.PENDING]

    def all_settled(self) -> bool:
        return all(r.is_settled() for r in self.runs.values())

    def timeout_us(self) -> Optional[int]:
        if self.workflow.instance_timeout_s is None:
            return None
        return self.started_at_us + int(self.workflow.instance_timeout_s * US_PER_S)

    def oldest_open_eligible_us(self) -> int:
        """When the earliest still-open step became eligible.

        Tie-break for zone-fallback correlation: when an uncorrelatable event
        could belong to two instances at the same station, the one that has
        been waiting longest for it is the better guess.
        """
        times = [
            r.eligible_at_us
            for r in self.runs.values()
            if r.is_open() and r.eligible_at_us is not None
        ]
        return min(times) if times else self.started_at_us

    def unmet_predecessors(self, run: StepRun) -> list[str]:
        """Predecessors of `run` that have not completed successfully.

        Used to distinguish a genuine out-of-order execution from a step that
        merely follows a failed one.
        """
        return [
            p
            for p in run.step.predecessors
            if self.runs[p].state is not StepState.COMPLETE
        ]
