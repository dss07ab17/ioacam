"""Loading and querying the declared policy.

The policy file is the declaration of what is supposed to happen. Because an
attacker who edits it turns 'violation' into 'conformant', in production this
file must be signed and its version anchored in the TPM. `Policy.version` is
stamped onto every Finding so any alert traces to the exact policy that judged
it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .model import US_PER_S, Deviation, Response, Route, Severity, Verdict

DAY_NAMES = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


class PolicyError(ValueError):
    """Raised when a policy file is internally inconsistent.

    Better to fail loudly at load than to run a policy with a dangling
    predecessor that silently never becomes eligible.
    """


@dataclass
class TimeWindow:
    days: list[str]
    start: str
    end: str

    def contains(self, day: str, hhmm: str) -> bool:
        return day in self.days and self.start <= hhmm <= self.end


@dataclass
class Zone:
    zone_id: str
    name: str
    criticality: str
    allowed_roles: list[str] = field(default_factory=list)
    time_windows: list[TimeWindow] = field(default_factory=list)
    max_occupancy: Optional[int] = None
    covered_by: list[str] = field(default_factory=list)

    def role_permitted(self, role: Optional[str]) -> bool:
        # No declared list means the zone places no role restriction.
        if not self.allowed_roles:
            return True
        if role is None:
            return False
        return role in self.allowed_roles

    def time_permitted(self, day: Optional[str], hhmm: Optional[str]) -> bool:
        # Window checks only apply when the event carried a wall clock.
        if not self.time_windows or day is None or hhmm is None:
            return True
        return any(w.contains(day, hhmm) for w in self.time_windows)

    @staticmethod
    def from_dict(d: dict) -> "Zone":
        return Zone(
            zone_id=d["zone_id"],
            name=d["name"],
            criticality=d["criticality"],
            allowed_roles=d.get("allowed_roles", []),
            time_windows=[TimeWindow(**w) for w in d.get("time_windows", [])],
            max_occupancy=d.get("max_occupancy"),
            covered_by=d.get("covered_by", []),
        )


@dataclass
class EvidenceReq:
    """One observation that must be seen for a step to count as performed."""

    observation: str
    value: Any = None
    min_confidence: float = 0.5

    def matches_observation(self, event) -> bool:
        """Whether the event is *about* this evidence item.

        Note this ignores confidence, role and zone deliberately. An event can
        be recognised as relevant to a step and still fail it on confidence or
        be flagged for wrong role. Conflating the two would make a wrong-role
        event look like an unrelated event and disappear.
        """
        if event.observation != self.observation:
            return False
        if self.value is None:
            return True
        return event.value == self.value

    def satisfied_by(self, event) -> bool:
        return self.matches_observation(event) and event.confidence >= self.min_confidence

    @staticmethod
    def from_dict(d: dict) -> "EvidenceReq":
        return EvidenceReq(
            observation=d["observation"],
            value=d.get("value"),
            min_confidence=float(d.get("min_confidence", 0.5)),
        )


@dataclass
class Step:
    step_id: str
    name: str
    evidence: list[EvidenceReq]
    predecessors: list[str] = field(default_factory=list)
    actor_role: Optional[str] = None
    zone_id: Optional[str] = None
    optional: bool = False
    min_duration_s: Optional[float] = None
    max_duration_s: Optional[float] = None
    duration_tolerance_s: float = 0.0
    underrun_tolerance_s: Optional[float] = None
    overrun_tolerance_s: Optional[float] = None
    deviation_severity: dict[str, Severity] = field(default_factory=dict)

    @property
    def underrun_tol(self) -> float:
        """Grace below the minimum. Falls back to the shared tolerance.

        Separate from the overrun grace because one value cannot serve both: a
        grace appropriate to a long maximum is usually absurd against a short
        minimum, and silently swallows the underrun rule.
        """
        if self.underrun_tolerance_s is not None:
            return self.underrun_tolerance_s
        return self.duration_tolerance_s

    @property
    def overrun_tol(self) -> float:
        """Grace above the maximum, which also extends the omission deadline."""
        if self.overrun_tolerance_s is not None:
            return self.overrun_tolerance_s
        return self.duration_tolerance_s

    def severity_for(self, deviation: Deviation) -> Severity:
        """Authored severity, with a conservative default.

        An unauthored deviation defaults to WARNING rather than
        INFORMATIONAL: an omission in the policy file should surface, not
        vanish.
        """
        return self.deviation_severity.get(deviation.value, Severity.WARNING)

    def deadline_us(self, eligible_at_us: int) -> Optional[int]:
        """When this step must have completed by.

        This deadline is what detects an OMITTED step. A skipped step produces
        no observation at all, so no detector can see it; the deadline expiring
        is the detection.
        """
        if self.max_duration_s is None:
            return None
        return eligible_at_us + int(
            (self.max_duration_s + self.overrun_tol) * US_PER_S
        )

    @staticmethod
    def from_dict(d: dict) -> "Step":
        sev = {
            k: Severity(v) for k, v in (d.get("deviation_severity") or {}).items()
        }
        return Step(
            step_id=d["step_id"],
            name=d["name"],
            evidence=[EvidenceReq.from_dict(e) for e in d["evidence"]],
            predecessors=d.get("predecessors", []),
            actor_role=d.get("actor_role"),
            zone_id=d.get("zone_id"),
            optional=bool(d.get("optional", False)),
            min_duration_s=d.get("min_duration_s"),
            max_duration_s=d.get("max_duration_s"),
            duration_tolerance_s=float(d.get("duration_tolerance_s", 0.0)),
            underrun_tolerance_s=(
                float(d["underrun_tolerance_s"])
                if "underrun_tolerance_s" in d
                else None
            ),
            overrun_tolerance_s=(
                float(d["overrun_tolerance_s"])
                if "overrun_tolerance_s" in d
                else None
            ),
            deviation_severity=sev,
        )


@dataclass
class Trigger:
    observation: str
    zone_id: Optional[str] = None
    value: Any = None
    role: Optional[str] = None

    def matches(self, event) -> bool:
        if event.observation != self.observation:
            return False
        if self.zone_id is not None and event.zone_id != self.zone_id:
            return False
        if self.value is not None and event.value != self.value:
            return False
        if self.role is not None and event.subject.role != self.role:
            return False
        return True

    @staticmethod
    def from_dict(d: dict) -> "Trigger":
        return Trigger(
            observation=d["observation"],
            zone_id=d.get("zone_id"),
            value=d.get("value"),
            role=d.get("role"),
        )


# Event attributes that may be used to tell concurrent instances apart.
CORRELATION_ATTRS = ("track_id", "subject.identity", "subject.asset_id", "zone_id")


@dataclass
class Workflow:
    workflow_id: str
    name: str
    actor_class: str
    trigger: Trigger
    steps: list[Step]
    instance_timeout_s: Optional[float] = None

    # Ordered list of event attributes that identify which instance an event
    # belongs to. Empty means singleton: one instance at a time, which is the
    # original behaviour and still correct for workflows that genuinely cannot
    # overlap.
    correlation: list[str] = field(default_factory=list)

    # Instances are created from observed events, so a tracker that churns ids
    # would otherwise spawn them without bound. This caps memory and, more
    # usefully, makes id churn visible instead of silent.
    max_concurrent_instances: int = 8

    # Silence, in seconds, before an instance is closed as correlation lost
    # rather than accruing step violations. Default None (disabled), because
    # it is only safe when the perception layer emits periodic liveness for
    # active tracks. Without that, a legitimately long step looks identical to
    # a lost subject, and enabling it would manufacture false closures.
    correlation_timeout_s: Optional[float] = None

    @property
    def singleton(self) -> bool:
        return not self.correlation

    def zones(self) -> set[str]:
        return {s.zone_id for s in self.steps if s.zone_id}

    def step(self, step_id: str) -> Step:
        for s in self.steps:
            if s.step_id == step_id:
                return s
        raise KeyError(step_id)

    def validate(self) -> None:
        for attr in self.correlation:
            if attr not in CORRELATION_ATTRS:
                raise PolicyError(
                    f"{self.workflow_id}: correlation attribute '{attr}' is not one "
                    f"of {list(CORRELATION_ATTRS)}"
                )
        if self.max_concurrent_instances < 1:
            raise PolicyError(
                f"{self.workflow_id}: max_concurrent_instances must be at least 1"
            )
        ids = {s.step_id for s in self.steps}
        if len(ids) != len(self.steps):
            raise PolicyError(f"{self.workflow_id}: duplicate step_id")
        for s in self.steps:
            for p in s.predecessors:
                if p not in ids:
                    raise PolicyError(
                        f"{self.workflow_id}/{s.step_id}: unknown predecessor '{p}'"
                    )
        # A cycle would leave steps permanently PENDING and the workflow would
        # silently never detect anything.
        resolved: set[str] = set()
        progress = True
        while progress:
            progress = False
            for s in self.steps:
                if s.step_id in resolved:
                    continue
                if all(p in resolved for p in s.predecessors):
                    resolved.add(s.step_id)
                    progress = True
        if resolved != ids:
            raise PolicyError(
                f"{self.workflow_id}: cycle in predecessors among {sorted(ids - resolved)}"
            )

    @staticmethod
    def from_dict(d: dict) -> "Workflow":
        wf = Workflow(
            workflow_id=d["workflow_id"],
            name=d["name"],
            actor_class=d["actor_class"],
            trigger=Trigger.from_dict(d["trigger"]),
            steps=[Step.from_dict(s) for s in d["steps"]],
            instance_timeout_s=d.get("instance_timeout_s"),
            correlation=list(d.get("correlation", [])),
            max_concurrent_instances=int(d.get("max_concurrent_instances", 8)),
            correlation_timeout_s=d.get("correlation_timeout_s"),
        )
        wf.validate()
        return wf


@dataclass
class ResponseRule:
    verdict: str
    response: Response
    severity: str = "any"
    min_confidence: float = 0.0
    max_confidence: float = 1.0
    retain_evidence: bool = False
    route: Route = Route.SOC_SECURITY

    def matches(
        self, verdict: Verdict, severity: Optional[Severity], confidence: float
    ) -> bool:
        if self.verdict != "any" and self.verdict != verdict.value:
            return False
        if self.severity != "any":
            if severity is None or self.severity != severity.value:
                return False
        return self.min_confidence <= confidence <= self.max_confidence

    @staticmethod
    def from_dict(d: dict) -> "ResponseRule":
        return ResponseRule(
            verdict=d["verdict"],
            response=Response(d["response"]),
            severity=d.get("severity", "any"),
            min_confidence=float(d.get("min_confidence", 0.0)),
            max_confidence=float(d.get("max_confidence", 1.0)),
            retain_evidence=bool(d.get("retain_evidence", False)),
            route=Route(d.get("route", "soc_security")),
        )


@dataclass
class Policy:
    policy_id: str
    version: str
    site_id: str
    mode: str
    zones: dict[str, Zone]
    workflows: list[Workflow]
    response_matrix: list[ResponseRule]
    calibration: dict = field(default_factory=dict)

    @property
    def shadow(self) -> bool:
        """Every site starts here. Switching to enforce is a deliberate,
        recorded commissioning decision made only after the shadow log has been
        reviewed against reality."""
        return self.mode == "shadow"

    def resolve(
        self, verdict: Verdict, severity: Optional[Severity], confidence: float
    ) -> ResponseRule:
        """First matching row wins, so order specific rules before general."""
        for rule in self.response_matrix:
            if rule.matches(verdict, severity, confidence):
                return rule
        # No row matched. Fail loud rather than silent: an unmatched finding
        # must not disappear.
        return ResponseRule(
            verdict=verdict.value,
            response=Response.LOG_AND_QUEUE_REVIEW,
            route=Route.REVIEW_QUEUE,
            retain_evidence=True,
        )

    @staticmethod
    def load(path: str | Path) -> "Policy":
        d = json.loads(Path(path).read_text())
        zones = {z["zone_id"]: Zone.from_dict(z) for z in d["zones"]}
        workflows = [Workflow.from_dict(w) for w in d["workflows"]]

        for w in workflows:
            for s in w.steps:
                if s.zone_id and s.zone_id not in zones:
                    raise PolicyError(
                        f"{w.workflow_id}/{s.step_id}: unknown zone '{s.zone_id}'"
                    )

        return Policy(
            policy_id=d["policy_id"],
            version=d["version"],
            site_id=d["site_id"],
            mode=d["mode"],
            zones=zones,
            workflows=workflows,
            response_matrix=[ResponseRule.from_dict(r) for r in d["response_matrix"]],
            calibration=d.get("calibration", {}),
        )
