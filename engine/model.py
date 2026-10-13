"""Core data types for the iOACAM workflow engine.

Three quantities are kept deliberately separate throughout this module:

  verdict     - from the rules, comparing observation against the declaration
  severity    - authored in the policy file, fixed per step per deviation type
  confidence  - from perception only, computed at runtime

Nothing about verdict or severity may influence confidence. Conflating them is
the design error this separation exists to prevent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


US_PER_S = 1_000_000


class Verdict(str, Enum):
    """How the observation compares to the declaration."""

    CONFORMANT = "conformant"
    TOLERATED = "tolerated"
    VIOLATION = "violation"
    UNKNOWN = "unknown"


class Severity(str, Enum):
    """Authored in the policy, never computed at runtime."""

    INFORMATIONAL = "informational"
    WARNING = "warning"
    CRITICAL = "critical"
    SAFETY_RELEVANT = "safety_relevant"


class Deviation(str, Enum):
    """The ways a step can go wrong. Each maps to an authored severity."""

    SKIPPED = "skipped"
    OUT_OF_ORDER = "out_of_order"
    OVERRUN = "overrun"
    UNDERRUN = "underrun"
    WRONG_ROLE = "wrong_role"
    WRONG_ZONE = "wrong_zone"
    REPEATED = "repeated"
    INCOMPLETE = "incomplete"


class Response(str, Enum):
    """Mitigation is only ever REQUESTED. iOACAM never actuates."""

    NONE = "none"
    LOG_ONLY = "log_only"
    LOG_AND_QUEUE_REVIEW = "log_and_queue_review"
    LOG_AND_NOTIFY_OPERATOR = "log_and_notify_operator"
    LOG_NOTIFY_SOC = "log_notify_soc"
    ALARM_AND_ESCALATE_SOC = "alarm_and_escalate_soc"
    ALARM_ESCALATE_AND_REQUEST_MITIGATION = "alarm_escalate_and_request_mitigation"


class Route(str, Enum):
    SOC_SECURITY = "soc_security"
    SAFETY_EMERGENCY = "safety_emergency"
    MAINTENANCE = "maintenance"
    REVIEW_QUEUE = "review_queue"


# Sources whose events are read or computed facts rather than inferences.
# Events from these carry confidence 1.0 by construction.
CERTAIN_SOURCES = {"robot_bus", "machine_bus", "access_control", "timer", "engine"}


@dataclass
class Subject:
    """Who or what the observation is about. Unknown fields stay None."""

    cls: Optional[str] = None
    identity: Optional[str] = None
    role: Optional[str] = None
    asset_id: Optional[str] = None

    @staticmethod
    def from_dict(d: Optional[dict]) -> "Subject":
        if not d:
            return Subject()
        return Subject(
            cls=d.get("class"),
            identity=d.get("identity"),
            role=d.get("role"),
            asset_id=d.get("asset_id"),
        )


@dataclass
class Event:
    """One observation, from any perception or telemetry source.

    This is the only thing the engine consumes. It never sees pixels, models
    or sensor internals.
    """

    event_id: str
    timestamp_us: int
    source: str
    observation: str
    confidence: float
    sensor_id: Optional[str] = None
    track_id: Optional[str] = None
    zone_id: Optional[str] = None
    value: Any = None
    unit: Optional[str] = None
    subject: Subject = field(default_factory=Subject)
    wall_time: Optional[str] = None

    @staticmethod
    def from_dict(d: dict) -> "Event":
        source = d["source"]
        confidence = d.get("confidence")
        if confidence is None:
            # A fact from a bus, badge or timer carries no perception
            # uncertainty. Defaulting it here stops scenario files having to
            # restate 1.0 on every telemetry event.
            confidence = 1.0 if source in CERTAIN_SOURCES else 0.0
        return Event(
            event_id=d["event_id"],
            timestamp_us=int(d["timestamp_us"]),
            source=source,
            observation=d["observation"],
            confidence=float(confidence),
            sensor_id=d.get("sensor_id"),
            track_id=d.get("track_id"),
            zone_id=d.get("zone_id"),
            value=d.get("value"),
            unit=d.get("unit"),
            subject=Subject.from_dict(d.get("subject")),
            wall_time=d.get("wall_time"),
        )

    def is_certain(self) -> bool:
        return self.source in CERTAIN_SOURCES


@dataclass
class Finding:
    """The engine's output: a judgement, not an action.

    `response` is what the response matrix selected. `suppressed` is True in
    shadow mode, meaning the response was computed and logged but nothing was
    emitted to the SOC, no alarm sounded and no mitigation requested.
    """

    timestamp_us: int
    verdict: Verdict
    confidence: float
    response: Response
    route: Route
    severity: Optional[Severity] = None
    deviation: Optional[Deviation] = None
    workflow_id: Optional[str] = None
    instance_id: Optional[str] = None
    step_id: Optional[str] = None
    zone_id: Optional[str] = None
    subject: Subject = field(default_factory=Subject)
    detail: str = ""
    retain_evidence: bool = False
    suppressed: bool = False
    policy_version: str = ""
    triggering_event_id: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "timestamp_us": self.timestamp_us,
            "verdict": self.verdict.value,
            "severity": self.severity.value if self.severity else None,
            "deviation": self.deviation.value if self.deviation else None,
            "confidence": round(self.confidence, 4),
            "response": self.response.value,
            "route": self.route.value,
            "workflow_id": self.workflow_id,
            "instance_id": self.instance_id,
            "step_id": self.step_id,
            "zone_id": self.zone_id,
            "subject": {
                k: v
                for k, v in {
                    "class": self.subject.cls,
                    "identity": self.subject.identity,
                    "role": self.subject.role,
                    "asset_id": self.subject.asset_id,
                }.items()
                if v is not None
            },
            "detail": self.detail,
            "retain_evidence": self.retain_evidence,
            "suppressed": self.suppressed,
            "policy_version": self.policy_version,
            "triggering_event_id": self.triggering_event_id,
        }

    def __str__(self) -> str:
        t = self.timestamp_us / US_PER_S
        sev = self.severity.value if self.severity else "-"
        dev = self.deviation.value if self.deviation else "-"
        tag = " [SHADOW]" if self.suppressed else ""
        return (
            f"[{t:7.1f}s] {self.verdict.value:<10} {sev:<15} {dev:<13} "
            f"conf={self.confidence:.2f} -> {self.response.value}{tag}  {self.detail}"
        )
