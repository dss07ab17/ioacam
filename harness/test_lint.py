"""Tests for the policy linter.

A linter that stops firing is worse than no linter, because the clean run
becomes evidence of correctness. Each check below is verified against a policy
deliberately broken in exactly that way.

Run with:  python3 harness/test_lint.py
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.lint_policy import lint  # noqa: E402

BASE = json.loads(
    (ROOT / "workflows" / "example_manufacturing_policy.json").read_text()
)

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"[{'PASS' if condition else 'FAIL'}] {name}")
    if not condition:
        failures.append(f"{name}: {detail}")


def codes(mutate) -> set[str]:
    d = copy.deepcopy(BASE)
    mutate(d)
    return {i.code for i in lint(d, "test")}


def catches(name: str, code: str, mutate) -> None:
    found = codes(mutate)
    check(f"catches {name}", code in found, f"got {sorted(found)}")


# The clean policy must stay clean, otherwise every real finding is noise.
baseline = {i.code for i in lint(copy.deepcopy(BASE), "base") if i.level != "INFO"}
check("baseline policy is clean", not baseline, f"got {sorted(baseline)}")


def _s(d, i=0, step=2):
    return d["workflows"][i]["steps"][step]


catches(
    "underrun made unreachable by tolerance",
    "underrun-unreachable",
    lambda d: _s(d).update(underrun_tolerance_s=99),
)

catches(
    "inverted duration band",
    "duration-inverted",
    lambda d: _s(d).update(min_duration_s=100, max_duration_s=10),
)

catches(
    "required step with no deadline",
    "omission-undetectable",
    lambda d: _s(d).pop("max_duration_s"),
)

catches(
    "step with no evidence",
    "no-evidence",
    lambda d: _s(d).update(evidence=[]),
)

catches(
    "observation outside the schema enum",
    "unknown-observation",
    lambda d: _s(d).update(evidence=[{"observation": "vibes_detected"}]),
)

catches(
    "unreachable confidence threshold on an inferred observation",
    "threshold-unreachable",
    lambda d: _s(d, step=3).update(
        evidence=[{"observation": "person_in_zone", "value": True, "min_confidence": 1.0}]
    ),
)

catches(
    "two steps proved by identical evidence",
    "ambiguous-evidence",
    lambda d: _s(d, step=4).update(evidence=copy.deepcopy(_s(d, step=1)["evidence"])),
)

catches(
    "typo in a deviation_severity key",
    "unknown-deviation",
    lambda d: _s(d).setdefault("deviation_severity", {}).update(skiped="critical"),
)

catches(
    "severity declared for a deviation that cannot occur",
    "dead-severity",
    lambda d: (
        _s(d).pop("min_duration_s"),
        _s(d)["deviation_severity"].update(underrun="critical"),
    ),
)

catches(
    "workflow where every step has a predecessor",
    "no-entry-step",
    lambda d: d["workflows"][0]["steps"][0].update(predecessors=["s5"]),
)

catches(
    "every step optional",
    "all-optional",
    lambda d: [s.update(optional=True) for s in d["workflows"][0]["steps"]],
)

catches(
    "zone with no covering sensor",
    "zone-no-coverage",
    lambda d: d["zones"][0].pop("covered_by"),
)

catches(
    "safety-critical zone with no matching response row",
    "no-safety-row",
    lambda d: d.__setitem__(
        "response_matrix",
        [r for r in d["response_matrix"] if r.get("severity") != "safety_relevant"],
    ),
)

catches(
    "response row shadowed by an earlier broader row",
    "matrix-row-shadowed",
    lambda d: d["response_matrix"].insert(
        0, {"verdict": "violation", "response": "log_only"}
    ),
)

catches(
    "confidence band with no response row",
    "matrix-confidence-gap",
    lambda d: d.__setitem__(
        "response_matrix",
        [
            r
            for r in d["response_matrix"]
            if not (r["verdict"] == "violation" and r.get("severity") == "critical"
                    and r.get("max_confidence", 1.0) <= 0.5)
        ],
    ),
)

catches(
    "one tolerance serving both bounds",
    "shared-tolerance",
    lambda d: (
        _s(d).pop("underrun_tolerance_s", None),
        _s(d).pop("overrun_tolerance_s", None),
        _s(d).update(duration_tolerance_s=2),
    ),
)

catches(
    "poor calibration",
    "poor-calibration",
    lambda d: d["calibration"].update(expected_calibration_error=0.35),
)

catches(
    "missing calibration",
    "no-calibration",
    lambda d: d.pop("calibration"),
)


catches(
    "human workflow with no correlation in a shared zone",
    "singleton-in-shared-zone",
    lambda d: d["workflows"][0].pop("correlation"),
)

catches(
    "unrecognised correlation attribute",
    "unknown-correlation-attr",
    lambda d: d["workflows"][0].__setitem__("correlation", ["subject.mood"]),
)

catches(
    "instance cap below the zone's occupancy",
    "capacity-below-occupancy",
    lambda d: d["workflows"][0].__setitem__("max_concurrent_instances", 1),
)

catches(
    "track_id used as the primary correlation key",
    "correlation-on-track-id",
    lambda d: d["workflows"][0].__setitem__("correlation", ["track_id"]),
)


# The cross-attestation exemption: a vision step that exists precisely to
# corroborate a telemetry claim must NOT be flagged as action-only.
found = {i.code for i in lint(copy.deepcopy(BASE), "base")}
check(
    "cross-attestation step is exempt from action-only warning",
    "critical-on-action-only" not in found,
    f"got {sorted(found)}",
)

# But a critical step proved by action recognition with no telemetry
# predecessor must still be flagged.
catches(
    "critical step proved by action recognition alone",
    "critical-on-action-only",
    lambda d: _s(d, step=3).update(
        evidence=[{"observation": "action_recognised", "value": "inspection"}],
        predecessors=[],
    ),
)


print()
if failures:
    print(f"{len(failures)} lint test(s) failed")
    for f in failures:
        print(f"  !! {f}")
    raise SystemExit(1)
print("all lint tests passed")
