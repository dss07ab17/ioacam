"""Policy linter.

Schema validation proves a policy file is well *formed*. It cannot prove the
policy is well *posed*. A policy can validate perfectly and still detect
nothing, and that failure is silent -- the system reports conformant operation
forever and nobody notices until an incident is missed.

Every check here comes from a mistake that actually shipped in this repo's own
example policy, or from a failure mode the engine's design makes possible.

Usage:
    python3 tools/lint_policy.py workflows/example_manufacturing_policy.json
    python3 tools/lint_policy.py workflows/*.json --strict

Exit code is non-zero if any ERROR is found, or any WARN when --strict is set.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ERROR, WARN, INFO = "ERROR", "WARN", "INFO"

# Observations that come from a bus read, a badge transaction or a timer.
# These arrive with confidence 1.0 by construction, so a confidence threshold
# on them is meaningless, and conversely a threshold of 1.0 on anything else
# is unreachable.
CERTAIN_OBSERVATIONS = {
    "robot_state",
    "robot_program",
    "machine_state",
    "door_state",
    "credential_presented",
    "sensor_health",
}

VALID_DEVIATIONS = {
    "skipped",
    "out_of_order",
    "overrun",
    "underrun",
    "wrong_role",
    "wrong_zone",
    "repeated",
    "incomplete",
}

VALID_SEVERITIES = {"informational", "warning", "critical", "safety_relevant"}

VALID_CORRELATION_ATTRS = {
    "track_id",
    "subject.identity",
    "subject.asset_id",
    "zone_id",
}


@dataclass
class Issue:
    level: str
    code: str
    where: str
    message: str
    why: str = ""

    def __str__(self) -> str:
        s = f"  [{self.level:<5}] {self.code:<22} {self.where}\n           {self.message}"
        if self.why:
            s += f"\n           why: {self.why}"
        return s


def _observation_enum() -> set[str]:
    """The closed observation list, read from the event schema itself so the
    linter cannot drift from the contract."""
    path = ROOT / "schema" / "event.schema.json"
    if not path.exists():
        return set()
    d = json.loads(path.read_text())
    return set(d["properties"]["observation"]["enum"])


def _evidence_signature(step: dict) -> tuple:
    """What this step's evidence looks like, ignoring confidence thresholds.

    Two steps in one workflow with the same signature cannot be told apart from
    the event stream. Whichever the engine offers the event to first wins, and
    the other becomes unreachable or fires spuriously.
    """
    return tuple(
        sorted(
            (e.get("observation"), json.dumps(e.get("value"), sort_keys=True))
            for e in step.get("evidence", [])
        )
    )


def lint(policy: dict, name: str = "") -> list[Issue]:
    issues: list[Issue] = []
    observations = _observation_enum()

    def add(level, code, where, message, why=""):
        issues.append(Issue(level, code, where, message, why))

    # ---------------------------------------------------------------
    # Mode and calibration
    # ---------------------------------------------------------------

    if policy.get("mode") == "enforce":
        add(
            INFO,
            "enforce-mode",
            name,
            "policy is in enforce mode",
            "confirm the shadow-mode log was reviewed against reality first; "
            "enforcing before that is how sites lose confidence in the system",
        )

    cal = policy.get("calibration") or {}
    if not cal:
        add(
            WARN,
            "no-calibration",
            name,
            "no calibration block",
            "confidence thresholds in the response matrix are arbitrary until "
            "the model's scores have been calibrated at this site",
        )
    elif cal.get("expected_calibration_error", 0) > 0.10:
        add(
            WARN,
            "poor-calibration",
            name,
            f"expected_calibration_error is {cal['expected_calibration_error']}",
            "when reported confidence does not match observed accuracy, every "
            "band in the response matrix means something other than intended",
        )

    # ---------------------------------------------------------------
    # Zones
    # ---------------------------------------------------------------

    zones = {z["zone_id"]: z for z in policy.get("zones", [])}

    for zid, z in zones.items():
        if not z.get("covered_by"):
            add(
                WARN,
                "zone-no-coverage",
                f"{name}:{zid}",
                "zone declares no covering sensors",
                "the engine cannot detect coverage loss for this zone, so it "
                "will keep reporting conformant even when nothing can see it",
            )
        if z.get("criticality") == "safety_critical":
            add(
                INFO,
                "safety-critical-zone",
                f"{name}:{zid}",
                "zone is safety_critical",
                "confirm an independent hard safety device protects this zone; "
                "iOACAM is supervisory and must not be the safety function",
            )
        if z.get("allowed_roles") == []:
            add(
                WARN,
                "zone-no-roles",
                f"{name}:{zid}",
                "allowed_roles is an empty list",
                "an empty list places no restriction at all, which is probably "
                "not what was meant; omit the key or list the roles",
            )

    # ---------------------------------------------------------------
    # Workflows and steps
    # ---------------------------------------------------------------

    for wf in policy.get("workflows", []):
        wid = wf["workflow_id"]
        steps = wf.get("steps", [])
        step_ids = [s["step_id"] for s in steps]

        # --- concurrency configuration --------------------------------
        corr = wf.get("correlation", [])
        cap = wf.get("max_concurrent_instances", 8)

        for attr in corr:
            if attr not in VALID_CORRELATION_ATTRS:
                add(
                    ERROR,
                    "unknown-correlation-attr",
                    f"{name}:{wid}",
                    f"correlation attribute '{attr}' is not recognised",
                    f"valid attributes are {sorted(VALID_CORRELATION_ATTRS)}",
                )

        wf_zones = {s.get("zone_id") for s in steps if s.get("zone_id")}
        crowded = [
            z
            for z in wf_zones
            if z in zones and (zones[z].get("max_occupancy") or 1) > 1
        ]
        if not corr and wf.get("actor_class") == "human" and crowded:
            add(
                WARN,
                "singleton-in-shared-zone",
                f"{name}:{wid}",
                f"no correlation declared, but {crowded} allow more than one person",
                "a second actor starting the same workflow will be folded into "
                "the first instance, so their steps interleave and produce "
                "deviations that nobody actually committed",
            )

        if corr:
            worst = max(
                (zones[z].get("max_occupancy") or 1) for z in crowded
            ) if crowded else 1
            if cap < worst:
                add(
                    WARN,
                    "capacity-below-occupancy",
                    f"{name}:{wid}",
                    f"max_concurrent_instances ({cap}) is below the zone's "
                    f"max_occupancy ({worst})",
                    "a legitimately full zone will hit the cap and further "
                    "triggers will be dropped during normal operation",
                )
            if "track_id" in corr and corr.index("track_id") == 0:
                add(
                    INFO,
                    "correlation-on-track-id",
                    f"{name}:{wid}",
                    "track_id is the primary correlation key",
                    "tracker ids swap when subjects cross and are reissued when "
                    "someone leaves and returns; prefer subject.identity first "
                    "where enrolment exists, with track_id as the fallback",
                )

        if wf.get("correlation_timeout_s") is not None:
            add(
                INFO,
                "correlation-timeout-set",
                f"{name}:{wid}",
                f"correlation_timeout_s is {wf['correlation_timeout_s']}",
                "only safe when the perception layer emits periodic liveness "
                "for active tracks; otherwise a legitimately long step is "
                "indistinguishable from a lost subject",
            )

        if not any(not s.get("predecessors") for s in steps):
            add(
                ERROR,
                "no-entry-step",
                f"{name}:{wid}",
                "every step has a predecessor, so none can ever start",
                "the workflow will trigger and then detect nothing at all",
            )

        # Evidence signatures must be distinctive within a workflow.
        by_sig: dict[tuple, list[str]] = defaultdict(list)
        for s in steps:
            by_sig[_evidence_signature(s)].append(s["step_id"])
        for sig, ids in by_sig.items():
            if len(ids) > 1:
                add(
                    ERROR,
                    "ambiguous-evidence",
                    f"{name}:{wid}:{'/'.join(ids)}",
                    f"steps {ids} are proved by identical evidence",
                    "the engine cannot tell these steps apart from the event "
                    "stream; one will absorb the other's evidence and the "
                    "second becomes unreachable or fires spuriously",
                )

        trigger_sig = (
            wf.get("trigger", {}).get("observation"),
            json.dumps(wf.get("trigger", {}).get("value"), sort_keys=True),
        )

        for s in steps:
            sid = s["step_id"]
            where = f"{name}:{wid}:{sid}"
            mn = s.get("min_duration_s")
            mx = s.get("max_duration_s")
            shared = s.get("duration_tolerance_s", 0)
            under = s.get("underrun_tolerance_s", shared)
            over = s.get("overrun_tolerance_s", shared)

            # The mistake that shipped in this repo's own example policy, and
            # then shipped twice more because only one step was fixed.
            if mn is not None and under >= mn:
                add(
                    ERROR,
                    "underrun-unreachable",
                    where,
                    f"underrun tolerance ({under}) >= min_duration_s ({mn})",
                    "every possible underrun falls inside the grace band, so "
                    "the underrun rule can never fire; the policy looks "
                    "complete and detects nothing",
                )

            if (
                "duration_tolerance_s" in s
                and mn is not None
                and mx is not None
                and "underrun_tolerance_s" not in s
                and "overrun_tolerance_s" not in s
            ):
                add(
                    WARN,
                    "shared-tolerance",
                    where,
                    "one duration_tolerance_s serves both bounds",
                    "a grace suited to a long maximum is usually absurd "
                    "against a short minimum; set underrun_tolerance_s and "
                    "overrun_tolerance_s separately",
                )

            if mn is not None and mx is not None and mn > mx:
                add(
                    ERROR,
                    "duration-inverted",
                    where,
                    f"min_duration_s ({mn}) > max_duration_s ({mx})",
                    "no duration can satisfy both, so the step is always a "
                    "deviation however it is performed",
                )

            if mx is None and not s.get("optional"):
                add(
                    ERROR,
                    "omission-undetectable",
                    where,
                    "no max_duration_s on a required step",
                    "a skipped step produces no observation, so the deadline "
                    "IS the detection; without one this step can be omitted "
                    "silently forever",
                )

            if mx is not None and over > mx:
                add(
                    WARN,
                    "tolerance-exceeds-max",
                    where,
                    f"overrun tolerance ({over}) > max_duration_s ({mx})",
                    "the grace band is larger than the step itself, which "
                    "more than doubles the time before an omission is caught",
                )

            # Severity declared for a deviation the step cannot produce.
            sev = s.get("deviation_severity") or {}
            for key, val in sev.items():
                if key not in VALID_DEVIATIONS:
                    add(
                        ERROR,
                        "unknown-deviation",
                        where,
                        f"deviation_severity key '{key}' is not a deviation type",
                        f"silently ignored at runtime and defaults to warning; "
                        f"valid keys are {sorted(VALID_DEVIATIONS)}",
                    )
                if val not in VALID_SEVERITIES:
                    add(
                        ERROR,
                        "unknown-severity",
                        where,
                        f"severity '{val}' for '{key}' is not valid",
                        f"valid severities are {sorted(VALID_SEVERITIES)}",
                    )
            if "underrun" in sev and mn is None:
                add(
                    WARN,
                    "dead-severity",
                    where,
                    "underrun severity declared but no min_duration_s",
                    "underrun can never be reported for this step",
                )
            if "overrun" in sev and mx is None:
                add(
                    WARN,
                    "dead-severity",
                    where,
                    "overrun severity declared but no max_duration_s",
                    "overrun can never be reported for this step",
                )
            if "out_of_order" in sev and not s.get("predecessors"):
                add(
                    WARN,
                    "dead-severity",
                    where,
                    "out_of_order severity declared but step has no predecessors",
                    "an entry step cannot be out of order",
                )
            if "wrong_role" in sev and not s.get("actor_role"):
                add(
                    WARN,
                    "dead-severity",
                    where,
                    "wrong_role severity declared but no actor_role",
                    "any role may perform this step, so wrong_role never fires",
                )

            if not s.get("evidence"):
                add(
                    ERROR,
                    "no-evidence",
                    where,
                    "step declares no evidence",
                    "it can never complete and will always report as skipped",
                )

            for e in s.get("evidence", []):
                obs = e.get("observation")
                if observations and obs not in observations:
                    add(
                        ERROR,
                        "unknown-observation",
                        where,
                        f"'{obs}' is not in the event schema's observation enum",
                        "no perception source will ever emit it, so this step "
                        "can never complete",
                    )
                mc = e.get("min_confidence", 0.5)
                if obs in CERTAIN_OBSERVATIONS and mc not in (1.0, 0.5):
                    add(
                        INFO,
                        "threshold-on-fact",
                        where,
                        f"min_confidence {mc} on '{obs}'",
                        "this observation is a bus or badge read and always "
                        "arrives at confidence 1.0; the threshold has no effect",
                    )
                if obs not in CERTAIN_OBSERVATIONS and mc >= 1.0:
                    add(
                        ERROR,
                        "threshold-unreachable",
                        where,
                        f"min_confidence 1.0 on inferred observation '{obs}'",
                        "calibrated perception confidence is never exactly 1.0, "
                        "so this evidence can never be accepted",
                    )
                if obs not in CERTAIN_OBSERVATIONS and mc < 0.4:
                    add(
                        WARN,
                        "threshold-permissive",
                        where,
                        f"min_confidence {mc} on inferred observation '{obs}'",
                        "accepts detections the model is barely confident in; "
                        "check this against the site's false-alarm budget",
                    )

                # A step proved solely by fine-grained action recognition is
                # the least dependable configuration available.
                preds = {p_: next((x for x in steps if x["step_id"] == p_), {})
                         for p_ in s.get("predecessors", [])}
                corroborated_by_telemetry = any(
                    pstep.get("evidence")
                    and all(
                        ev.get("observation") in CERTAIN_OBSERVATIONS
                        for ev in pstep["evidence"]
                    )
                    for pstep in preds.values()
                )
                if (
                    obs == "action_recognised"
                    and len(s.get("evidence", [])) == 1
                    and sev.get("skipped") in ("critical", "safety_relevant")
                    and not corroborated_by_telemetry
                ):
                    add(
                        WARN,
                        "critical-on-action-only",
                        where,
                        "critical step proved solely by action_recognised",
                        "fine-grained action recognition is the least reliable "
                        "evidence type; corroborate it with presence, zone, "
                        "object or bus state before treating a miss as critical",
                    )

            if s.get("zone_id") and s["zone_id"] not in zones:
                add(
                    ERROR,
                    "unknown-zone",
                    where,
                    f"zone '{s['zone_id']}' is not declared",
                    "",
                )

            if s.get("optional") and sev.get("skipped"):
                add(
                    INFO,
                    "optional-with-severity",
                    where,
                    "step is optional but declares a skipped severity",
                    "an optional step's absence is reported as tolerated, so "
                    "the severity has limited effect",
                )

            if _evidence_signature(s) == (
                (trigger_sig[0], trigger_sig[1]),
            ) and not s.get("predecessors"):
                add(
                    WARN,
                    "trigger-equals-step",
                    where,
                    "entry step evidence is identical to the workflow trigger",
                    "the triggering event will immediately complete this step, "
                    "so it measures nothing",
                )

        optional_count = sum(1 for s in steps if s.get("optional"))
        if steps and optional_count == len(steps):
            add(
                ERROR,
                "all-optional",
                f"{name}:{wid}",
                "every step is optional",
                "no absence is ever a violation, so the workflow detects nothing",
            )
        elif optional_count > len(steps) / 2:
            add(
                WARN,
                "mostly-optional",
                f"{name}:{wid}",
                f"{optional_count} of {len(steps)} steps are optional",
                "over-use of optional is how a workflow model quietly stops "
                "detecting anything",
            )

    # ---------------------------------------------------------------
    # Response matrix
    # ---------------------------------------------------------------

    matrix = policy.get("response_matrix", [])
    verdicts = ["conformant", "tolerated", "violation", "unknown"]
    severities = ["informational", "warning", "critical", "safety_relevant"]

    def _rows_for(v: str, s: str) -> list[dict]:
        out = []
        for r in matrix:
            if r["verdict"] not in (v, "any"):
                continue
            if r.get("severity", "any") not in (s, "any"):
                continue
            out.append(r)
        return out

    for v in verdicts:
        for s in severities:
            rows = _rows_for(v, s)
            if not rows:
                if v == "conformant":
                    continue
                add(
                    WARN,
                    "matrix-gap",
                    f"{name}:response_matrix",
                    f"no row matches verdict={v} severity={s}",
                    "such findings fall through to the log_and_queue_review "
                    "fallback rather than the response you intended",
                )
                continue

            # Confidence coverage: walk 0..1 and find uncovered gaps.
            covered: list[tuple[float, float]] = []
            for r in rows:
                covered.append(
                    (float(r.get("min_confidence", 0.0)), float(r.get("max_confidence", 1.0)))
                )
            covered.sort()
            cursor = 0.0
            for lo, hi in covered:
                if lo > cursor + 1e-9:
                    add(
                        WARN,
                        "matrix-confidence-gap",
                        f"{name}:response_matrix",
                        f"verdict={v} severity={s}: confidence "
                        f"[{cursor:.2f}, {lo:.2f}) has no row",
                        "findings in this band fall through to the fallback",
                    )
                cursor = max(cursor, hi)
            if cursor < 1.0 - 1e-9:
                add(
                    WARN,
                    "matrix-confidence-gap",
                    f"{name}:response_matrix",
                    f"verdict={v} severity={s}: confidence "
                    f"[{cursor:.2f}, 1.00] has no row",
                    "findings in this band fall through to the fallback",
                )

    # Rows made unreachable by an earlier, broader row.
    for i, row in enumerate(matrix):
        for earlier in matrix[:i]:
            if earlier["verdict"] not in (row["verdict"], "any"):
                continue
            if earlier.get("severity", "any") not in (row.get("severity", "any"), "any"):
                continue
            e_lo = float(earlier.get("min_confidence", 0.0))
            e_hi = float(earlier.get("max_confidence", 1.0))
            r_lo = float(row.get("min_confidence", 0.0))
            r_hi = float(row.get("max_confidence", 1.0))
            if e_lo <= r_lo and e_hi >= r_hi:
                add(
                    WARN,
                    "matrix-row-shadowed",
                    f"{name}:response_matrix[{i}]",
                    f"row {i} ({row['verdict']}/{row.get('severity','any')}) is "
                    f"fully covered by an earlier row",
                    "first match wins, so this row can never be selected; "
                    "order specific rules before general ones",
                )
                break

    # Safety-critical zones need somewhere for their findings to go.
    has_safety_zone = any(
        z.get("criticality") == "safety_critical" for z in policy.get("zones", [])
    )
    has_safety_row = any(
        r.get("severity") == "safety_relevant" for r in matrix
    )
    if has_safety_zone and not has_safety_row:
        add(
            ERROR,
            "no-safety-row",
            f"{name}:response_matrix",
            "a safety_critical zone exists but no row handles safety_relevant",
            "its violations fall through to the generic fallback instead of "
            "the safety route",
        )

    return issues


def lint_file(path: str | Path) -> list[Issue]:
    p = Path(path)
    return lint(json.loads(p.read_text()), p.stem)


def main(argv: list[str]) -> int:
    strict = "--strict" in argv
    paths: list[str] = [a for a in argv if not a.startswith("-")]
    if not paths:
        paths = [str(ROOT / "workflows" / "example_manufacturing_policy.json")]

    total = {ERROR: 0, WARN: 0, INFO: 0}
    for path in paths:
        issues = lint_file(path)
        print(f"\n{path}")
        if not issues:
            print("  clean")
        for issue in issues:
            print(issue)
            total[issue.level] += 1

    print(
        f"\n{total[ERROR]} error(s), {total[WARN]} warning(s), {total[INFO]} note(s)"
    )
    if total[ERROR]:
        return 1
    if strict and total[WARN]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
