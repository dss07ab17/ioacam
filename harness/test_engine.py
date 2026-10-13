"""Property tests that the scenario files cannot express.

Run with:  python3 harness/test_engine.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine import Policy, PolicyError, WorkflowEngine  # noqa: E402
from engine.model import US_PER_S, Verdict  # noqa: E402
from harness.runner import _event_from_scenario  # noqa: E402

POLICY = ROOT / "workflows" / "example_manufacturing_policy.json"
SCENARIOS = ROOT / "harness" / "scenarios"

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"[{'PASS' if condition else 'FAIL'}] {name}")
    if not condition:
        failures.append(f"{name}: {detail}")


def _run(spec: dict, tick_step_s: float | None) -> list[dict]:
    """Run a scenario, optionally ticking at a fixed cadence rather than
    jumping straight to the end."""
    policy = Policy.load(POLICY)
    if "mode_override" in spec:
        policy.mode = spec["mode_override"]
    engine = WorkflowEngine(policy)

    findings = []
    for i, raw in enumerate(spec.get("events", [])):
        findings += engine.ingest(_event_from_scenario(raw, i))

    end_us = int(round(float(spec.get("final_tick_s", 0)) * US_PER_S))
    if tick_step_s is None:
        findings += engine.tick(end_us)
    else:
        step = int(tick_step_s * US_PER_S)
        t = engine.now_us
        while t < end_us:
            t = min(t + step, end_us)
            findings += engine.tick(t)
    if spec.get("flush"):
        findings += engine.flush()

    # Drop the policy version so the comparison is about behaviour, not
    # bookkeeping.
    return [
        {k: v for k, v in f.to_dict().items() if k != "policy_version"}
        for f in findings
    ]


# ----------------------------------------------------------------------
# 1. Tick granularity must not change the verdicts.
#
# The device ticks every few hundred milliseconds; the harness jumps minutes.
# If those disagree, every duration measurement is a function of scheduler
# timing, and a site's tuning would not survive a change in tick rate.
# ----------------------------------------------------------------------

for path in sorted(SCENARIOS.glob("*.json")):
    spec = json.loads(path.read_text())
    coarse = _run(spec, tick_step_s=None)
    fine = _run(spec, tick_step_s=0.5)
    check(
        f"tick invariance: {path.stem}",
        coarse == fine,
        f"coarse={len(coarse)} findings, fine={len(fine)}",
    )


# ----------------------------------------------------------------------
# 2. A malformed policy must fail at load, not at runtime.
#
# A dangling predecessor leaves a step permanently PENDING and a cycle leaves
# a whole branch unreachable. Either way the workflow silently stops detecting
# anything, which is the worst possible failure for this product.
# ----------------------------------------------------------------------

base = json.loads(POLICY.read_text())


def _load_mutated(mutate) -> Exception | None:
    d = json.loads(json.dumps(base))
    mutate(d)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(d, fh)
        tmp = fh.name
    try:
        Policy.load(tmp)
        return None
    except Exception as exc:  # noqa: BLE001 - we want the type back
        return exc
    finally:
        Path(tmp).unlink(missing_ok=True)


err = _load_mutated(
    lambda d: d["workflows"][0]["steps"][2].__setitem__("predecessors", ["s99"])
)
check("dangling predecessor rejected", isinstance(err, PolicyError), repr(err))


def _make_cycle(d):
    steps = d["workflows"][0]["steps"]
    steps[1]["predecessors"] = ["s3"]
    steps[2]["predecessors"] = ["s2"]


check("predecessor cycle rejected", isinstance(_load_mutated(_make_cycle), PolicyError))

err = _load_mutated(
    lambda d: d["workflows"][0]["steps"][0].__setitem__("zone_id", "zone-nowhere")
)
check("unknown zone rejected", isinstance(err, PolicyError), repr(err))


# ----------------------------------------------------------------------
# 3. Verdict and severity must not leak into confidence.
#
# Confidence is a property of perception alone. If changing an authored
# severity moves a confidence value, the two have been conflated and the
# response matrix stops meaning what it says.
# ----------------------------------------------------------------------

spec = json.loads((SCENARIOS / "05_underrun.json").read_text())
before = _run(spec, None)

policy_d = json.loads(POLICY.read_text())
policy_d["workflows"][0]["steps"][2]["deviation_severity"]["underrun"] = "informational"
with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
    json.dump(policy_d, fh)
    alt = fh.name

policy = Policy.load(alt)
engine = WorkflowEngine(policy)
after = []
for i, raw in enumerate(spec["events"]):
    after += engine.ingest(_event_from_scenario(raw, i))
after += engine.tick(int(spec["final_tick_s"] * US_PER_S))
Path(alt).unlink(missing_ok=True)

conf_before = [f["confidence"] for f in before]
conf_after = [f.confidence for f in after]
sev_changed = any(f.severity and f.severity.value == "informational" for f in after)
check(
    "severity change does not move confidence",
    conf_before == conf_after and sev_changed,
    f"{conf_before} vs {conf_after}, severity changed={sev_changed}",
)


# ----------------------------------------------------------------------
# 4. Every finding must resolve to a response.
#
# An unmatched finding must never vanish. The fallback exists so a gap in the
# response matrix surfaces as a review item rather than as silence.
# ----------------------------------------------------------------------

policy = Policy.load(POLICY)
policy.response_matrix = []
resolved = policy.resolve(Verdict.VIOLATION, None, 0.9)
check(
    "empty response matrix still resolves",
    resolved.response.value == "log_and_queue_review",
    resolved.response.value,
)


print()
if failures:
    print(f"{len(failures)} property test(s) failed")
    for f in failures:
        print(f"  !! {f}")
    raise SystemExit(1)
print("all property tests passed")
