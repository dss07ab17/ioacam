"""Scenario harness.

Feeds scripted event sequences to the engine and checks the findings. No
cameras, no models, no hardware. Times in scenario files are written in
seconds (`t_s`) purely for readability; the engine works in microseconds
throughout.

Scenario file:

    {
      "name": "skipped_step",
      "description": "...",
      "policy": "workflows/example_manufacturing_policy.json",
      "mode_override": "enforce",        # optional
      "final_tick_s": 120,               # advance the clock to fire deadlines
      "flush": false,                    # also close any still-open instance
      "events": [ { "t_s": 0, "source": "camera", ... } ],
      "expect":  [ { "verdict": "violation", "step_id": "s3", ... } ],
      "forbid":  [ { "verdict": "violation" } ]
    }

An `expect` entry is a partial match: only the keys present are compared, so a
test states what it cares about and stays readable. `conf_min` / `conf_max`
bound the confidence rather than demanding an exact float.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine import Event, Finding, Policy, WorkflowEngine  # noqa: E402
from engine.model import US_PER_S  # noqa: E402


@dataclass
class ScenarioResult:
    name: str
    description: str
    findings: list[Finding]
    failures: list[str]

    @property
    def passed(self) -> bool:
        return not self.failures


def _event_from_scenario(raw: dict, index: int) -> Event:
    d = dict(raw)
    if "t_s" in d:
        d["timestamp_us"] = int(round(float(d.pop("t_s")) * US_PER_S))
    d.setdefault("event_id", f"ev-{index:04d}")
    return Event.from_dict(d)


def _matches(finding: Finding, want: dict[str, Any]) -> bool:
    got = finding.to_dict()
    for key, expected in want.items():
        if key == "conf_min":
            if finding.confidence < expected:
                return False
        elif key == "conf_max":
            if finding.confidence > expected:
                return False
        elif key == "detail_contains":
            if expected.lower() not in finding.detail.lower():
                return False
        else:
            if got.get(key) != expected:
                return False
    return True


def run_scenario(path: str | Path) -> ScenarioResult:
    spec = json.loads(Path(path).read_text())

    policy_path = ROOT / spec.get(
        "policy", "workflows/example_manufacturing_policy.json"
    )
    policy = Policy.load(policy_path)
    if "mode_override" in spec:
        policy.mode = spec["mode_override"]

    engine = WorkflowEngine(policy)

    findings: list[Finding] = []
    for i, raw in enumerate(spec.get("events", [])):
        findings += engine.ingest(_event_from_scenario(raw, i))

    if "final_tick_s" in spec:
        findings += engine.tick(int(round(float(spec["final_tick_s"]) * US_PER_S)))
    if spec.get("flush"):
        findings += engine.flush()

    failures: list[str] = []
    for want in spec.get("expect", []):
        if not any(_matches(f, want) for f in findings):
            failures.append(f"expected but not found: {want}")
    for forbidden in spec.get("forbid", []):
        hits = [f for f in findings if _matches(f, forbidden)]
        if hits:
            failures.append(
                f"forbidden but present: {forbidden} -> {hits[0].detail}"
            )

    return ScenarioResult(
        name=spec.get("name", Path(path).stem),
        description=spec.get("description", ""),
        findings=findings,
        failures=failures,
    )


def run_all(directory: Optional[str | Path] = None, verbose: bool = False) -> int:
    d = Path(directory) if directory else ROOT / "harness" / "scenarios"
    paths = sorted(d.glob("*.json"))
    if not paths:
        print(f"no scenarios found in {d}")
        return 1

    passed = 0
    for p in paths:
        result = run_scenario(p)
        mark = "PASS" if result.passed else "FAIL"
        print(f"[{mark}] {result.name:<26} {result.description}")
        if verbose or not result.passed:
            for f in result.findings:
                print(f"          {f}")
        for msg in result.failures:
            print(f"       !! {msg}")
        passed += 1 if result.passed else 0

    print(f"\n{passed}/{len(paths)} scenarios passed")
    return 0 if passed == len(paths) else 1


if __name__ == "__main__":
    import sys

    verbose = "-v" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    raise SystemExit(run_all(args[0] if args else None, verbose=verbose))
