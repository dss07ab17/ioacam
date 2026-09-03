# Development Guide

## Project Structure

```
engine/              Workflow validation logic (no external dependencies, stdlib only)
  engine.py         Main workflow engine class
  instance.py       Step lifecycle state machine
  model.py          Enums, Event, Finding data classes
  policy.py         Policy loading and validation

perception/         Camera detection layer (separate process)
  perceive.py       Main perception entry point
  tracking.py       Person tracking across frames
  association.py    Wrist-in-box object-to-person attribution
  zones.py          Zone membership checking
  confidence.py     Confidence score calibration
  emit.py           Event emission
  detectors/        Pluggable detector implementations
    base.py         Detection + Detector protocol
    yolox_onnx.py   Apache-2.0 default (cv2.dnn)
    ultralytics_yolo.py  AGPL-3.0, evaluation only
    stub.py         Scripted boxes for tests
  actions/          Pose tubes, RTMPose, PoseC3D (see actions/README.md)
  tools/            fetch_model.py, define_zone.py, validate_events.py,
                    preview_pose.py, export_rtmpose.py, export_posec3d.py
  test_perception.py
  test_objects.py

identity/           Badge roster + face-matcher seam (tests use StubMatcher)
  roster.py matcher.py resolver.py test_identity.py

harness/            Comprehensive test suite
  runner.py         Scenario test runner
  test_engine.py    Property-based tests
  test_lint.py      Policy linter tests
  scenarios/        19 JSON test scenarios

schema/             JSON contracts
  event.schema.json      Perception → Engine event format
  workflow.schema.json   Policy declaration format

workflows/          Declared policies
  example_manufacturing_policy.json
  pick_and_place_policy.json

tools/              Policy and checkpoint utilities
  lint_policy.py           Policy validation and linting
  inspect_checkpoint.py    Read a .pth input contract without torch
```

There is no `setup.py` / `pyproject.toml`. Install with
`pip install -r requirements.txt` from the repo root. Optional packages
(`onnxruntime`, `torch`, `onnx`) stay commented until you need pose export
or the faster ONNX Runtime path.

## Running Tests

### Quick Test Suite

```bash
# Run all 19 scenarios (should see all pass)
python3 harness/runner.py

# Verbose output with detailed findings
python3 harness/runner.py -v
```

`harness/runner.py` takes an optional directory of `*.json` files, not a
single scenario filename.

### Property-Based Tests

Tests tick-invariance across every file in `harness/scenarios/`:

```bash
python3 harness/test_engine.py
```

These verify that the engine produces consistent results regardless of when
`tick()` is called.

### Linter Tests

```bash
python3 harness/test_lint.py
```

This suite exercises `tools/lint_policy.py`. It does **not** check
perception/engine import isolation.

### Other suites

```bash
python3 identity/test_identity.py
python3 perception/test_perception.py
python3 perception/test_objects.py
python3 perception/actions/test_actions.py
python3 perception/actions/test_rtmpose.py
python3 perception/actions/test_posec3d.py
```

`perception/test_perception.py` parses every module under `perception/` and
fails if anything imports `engine`.

### Policy Validation

```bash
# Check a specific policy file
python3 tools/lint_policy.py workflows/example_manufacturing_policy.json

# Strict validation (fails on warnings). Bash expands the glob; PowerShell does not.
python3 tools/lint_policy.py workflows/*.json --strict
python tools/lint_policy.py workflows/example_manufacturing_policy.json workflows/pick_and_place_policy.json --strict
```

## Key Design Principles to Preserve

When extending the engine:

1. **Separation of concerns**: Perception never judges; engine never sees pixels
2. **Three independent quantities**: Verdict, severity, and confidence are computed separately
3. **Open steps get first claim**: Earlier steps have priority on evidence (prevents false out-of-order violations)
4. **Weakest link principle**: Confidence is the minimum of all evidence components
5. **Failed steps release successors**: Prevents cascading false violations
6. **Strict policy validation**: Malformed workflows fail at load time (cycles, dangling predecessors)

## Extending the Engine

### Adding a New Observation Type

1. Add the name to the closed `observation` enum in `schema/event.schema.json`.

2. `Event.observation` is a string, not an engine-side enum. Handle the new
   value in `engine/engine.py` (`ingest` / `tick`) where the rules need it.

3. Add a test scenario to `harness/scenarios/`.

4. Update linter checks in `tools/lint_policy.py` if the observation can appear
   in a policy evidence list.

### Adding a New Detector

1. Implement the `Detector` protocol in `perception/detectors/` (`name`,
   `licence`, `detect(frame) -> Sequence[Detection]`). Return `Detection`
   objects (`x1, y1, x2, y2, score`), not raw tuples.

2. Register the backend name in `perception/detectors/__init__.py`
   (`BACKENDS` and `build_detector`). `perceive.py` already calls
   `build_detector`; it should not hard-code a new class.

3. Test with the stub path or real frames. `perception/test_perception.py`
   runs the real CLI against a synthetic video and the stub detector.

### Adding a New Test Scenario

1. Create JSON scenario in `harness/scenarios/NN_description.json`
2. Define events in chronological order with expected findings
3. Run: `python3 harness/runner.py` to validate
4. Update `harness/test_lint.py` if adding new policy validation checks

## Debugging Workflow Violations

### Tracing a Scenario

```bash
python3 harness/runner.py -v
```

### Inspecting Engine State

In `engine/engine.py`, the state is tracked in:
- `self.instances` — active workflow instances
- `self.occupancy` — current zone occupancy
- `self.sensor_health` — sensor status

### Understanding Findings

A Finding contains:
- `verdict` — `conformant` / `tolerated` / `violation` / `unknown`
- `severity` — authored (`informational` / `warning` / `critical` / `safety_relevant`)
- `confidence` — perception certainty (0.0–1.0)
- `response` — what the response matrix selected (`none`, `log_only`,
  `log_and_queue_review`, `log_and_notify_operator`, `log_notify_soc`,
  `alarm_and_escalate_soc`, `alarm_escalate_and_request_mitigation`)
- `route` — `soc_security` / `safety_emergency` / `maintenance` / `review_queue`

The response matrix in the policy maps (verdict, severity, confidence) →
`response` and `route`.

## Policy Development Workflow

1. **Define zones** using `perception/tools/define_zone.py`
2. **Create workflow policy** with steps, predecessors, evidence requirements
3. **Lint the policy**: `python3 tools/lint_policy.py workflows/your_policy.json --strict`
4. **Test with scenarios**: Create synthetic event sequences in `harness/scenarios/`
5. **Validate live emission** against `schema/event.schema.json` with
   `perception/tools/validate_events.py`. There is no stock stdin→engine
   consumer; the engine is a library until transport exists.
6. **Calibrate confidence thresholds** based on real detections

## Performance Considerations

### Perception Layer
- Runs at camera framerate (typically 30 FPS on paper; YOLOX-tiny on a
  laptop CPU is closer to ~5 fps — debounce is in **frames**, not seconds)
- Emits only edge-triggered events (~dozen per 300 frames)
- Confidence calculation is deterministic and fast

### Engine
- Processes events in O(N) where N = number of active instances
- tick() must be called periodically to detect omitted steps
- On-device: tick() runs from periodic timer (every few hundred ms)

### Optimization Tips
- Use YOLOX tiny for real-time inference on edge devices
- Reduce detection confidence thresholds to catch more violations (but more false positives)
- Use sensor_health events to skip perception when cameras are offline

## Useful Commands

```bash
# Dependencies (no editable package)
pip install -r requirements.txt

# Format / typecheck / lint (listed in requirements.txt)
black engine/ perception/ harness/ tools/ identity/
mypy engine/ --strict
flake8 engine/ perception/ harness/ tools/ identity/

# Run a single scenario by path
python3 -c "
from harness.runner import run_scenario
r = run_scenario('harness/scenarios/01_clean_run.json')
print(r.name, 'PASS' if r.passed else 'FAIL')
for f in r.findings:
    print(f)
"
```

## Troubleshooting Development

### Import Errors

Ensure the workspace root is in PYTHONPATH:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python3 harness/runner.py
```

On PowerShell: `$env:PYTHONPATH = "$(Get-Location);$env:PYTHONPATH"`.
Most of the test scripts insert the repo root themselves.

### Perception and Engine Don't Separate

Both modules are intentionally independent. Check that:
- Perception imports nothing from `engine/` (enforced by
  `perception/test_perception.py`)
- Engine imports nothing from `perception/`
- Identity is a third package; keep the same rule unless you deliberately
  change the contract

### Policy Won't Load

Run the strict linter:
```bash
python3 tools/lint_policy.py workflows/your_policy.json --strict
```

Common issues:
- Dangling predecessor references (step doesn't exist)
- Cycles in step dependencies
- Missing required fields
- Invalid zone references

## Contributing

When making changes:
1. Run the harness: `python3 harness/runner.py`
2. Run the other suites you touched (identity, perception, actions)
3. Lint policies if you modified workflows: `python3 tools/lint_policy.py workflows/*.json --strict`
4. Ensure separation of concerns (no cross-layer imports)
5. Add scenarios for new observation types
6. Update this guide if adding new development workflows
