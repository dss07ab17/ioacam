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
  zones.py          Zone membership checking
  confidence.py     Confidence score calibration
  emit.py           Event emission
  detectors/        Pluggable detector implementations
    base.py         Base detector interface
    ultralytics_yolo.py
    yolox_onnx.py
  
harness/            Comprehensive test suite
  runner.py         Scenario test runner
  test_engine.py    Property-based tests
  test_lint.py      Policy linter tests
  scenarios/        12 JSON test scenarios
  
schema/             JSON contracts
  event.schema.json      Perception → Engine event format
  workflow.schema.json   Policy declaration format
  
tools/              Utilities
  lint_policy.py    Policy validation and linting
  define_zone.py    Interactive zone definition tool
  fetch_model.py    Model download utility
  validate_events.py Event validation tool
```

## Running Tests

### Quick Test Suite

```bash
# Run all scenarios (should see all pass)
python3 harness/runner.py

# Verbose output with detailed findings
python3 harness/runner.py -v
```

### Property-Based Tests

Tests tick-invariance across all 12 scenarios:

```bash
python3 harness/test_engine.py
```

These verify that the engine produces consistent results regardless of when `tick()` is called.

### Linter Tests

```bash
python3 harness/test_lint.py
```

### Policy Validation

```bash
# Check a specific policy file
python3 tools/lint_policy.py workflows/example_manufacturing_policy.json

# Strict validation (fails on warnings)
python3 tools/lint_policy.py workflows/*.json --strict
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

1. Add the observation to `schema/event.schema.json`:
   ```json
   {
     "type": "object",
     "properties": {
       "observation": {
         "enum": ["existing_observation", "new_observation"]
       }
     }
   }
   ```

2. Add the enum to `engine/model.py`

3. Add handling logic to `engine/engine.py` (ingest/tick methods)

4. Add test scenario to `harness/scenarios/`

5. Update linter checks in `tools/lint_policy.py`

### Adding a New Detector

1. Create a new detector class in `perception/detectors/`:
   ```python
   from .base import BaseDetector
   
   class MyDetector(BaseDetector):
       def detect(self, frame):
           # Return detections as (x1, y1, x2, y2, confidence)
           pass
   ```

2. Update `perception/perceive.py` to instantiate and use your detector

3. Test with real frames or test data

### Adding a New Test Scenario

1. Create JSON scenario in `harness/scenarios/NN_description.json`
2. Define events in chronological order with expected findings
3. Run: `python3 harness/runner.py` to validate
4. Update `harness/test_lint.py` if adding new policy validation checks

## Debugging Workflow Violations

### Tracing a Scenario

```bash
# Add debug logging in harness/runner.py
python3 harness/runner.py -v 2>&1 | grep "scenario_name"
```

### Inspecting Engine State

In `engine/engine.py`, the state is tracked in:
- `self.instances` — active workflow instances
- `self.occupancy` — current zone occupancy
- `self.sensor_health` — sensor status

### Understanding Findings

A Finding contains:
- `verdict` — what happened (violation/conformant/tolerated/unknown)
- `severity` — how serious (informational/warning/critical)
- `confidence` — perception certainty (0.0-1.0)
- `route` — where it should go (alarm/review/none)

The response matrix in the policy maps (verdict, severity) → action.

## Policy Development Workflow

1. **Define zones** using `perception/tools/define_zone.py`
2. **Create workflow policy** with steps, predecessors, evidence requirements
3. **Lint the policy**: `python3 tools/lint_policy.py workflows/your_policy.json --strict`
4. **Test with scenarios**: Create synthetic event sequences in `harness/scenarios/`
5. **Validate against live data**: Pipe perception → engine consumer with your policy
6. **Calibrate confidence thresholds** based on real detections

## Performance Considerations

### Perception Layer
- Runs at camera framerate (typically 30 FPS)
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
# Install in development mode
pip install -e .

# Format code
black engine/ perception/ harness/ tools/

# Type check
mypy engine/ --strict

# Lint
flake8 engine/ perception/ harness/ tools/

# Run a single scenario
python3 -c "
import json
from harness.runner import run_scenario

with open('harness/scenarios/01_clean_run.json') as f:
    scenario = json.load(f)
    run_scenario(scenario, verbose=True)
"
```

## Troubleshooting Development

### Import Errors

Ensure the workspace root is in PYTHONPATH:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python3 harness/runner.py
```

### Perception and Engine Don't Separate

Both modules are intentionally independent. Check that:
- Perception imports nothing from `engine/`
- Engine imports nothing from `perception/`
- The test in `harness/test_lint.py` enforces this

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
1. Run the full test suite: `python3 harness/runner.py`
2. Lint your policy if you modified workflows: `python3 tools/lint_policy.py workflows/*.json --strict`
3. Ensure separation of concerns (no cross-layer imports)
4. Add scenarios for new observation types
5. Update this guide if adding new development workflows
