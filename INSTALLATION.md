# Installation Guide

## Prerequisites

- **Python 3.10 or higher**
- **pip** package manager

On Windows, `python` is the usual interpreter name. The examples below use
`python3` as in the rest of the repo; substitute `python` if that is what
your venv provides. PowerShell does not expand `workflows/*.json` the way
bash does — pass each policy file explicitly, or run the commands from Git
Bash / WSL.

## Quick Start

### 1. Clone and Set Up Environment

```bash
# Clone the repository
git clone <repository-url>
cd ioacam

# Create a virtual environment (recommended)
python3 -m venv venv

# Activate the virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Set Up the Perception Layer (Optional, for camera-based detection)

The perception layer requires downloading a pre-trained model and configuring zones.

#### Download the YOLOX Model

```bash
# Fetch the YOLOX tiny model (Apache-2.0 licensed, ~20 MB)
python perception/tools/fetch_model.py
```

This downloads `perception/models/yolox_tiny.onnx` (the argparse default).
For the larger accuracy-reference weights:

```bash
python perception/tools/fetch_model.py --model yolox_s
```

Then point `detector.model_path` at `perception/models/yolox_s.onnx` in your
config. `onnxruntime` is not required for YOLOX: the default backend uses
OpenCV DNN.

#### Define Your Zones

```bash
# Interactive zone definition tool
python perception/tools/define_zone.py
```

This opens a GUI where you can:
1. Click on your camera feed to define zone boundaries
2. Copy the generated JSON configuration
3. Paste it into `perception/config/zones.example.json`

#### Test with Webcam (Optional)

Use the provided webcam configuration to quickly test:

```bash
# Preview the detection output
python perception/perceive.py --config perception/config/webcam.json --preview

# Emit events (without preview)
python perception/perceive.py --config perception/config/webcam.json
```

**Note:** The webcam config uses `y = 1.0` as the zone boundary because seated subjects are clipped by the frame bottom.

#### Validate Events

```bash
# Test perception layer with validation
python perception/perceive.py --max-frames 300 | python perception/tools/validate_events.py
```

### 3. Run the Engine and Tests

#### Run the Harness (Test Scenarios)

```bash
# Run all 19 scenarios
python3 harness/runner.py

# Run with verbose output (all findings printed)
python3 harness/runner.py -v
```

The runner treats an extra argument as a **directory** of JSON files, not a
single scenario path. To exercise one file, put it in a folder of its own
or call `run_scenario("harness/scenarios/01_clean_run.json")` from Python
(it takes a path, not a loaded dict).

#### Run Property Tests

```bash
# Property-based tests for tick invariance across every scenario file
python3 harness/test_engine.py
```

#### Run Linter Tests

```bash
# Test policy file validation
python3 harness/test_lint.py
```

#### Lint Policy Files

```bash
# Check workflow policies for correctness (bash / Git Bash)
python3 tools/lint_policy.py workflows/*.json --strict

# Same check with explicit paths (PowerShell)
python tools/lint_policy.py workflows/example_manufacturing_policy.json workflows/pick_and_place_policy.json --strict
```

#### Identity, perception, and action tests

These need no camera and no downloaded weights except where noted:

```bash
python3 identity/test_identity.py
python3 perception/test_perception.py
python3 perception/test_objects.py
python3 perception/actions/test_actions.py
python3 perception/actions/test_rtmpose.py
python3 perception/actions/test_posec3d.py
```

Pose/action **live** preview is optional. Uncomment `onnxruntime` in
`requirements.txt` (or rely on the slower `cv2.dnn` fallback), export
weights once with `perception/tools/export_rtmpose.py` and
`perception/tools/export_posec3d.py` (those need `torch`, also commented
in `requirements.txt`), then see `perception/actions/README.md`.

### 4. Full Workflow Integration

The engine is a **library**. It has no stdin consumer yet: findings are
produced in-process (the harness feeds JSON scenario files) and go nowhere
until a transport exists. Perception emits JSON lines on stdout for a
consumer you write:

```bash
python perception/perceive.py --config perception/config/webcam.json
python perception/perceive.py --config perception/config/webcam.json | python perception/tools/validate_events.py
```

On PowerShell, do not redirect stdout with `>`. That writes UTF-16, which
JSON Lines readers reject. Pipe to another process, or write the file from
Python with UTF-8. See the known issue in [README.md](README.md).

## Configuration

### Perception Configuration

- **Webcam config:** `perception/config/webcam.json` (pre-configured for laptop webcams)
- **Zone definitions:** `perception/config/zones.example.json`
- **Camera calibration:** Adjust `temperature` scaling in workflow policy (default: 1.8)

### Workflow Policy

- **Example policy:** `workflows/example_manufacturing_policy.json` (assembly station + robot cross-check; what the harness scenarios load)
- **Second example:** `workflows/pick_and_place_policy.json` (transfer-bay / pick-and-place)
- **Schema reference:** `schema/workflow.schema.json`

Edit the policy file to:
- Define zones and their access rules
- Specify workflow steps and evidence requirements
- Set confidence thresholds
- Configure response actions

### Event Schema

- **Event structure:** `schema/event.schema.json`
- All perception events conform to this JSON schema

## Troubleshooting

### Model Download Fails

```bash
# Manually download YOLOX model
# Visit: https://github.com/Megvii-BaseDetection/YOLOX/releases
# Place in: perception/models/yolox_tiny.onnx
```

### Camera Not Found

```bash
# List available cameras
python -c "import cv2; print([cv2.VideoCapture(i).isOpened() for i in range(5)])"

# Update `perception/config/webcam.json` with correct camera index
```

### Tests Fail

```bash
# Clear cached test artifacts
rm -rf __pycache__ .pytest_cache

# Re-run tests
python3 harness/runner.py
```

### Perception Events Not Emitted

Ensure:
- Zone configuration is valid (run `perception/tools/define_zone.py`)
- Camera is running and detecting people
- Confidence thresholds in the zone config (`emission.min_confidence`, typically much lower than the policy's decision thresholds) are appropriate
- The zone polygon reaches `y = 1.0` if subjects are clipped by the bottom of the frame (see `perception/config/README.md`)

## Licence Notes

The perception layer uses **YOLOX** (Apache-2.0), not Ultralytics YOLO (AGPL-3.0). 

For details on why and licensing alternatives, see [perception/LICENCE-NOTES.md](perception/LICENCE-NOTES.md).

## Next Steps

- Read the main [README.md](README.md) for architecture and design principles
- See [DEVELOPMENT.md](DEVELOPMENT.md) for development workflow
- Review `schema/` for data contract details
- Study `workflows/example_manufacturing_policy.json` as a template
- Optional pose/action path: [perception/actions/README.md](perception/actions/README.md)
