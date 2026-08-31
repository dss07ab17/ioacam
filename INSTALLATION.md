# Installation Guide

## Prerequisites

- **Python 3.10 or higher**
- **pip** package manager

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

This downloads `perception/models/yolox_tiny.onnx` (or `yolox_s.onnx` for higher accuracy).

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
# Run all 12 scenarios
python3 harness/runner.py

# Run with verbose output (all findings printed)
python3 harness/runner.py -v
```

#### Run Property Tests

```bash
# Property-based tests for tick invariance
python3 harness/test_engine.py
```

#### Run Linter Tests

```bash
# Test policy file validation
python3 harness/test_lint.py
```

#### Lint Policy Files

```bash
# Check your workflow policy for correctness
python3 tools/lint_policy.py workflows/*.json --strict
```

### 4. Full Workflow Integration

To run the complete pipeline (perception → engine):

```bash
# Terminal 1: Run perception layer
python perception/perceive.py

# Terminal 2: Run your consumer/engine
# (This is where you integrate with your workflow logic)
python perception/perceive.py | python your_consumer.py
```

## Configuration

### Perception Configuration

- **Webcam config:** `perception/config/webcam.json` (pre-configured for laptop webcams)
- **Zone definitions:** `perception/config/zones.example.json`
- **Camera calibration:** Adjust `temperature` scaling in workflow policy (default: 1.8)

### Workflow Policy

- **Example policy:** `workflows/example_manufacturing_policy.json`
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
- Confidence thresholds in zones.json are appropriate (typically 0.6-0.7)

## Licence Notes

The perception layer uses **YOLOX** (Apache-2.0), not Ultralytics YOLO (AGPL-3.0). 

For details on why and licensing alternatives, see [perception/LICENCE-NOTES.md](perception/LICENCE-NOTES.md).

## Next Steps

- Read the main [README.md](README.md) for architecture and design principles
- See [DEVELOPMENT.md](DEVELOPMENT.md) for development workflow
- Review `schema/` for data contract details
- Study `workflows/example_manufacturing_policy.json` as a template
