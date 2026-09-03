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

### 2. Verify the Install Before Downloading Anything

Nine suites run with no camera and no model weights. Run them first: if they
pass, the checkout is sound, and any later problem is a model or environment
issue rather than a broken clone.

```bash
python3 harness/runner.py                    # 19 workflow scenarios
python3 harness/test_engine.py               # property tests
python3 harness/test_lint.py                 # linter tests
python3 identity/test_identity.py
python3 perception/test_perception.py
python3 perception/test_objects.py
python3 perception/actions/test_actions.py
python3 perception/actions/test_rtmpose.py
python3 perception/actions/test_posec3d.py
```

### 3. Set Up the Perception Layer (Optional, for camera-based detection)

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

### 4. Pose and Action Recognition (Optional)

`perception/perceive.py` is the **events** pipeline: detection, zones,
tracking, and JSON-lines output for the engine. It has no pose or action
stage.

The full chain — YOLOX → RTMPose → pose tube → PoseC3D → abstention, plus
object detection and wrist attribution — lives in
**`perception/tools/preview_pose.py`**, with a preview window and JSONL
logging.

#### Export the pose and action models

These need `torch` and `onnx`, both commented out in `requirements.txt`
because they are **export-only** and never on the inference path. Install
them, run the two exports once, and uninstall them again if you like.

```bash
pip install torch onnx

python perception/tools/export_rtmpose.py     # -> perception/models/rtmpose_t.onnx
python perception/tools/export_posec3d.py     # -> perception/models/posec3d_pose_only.onnx
```

Both scripts download the published checkpoints themselves and rebuild the
architecture in plain torch. **MMPose and MMAction2 are deliberately not
required** — they pin their own torch and numpy and do not install cleanly,
and dragging that stack in to call `torch.onnx.export` once is a poor trade.
Keeping everything in ONNX is also what leaves the RKNN path open for the
board.

`onnxruntime` is recommended for the pose backend. Without it there is a
`cv2.dnn` fallback at roughly 4x the cost and one person per session call.

```bash
pip install onnxruntime
```

#### Run it

```bash
# Pose only, using the detector for boxes
python perception/tools/preview_pose.py --detector

# Pose + PoseC3D action recognition (--actions implies --detector)
python perception/tools/preview_pose.py --actions

# Everything, logged to a file for later analysis
python perception/tools/preview_pose.py --actions --objects --log logs/run.jsonl
```

Useful flags:

| Flag | Effect |
|---|---|
| `--detector` | Use YOLOX for person boxes. **Off by default**, in which case the whole frame is fed to RTMPose — measurably worse (mean keypoint score 0.26 against 0.53 in a side-by-side run). |
| `--actions` | Full pose → tube → PoseC3D → abstention chain. Implies `--detector`. |
| `--objects` | Also detect the configured object classes and attribute them to the wrist holding them. |
| `--action-every` | Seconds between action inferences. PoseC3D costs ~130–185 ms, so it deliberately does not run per frame. |
| `--min-confidence`, `--min-margin` | Abstention thresholds. Defaults 0.55 and 0.15. |
| `--source` | Camera index or a video file path. |
| `--no-window`, `--max-frames`, `--save` | Headless smoke testing. |
| `--log PATH` | JSON-lines record of the run: per-frame timings, keypoints, objects, action verdicts, and a summary. |

#### Framing matters more than anything else here

**Action recognition needs the whole body in frame.** PoseC3D was trained on
full-body skeletons, and sitting-versus-standing lives almost entirely in the
hips and knees. A laptop webcam at desk distance sees neither.

Symptom: RTMPose reports head and shoulders at 0.75–0.95 while hips, knees and
ankles sit below 0.1, and PoseC3D then abstains on nearly every window. That is
the abstention policy working correctly, not a broken model — the information
simply is not in the picture.

Stand back far enough that your feet are visible before judging action
accuracy. A useful sanity check: the person box should be roughly 2.5x taller
than wide. Around 1.0 means upper body only.

Pose **estimation** has no such requirement. RTMPose reports what it can see
and scores the rest low, which is the correct behaviour.

#### What to expect from the stock model

The exported PoseC3D checkpoint ships a **60-class NTU-60 head** — "drink
water", "taking a selfie", "hand waving". Nothing about factories or
workflows. It is useful for proving the chain runs end to end, not for real
output. Site classes mean replacing `cls_head.fc_cls` and fine-tuning; the
512-d backbone features are what you keep.

Expect frequent abstention on unfamiliar poses. That is by design: a
classifier cannot say "I have not seen this before" on its own, so the
abstention policy refuses any answer that fails either a confidence or a
margin test. Silence is the correct output for an action the model does not
know.

### 5. Run the Engine and Tests

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

Run this before deploying any policy. It catches faults the schema cannot:
unreachable underrun rules, required steps with no deadline (which makes an
omission undetectable), evidence that cannot distinguish two steps, and gaps
in the response matrix.

#### Inspect a Model Checkpoint

```bash
python3 tools/inspect_checkpoint.py perception/models/rtmpose_t.pth
```

Reads a `.pth`'s input contract — keypoint count, clip length, heatmap size,
head shape — **without needing torch installed**. Worth running whenever a new
checkpoint arrives: those numbers are baked into the weights, not the
documentation, and getting them wrong does not always fail loudly. A volume of
the wrong temporal length can run and quietly return nonsense.

#### Identity, perception, and action tests

These need no camera and no downloaded weights:

```bash
python3 identity/test_identity.py
python3 perception/test_perception.py
python3 perception/test_objects.py
python3 perception/actions/test_actions.py
python3 perception/actions/test_rtmpose.py
python3 perception/actions/test_posec3d.py
```

### 6. Full Workflow Integration

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
Python with UTF-8. `preview_pose.py --log PATH` writes UTF-8 directly and
avoids the problem. See the known issue in [README.md](README.md).

## Model Files

Model weights are **not** in the repository — `.gitignore` excludes
`perception/models/*.onnx` and `*.pth`. They are build artefacts, and 63 MB of
binaries in git history is permanent.

Regenerate them instead:

```bash
python perception/tools/fetch_model.py        # YOLOX detector
python perception/tools/export_rtmpose.py     # pose  (needs torch)
python perception/tools/export_posec3d.py     # action (needs torch)
```

## Configuration

### Perception Configuration

- **Webcam config:** `perception/config/webcam.json` (pre-configured for laptop webcams)
- **Zone definitions:** `perception/config/zones.example.json`
- **Action config:** `perception/actions/config/actions.example.json`
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

If a downloaded `.pth` will not load, check its size first. A file of a few KB
is usually an HTML error page saved with the wrong extension.
`tools/inspect_checkpoint.py` detects this and says so.

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

### ONNX Export Fails

The exports need `torch` and `onnx`, both commented out in
`requirements.txt`. 3D convolutions occasionally hit unsupported operators.
Worth resolving rather than working around: RKNN converts *from* ONNX, so a
model that will not export is a model that cannot go on the board.

### Action Recognition Always Abstains

Usually correct behaviour, not a fault. Check in this order:

1. **Framing.** Are hips, knees and ankles visible? Check the per-joint scores
   in a `--log` run. Below 0.1 means they are out of shot.
2. **Detector.** Without `--detector`, the whole frame goes to RTMPose and
   keypoint quality drops sharply.
3. **The pose the model knows.** NTU-60 has no class for most workplace
   activity. Abstaining on an unknown action is the intended outcome.

Only after those, consider lowering `--min-confidence`.

### Perception Events Not Emitted

Ensure:
- Zone configuration is valid (run `perception/tools/define_zone.py`)
- Camera is running and detecting people
- Confidence thresholds in the zone config (`emission.min_confidence`, typically much lower than the policy's decision thresholds) are appropriate
- The zone polygon reaches `y = 1.0` if subjects are clipped by the bottom of the frame (see `perception/config/README.md`)

## Licence Notes

The perception layer uses **YOLOX** (Apache-2.0), not Ultralytics YOLO
(AGPL-3.0). See [perception/LICENCE-NOTES.md](perception/LICENCE-NOTES.md).

The action models carry a separate concern: PoseC3D checkpoints are commonly
pretrained on **NTU RGB+D**, which is research-only under an academic use
agreement, and a fine-tuned model inherits the restriction. Resolve this
before shipping, not after training a classifier on it. See
[perception/actions/LICENCE-NOTES.md](perception/actions/LICENCE-NOTES.md).

## Next Steps

- Read the main [README.md](README.md) for architecture and design principles
- See [DEVELOPMENT.md](DEVELOPMENT.md) for development workflow
- Review `schema/` for data contract details
- Study `workflows/example_manufacturing_policy.json` as a template
- Optional pose/action path: [perception/actions/README.md](perception/actions/README.md)