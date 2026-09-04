# Installation Guide

## Prerequisites

- **Python 3.10 or higher**
- **pip** package manager

On Windows, `python` is the usual interpreter name. The examples below use
`python3` as in the rest of the repo; substitute `python` if that is what
your venv provides. PowerShell does not expand `workflows/*.json` the way
bash does — pass each policy file explicitly, or run the commands from Git
Bash / WSL.

## Which script do I run?

Two entry points. They do the **same detection work** and differ only in what
comes out of them.

| | `perception/perceive.py` | `perception/tools/preview_pose.py` |
|---|---|---|
| Purpose | **Production pipeline** | **Visual demo** |
| Output | JSON-lines events on stdout, for the engine | Preview window, drawn skeleton and boxes |
| Logging | pipe to a consumer | `--log PATH` writes JSONL directly |
| Stages | all, opt-in via config or flags | all, opt-in via flags |

Both run on live video. Both share one implementation — `perception/objects.py`,
`perception/pose.py`, `perception/actions_stage.py` — so what you see in the
demo is what the pipeline emits.

Use `perceive.py` when you want events. Use `preview_pose.py` when you want to
see what the camera thinks is happening.

## Quick Start

### 1. Clone and Set Up Environment

```bash
git clone <repository-url>
cd ioacam

python3 -m venv venv

# Linux/macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Verify the Install Before Downloading Anything

Ten suites run with no camera and no model weights. Run them first: if they
pass, the checkout is sound, and any later problem is a model or environment
issue rather than a broken clone.

```bash
python3 harness/runner.py                    # 19 workflow scenarios
python3 harness/test_engine.py               # property tests
python3 harness/test_lint.py                 # linter tests
python3 identity/test_identity.py
python3 perception/test_perception.py
python3 perception/test_objects.py
python3 perception/test_stages.py            # opt-in stage wiring
python3 perception/actions/test_actions.py
python3 perception/actions/test_rtmpose.py
python3 perception/actions/test_posec3d.py
```

### 3. Download the Models

**YOLOX** (person and object detection) — needed for anything camera-related:

```bash
python perception/tools/fetch_model.py
```

Downloads `perception/models/yolox_tiny.onnx` (~20 MB, Apache-2.0). For the
larger accuracy-reference weights, `--model yolox_s`, then point
`detector.model_path` at it. `onnxruntime` is not required for YOLOX; the
default backend uses OpenCV DNN.

**RTMPose and PoseC3D** — only for the pose and action stages. These need
`torch` and `onnx`, both commented out in `requirements.txt` because they are
**export-only** and never on the inference path. Install them, export once,
uninstall if you like.

```bash
pip install torch onnx

python perception/tools/export_rtmpose.py     # -> perception/models/rtmpose_t.onnx
python perception/tools/export_posec3d.py     # -> perception/models/posec3d_pose_only.onnx
```

Both scripts download the published checkpoints themselves and rebuild the
architecture in plain torch. **MMPose and MMAction2 are deliberately not
required** — they pin their own torch and numpy, do not install cleanly, and
dragging that stack in to call `torch.onnx.export` once is a poor trade.
Keeping everything in ONNX is also what leaves the RKNN path open for the
board.

`onnxruntime` is recommended for the pose backend. Without it there is a
`cv2.dnn` fallback at roughly 4x the cost.

```bash
pip install onnxruntime
```

Model weights are **not** in the repository — `.gitignore` excludes
`perception/models/*.onnx` and `*.pth`. They are build artefacts, and 63 MB of
binaries in git history is permanent. Regenerate with the three commands above.

### 4. Define Your Zones

```bash
python perception/tools/define_zone.py
```

Opens a GUI. Click to define zone boundaries, copy the generated JSON, paste
into `perception/config/zones.example.json`.

**Note:** the webcam config uses `y = 1.0` as the zone boundary because seated
subjects are clipped by the frame bottom.

### 5. Run the Events Pipeline

```bash
# People and zones only — the default
python perception/perceive.py --config perception/config/webcam.json

# With a preview window
python perception/perceive.py --config perception/config/webcam.json --preview

# Validate the output against the schema
python perception/perceive.py --max-frames 300 | python perception/tools/validate_events.py
```

#### Enabling the optional stages

All three are **off by default**, so an unchanged config behaves exactly as the
people-and-zones path did before they existed. Turn them on by flag:

```bash
# Object detection and wrist attribution
python perception/perceive.py --objects

# PoseC3D action recognition
python perception/perceive.py --actions

# Both
python perception/perceive.py --objects --actions --action-every 1.5
```

Or by config, in `objects.enabled`, `pose.enabled`, `actions.enabled`. Flags
override config.

**Enabling objects or actions turns pose on implicitly.** Wrist attribution
needs keypoints, and PoseC3D consumes keypoints rather than pixels, so neither
stage means anything without pose.

**A missing model stops the run.** If actions are enabled and the PoseC3D ONNX
is absent, the process exits naming the export command. A stage that is enabled
but silently inactive is worse than one that refuses to start.

| Flag | Effect |
|---|---|
| `--objects` | Object detection and wrist attribution. Classes come from `detector.classes` |
| `--actions` | Full pose → tube → PoseC3D → abstention chain |
| `--action-every` | Seconds between action inferences. PoseC3D costs ~130–185 ms and reads a ~2 s window, so it deliberately does not run per frame |
| `--preview` | Show a window |
| `--source` | Camera index or video file |
| `--max-frames` | Bound the run |

### 6. Run the Visual Demo

Same work, with a window and drawn output:

```bash
python perception/tools/preview_pose.py --detector
python perception/tools/preview_pose.py --actions
python perception/tools/preview_pose.py --actions --objects --log logs/run.jsonl
```

`--detector` is **off by default** here, and it costs you: without it the whole
frame goes to RTMPose instead of a person crop. In a side-by-side run that was
mean keypoint score 0.26 against 0.53. `--actions` implies `--detector`.

`--min-confidence` and `--min-margin` tune the abstention thresholds;
`--no-window` runs headless; `--log PATH` writes a JSONL record with per-frame
timings, keypoints, objects, action verdicts and a summary.

### Framing matters more than anything else here

**Action recognition needs the whole body in frame.** PoseC3D was trained on
full-body skeletons, and sitting-versus-standing lives almost entirely in the
hips and knees. A laptop webcam at desk distance sees neither.

Symptom: RTMPose reports head and shoulders at 0.75–0.95 while hips, knees and
ankles sit below 0.1, and PoseC3D then abstains on nearly every window. That is
the abstention policy working correctly, not a broken model — the information
simply is not in the picture.

Sanity check: the person box should be roughly **2.5x taller than wide**.
Around 1.0 means upper body only.

Pose **estimation** has no such requirement. RTMPose reports what it can see
and scores the rest low, which is the correct behaviour.

### What to expect from the stock action model

The exported PoseC3D checkpoint ships a **60-class NTU-60 head** — "drink
water", "taking a selfie", "hand waving". Nothing about factories or
workflows. It proves the chain runs end to end; it is not useful output. Site
classes mean replacing `cls_head.fc_cls` and fine-tuning; the 512-d backbone
features are what you keep.

Expect frequent abstention on unfamiliar poses. A classifier cannot say "I have
not seen this before" on its own — it returns the nearest class it knows with a
mediocre score. The abstention policy refuses any answer failing either a
confidence or a margin test, and emits `value: "unknown"` so the engine can
queue it for review rather than losing it.

## Run the Engine and Tests

### The Harness

```bash
python3 harness/runner.py        # all 19 scenarios
python3 harness/runner.py -v     # with every finding printed
```

The runner treats an extra argument as a **directory** of JSON files, not a
single scenario path. To exercise one file, put it in a folder of its own or
call `run_scenario("harness/scenarios/01_clean_run.json")` from Python.

### Lint Policy Files

```bash
# bash / Git Bash
python3 tools/lint_policy.py workflows/*.json --strict

# PowerShell — no glob expansion
python tools/lint_policy.py workflows/example_manufacturing_policy.json workflows/pick_and_place_policy.json --strict
```

Run this before deploying any policy. It catches faults the schema cannot:
unreachable underrun rules, required steps with no deadline (which makes an
omission undetectable), evidence that cannot distinguish two steps, and gaps in
the response matrix.

### Inspect a Model Checkpoint

```bash
python3 tools/inspect_checkpoint.py perception/models/rtmpose_t.pth
```

Reads a `.pth`'s input contract — keypoint count, clip length, heatmap size,
head shape — **without needing torch installed**. Worth running whenever a new
checkpoint arrives: those numbers are baked into the weights, not the
documentation, and getting them wrong does not always fail loudly. A volume of
the wrong temporal length can run and quietly return nonsense.

## Full Workflow Integration

The engine is a **library**. It has no stdin consumer yet: findings are
produced in-process (the harness feeds JSON scenario files) and go nowhere
until a transport exists. Perception emits JSON lines on stdout for a consumer
you write.

On PowerShell, do not redirect stdout with `>`. That writes UTF-16, which JSON
Lines readers reject. Pipe to another process, or write the file from Python
with UTF-8. `preview_pose.py --log PATH` writes UTF-8 directly and avoids the
problem.

## Configuration

- **Webcam config:** `perception/config/webcam.json`
- **Zone definitions:** `perception/config/zones.example.json`
- **Action config:** `perception/actions/config/actions.example.json`
- **Identity config:** `identity/config/identity.example.json`
- **Example policy:** `workflows/example_manufacturing_policy.json`
- **Second example:** `workflows/pick_and_place_policy.json`
- **Schemas:** `schema/event.schema.json`, `schema/workflow.schema.json`

`detector.classes` decides which of YOLOX's 80 COCO classes this site cares
about, and `detector.class_thresholds` gives each its own floor — one threshold
cannot serve a person filling the frame and a phone 40 px across. Tune them
against a logged run before trusting any of them.

## Troubleshooting

### Model Download Fails

Manually: https://github.com/Megvii-BaseDetection/YOLOX/releases, place in
`perception/models/yolox_tiny.onnx`.

If a downloaded `.pth` will not load, check its size first. A file of a few KB
is usually an HTML error page saved with the wrong extension.
`tools/inspect_checkpoint.py` detects this and says so.

### Camera Not Found

```bash
python -c "import cv2; print([cv2.VideoCapture(i).isOpened() for i in range(5)])"
```

Update the camera index in `perception/config/webcam.json`.

### ONNX Export Fails

The exports need `torch` and `onnx`, both commented out in `requirements.txt`.
3D convolutions occasionally hit unsupported operators. Worth resolving rather
than working around: RKNN converts *from* ONNX, so a model that will not export
is a model that cannot go on the board.

### No Object Events

1. Is the stage on? `--objects`, or `objects.enabled` in config.
2. Is the class in `detector.classes`? Only COCO names are valid.
3. Is `class_thresholds` too high for that class?
4. For attribution specifically, check wrist scores in a `--log` run.
   Attribution needs a confident wrist, and in a typical webcam run wrists sit
   near 0.30 — a phone was seen 90 times and attributed 3 times, with the wrist
   as the limiting factor, not the object.

### Action Recognition Always Abstains

Usually correct behaviour. Check in this order:

1. **Framing.** Are hips, knees and ankles visible? Check per-joint scores in a
   `--log` run. Below 0.1 means they are out of shot.
2. **Detector.** In `preview_pose.py`, without `--detector` the whole frame
   goes to RTMPose and keypoint quality drops sharply.
3. **The pose the model knows.** NTU-60 has no class for most workplace
   activity. Abstaining on an unknown action is the intended outcome.

Only after those, consider lowering `--min-confidence`.

### Perception Events Not Emitted

- Zone configuration valid (`perception/tools/define_zone.py`)
- Camera running and detecting people
- `emission.min_confidence` appropriate — typically much lower than the
  policy's decision thresholds
- Zone polygon reaches `y = 1.0` if subjects are clipped by the frame bottom

### Tests Fail

```bash
rm -rf __pycache__ .pytest_cache
python3 harness/runner.py
```

## Licence Notes

The perception layer uses **YOLOX** (Apache-2.0), not Ultralytics YOLO
(AGPL-3.0). See [perception/LICENCE-NOTES.md](perception/LICENCE-NOTES.md).

The action models carry a separate concern: PoseC3D checkpoints are commonly
pretrained on **NTU RGB+D**, which is research-only under an academic use
agreement, and a fine-tuned model inherits the restriction. Resolve this before
shipping, not after training a classifier on it. See
[perception/actions/LICENCE-NOTES.md](perception/actions/LICENCE-NOTES.md).

## Next Steps

- [README.md](README.md) — architecture and design principles
- [DEVELOPMENT.md](DEVELOPMENT.md) — development workflow
- `schema/` — data contract details
- `workflows/example_manufacturing_policy.json` — policy template
- [perception/actions/README.md](perception/actions/README.md) — pose and action detail