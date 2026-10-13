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

Fourteen suites and the scenario runner go through with no camera and no model
weights. Run them first: if they pass, the checkout is sound, and any later
problem is a model or environment issue rather than a broken clone.

```bash
python3 harness/runner.py                    # 19 workflow scenarios
python3 harness/test_engine.py               # property tests
python3 harness/test_lint.py                 # linter tests
python3 identity/test_identity.py            # cascade, thresholds, person IDs
python3 identity/test_alignment.py           # face alignment
python3 identity/test_directory.py           # employee directory lookup
python3 perception/test_perception.py
python3 perception/test_objects.py
python3 perception/test_yolo26.py            # YOLO26 decode + both backends agree
python3 perception/test_ppe.py               # PPE classes, floors, attribution
python3 perception/test_stages.py            # opt-in stage wiring
python3 perception/test_intervals.py         # action interval join
python3 perception/actions/test_actions.py
python3 perception/actions/test_rtmpose.py
python3 perception/actions/test_posec3d.py
```

### 3. Download the Models

**The detector** — needed for anything camera-related. Which model runs is the
`detector.backend` config field, not a fixture. Two COCO backends ship:
`yolox-onnx` (Apache-2.0, the default) and `yolo26` (AGPL-3.0, evaluation
only). You need the weights for whichever one you select.

**YOLOX** (the default), person and object detection:

```bash
python perception/tools/fetch_model.py
```

Downloads `perception/models/yolox_tiny.onnx` (~20 MB, Apache-2.0). For the
larger accuracy-reference weights, `--model yolox_s`, then point
`detector.model_path` at it. `onnxruntime` is not required for YOLOX; the
default backend uses OpenCV DNN.

**YOLO26** — optional, and only if you are evaluating it. Ultralytics
publishes `.pt` checkpoints rather than ONNX, so this one is an export rather
than a download:

```bash
pip install ultralytics onnxruntime          # export-only; see the note below
python perception/tools/export_yolo26.py --download
```

Fetches `yolo26n.pt` and writes `perception/models/yolo26n.onnx` (~10 MB),
NMS-free and end to end, then verifies it two ways before reporting success:
the shipped decode against Ultralytics' own postprocess on the same graph, and
the converted graph against the original checkpoint. A non-zero exit means do
not use the file.

`ultralytics` is needed for that one offline step and **must not** end up on
the inference path — `perception/detectors/yolo26.py` runs on onnxruntime, numpy
and the OpenCV already needed for capture, which is what keeps the RK3568 port
a conversion job. Same rule as
`torch` for the pose models below.

> **Licence:** YOLO26 is an Ultralytics model. Its code **and its pretrained
> weights** are AGPL-3.0, and exporting to ONNX does not change that. That is
> fine for internal evaluation on your own footage; it is not shippable without
> an Ultralytics commercial licence. The backend prints an AGPL banner on
> startup and says so in the licence line `perceive.py` logs. Read
> [perception/LICENCE-NOTES.md](perception/LICENCE-NOTES.md) first.

**Switching backends.** One field, plus the weights for the model you picked:

```jsonc
// perception/config/zones.example.json
"detector": {
  "backend": "yolo26",                              // or "yolox-onnx" (default)
  "model_path": "perception/models/yolo26n.onnx",   // yolox_tiny.onnx for YOLOX
  "input_size": [640, 640]                          // [416, 416] for yolox_tiny
}
```

Or for one run, without editing anything:

```bash
python perception/perceive.py --backend yolo26 --config <a config pointing at the .onnx>
```

`classes`, `class_thresholds` and `score_threshold` mean the same thing under
both and need no edit. `nms_threshold` and `p6` are YOLOX-only — YOLO26's head
is NMS-free — and are accepted and ignored rather than rejected, so a config
switches over without being rewritten. `input_size` is the one that must match:
YOLO26's graph has a static input axis, and a mismatch is refused at load with
both sizes named.

**RTMPose and PoseC3D** — only for the pose and action stages. These need
`torch` and `onnx`, both commented out in `requirements.txt` because they are
**export-only** and never on the inference path. Install them, export once,
uninstall if you like.

```bash
pip install torch onnx

# RTMPose: fetch the checkpoint first -- the exporter has no download step
curl -L -o perception/models/rtmpose-tiny_simcc-aic-coco_pt-aic-coco_420e-256x192-cfc8f33d_20230126.pth \
  https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-tiny_simcc-aic-coco_pt-aic-coco_420e-256x192-cfc8f33d_20230126.pth
python perception/tools/export_rtmpose.py               # -> perception/models/rtmpose_t.onnx

python perception/tools/export_posec3d.py --download    # -> perception/models/posec3d_pose_only.onnx
```

The two differ, and the difference is not cosmetic. `export_posec3d.py`
fetches its own checkpoint, but only when you pass `--download`; without the
flag it stops rather than reaching the network. `export_rtmpose.py` has **no
download step at all** — hand it a `.pth` or it exits with `checkpoint not
found`, which is why the `curl` above comes first. Give it a different
checkpoint with `--checkpoint`.

Both then rebuild the architecture in plain torch. **MMPose and MMAction2 are
deliberately not required** — they pin their own torch and numpy, do not
install cleanly, and dragging that stack in to call `torch.onnx.export` once is
a poor trade. Keeping everything in ONNX is also what leaves the RKNN path open
for the board.

`onnxruntime` is recommended for the pose backend. Without it there is a
`cv2.dnn` fallback at roughly 4x the cost.

```bash
pip install onnxruntime
```

Model weights are **not** in the repository — `.gitignore` excludes
`perception/models/*.onnx`, `*.pth` and `*.pt`. They are build artefacts, and
63 MB of binaries in git history is permanent. Regenerate with the commands
above. The `*.pt` rule matters for a second reason: YOLO26 checkpoints are
AGPL-3.0, and committing them would put AGPL weights in the repository by
accident.

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

All four are **off by default**, so an unchanged config behaves exactly as the
people-and-zones path did before they existed. Turn them on by flag:

```bash
# Object detection and hand attribution
python perception/perceive.py --objects

# PoseC3D action recognition
python perception/perceive.py --actions

# Hard hats and high-vis vests, from a second detector
python perception/perceive.py --config perception/config/webcam.yolo26.json --ppe

# Both of the first two
python perception/perceive.py --objects --actions --action-every 1.5
```

Or by config, in `objects.enabled`, `pose.enabled`, `actions.enabled`,
`ppe.enabled`. Flags override config.

**Enabling objects or actions turns pose on implicitly.** Hand attribution
needs keypoints, and PoseC3D consumes keypoints rather than pixels, so neither
stage means anything without pose.

**PPE implies nothing.** Its association is box containment against the person
boxes the tracker already has, so it needs no keypoints — turning pose on for
it would buy an RTMPose pass per person for nothing. What it does cost is a
whole second forward pass on every frame, roughly doubling detection: see
[perception/config/README.md](perception/config/README.md#what-the-ppe-stage-costs).

**A missing model stops the run.** If actions are enabled and the PoseC3D ONNX
is absent, the process exits naming the export command. A stage that is enabled
but silently inactive is worse than one that refuses to start.

| Flag | Effect |
|---|---|
| `--objects` | Object detection and hand attribution. Classes come from `detector.classes` |
| `--actions` | Full pose → tube → PoseC3D → abstention chain |
| `--ppe` | Second detector for hard hats and high-vis vests. Needs a config with a `ppe` block and `onnxruntime` |
| `--action-every` | Seconds between action inferences. PoseC3D costs ~157 ms per tube and reads a 1.5 s window, so it deliberately does not run per frame |
| `--identity` | Path to an identity config; binds tracks to enrolled person IDs. Every camera event for a bound track then carries `subject.identity` -- zone entries and exits, `action_recognised`, `action_interval`, `ppe_helmet` and `ppe_vest`. An unbound track carries **no** `identity` key rather than an empty one |
| `--preview` | Show a window |
| `--source` | Camera index or video file |
| `--max-frames` | Bound the run |

### 6. The Employee Directory (optional)

Everything in this system is keyed on a badge ID, so an operator watching a
screen sees `EMP-4471`. This file is what turns that back into a name.

```bash
cp identity/data/employees.example.csv identity/data/employees.csv
```

Then edit it. One row per person:

```
employee_id,name,role,department,active
EMP-4471,Rajesh Kumar,operator,Assembly,yes
EMP-5522,Priya Nair,maintenance_engineer,Maintenance,yes
EMP-9012,Daniel Okafor,operator,Assembly,no
```

| Column | Required | Meaning |
|---|---|---|
| `employee_id` | yes | The badge ID, and the only key |
| `name` | no | Shown on screen. Blank means the ID is shown |
| `role` | no | `operator`, `maintenance_engineer`, `visitor`. Not purely cosmetic � see below |
| `department` | no | Display, and the first thing asked about a leaver match |
| `active` | no | `no` means they have left |

CSV, not `.xlsx`: `csv` is in the standard library, so this adds no dependency,
and the file still opens in Excel for whoever maintains it. Extra columns from
an HR export are ignored rather than rejected.

**It is read for display and nothing else.** Events, logs and the event store
carry `employee_id` and never `name`. That is the whole point of keying on an
ID � years of event history hold no personal names on their face, and the
mapping to a person is this one file, which can be access-controlled, audited
and deleted on its own schedule.

**So nothing here may need it.** The file is optional in the strong sense:

```bash
rm identity/data/employees.csv       # the system runs, labelling with IDs
```

If something breaks without it, a name has leaked into a path that should only
ever have carried an ID. **Do not put anything in this file that the event
store needs.**

Three behaviours worth knowing before you deploy it:

- **It is gitignored and it is personal data.** `employees.example.csv` with
  its three fictional rows is committed; `employees.csv` is not. On a device,
  `chmod 640 identity/data/employees.csv`.
- **It reloads without a restart.** Edited rows are picked up within a few
  seconds (`directory.auto_reload`), or call `reload()`. Staff change weekly
  and restarting a device to add one person is not a procedure anybody
  follows. A file that appears *after* startup is picked up too.
- **`active=no` is enforced.** `identity/enroll.py` refuses to enrol the ID
  and says why, and a camera match against one is reported as a leaver rather
  than drawn as a normal recognition � `live_recognize.py` marks the box
  amber and `[INACTIVE]`, and logs one line per ID.

`role` is the one column that is not cosmetic: the engine uses role for zone
rules and for a step's `actor_role`. By default the directory's role is display
only. Setting `directory.supply_roles: true` in the identity config lets it fill
in a role for a badge-in event that carried none � useful where readers send an
ID only, and off by default because a stale role here produces a confident
`wrong_role` finding against somebody who did nothing wrong.

```bash
python3 identity/test_directory.py      # including the no-file-at-all case
```

### 7. Run the Visual Demo

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

With `--actions` the log also carries one line per closed action interval:

```json
{"type":"action_interval","frame":156,"t_us":184268000000,
 "track_id":"trk-0001","label":"writing",
 "start_us":184261000000,"end_us":184264500000,
 "wall_time_start":"2026-08-31T17:40:26.000Z",
 "wall_time_end":"2026-08-31T17:40:29.500Z",
 "duration_s":3.5,"confidence":0.29,"windows":3}
```

Shape only: those numbers are illustrative, not from a measured run. The real
records in `logs/` were written before the two wall-clock fields existed.

`start_us` / `end_us` are monotonic microseconds since boot. They subtract
correctly and no clock correction can reorder them, which is what every
duration calculation needs, and on their own they do not say when anything
happened -- 338483682608 is not a time of day. `wall_time_start` /
`wall_time_end` are those same two window edges as ISO UTC, for whoever reads
the log afterwards and has to line it up against a shift or a clip. Both are
derived from one monotonic-to-wall offset captured at startup, not from the
clock at the moment the line is written: the line is written when the interval
**closes**, a couple of windows after the action ended, so a clock read there
would date every interval late by exactly that delay. `t_us` beside them is
the separate fact of when the line was written. All of these carry
window-level precision, about 1.5 s -- do not read them as exact instants.

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
- **Employee directory:** `identity/data/employees.example.csv`
- **Example policy:** `workflows/example_manufacturing_policy.json`
- **Second example:** `workflows/pick_and_place_policy.json`
- **Schemas:** `schema/event.schema.json`, `schema/workflow.schema.json`

`detector.classes` decides which of the detector's 80 COCO classes this site
cares about, and `detector.class_thresholds` gives each its own floor — one
threshold cannot serve a person filling the frame and a phone 40 px across.
Tune them against a logged run before trusting any of them. Both COCO backends
read these keys with the same meaning, so they survive a backend switch.

## Troubleshooting

### Model Download Fails

YOLOX, manually: https://github.com/Megvii-BaseDetection/YOLOX/releases, place
in `perception/models/yolox_tiny.onnx`.

YOLO26 has no ONNX release to download — `export_yolo26.py` converts the `.pt`
locally. If the export fails, it says which of its two checks failed rather
than leaving an unverified graph behind.

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
4. For attribution specifically, replay the log rather than guessing:
   `python3 perception/tools/diagnose_association.py logs/run.jsonl --verbose`
   prints every hand anchor, its distance to the object, and which gate
   rejected it. An earlier note here blamed `min_wrist_score` for a run in
   which a phone was seen 90 times and attributed 3 times; that was wrong.
   Replaying the same log through the current geometry attributes 89 of 90
   with the same keypoints and the same threshold — the reach was the limiting
   factor, not the wrist. Measure before changing any setting.

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

The perception layer defaults to **YOLOX** (Apache-2.0), not Ultralytics YOLO
(AGPL-3.0). The detector is a config choice, and two of the available backends
— `ultralytics` and `yolo26` — are AGPL-3.0 in both code and weights. They are
there for bench comparison on your own footage and must not be present in a
delivered build. An ONNX export does not relicense Ultralytics weights. See
[perception/LICENCE-NOTES.md](perception/LICENCE-NOTES.md).

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