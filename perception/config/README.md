# Config reference

Three configs ship here:

- **`zones.example.json`** -- the documented template, an industrial floor view.
- **`webcam.json`** -- known-good for a laptop webcam, measured on a real one.
  Zone taken to `y = 1.0`, `reference_height_px` raised to 400 for a close
  subject, and shorter debounce windows to suit ~6 fps.
- **`webcam.yolo26.json`** -- the same webcam view on `backend: "yolo26"` at
  640, and the only one carrying a `ppe` block, because the PPE stage is a
  second YOLO26 pass and both are AGPL-3.0 evaluation paths. Read
  [LICENCE-NOTES.md](../LICENCE-NOTES.md) before running it.

One file describes one camera. Fields not listed are inherited from `DEFAULTS`
in `perceive.py`.

| Key | Meaning |
|---|---|
| `sensor_id` | Goes into every event. Must match a `covered_by` entry in the policy's zone list, or the engine's coverage check cannot associate this camera with a zone. |
| `camera.index` | Webcam index. `0` is the built-in laptop camera. Override with `--source` for a video file. |
| `detector.backend` | Which detector runs. `yolox-onnx` (Apache-2.0, **default**, ships) / `yolo26` (AGPL-3.0, NMS-free, evaluation only) / `ultralytics` (AGPL-3.0, evaluation only, also needs `acknowledge_agpl`) / `stub` (tests). The detector is a config choice, not a fixture: nothing outside `detectors/` knows which one is running. Read [LICENCE-NOTES.md](../LICENCE-NOTES.md) before choosing either AGPL one. |
| `detector.model_path` | The weights for that backend, so it changes with `backend`: `perception/models/yolox_tiny.onnx` (fetched by `tools/fetch_model.py`) or `perception/models/yolo26n.onnx` (exported by `tools/export_yolo26.py`). |
| `detector.input_size` | `[416, 416]` for `yolox_tiny`, `[640, 640]` for YOLO26 — the size the graph was exported at. YOLO26's input axis is static, so a mismatch is rejected at load with the two sizes named rather than tolerated. |
| `detector.score_threshold` | Raw model score floor, applied before calibration. Lower it to trade precision for recall; the confidence pipeline downstream will discount weak detections anyway. |
| `detector.nms_threshold` | IoU floor for within-class NMS. `yolox-onnx` only: YOLO26's head is one-to-one and emits no duplicates, so the key is accepted and ignored there rather than rejected — switching `backend` should not mean editing the rest of the block. |
| `calibration.temperature` | Divides the logit. **Must match `calibration.temperature` in the policy file**, or the engine compares confidences against thresholds tuned on a different scale. |
| `zones[].coordinates` | `normalized` (0..1, default) or `pixel`. Normalised survives a resolution change; pixel does not. |
| `zones[].polygon` | At least 3 points, in order around the boundary. Generate it with `tools/define_zone.py`. |
| `emission.enter_frames` | Consecutive frames inside before an entry is emitted. Raise it if the boundary flickers. |
| `emission.exit_frames` | Consecutive frames outside before an exit is emitted. Keep it above `enter_frames`: a subject who steps out briefly has not left. |
| `emission.persistence_window` | Frames retained for the `persistence` confidence component. |
| `emission.min_confidence` | Events below this are dropped as uninformative. Not a detection threshold. |
| `quality.reference_height_px` | Subject pixel height at which viewing quality is considered full. Depends on how far the camera is from the zone; measure it once with `--preview`. |
| `quality.min_height_px` | Below this a subject scores zero quality. |
| `integrity.enabled` | Emit `integrity.seq` and `prev_hash`. Off by default; see the README. |
| `identity.config` | Path to an identity config (e.g. `identity/config/identity.example.json`), or `null`. `--identity` overrides it. Binds person tracks to enrolled IDs inside the loop, after which **every** camera event for a bound track carries `subject.identity`: `person_in_zone`, `person_left_zone`, `action_recognised`, `action_interval`, `ppe_helmet`, `ppe_vest`. A track the cascade did not bind carries **no** `identity` key rather than an empty one, because the engine's `wrong_role` rules read an absent key and a falsy value differently. `track_id` is kept beside the identity, never replaced by it. Off by default: naming people is a heavier proposition, legally and operationally, than counting them. |
| `objects.enabled` | Opt-in. Track non-person COCO classes; emit `object_at_station` on the production path. Implies pose. |
| `ppe.enabled` | Opt-in, `--ppe`. Runs a **second detector** on every frame for hard hats and high-vis vests, and emits `ppe_helmet` / `ppe_vest`. Costs a whole extra forward pass — read "What the PPE stage costs" below before switching it on. Implies nothing: attribution is box containment, so no pose is needed. |
| `ppe.backend` | `yolo26-ppe` (AGPL-3.0), the only one. This is **not** the person detector: `detector.backend` still selects that and still owns tracking, zones and pose. Naming a person backend here is rejected at startup. |
| `ppe.model_path` | `perception/models/ppe_yolo26.onnx`. A YOLO26n fine-tune on a construction-site-safety set, not a download: `export_yolo26.py` does not produce it. |
| `ppe.classes` | Which of the model's ten classes run. Default is the four that map onto a schema observation: `Hardhat`, `NO-Hardhat`, `Safety Vest`, `NO-Safety Vest`. Names are **case-sensitive** and are not COCO's. `Mask`/`NO-Mask` are detectable and emit nothing — the schema has no mask observation and they are not squeezed into `ppe_glasses`. `Person` is this model's own person class and must stay off. |
| `ppe.class_thresholds` | Per class, and **deliberately asymmetric**: `NO-*` at 0.20 against 0.40 for the positives. Measured recall on the `NO-*` classes is 0.55–0.62, and a missed bare head costs more than a false alarm does, so those classes are floored low and the resulting over-flagging is the intended trade. Omit the key entirely to take the model's own measured defaults from `detectors/ppe.py`. A name that is not one of the ten is rejected rather than silently ignored. |
| `ppe.min_containment` | Fraction of a PPE box that must fall inside a person box before that person is a candidate to be wearing it. `0.70`, not `1.0`: a hat on someone at the frame edge is clipped by the person box as often as not. |
| `ppe.loose_frames` | Consecutive frames a PPE box must stay unattributed before it is announced that way. `3`. The same debounce `emission.enter_frames` applies to zone entry: measured on a live webcam run, containment is bimodal (0.98–1.00 wherever a person box exists, 0.0 where none does), so a lone unattributed frame is the person detector missing a frame rather than a marginal association. **Attributed findings never wait** — delaying a non-compliance is the one cost this stage must not pay. |
| `ppe.forget_after` | Frames a person track may be missing before their compliance state, and any PPE attributed to them, is dropped. `perceive.py` defaults it to `tracking.max_misses`, which is what it has to match: `person_boxes` carries only the tracks matched *this* frame. |
| `ppe.tie_ratio` | Two candidate people with containment and box area this close are treated as tied, and the PPE box is attributed to **neither** — but is still emitted, with no `track_id`. Same reasoning as `objects.tie_ratio`. |
| `pose.enabled` | Opt-in RTMPose (or `backend: stub` for tests). Implied by objects/actions. |
| `actions.enabled` | Opt-in PoseC3D on `actions.every_s` cadence; emit `action_recognised` (including abstentions as `value: "unknown"`) and, on close, `action_interval`. The interval event carries its span on both clocks: `start_us` / `end_us` monotonic for anything that measures or orders, `wall_time_start` / `wall_time_end` as ISO UTC for anyone reading the log. Both come from one offset captured at startup, so a span is dated to when it ran rather than to when the interval closed. |
| `actions.intervals.close_after` | Consecutive windows that must disagree with an open interval before it closes. Default 2. One flickering window (a different label, or an abstention mid-action) must not split `reading` into two intervals, so the close is delayed by this many windows. The interval still ends at the last window that carried its label. |
| `actions.intervals.min_duration_s` | Intervals shorter than this are dropped rather than emitted — usually one misclassified window. Default 1.0. Read it against the tube length: a single 1.5s window already clears a 1.0s floor, so a site that wants single windows gone sets this above `actions.tube.window_s`. |
| `actions.intervals.track_timeout_s` | A track producing no windows for this long has its open interval closed. Default 3.0. Nothing else would ever close it: a person who leaves frame produces no later label to disagree with. |

## Defining the polygon

```bash
python perception/tools/define_zone.py --zone-id zone-assembly-4
```

Click the corners in order, `s` prints the JSON, `u` undoes a point, `q` quits.
Paste the result into `zones`. Then check it with:

```bash
python perception/perceive.py --preview
```

Stand at the edge of the real zone and watch the ground-contact dot: that dot,
not the box, is what decides membership.

### Extend the polygon to y = 1.0 whenever subjects are cut off at the bottom

The one that will cost you an afternoon. If the camera is close enough that a
subject's bounding box is clipped by the bottom of the frame -- a laptop webcam
at a desk, a camera over a doorway, anything where feet are out of shot -- then
the ground point is pinned at **y ~ 0.99**, and a polygon whose bottom edge sits
at 0.98 excludes every single person.

It fails silently. Detection is perfect, the box is drawn, the dot is drawn, and
no event is ever emitted, because the dot is a pixel below the zone.

Measured on this repo's own laptop webcam: subject detected in 39 of 40 frames
at score 0.90, ground point `(0.53, 0.99)`, polygon bottom `0.98`, events
emitted: zero.

So: if feet are visible on the floor, put the boundary where the floor is. If
they are not, take the polygon to `1.0`. When a zone looks right in `--preview`
but emits nothing, check the bottom edge first.

## What the PPE stage costs

`ppe.enabled` buys a second full forward pass on every frame. It is not a few
more classes on the detector already running: hard hats and high-vis vests are
not COCO classes, so they need a different checkpoint.

Measured on this repo's own machine (Ryzen 4000-series mobile, 16 logical
cores, onnxruntime 1.29 CPU provider, 1280x720 frames):

| | mean | median | p95 |
|---|---|---|---|
| person detector, `yolo26` @ 640 | 55.5 ms | 55.4 ms | 59.7 ms |
| PPE detector, `yolo26-ppe` @ 640 | **54.1 ms** | 52.7 ms | 60.5 ms |
| ...of which letterbox + normalise | 10.0 ms | 9.7 ms | 11.4 ms |
| ...of which `session.run` | 40.8 ms | 40.6 ms | 43.8 ms |
| containment + emission, 6 people | 0.4 ms | 0.3 ms | 0.4 ms |
| **in the live loop, `perceive.py --ppe`** | **62.8 ms** | | |

Two things to take from it rather than the absolute numbers, which are this
machine's and nobody else's:

**It roughly doubles detection cost.** The PPE model is YOLO26n at 640, the
same architecture and the same input size as the person detector under the
`yolo26` backend — 10 output classes against 80, which the head barely feels.
The ratio measured here is 0.97. So on a machine where the person detector
costs 31 ms, expect the PPE pass to cost about the same and the pair to cost
about 62 ms, for a ceiling near 16 fps before anything else runs.

**A 416-input person detector is not the baseline.** If the 31 ms is
`yolox-onnx` at 416, the PPE pass is over 2.3x the pixels and will cost
considerably more than 31 ms, not the same.

The association and emission work is a rounding error — 0.4 ms for six people
— so the cost is the forward pass and nothing else. About 10 ms of it is the
letterbox, which both detectors currently do independently on the same frame;
sharing one blob would recover roughly 18% when both run at 640, and is the
first thing to try if the second pass has to fit in a tighter budget.

Every run prints its own number at shutdown, so the figure for the machine that
matters never has to be inferred from this table:

```
[perception] ppe detector: 62.8 ms/frame mean over 60 frames (45.1% of the 139.2 ms frame budget at 7.2 fps)
```
