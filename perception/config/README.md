# Config reference

Two configs ship here:

- **`zones.example.json`** -- the documented template, an industrial floor view.
- **`webcam.json`** -- known-good for a laptop webcam, measured on a real one.
  Zone taken to `y = 1.0`, `reference_height_px` raised to 400 for a close
  subject, and shorter debounce windows to suit ~6 fps.

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
| `objects.enabled` | Opt-in. Track non-person COCO classes; emit `object_at_station` on the production path. Implies pose. |
| `pose.enabled` | Opt-in RTMPose (or `backend: stub` for tests). Implied by objects/actions. |
| `actions.enabled` | Opt-in PoseC3D on `actions.every_s` cadence; emit `action_recognised` (including abstentions as `value: "unknown"`). |

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
