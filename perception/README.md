# Perception layer

Webcam → person detection → zone membership → events on stdout.

A **separate process** from the engine. It imports nothing from `engine/`, and a
test enforces that by parsing every module's imports. The only thing crossing
between them is a stream of JSON lines matching `schema/event.schema.json`.

> **Licence, up front:** the obvious choice — Ultralytics YOLO — is **AGPL-3.0
> and cannot ship in a commercial iOACAM**. The default here is **YOLOX,
> Apache-2.0** for both code and weights. Full reasoning and the alternatives
> considered: **[LICENCE-NOTES.md](LICENCE-NOTES.md)**. Read it before choosing
> a model.

## Quick start

```bash
pip install opencv-python numpy                 # the only dependencies
python perception/tools/fetch_model.py          # yolox_tiny, Apache-2.0, ~20 MB

python perception/tools/define_zone.py          # click your zone, copy the JSON
                                                # into config/zones.example.json
python perception/perceive.py --preview         # check it against the live view
python perception/perceive.py                   # emit events
```

On a laptop webcam, start from `config/webcam.json` instead -- it is a
known-good config measured on a real one, with the zone taken to `y = 1.0`
because a seated subject's box is clipped by the bottom of the frame:

```bash
python perception/perceive.py --config perception/config/webcam.json --preview
```

Piping into a consumer, which is how it is meant to run:

```bash
python perception/perceive.py | python your_consumer.py
python perception/perceive.py --max-frames 300 | python perception/tools/validate_events.py
```

Tests need no camera, no weights and no network:

```bash
python perception/test_perception.py
```

## What it emits

| Observation | When | Source | Confidence |
|---|---|---|---|
| `person_in_zone` | A tracked person's ground point has been inside the polygon for `enter_frames` frames | `camera` | calibrated, < 1.0 |
| `person_left_zone` | Outside for `exit_frames` frames, or the track was lost while inside | `camera` | calibrated, < 1.0 |
| `person_count` | Zone occupancy changed and held for `count_debounce_frames` | `camera` | weakest occupant |
| `sensor_health` | Start-up, frame-grab failure, recovery, shutdown | `timer` | exactly `1.0` |

A real 301-frame clip with four people produces **on the order of a dozen
events, not one per frame**. Edge-triggered emission is the whole design: the
engine reasons about state changes, and a per-frame stream would bury them.

Sample entry event:

```json
{"event_id":"0178ac81-cfe4-4315-b84a-6103204f4ab4","timestamp_us":975081417180,
 "wall_time":"2026-08-31T17:40:27.193Z","source":"camera","sensor_id":"cam-01",
 "track_id":"trk-0001","observation":"person_in_zone","zone_id":"zone-assembly-4",
 "value":true,"confidence":0.6663,"subject":{"class":"human"},
 "confidence_components":{"raw_score":0.8982,"calibrated_score":0.7702,
   "quality":0.8651,"persistence":1.0,"agreement":1.0,"frames_observed":4}}
```

Measured on this repo's laptop webcam. The whole chain is legible in it: YOLOX
raw 0.898, softened to 0.770 by temperature scaling at T=1.8, then discounted to
0.666 by a quality term that noticed the subject was clipped by the frame edge.
That it lands near the 0.6 in the schema's own worked example is a reasonable
sign the calibration is in a sane range.

## Objects, and whose hand they are in

YOLOX has always been COCO-80; this layer used to discard 79 of the classes and
keep `person`. The enabled list is now config, because a different site wants a
different list:

```json
"classes": ["person", "cell phone", "laptop", "bottle", "cup",
            "backpack", "handbag", "book", "scissors", "knife"],
"class_thresholds": {"person": 0.35, "cell phone": 0.50, "knife": 0.60}
```

Only COCO names are accepted, and an unknown one is rejected at construction
rather than silently detecting nothing — the model knows 80 things and a torque
wrench is not one of them.

**It is nearly free.** One forward pass produces all 80 class scores whatever
you filter to. Measured on this laptop: 1 class 132.7 ms, 10 classes 123.0 ms —
the difference is noise, and post-processing is ~15 ms either way against a
~120 ms forward pass. The classes were always being computed and thrown away.

**Thresholds are per class** because one number cannot serve a person filling
the frame at 0.95 and a phone forty pixels across at 0.50. `knife` and
`scissors` sit highest deliberately: COCO confuses them with pens, cutlery and
phone edges, and a false knife is the most expensive false positive in the list.

**Objects get their own tracks**, one tracker per class, so a phone lying on a
laptop cannot inherit the laptop's id when it drops out for a frame. Without
tracks a phone seen in 3 frames of 10 emits three events for one phone and the
engine cannot tell it is the same phone. Track persistence reaches the event's
confidence, so a two-frame flicker is visibly weaker than a steady observation.

**Association is the inference**, and it lives in `association.py` alone rather
than spread through a loop. An object is held by a person if one of that
person's wrist keypoints falls inside the object's box, grown by a margin;
nearest wrist wins. A wrist, not a box overlap — a phone on a desk sits inside
its owner's bounding box whenever they lean over it. A margin, because the hand
occludes what it holds and the wrist lands just outside. And a wrist below the
score threshold cannot claim anything: an occluded wrist's position is a
prediction, and attributing a knife to a prediction is precisely the confident
wrong answer this system is built to avoid.

An object nobody is holding emits **with no `track_id`**, not dropped. A knife
on a bench with no one near it is a real observation, and often the more
interesting one; dropping it would make "no object" and "unattributed object"
the same stream.

```json
{"observation":"object_at_station","value":"cell phone","track_id":"trk-0001",
 "confidence":0.6071,"subject":{"class":"object"}}
{"observation":"object_at_station","value":"cup","confidence":0.5411,
 "subject":{"class":"object"}}
```

Both shapes are validated against `schema/event.schema.json` in
`test_objects.py`, using the repo's own validator.

**Not wired to the engine yet, on purpose.** `perceive.py` does not emit these.
The events are built by the real `EventEmitter` inside
`preview_pose.py --objects` and written to the demo's JSONL log, so the false
positive rate can be counted before anything acts on them:

```bash
python3 perception/tools/preview_pose.py --objects --log
```

Two honest gaps. The `confidence` on these events is the raw detector score
times track persistence — **not** the calibrated composition `confidence.py`
builds for people, because quality is computed against person box height and no
temperature has ever been fitted for objects. And association is monocular with
no depth: two people shoulder to shoulder will sometimes attribute to the wrong
one. It is evidence, not proof.

## Layout

```
perceive.py        capture -> detect -> track -> zones -> emit. The process.
zones.py           polygon config, point-in-polygon, enter/exit hysteresis
tracking.py        greedy IoU tracker; mints track_id, measures persistence
confidence.py      temperature scaling, quality, the components block
emit.py            event construction, monotonic clock, JSONL to stdout
detectors/
  base.py            Detection + the Detector protocol. The licence seam.
  yolox_onnx.py      Apache-2.0. Default. cv2.dnn, no extra runtime.
  ultralytics_yolo.py AGPL-3.0. Gated, evaluation only. Do not ship.
  stub.py            scripted boxes; makes the tests camera-free
config/            example config + field reference
tools/             fetch_model.py, define_zone.py, validate_events.py
test_perception.py offline suite, runs the real CLI end to end
```

## Which model

Measured on this laptop's CPU (16 threads, OpenCV DNN), 640×424 input, and on
the same 4-person scene:

| Model | Input | ms/frame | fps | People found |
|---|---|---|---|---|
| yolox_nano | 416 | 139 | 7.2 | 2 of 4 |
| **yolox_tiny** | **416** | **189** | **5.3** | **4 of 4** |
| yolox_tiny | 640 | 396 | 2.5 | 4 of 4 |
| yolox_s | 640 | 510 | 2.0 | 4 of 4 |

`yolox_tiny` at 416 is the default: it matches `yolox_s` recall at 2.6× the
speed. `yolox_nano` is faster still and **misses half the people**, which is the
one failure this system cannot tolerate — a missed person is a workflow step
that silently never happened.

5 fps is enough for workflow-step timing at a fixed station, and it sets the
tuning: `enter_frames: 5` is one second of confirmation, `exit_frames: 10` is
two. If you change the model, re-check those numbers — they are in frames, not
seconds, deliberately, because the debounce is about detector noise.

## Will it run on another laptop

Mostly yes, with caveats worth knowing before you promise it to anyone.

**Verified here:** Windows 11, Python 3.13, x86-64, `opencv-python` 4.12,
integrated webcam at 1280x720, 5.6-5.9 fps. Everything below that is not
Windows is reasoned from the code, not tested.

| Requirement | Notes |
|---|---|
| Python 3.10+ | Matches the engine. No compiled extensions of our own. |
| `opencv-python` + `numpy` | The only dependencies. No torch, no onnxruntime, no GPU. |
| ~20 MB for weights | Fetched separately; not in the repo. |
| A CPU | Works on any x86-64 or arm64 machine with an OpenCV wheel, Apple Silicon included. |

**`--preview` needs a GUI build.** `opencv-python-headless` has no `imshow` at
all, and plenty of environments install it by default. The process now probes
for a window before opening the camera and tells you exactly that, rather than
dying inside the capture loop. Everything except `--preview` and
`define_zone.py` works headless, so on a server you drop the flag and read the
events.

**Camera access differs by OS.** Index 0 is not always the built-in camera when
a virtual camera (OBS, Teams) is installed -- try `--source 1`. On macOS the
terminal or IDE running this needs Camera permission in System Settings, or
capture fails or returns black frames. On Linux you need `/dev/video*` and
membership of the `video` group. The error message names the right one for the
platform it is running on.

**Speed varies, and speed changes the tuning.** `enter_frames` and
`exit_frames` are counted in frames, not seconds, so the same config means
different things on a faster or slower machine. The process now reports its
measured throughput and translates it for you at shutdown:

```
[perception] 60 frames in 10.7s (5.6 fps), 2 events emitted
[perception] at 5.6 fps, enter_frames=4 is 0.7s and exit_frames=8 is 1.4s
```

Read that line on any new machine and adjust. A laptop half this speed makes
`enter_frames: 4` a 1.4-second confirmation, which may be too sluggish for a
fast workflow step; drop to `yolox_nano` only if you accept the recall loss
documented above, or reduce `input_size`.

**What will not change:** the engine is stdlib-only and runs anywhere Python
does. Nothing in this directory is required for it.

## Design decisions worth preserving

**Zone membership is tested at the ground point, not the box centre.** A person
standing just outside a zone leans over it constantly; their feet do not. The
centroid produces a stream of spurious enter/leave pairs the engine cannot tell
from real traffic.

**Emission is edge-triggered and hysteretic.** A raw per-frame polygon test on a
subject at the boundary flips at the detector's noise frequency. The debounce
belongs at the source: nothing downstream can reconstruct which of forty
crossings were real. 120 flickering boundary frames yield at most 4 events, and
a test pins that.

**Polygons are normalised (0..1), not pixels.** A webcam that negotiates
1280×720 today and 640×480 tomorrow would silently move every zone boundary.
Normalised coordinates survive that, and survive swapping in the board's sensor.

**Persistence measures detection stability, not time-in-zone.** Defining it as
"how much of the window agrees with the current zone state" scores *every entry
event near zero*, because at the instant of entry the subject has by definition
only just arrived. That was a real bug here: entry confidences came out at 0.05
and would have been discarded by every threshold in the policy. A subject
detected in 29 of the last 30 frames is solid whether they arrived a second ago
or a minute ago.

**The emitted stream is balanced per track, independently of the geometry.** An
entry suppressed by the `min_confidence` gate still flips the internal state, so
without care the later exit is emitted unpaired and the engine sees someone
leaving a zone it never saw them enter. `Membership.announced` tracks what was
actually *emitted*, not what happened. Also a real bug, caught by running the
thing rather than by reading it.

**Quality is a product, not a mean.** Pixel height, truncation, occlusion, blur
and luminance are independent ways of being unusable. A tack-sharp 12-pixel
person is still not identifiable, and averaging lets good lighting hide that.

**Temperature scaling happens here.** The README's open question — "where that
correction lives is a perception-layer decision" — is answered in
`confidence.py`. The engine keeps receiving confidence it can take at face
value. `calibration.temperature` **must match the policy file's value**, or the
engine compares confidences against thresholds tuned on a different scale.

**Identity and role are never guessed.** This layer runs no face recognition and
reads no badges, so `subject` carries `class: human` and nothing else. The
schema says unknown fields are omitted rather than guessed, and the engine's
`wrong_role` rules depend on that honesty. Role arrives from `access_control`,
not from a person detector.

**A clean shutdown still reports coverage lost.** Stopping the process is as
much a loss of coverage as a failed camera. Without the closing `sensor_health`,
an unwatched zone keeps reporting conformant — the exact failure scenario 12
exists to prevent. Note what is *not* done: people still inside a zone at
shutdown get no synthetic `person_left_zone`. They did not leave. The coverage
event says the zone is no longer observed, which is true; a fabricated exit
would be a lie the engine would happily act on.

**stdout is the contract; everything else is stderr.** A stray `print()` to
stdout corrupts the event stream for every downstream consumer.

## Two defects this work surfaced

**`schema/event.schema.json` rejected every integer value.** `value` used
`oneOf` across `boolean / integer / number / string`. In JSON Schema an integer
is *also* a number, so every integer matched two branches and `oneOf` failed —
meaning `person_count`, which the schema's own description calls for ("integer
for counts"), could never validate. The repo's scenarios only ever used booleans
and strings, so nothing had hit it. Changed to `anyOf`; the schema's three
examples and all 12 engine scenarios still pass.

**The two perception bugs above** were both found by running the pipeline on
real footage and reading the output, not by the unit tests, which passed
throughout. Both now have tests.

## Limits, stated plainly

- **One camera, one process.** `agreement` is fixed at 1.0 because there is no
  second sensor to corroborate. Multi-camera fusion is where that term earns its
  place in the schema.
- **The tracker swaps ids when people cross.** Greedy IoU is enough to mint a
  `track_id` and measure persistence; it is not enough for per-actor attribution
  in a crowd. ByteTrack (MIT — no licence problem) is the intended replacement.
- **No PPE, pose, action or identity.** The schema's `ppe_*`,
  `action_recognised` and `person_identified` observations need RTMPose and
  ST-GCN, per the README's pipeline. This layer emits four of the nineteen
  observations and nothing else — extending that list means committing to a
  detector, a feasibility assessment and an ATP test, as the schema says.
- **`min_height_px: 40` is a guess until you measure it.** Stand in the real
  zone with `--preview` and read the actual box height.
- **A zone polygon must reach `y = 1.0` if subjects are clipped by the bottom of
  the frame** (desk webcam, doorway camera). The ground point pins to ~0.99 and
  a boundary at 0.98 excludes everyone, silently, with detection working
  perfectly. See `config/README.md`. `config/webcam.json` is a working
  laptop-webcam config.
- **Calibration temperature 1.8 is copied from the example policy, not fitted.**
  A fitted value needs labelled site data. Until then the number is a
  placeholder that happens to be conservative.
- **`integrity` emits `seq` and `prev_hash` only** (off by default). The
  TPM-keyed signature belongs at log-write time, not in this hot path.
- **RKNN conversion is unverified.** YOLOX was chosen partly because its
  conversion path is well trodden, but as the repo README insists: confirm it on
  the actual board before committing. That check costs nothing now.
