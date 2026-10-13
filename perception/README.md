# Perception layer

Webcam → person detection → zone membership → events on stdout.

**`perceive.py` is the production events pipeline** (JSON-lines for the engine).
**`tools/preview_pose.py` is the live demo** (preview window + JSONL log).
Object detection, pose and action recognition are **opt-in stages** shared by
both (`objects.py`, `pose.py`, `actions_stage.py`). With stages off,
`perceive.py` is people-and-zones only.

A **separate process** from the engine. It imports nothing from `engine/`, and a
test enforces that by parsing every module's imports. The only thing crossing
between them is a stream of JSON lines matching `schema/event.schema.json`.

> **Licence, up front:** the obvious choice — Ultralytics YOLO, including
> YOLO26 — is **AGPL-3.0 and cannot ship in a commercial iOACAM**. The detector
> is a config choice (`detector.backend`), and the default is **YOLOX,
> Apache-2.0** for both code and weights. A `yolo26` backend exists alongside
> it for bench comparison only; an ONNX export does not relicense Ultralytics
> weights. Full reasoning and the alternatives considered:
> **[LICENCE-NOTES.md](LICENCE-NOTES.md)**. Read it before choosing a model.

## Quick start

```bash
pip install opencv-python numpy                 # the whole default path
# onnxruntime as well for the yolo26 backend, the PPE stage, and identity --
# the yolox-onnx default runs on the cv2.dnn already needed for capture
python perception/tools/fetch_model.py          # yolox_tiny, Apache-2.0, ~20 MB
                                                # the default backend; see
                                                # "Which model" for the yolo26
                                                # alternative

python perception/tools/define_zone.py          # click your zone, copy the JSON
                                                # into config/zones.example.json
python perception/perceive.py --preview         # check it against the live view
python perception/perceive.py                   # emit events (people + zones)

# Opt-in stages (same modules as preview_pose.py; need exported ONNX for real pose/actions)
python perception/perceive.py --config perception/config/webcam.json --objects
python perception/perceive.py --config perception/config/webcam.json --objects --actions
python perception/perceive.py --config perception/config/webcam.yolo26.json --ppe
```

On a laptop webcam, start from `config/webcam.json` instead -- it is a
known-good config measured on a real one, with the zone taken to `y = 1.0`
because a seated subject's box is clipped by the bottom of the frame:

```bash
python perception/perceive.py --config perception/config/webcam.json --preview
```

Config keys `objects.enabled`, `pose.enabled`, `actions.enabled` default to
false. CLI `--objects` / `--actions` override them (and imply pose). A stage
that is enabled while its model file is missing **exits** and names
`export_rtmpose.py` / `export_posec3d.py` — it does not run quietly without
the stage. PoseC3D uses `--action-every` / `actions.every_s` (default 1s), not
per-frame.

Piping into a consumer, which is how it is meant to run:

```bash
python perception/perceive.py | python your_consumer.py
python perception/perceive.py --max-frames 300 | python perception/tools/validate_events.py
```

Tests need no camera, no weights and no network:

```bash
python perception/test_perception.py
python perception/test_objects.py
python perception/test_yolo26.py
python perception/test_ppe.py
python perception/test_stages.py
python perception/test_intervals.py
```

## What it emits

| Observation | When | Source | Confidence |
|---|---|---|---|
| `person_in_zone` | A tracked person's ground point has been inside the polygon for `enter_frames` frames | `camera` | calibrated, < 1.0 |
| `person_left_zone` | Outside for `exit_frames` frames, or the track was lost while inside | `camera` | calibrated, < 1.0 |
| `person_count` | Zone occupancy changed and held for `count_debounce_frames` | `camera` | weakest occupant |
| `sensor_health` | Start-up, frame-grab failure, recovery, shutdown | `timer` | exactly `1.0` |
| `object_at_station` | Opt-in. Class in `value`; holder's `track_id` when attributed, omitted when unheld | `camera` | score × persistence |
| `action_recognised` | Opt-in. Class label, or `unknown` on abstention (not dropped) | `camera` | model confidence |
| `action_interval` | Opt-in. Consecutive windows of one label joined into a span; `start_us` / `end_us` carry it, with `wall_time_start` / `wall_time_end` as the same two instants on the wall clock, accurate to about one window. Emitted when the interval closes, which is `close_after` windows after it ended | `camera` | mean over its windows |
| `ppe_helmet` | Opt-in. `true` from a `Hardhat` box, `false` from a `NO-Hardhat` one, over a person. Nothing at all when the model said nothing | `camera` | score × persistence |
| `ppe_vest` | Opt-in. Same, from `Safety Vest` / `NO-Safety Vest` | `camera` | score × persistence |

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

Measured on this repo's laptop webcam, with the default YOLOX backend. The
whole chain is legible in it: the detector's raw 0.898, softened to 0.770 by temperature scaling at T=1.8, then discounted to
0.666 by a quality term that noticed the subject was clipped by the frame edge.
That it lands near the 0.6 in the schema's own worked example is a reasonable
sign the calibration is in a sane range.

## Who the event is about

`subject` says who or what an observation concerns. With no identity stage it
is `{"class": "human"}` on every camera event about a person, which is the
complete and correct answer for an anonymous detection and is what this layer
emitted before identity existed.

With `--identity` (or `identity.config`), `identity_stage.py` binds person
tracks to enrolled IDs and the same `subject` block carries the ID on **every**
event the loop emits for a bound track — `person_in_zone`, `person_left_zone`,
`action_recognised`, `action_interval`, `ppe_helmet` and `ppe_vest`:

```json
{"observation":"action_interval","track_id":"trk-0448","value":"reading",
 "start_us":184261000000,"end_us":184264500000,
 "wall_time_start":"2026-08-31T17:40:26.000Z",
 "wall_time_end":"2026-08-31T17:40:29.500Z","confidence":0.78,
 "subject":{"class":"human","identity":"EMP-4471","role":"operator"}}
```

**An unidentified track carries no `identity` key at all.** Not `null`, not
`""`, not `"unknown"`:

```json
{"observation":"ppe_helmet","track_id":"trk-0449","value":false,
 "confidence":0.28,"subject":{"class":"human"}}
```

The difference matters because the engine's `wrong_role` rules test for a role
or identity they can act on, and a falsy placeholder is a value those tests
would read as an answer. An absent key records that nobody has claimed to
know; it never records that nobody was there. `IdentityStage.subject_for`
omits both `identity` and `role` for an unbound track, and `test_stages.py`
asserts the key is absent rather than merely falsy, because that is the one
property a well-meaning refactor breaks.

**`track_id` never goes away.** The track id is what the camera observed; the
identity is an inference about who that was, carrying a match confidence of
its own. Keeping both is what lets an attribution be re-examined later against
the raw tracks — collapse them and the claim becomes the only record of
itself. See `identity/README.md` on authentication versus attribution.

**An interval resolves its subject when it closes, not when it opens.** A
track is routinely unbound for its first window — the person has to face the
camera before there is a face to match — and binds partway through. The
identity known at close is the better answer for the whole span, so
`IntervalBuilder` resolves there (`_finish`). It is still an inference applied
backwards over the span, which is sound only because the resolver binds once
per track and holds.

Identity stays **opt-in and off by default**. Naming people is a heavier
proposition, legally and operationally, than counting them — see the
regulatory note in `identity/README.md`.

## Both clocks, on a span

An `action_interval` event carries its span in four fields, on two clocks:
`start_us` / `end_us` in monotonic microseconds, and `wall_time_start` /
`wall_time_end` as ISO UTC. That is not redundancy, because neither clock
answers both questions.

The monotonic pair is what **orders and subtracts**. It cannot be reordered by
an NTP correction, which is exactly why `timestamp_us` is monotonic too, and
every duration rule in the engine reads it. What it cannot do is say when
anything happened: `338483682608` is microseconds since boot, not a time of
day.

The wall-clock pair is what a **human** reads — which shift, which CCTV clip,
which ticket. It is additive and is never used for sequencing or durations.

Both are converted through **one monotonic-to-wall offset captured at process
start** (`emit.wall_time_at`), not by reading the clock when the event is
built. An interval is emitted only once later windows have confirmed it ended,
so a clock read at emission would date a span that ran at 14:02:10 to 14:02:14
several seconds later — wrong by exactly the close delay. The shared offset
also means a duration read off the wall clock agrees with the monotonic one to
the millisecond.

The same two fields appear in `preview_pose.py --log` interval records, from
the same helper.

## PPE, and whose head the hat is on

`--ppe` (or `ppe.enabled`) runs a **second detector** on every frame. Hard hats
and high-vis vests are not COCO classes, so this is a different checkpoint and
therefore a whole extra forward pass — roughly doubling detection cost, which
is why it is off by default. The measured table is in
[config/README.md](config/README.md#what-the-ppe-stage-costs), and every run
prints its own figure at shutdown.

The person detector is untouched by it. Tracking, zones, occupancy and pose all
still come from `detector`, and nothing in the PPE path mints a track id — the
PPE model's own `Person` class is off by default for exactly that reason.

Three decisions are worth knowing before reading a log of it:

**The `NO-*` thresholds are low on purpose.** 0.20, against 0.40 for the
positive classes. Measured recall on the `NO-*` classes is 0.55–0.62: an unworn
hard hat is a smaller, less distinctive visual event than a worn one, because
there is no object to find. The costs run the other way — a false "no helmet"
costs somebody walking over to look, a missed one costs an unprotected head —
so the floors are asymmetric and the over-flagging that follows is the trade,
not a defect. `preview_pose.py --ppe --ppe-threshold 0.05` prints a per-class
score distribution to set them against your own footage.

**Association is box containment, and it refuses ties.** A hat inside one
person's box is theirs. A hat inside two boxes that are indistinguishable in
containment and size is attributed to *neither* — the same refusal
`association.py` makes for objects in hands, for the same reason. But it is
still emitted, with no `track_id`: an unattributable bare head is not the same
fact as no bare head, and dropping it would make those two the same stream.

**Absence of a detection is not a violation.** A person with no hat detection
at all produces no `ppe_helmet` event in either direction. A bare head is never
inferred from a missing hat — somebody facing away, half out of frame, or 20 px
tall produces no detection for reasons that have nothing to do with what they
are wearing.

## Objects, and whose hand they are in

Both COCO backends have always been COCO-80; this layer used to discard 79 of
the classes and keep `person`. The enabled list is now config, because a
different site wants a different list — and it is read by the same shared code
(`detectors/coco.py`) whichever backend is running, so it survives a backend
switch unchanged:

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
person's **hand anchors** lands within **reach** of the object's box; the
nearest anchor wins, and a near-tie between two people wins for neither.

*The anchor* is where the hand is, which is not where the wrist keypoint is.
COCO-17 has no finger keypoints, so a phone gripped in the fingers sits several
centimetres beyond the wrist, in the direction the forearm points. When wrist
and elbow are both confident the anchor is the wrist pushed forward along the
elbow-to-wrist vector by a third of the forearm. The plain wrist is kept as a
second anchor because extrapolation overshoots on a foreshortened arm, and the
elbow serves as a third when the wrist is not confident enough to be believed —
an occluded wrist should degrade the answer, not erase it. A wrist, not a box
overlap: a phone on a desk sits inside its owner's bounding box whenever they
lean over it.

*The reach* is a fraction of **body scale**, not of the object's box. This was
the bug. Reach as a fraction of the object made a 60 px phone allow 15 px,
which is smaller than the wrist-to-grip offset it needed to cover, while a
400 px laptop allowed 100 px for no reason but its own size. How far a grip
sits from a wrist keypoint is a property of the arm. Body scale is the longest
apparent limb on that person — the longest because perspective only ever
shortens a limb in the image, so the largest measurement is the least
foreshortened and the best estimate, and it tracks the subject automatically as
they walk towards the camera.

*The doubt* is two gates that decline rather than guess. A keypoint below
`min_wrist_score` is a predicted position, not an observed one, and attributing
a knife to a prediction is precisely the confident wrong answer this system is
built to avoid. And when two different people's anchors are within `tie_ratio`
of each other, the object goes to **neither**: nearest-anchor between two
equally close people is a coin toss, and the engine accumulates evidence
against whichever `track_id` it lands on.

**When attribution goes wrong, measure it before tuning it.**
`objects_seen: {"cell phone": 8}` beside `objects_attributed: {}` is a symptom
consistent with a bad detector, a bad pose, a bad threshold, a bad margin and
two stages running on different frames.
`tools/diagnose_association.py` replays a run log through the real
`associate()` and prints, per detection, every anchor with its score, the
distance to the object in pixels and as a fraction of both the object box and
body scale, and which gate rejected it:

```bash
python3 perception/tools/diagnose_association.py logs/run.jsonl --verbose
python3 perception/tools/diagnose_association.py logs/run.jsonl --compare
```

`--compare` also replays the previous object-box rule, so a change to this
geometry is reported as a before-and-after on real footage rather than
asserted. Every `HeldObject` carries the same `.report` in-process.

The settings live in the `objects` block of the config, and each is calibrated
against logged runs rather than chosen by taste — `body_margin` in particular
sits in the middle of the 0.30–0.35 plateau where every held phone in
`logs/run.jsonl` and `logs/both-live.jsonl` attributes and the desk cup in
`logs/objects-run2.jsonl` does not. At 0.40 the cup starts being attributed.

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

**Wired into `perceive.py` when `objects.enabled` or `--objects`.** The same
`ObjectStage` builds events for both the production stdout path and the demo
log. Count false positives with the demo first if you like:

```bash
python3 perception/tools/preview_pose.py --objects --log
python3 perception/perceive.py --config perception/config/webcam.json --objects
```

Two honest gaps. The `confidence` on these events is the raw detector score
times track persistence — **not** the calibrated composition `confidence.py`
builds for people, because quality is computed against person box height and no
temperature has ever been fitted for objects. And association is monocular with
no depth: an object directly behind a hand looks exactly like an object in it,
and two people shoulder to shoulder will sometimes attribute to the wrong one —
the tie gate abstains on the closest calls, but it cannot see depth either. It
is evidence, not proof.

## Layout

```
perceive.py        production events pipeline (stdout JSONL)
objects.py         object tracks + object_at_station events (shared)
ppe_stage.py       second detector + containment → ppe_helmet / ppe_vest (shared)
pose.py            RTMPose / stub pose stage (shared)
actions_stage.py   tubes → PoseC3D → action_recognised (shared)
action_intervals.py consecutive windows → action_interval (shared)
identity_stage.py  binds tracks to enrolled faces inside the loop; opt-in,
                   via --identity or the identity.config key
association.py     hand-anchor object attribution + per-object reports
zones.py           polygon config, point-in-polygon, enter/exit hysteresis
tracking.py        greedy IoU tracker; mints track_id, measures persistence
confidence.py      temperature scaling, quality, the components block
emit.py            event construction, monotonic clock, the startup
                   wall-clock offset, subject resolution, JSONL to stdout
detectors/
  base.py            Detection + the Detector protocol. The licence seam.
  vocabulary.py      a model's class list + validation + per-class floors
  coco.py            the COCO-80 list, as a Vocabulary. Shared by both backends.
  ppe.py             the 10-class PPE fine-tune's list, floors, and thin backend
  yolox_onnx.py      Apache-2.0. Default. cv2.dnn, no extra runtime.
  yolo26.py          AGPL-3.0. NMS-free. onnxruntime. Evaluation only.
  ultralytics_yolo.py AGPL-3.0. Gated, evaluation only. Do not ship.
  stub.py            scripted boxes; makes the tests camera-free
face/
  base.py            FaceBox + the FaceDetector protocol. Same seam shape.
  rfb_onnx.py        RFB-320 int8 via onnxruntime; ported 1:1 from the C++
actions/           RTMPose, pose tubes, PoseC3D — see actions/README.md
config/            example config + field reference
tools/             fetch_model, define_zone, validate_events, preview_pose (demo),
                   diagnose_association, export_rtmpose, export_posec3d,
                   export_yolo26
test_perception.py offline suite, runs the real CLI end to end
test_objects.py    object events against schema/event.schema.json
test_yolo26.py     YOLO26 decode, and that both backends agree exactly
test_ppe.py        PPE classes, asymmetric floors, containment, refusals
test_stages.py     opt-in stage wiring: which stages a config turns on
test_intervals.py  action interval join: flicker, close timing, timeouts
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

### The other backend

`detector.backend: "yolo26"` runs Ultralytics YOLO26 through onnxruntime
instead. It is NMS-free end to end — a one-to-one head, so no NMS threshold to
tune and no NMS operator in the graph, which is the part that matters for the
RK3568 conversion still outstanding.

**It is AGPL-3.0 in both code and weights and cannot ship**, exporting to ONNX
notwithstanding. It is here to answer one question on your own footage: does a
modern NMS-free detector buy enough recall over `yolox_tiny` to justify either
buying Ultralytics' commercial licence or porting to a permissive NMS-free
model (RT-DETR, D-FINE). See [LICENCE-NOTES.md](LICENCE-NOTES.md).

```bash
pip install ultralytics onnxruntime               # export-only, AGPL
python perception/tools/export_yolo26.py --download   # -> models/yolo26n.onnx
```

The row above has no YOLO26 entry on purpose: no measurement on the 4-person
scene has been taken yet, and a made-up number in a table people tune against
is worse than a gap. When you take one, note that **`yolo predict` is not a
valid comparison** — the `.pt` runs a one-to-many head plus NMS, while the
exported graph runs the one-to-one head, and they give different answers.
Compare `Yolo26Detector` against `YoloxOnnxDetector`, or ultralytics loading
the exported `.onnx`.

## Will it run on another laptop

Mostly yes, with caveats worth knowing before you promise it to anyone.

**Verified here:** Windows 11, Python 3.13, x86-64, `opencv-python` 4.12,
integrated webcam at 1280x720, 5.6-5.9 fps. Everything below that is not
Windows is reasoned from the code, not tested.

| Requirement | Notes |
|---|---|
| Python 3.10+ | Matches the engine. No compiled extensions of our own. |
| `opencv-python` + `numpy` | All the default path needs. No torch, no GPU. `onnxruntime` on top for the `yolo26` backend, the PPE stage and identity; pose falls back to `cv2.dnn` without it. |
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

**Identity and role are never guessed.** With no identity stage this layer
runs no face recognition and reads no badges, so `subject` carries
`class: human` and nothing else — that remains the default. With
`--identity` it carries an enrolled ID **only where the cascade actually bound
the track**: `subject_for` omits `identity` and `role` for an unbound one
rather than filling them in. The schema says unknown fields are omitted rather
than guessed, and the engine's `wrong_role` rules depend on that honesty — a
guessed role would produce a confident `wrong_role` finding against a name
nobody verified. Role otherwise arrives from `access_control`, not from a
person detector.

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
- **PPE covers helmets and vests only, and costs a second forward pass.**
  `--ppe` emits `ppe_helmet` and `ppe_vest`. `ppe_glasses` and `ppe_gloves`
  stay unimplemented: this model detects neither, and the schema's list is
  closed on purpose. It detects masks, which map to no observation and are
  therefore not emitted rather than being squeezed into `ppe_glasses`. The
  detector is AGPL-3.0 with **no permissive alternative in this repo**, unlike
  the person one — see `LICENCE-NOTES.md` before it becomes load-bearing.
- **PPE association is monocular containment, and can be confidently wrong.**
  A hard hat carried at waist height, or worn by somebody standing directly
  behind the person whose box contains it, looks exactly like one being worn.
  The tie gate abstains on the closest calls and emits them unattributed, but
  it cannot see depth. Same class of limitation as the object-in-hand rule
  above, and the same answer: evidence, not proof.
- **The rest of the observation list is not implemented.** Pose and action
  plumbing live in `actions/` / `pose.py` / `actions_stage.py` (RTMPose +
  PoseC3D). Identity is a separate package (`identity/`). With stages off,
  `perceive.py` emits four observations (`person_in_zone`, `person_left_zone`,
  `person_count`, `sensor_health`). With `--objects` / `--actions` / `--ppe` it
  also emits `object_at_station`, `action_recognised`, `action_interval`,
  `ppe_helmet` and `ppe_vest`. Extending further means committing to a detector, a feasibility
  assessment and an ATP test, as the schema says.
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
  the actual board before committing. That check costs nothing now. The same is
  true of the `yolo26` graph, which has one advantage there — being NMS-free,
  it contains no NMS operator for the converter to reject — and one
  disqualifying disadvantage, which is its licence.
