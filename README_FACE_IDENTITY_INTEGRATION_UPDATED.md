# Face Detection & Recognition Integration

This update adds real **face detection, face enrollment, and face recognition** support using RFB-320 and ArcFace ONNX models.

## New Files

| File                                            | Purpose                                                            |
| ----------------------------------------------- | ------------------------------------------------------------------ |
| `perception/models/version-RFB-320-int8.onnx`   | RFB-320 model used to detect faces                                 |
| `identity/models/arcfaceresnet100-11-int8.onnx` | ArcFace model used to generate face embeddings                     |
| `perception/face/base.py`                       | Defines `FaceBox` and common `FaceDetector` interface              |
| `perception/face/rfb_onnx.py`                   | Runs RFB-320 face detection using ONNX Runtime                     |
| `perception/face/__init__.py`                   | Provides `build_face_detector(cfg)` factory                        |
| `identity/arcface_matcher.py`                   | Generates ArcFace embeddings and compares them with enrolled users |
| `identity/enrollment.py`                        | Stores and loads enrolled face templates                           |
| `identity/enroll.py`                            | CLI utility for registering a user's face                          |
| `identity/live_recognize.py`                    | Standalone live camera face recognition using enrolled templates   |
| `identity/alignment.py`                         | Similarity warp onto ArcFace's 112x112 template before embedding   |
| `identity/landmarks.py`                         | Eye landmarks from the OpenCV Haar cascade; no extra model to fetch |
| `identity/test_alignment.py`                    | Alignment tests                                                    |
| `perception/identity_stage.py`                  | The join: binds person tracks to identities inside `perceive.py`   |

## Edited Files

| File                                    | Change                                                                    |
| --------------------------------------- | ------------------------------------------------------------------------- |
| `identity/matcher.py`                   | Added `build_matcher(cfg)` for selecting the real ArcFace or stub matcher |
| `identity/config/identity.example.json` | Added face model, template, backend, and model-hash configuration         |
| `perception/perceive.py`                | Runs `IdentityStage` when `--identity` (or `identity.config`) is given     |

## Enrollment

Register a person using the interactive enrollment CLI:

```bash
python3 identity/enroll.py
```

Then:

```text
Enter person ID: EMP-4471

Display name (optional). Shown on screen only; never used for matching, and
never written to an event. Press ENTER to skip.

Enter display name: Rajesh Kumar

Paste image paths one per line.
Press Enter on an empty line after the last image.
```

The key is the **person ID**, validated against
`face.person_id_pattern` (default `[A-Z]{2,5}-[0-9]{3,6}`), not a name. Names
are not unique and they change, and the ID is already what the badge reader
emits, so nothing downstream needs a translation step. `display_name` is a
separate field on the template, used for on-screen labels and nothing else.

Example:

```text
/home/rahul/Pictures/Webcam/front.jpg
/home/rahul/Pictures/Webcam/left.jpg
/home/rahul/Pictures/Webcam/right.jpg

```

Flow:

```text
Photo
  ↓
RFB-320 Face Detection
  ↓
Face Crop
  ↓
Align (identity/alignment.py)         onto ArcFace's 112x112 template
  ↓
ArcFace
  ↓
512-D Face Embedding
  ↓
templates.json                        keyed on the person ID
```

Enrollment requires **exactly one detected face** in the image.

## Face Recognition

Run standalone live face recognition:

```bash
python3 identity/live_recognize.py
```

The recognizer opens camera `0`, detects faces with RFB-320, generates ArcFace embeddings, compares them with enrolled templates, and applies the configured threshold and margin policy.

Press `Q` or `ESC` to exit.

The recognition components support:

```text
Face Crop
   ↓
ArcFace
   ↓
512-D Embedding
   ↓
Compare with Enrolled Templates
   ↓
Similarity Score
   ↓
Threshold + Margin Check
   ↓
Identity / Unverified
```

The matcher also verifies that enrolled templates were generated using the same ArcFace model.

## Current Capabilities

The project now has:

* RFB-320 face detection
* ArcFace face recognition
* 512-D face embedding generation
* photo-based user enrollment
* persistent face-template storage
* similarity-based identity matching
* configurable recognition thresholds
* model SHA-256 integrity checking
* integration with the existing `IdentityResolver`, inside the live loop

## In the Live Loop

`perception/identity_stage.py` is the join, and `perceive.py` runs it. Identity
is **opt-in**: give it a config and every event the loop emits for a bound
track carries the person ID. Without one, the loop behaves exactly as it did before
and every subject is an anonymous human, which stays the right default —
naming people is a heavier proposition, legally and operationally, than
counting them.

```bash
python3 perception/perceive.py --identity identity/config/identity.example.json
```

Or set `identity.config` in the perception config instead of passing the flag.
What happens per frame:

```text
Live Camera
    ↓
Person Detection + Tracking
    ↓
Face Detection (RFB-320)              only while some track is still unbound
    ↓
Associate Face with Track             by containment
    ↓
Align (identity/alignment.py) + ArcFace Recognition
    ↓
IdentityResolver                      the tiered cascade
    ↓
Identity ↔ track_id                   bound once, for the life of the track
    ↓
Workflow Events                       carrying the person ID
```

Two properties of that binding are deliberate and worth knowing before you
read a log:

**A bound track is never re-matched.** The answer cannot change while the
track persists, and re-running ArcFace every frame would spend an inference to
confirm what is already known. The cost is that a tracker id swap carries the
wrong name until that track dies. That is a real error, and it is in the open
list rather than hidden behind a per-frame re-check that would mask it.

**Face detection is skipped when there is nothing to do.** A frame in which
everyone visible is already identified costs nothing.

**Which events carry it.** Every camera observation the loop emits about a
bound track: `person_in_zone`, `person_left_zone`, `action_recognised`,
`action_interval` and the PPE pair, alongside the three identity observations
the cascade emits itself. The stages that build those events are handed a
`subject_for` callable rather than the identity stage, so `actions_stage.py`,
`action_intervals.py` and `ppe_stage.py` keep no dependency on `identity/`.

**An unbound track carries no `identity` key.** Not `null`, not an empty
string, not `"unknown"` -- the key is absent, because the engine's
`wrong_role` rules test for a role they can act on and would read a
placeholder as an answer. And `track_id` stays beside the identity rather than
being replaced by it: the track id is what the camera observed, the identity
is a claim about who that was, and an attribution has to remain questionable.

Where there is no access control system the roster is empty, and an empty
roster gives the cascade nobody to compare against; `access_control=False`
seeds it from the enrolled templates instead — see `_seed_from_templates` for
what that costs.

`identity/live_recognize.py` stays as a standalone recogniser, useful for
checking the models on their own without the perception loop around them.

