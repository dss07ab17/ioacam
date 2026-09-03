# Action recognition

Per-person action classification, feeding `action_recognised` events into the
engine.

```bash
python3 perception/actions/test_actions.py
python3 perception/actions/test_rtmpose.py
python3 perception/actions/test_posec3d.py
```

## Status

**The whole chain runs.** Camera → YOLOX → IoU tracker → RTMPose → pose tube →
PoseC3D → abstention, live, measured:

```bash
python3 perception/tools/export_posec3d.py --download   # once, offline
python3 perception/tools/preview_pose.py --actions      # live, with a window
```

```
detect        103.5 ms per frame   (YOLOX-tiny, cv2.dnn)
pose            8.7 ms per frame   (RTMPose-t, onnxruntime)
tube build     16.8 ms per tube    (numpy)
posec3d       157.5 ms per tube    (onnxruntime)
```

PoseC3D runs on a cadence — once a second per track by default — not per frame.
One tube costs more than a whole frame budget, and the evidence window is 1.5
seconds long, so re-answering every frame would halve the capture rate to
re-ask a question whose input has barely changed.

**But it classifies nothing useful yet, and that is expected.** The stock head
is NTU-60: a daily-living vocabulary recorded in a lab — *drink water*,
*brushing teeth*, *taking a selfie* — and the last eleven classes need two
people. On real footage it abstains nearly every time, with the nearest class
scoring 0.1–0.3. That is the abstention policy doing its job, not a fault. The
working chain proves the plumbing; the head has to be retrained on site classes
before any of it means anything.

`posetube.py`, `base.py` and `fusion.py` remain the tested core.

### Abstention is not an out-of-distribution detector

Worth knowing before the prohibited-action argument leans on it. Measured
against the real exported graph:

| Input | Verdict | Top class |
|---|---|---|
| Empty volume (nothing above threshold) | abstains | flat, 0.07 |
| Real camera footage, seated at a desk | abstains | 0.10 – 0.27 |
| Uniform noise | **decides** | *cheer up*, **1.000** |
| Static synthetic figure | **decides** | *touch back*, **0.936** |

Abstention catches a model that is *unsure*. It does not catch a closed-set
softmax that is confidently wrong on input unlike anything it trained on — and
with `temperature` still at its uncalibrated 1.0, sixty classes saturate
easily. A garbage volume, a badly cropped person or a tracker splice can all
produce a confident label. `test_posec3d.py` asserts this rather than assuming
it, so the claim cannot quietly rot.

The mitigations are the ones the repo already names: fit `temperature` on
held-out site data at commissioning, keep `coverage` in the record so a volume
built mostly from interpolation can be discounted, and do not let a blocklist
be the only thing standing between a wrong label and a response.

`rtmpose.py` **is** executed. The published RTMPose-t checkpoint runs, and
`test_rtmpose.py` puts a synthetic figure through it and checks the skeleton
comes back anatomically ordered — nose above shoulders above hips above ankles,
left shoulder on the opposite side from the right. That last check is the one
worth having: the crop geometry, the SimCC decode and the mapping back to frame
coordinates can each be wrong on their own and still produce seventeen
plausible keypoints, and only all three being right puts them in the right
places.

`stub.py` and `StubPoseEstimator` are still what the action tests use.

## Pose: RTMPose-t through ONNX

Same shape as the detector. The checkpoint is converted once, offline, and what
ships is a graph plus a few dozen lines of numpy:

```bash
python3 perception/tools/export_rtmpose.py     # .pth -> perception/models/rtmpose_t.onnx
```

MMPose is deliberately not a dependency. The mm\* stack pins its own torch and
numpy, does not install cleanly on the board, and would be pulled in only to
run `torch.onnx.export` once — so `rtmpose_arch.py` rebuilds the same module
graph in plain torch and loads the published weights into it, refusing any name
or shape that does not match. Torch is needed for the export and nowhere else.
Staying in ONNX is also what keeps the RKNN path open, which is the same reason
the detector is an ONNX model. The export self-verifies against torch before
you trust it — not just numerically, but by checking no keypoint decodes into a
different bin, because a small drift is fine and a moved argmax is a moved
joint.

**Look at it running** before trusting any of the above:

```bash
python3 perception/tools/preview_pose.py               # live window, whole-frame box
python3 perception/tools/preview_pose.py --detector    # real boxes, needs yolox_tiny.onnx
```

Joint colour is the model's confidence, which is the number the tube turns into
blob amplitude. Put a hand behind your back and watch the wrist go dim rather
than vanish or stay bright in the wrong place — that fading is the property the
whole heatmap design rests on, and a preview window is the only place you can
watch it happen.

**Runtime.** onnxruntime, at about 4 ms per person on a laptop CPU. If it is
not installed the backend falls back to the `cv2.dnn` already needed for
capture — same graph, agrees to 4e-6, but ~16 ms per person and one person per
call, so it says so on stderr rather than quietly costing you three quarters of
the frame budget. A test cross-checks the two runtimes decode the same
skeleton, since that agreement is the only thing that makes the fallback worth
having.

**The numbers are read, not chosen.** The SimCC head emits 384 x-bins and 512
y-bins — 192×256 at a split ratio of 2.0 — so decoding is an argmax per axis
divided by 2.0, then the inverse of the crop affine. `rtmpose.py` checks the
graph against those numbers at first inference, because a bin-count mismatch
does not crash: it scales every keypoint by the ratio of the two counts and
returns a perfectly plausible skeleton in the wrong place.

**The peak value is the point.** The winning bin's score is the model's
confidence in that joint, and `posetube.py` uses it directly as blob amplitude.
A backend that returned a constant would silently delete the uncertainty
weighting that makes a heatmap action model beat a skeleton-graph one at 8
metres — so the score is carried through unaltered apart from a clamp to 0..1,
and the tube's proportionality to it is tested.

Top-down (one crop per tracked person) rather than bottom-up: the boxes already
exist, per-person accuracy at range is much better, and identity comes from the
tracker instead of from a grouping step that can swap two workers' limbs
between frames. The cost is one forward pass per person, which is why the batch
is a dynamic axis and why the default is the *tiny* variant.

## The model

**PoseC3D**, with the RGB branch (RGBPose-Conv3D) available when object actions
need it. It is the only backend; `stub` exists for tests.

It wins on the three things asked of it at once:

**Body motion, live.** It convolves small heatmap volumes rather than pixels,
so it is roughly an order of magnitude cheaper than a video transformer and
plausibly stays on the device.

**Robust to the pose you actually get.** A skeleton-graph model like ST-GCN
treats every joint coordinate as fact, so a hallucinated wrist is
indistinguishable from a real one. Heatmaps encode uncertainty as blob
amplitude: an unsure joint is faint, and the convolution downweights it with
nothing special-cased. That property is why it holds up at 8 metres with
helmets, and it is the main reason to prefer it over ST-GCN.

**Extensible to objects.** A skeleton cannot see what is in the hand — phone
versus spanner is invisible. The RGB branch adds a pixel pathway with
cross-connections to the pose pathway, in the same framework and the same
training pipeline. One config flag.

Workflow step actions are just more classes. Train "loading fixture" and
"torque applied" and they emit `action_recognised` events that step evidence
already consumes; the engine needs no change.

**Staging:** pose-only first, for live body motion. Turn the RGB branch on when
you have object-involving classes to train, and accept the compute cost then.

Note pose runs before the action model, so this is two models, not one. Whether
that fits the RK3568 is the board measurement still outstanding.

## Abstention is the important part

A classifier trained on N classes does not say "I have not seen this before."
It returns the nearest class it knows, with a mediocre score. So an action
nobody trained it on comes back as `standing, 0.4` and silently disappears.

That matters most for a prohibited-action list, which inverts the safety
default this system otherwise holds: everything undeclared becomes `unknown`
and a human looks at it, but a blocklist says everything not listed is fine.
Without abstention, such a list only catches what the model was already taught.

Two independent tests, both must pass:

- **Confidence** — the top class clears a threshold, per class, because
  fine-grained classes are far less separable than coarse ones
- **Margin** — the top class beats the runner-up. Two classes at 0.41 and 0.39
  is not a recognition, it is a coin flip, and on a real floor the confusable
  pairs (tightening vs inspecting) are exactly the ones that matter

Fail either and the result is an abstention, which reaches the engine as
`unknown` and gets reviewed.

## Latency is set by the window, not the model

No action model can classify an action before it has happened. A 1.5-second
window means action findings arrive at least 1.5 seconds late, and that is
true of PoseC3D, ST-GCN and video transformers alike.

So for live surveillance the split is: geometry (zones, occupancy, PPE, falls
by pose aspect ratio) carries the sub-second alerting, and action recognition
adds context slightly behind. Anything you would actually interrupt has to come
from the geometry path.

The window is therefore a latency budget as much as an accuracy knob.

## The numbers are read, not chosen

`num_frames=32`, `heatmap_size=56`, `num_keypoints=17`, `sigma=0.6` come from
the published `pose_only` checkpoint itself, not from a guess:

```
backbone.conv1.conv.weight   (32, 17, 1, 7, 7)   -> 17 keypoint channels
cls_head.fc_cls.weight       (60, 512)           -> 60 NTU classes, 512-d features
clip_len=32   scale=(56,56)   hw_ratio=1.0
with_kp=True  with_limb=False
```

`tools/inspect_checkpoint.py` extracts this from any `.pth` without needing
torch installed. Run it whenever a new checkpoint arrives, before wiring it up.

This matters more than it looks. A volume of the wrong temporal length does not
reliably crash — it can run and return nonsense, which is far worse than an
error. `posec3d.py` therefore validates the volume shape at the boundary,
before loading the model, and the defaults are covered by a test.

The RGBPose variant uses different numbers (RGB clip_len 8 against pose 32, and
64x64 heatmaps). They are not interchangeable with the pose-only path.

The stock head is 60 NTU classes, so out of the box it predicts things like
"drinking water". For site classes, replace `cls_head.fc_cls` with your class
count and fine-tune; the 512-d backbone features are what you are keeping.

## Pose tubes

PoseC3D consumes a heatmap volume, not an RGB stack, so `posetube.py` builds it.
Three decisions there shape how well it generalises, and all three are tested:

**Normalise to the person's box, not the frame.** A worker at 3m and at 15m
perform the same action. Normalising to the frame would make the model learn
camera distance and fail at the first site with a different mounting height.

**One box across the window, not per frame.** Per-frame normalisation centres
the person in every slice and deletes their motion across the scene — walking
and standing would look identical.

**Scale each blob by keypoint confidence.** The property that makes heatmaps
beat graphs on noisy pose. A keypoint below `min_keypoint_score` contributes
nothing at all, because plotting it at its guessed position would assert a
joint location there is no evidence for.

Gaps are filled from the last known keypoints so the frame count and temporal
spacing stay constant — an unevenly sampled volume changes how fast the action
appears to happen. Coverage is reported rather than hidden, so a volume built
mostly from filled frames can be discounted. Tracks visible for less than half
the window are not offered for classification at all, or the model would be
classifying its own interpolation.

## Two streams

RGB sees objects. Skeletons survive lighting, clothing and site-to-site
appearance shift, and let you retain stick figures instead of video.

Fusion is deliberately conservative: agreement raises confidence slightly,
disagreement **abstains**. At least one stream is wrong and nothing says which,
so acting on the louder one makes two streams less reliable than one.


## Before shipping

Read `LICENCE-NOTES.md`. PoseC3D checkpoints are commonly NTU RGB+D pretrained,
which is research-only, and a fine-tuned model inherits the restriction of what
it was fine-tuned from.

Also note this sits outside the declared-workflow scope. Wrong zone, wrong
role, wrong order, skipped step and timing need none of it. Action recognition
is an extension, and it carries a heavier regulatory position — see the
regulatory triage document.
