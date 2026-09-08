# Model licences: why the default is not Ultralytics

Engineering guidance, not legal advice. Before shipping, have counsel confirm
the conclusion. What follows is the reasoning that shaped this directory, so
that review is cheap.

## The short version

**Ultralytics YOLO (v5, v8, YOLO11, YOLO26) is AGPL-3.0, and that is
disqualifying for a commercial iOACAM.** The default backend here is **YOLOX**,
which is Apache-2.0 for both code and weights, and which the repo README
already names in the target pipeline.

There are now two Ultralytics-derived backends in `detectors/` — `ultralytics`
and `yolo26` — and both are evaluation tools, not delivery options. See
[YOLO26](#yolo26) below, which is the newer and more tempting of the two.

## Why AGPL is the wrong licence for this product

AGPL-3.0 is the strongest copyleft in common use. Three properties matter here.

**It reaches the whole combined work.** A product that incorporates AGPL code
must be released, in its entirety, under AGPL — complete corresponding source,
to every recipient. For iOACAM that is not just the perception layer. It is
plausibly the engine, the policy files, the site integrations and the response
matrix: the entire commercial asset.

**Section 13 adds a network trigger.** Ordinary GPL obligations attach on
distribution. AGPL adds that anyone interacting with the software *over a
network* must be offered the source. A SOC dashboard, a REST endpoint or a
remote review queue can trigger it without anything being shipped at all.

**Shipping an appliance is distribution.** "It runs on our box at the
customer's site" is conveying under the licence, not an escape from it. The
board build is as exposed as a downloadable one.

The README names defence and aerospace use cases. Procurement in those sectors
routinely prohibits copyleft in deliverables outright, so this is likely to be
a bid-blocker before it is ever a legal problem.

## Mitigations that do not work

**"It is a separate process, so it is not a derivative work."** This is the one
worth being most careful about, because this directory *is* a separate process
and it would be easy to read that as the fix. It is not. The process boundary
exists for the reasons the README gives — the board port swaps one process —
and it buys nothing legally. The FSF's own position treats communicating
programs case by case, on how intimate the communication is; it has not been
settled in court for this shape; and in any case the perception process itself
would still be a derivative work of the AGPL code and would still have to ship
as AGPL. You cannot launder a licence through a pipe.

**"We only ship the weights, not the training code."** Ultralytics applies
AGPL-3.0 to the pretrained weights as well as the source.

**"We never distribute — it is all on-premise."** Handing a customer a device
is distribution. And §13 can trigger on network interaction regardless.

**"We will strip it out before release."** By then the architecture has been
shaped around one model's API, its output format and its tracker. That is the
month-four version of exactly the failure the README warns about with RKNN
conversion.

**What does work: internal evaluation.** AGPL obligations attach on
distribution and network interaction, not on private use. Benchmarking
Ultralytics on your own footage, on your own machines, to decide whether YOLOX
is good enough, is fine. That is the only thing the `ultralytics` backend in
this directory is for, and it is why that backend refuses to load unless the
config sets `acknowledge_agpl: true`.

## YOLO26

Added as the `yolo26` backend in September 2026, recorded here because this
file exists to record exactly this kind of decision and a stale version of it
is worse than none.

**The licence position, factually.** YOLO26 is an Ultralytics model, released
by Ultralytics under the same dual scheme as YOLO11: **AGPL-3.0** for the code
and for the pretrained COCO checkpoints, or an Ultralytics Enterprise licence.
Nothing above it in this document changes. In particular:

- **Exporting to ONNX does not relicense the weights.** `detectors/yolo26.py`
  never imports `ultralytics` — it is onnxruntime and numpy, and the package is
  needed only by `tools/export_yolo26.py`, once, offline. That is a real
  engineering benefit (see below) and it is **not** a licence benefit. The
  numbers inside the graph are the AGPL checkpoint's numbers. This is the same
  mistake as "we only ship the weights, not the training code", one step
  further along, and it should be expected to fail for the same reason.
- **It is therefore in the same position as the `ultralytics` backend**:
  lawful for internal evaluation on your own footage and your own machines,
  because AGPL obligations attach on distribution and network interaction and
  not on private use; unlawful in a delivered product, including an appliance,
  without buying the Enterprise licence.

**Why it is here anyway, and why that is a risk worth naming.** YOLO26's
detection head is trained with one-to-one assignment, so it is NMS-free end to
end. That removes a tuned threshold from the pipeline and removes NMS from the
graph — and NMS is one of the operators an RKNN conversion is least likely to
accept. It is, in other words, technically attractive for exactly the board
problem this repo keeps flagging, which makes it the most likely of the AGPL
options to be adopted by accident.

**What is different from `ultralytics`, and deliberately so.** That backend
refuses to load without `acknowledge_agpl: true`. `yolo26` does not: it is
selected by `backend: "yolo26"` alone, so that a bench comparison is one config
string. The countermeasures are that it prints an AGPL banner to stderr on
first construction, that its `licence` attribute says
`NOT SHIPPABLE without a commercial licence from Ultralytics`, and that
`perceive.py` logs that attribute at the top of every run. **If you want the
harder gate, add the same `acknowledge_agpl` check `ultralytics_yolo.py` uses —
it is four lines.** Whoever ships this should decide that on purpose rather
than inherit it.

**If YOLO26's recall does turn out to justify a change**, the two lawful routes
below are the only ones. RT-DETR and D-FINE in the permissive table are also
NMS-free and Apache-2.0, and are the first thing to measure before pricing an
Enterprise licence.

## The two lawful routes

**Buy the Ultralytics Enterprise licence.** A real option. Price it before
dismissing it — but it is a recurring per-product cost, it is a dependency on
one vendor's commercial terms for the life of the product, and it has to be
renegotiated for the board variant.

**Use a permissively licensed model.** Cheaper, and it removes the question
permanently. This is what the code does.

## Permissive alternatives

| Model | Licence (code + weights) | Notes |
|---|---|---|
| **YOLOX** (Megvii) | **Apache-2.0** | **The choice here.** Already named in the README pipeline. ONNX export is first-class, RKNN conversions are well trodden, and it runs under OpenCV's DNN module with no extra runtime. |
| RTMDet (OpenMMLab) | Apache-2.0 | Stronger accuracy than YOLOX at similar cost. ONNX via MMDeploy. Heavier toolchain. |
| RT-DETR | Apache-2.0 | Transformer, NMS-free, so no NMS threshold to tune. Confirm RKNN support before committing — attention ops are where conversions fail. |
| D-FINE | Apache-2.0 | Newer, strong accuracy. Same conversion caveat. |
| DAMO-YOLO (Alibaba) | Apache-2.0 | Reasonable fallback if YOLOX recall is short. |
| MobileNet-SSD via OpenCV DNN | permissive | Much weaker, but a genuine zero-dependency floor if the board disappoints. |

### Licence traps in the same neighbourhood

Do not assume "not Ultralytics" means "safe":

- **YOLOv6** (Meituan), **YOLOv7**, **YOLOv9** — GPL-3.0. Copyleft, without the
  network clause but still disqualifying for a proprietary deliverable.
- **YOLO-NAS** (Deci) — the `super-gradients` library is Apache-2.0 but the
  **pretrained weights carry a separate restrictive licence** that excludes
  commercial use. The trap is that the repo badge says Apache and the weights
  do not.
- Ultralytics itself **relicensed from GPL-3.0 to AGPL-3.0 at v8.** A licence
  you checked once is not a licence you have checked.
- **YOLO26** is Ultralytics, despite the name carrying neither "Ultralytics"
  nor a "v". It is AGPL-3.0, exactly like YOLO11. The trap is that it is
  NMS-free and exports to a clean ONNX graph that runs with no Ultralytics
  code anywhere near it, which makes it *look* like a permissive option in a
  way YOLO11 never did. See [YOLO26](#yolo26) above.

Record the licence, the repo and the commit hash for every model you adopt, at
the moment you adopt it.

## The rest of the pipeline has the same problem

The README's target is `GStreamer -> YOLOX -> ByteTrack -> RTMPose -> ST-GCN`.
Worth clearing now rather than in month four:

| Component | Licence | Watch for |
|---|---|---|
| GStreamer core | LGPL-2.1 | Fine when dynamically linked. **But plugin sets differ**: `gst-plugins-ugly` and parts of `-bad` pull in GPL code, and **x264 is GPL**. An H.264 encode path can quietly make the whole appliance GPL. Audit the exact plugin set you ship, not "GStreamer". |
| ByteTrack | MIT | Clean. |
| RTMPose (MMPose) | Apache-2.0 | Code is clean, and `actions/rtmpose.py` uses it without installing MMPose at all. The **weights** are the question: the checkpoint in use is `rtmpose-tiny_simcc-aic-coco`, trained on COCO **and AI Challenger**. See `actions/LICENCE-NOTES.md`. |
| ST-GCN | varies by implementation | The original and the several forks are not all licensed alike. Check the specific repo. |
| OpenCV | Apache-2.0 (4.5+) | Clean. Note `opencv-contrib` includes non-free patented algorithms in some builds. |
| ONNX Runtime | MIT | Clean, if you use it instead of OpenCV DNN. |

## How this directory enforces it

- `detectors/` is a seam. The backend is a config string; nothing else in the
  codebase knows which model is running.
- The default is `yolox-onnx`, Apache-2.0 end to end, with no dependency beyond
  the OpenCV already needed for capture. Changing the default is a decision
  this file has to be updated for.
- `detectors/ultralytics_yolo.py` raises `PermissionError` unless the config
  explicitly sets `acknowledge_agpl: true`, and prints a banner to stderr on
  every run. It exists for bench comparison and must not be present in a
  shipped build.
- `detectors/yolo26.py` is AGPL-3.0 too and carries the same "must not ship"
  status, but is **not** gated behind `acknowledge_agpl` — see
  [YOLO26](#yolo26). It warns on stderr and in its `licence` string instead.
- `tools/export_yolo26.py` downloads AGPL weights and needs the AGPL
  `ultralytics` package. It is an offline tool; nothing under `detectors/` may
  import `ultralytics`, and the inference path stays onnxruntime and OpenCV.
- Every backend carries a `licence` attribute, and `perceive.py` logs it at
  startup. The licence of the running model appears in the log of every session.
