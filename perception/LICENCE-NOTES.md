# Model licences: why the default is not Ultralytics

Engineering guidance, not legal advice. Before shipping, have counsel confirm
the conclusion. What follows is the reasoning that shaped this directory, so
that review is cheap.

## The short version

**Ultralytics YOLO (v5, v8, YOLO11) is AGPL-3.0, and that is disqualifying for
a commercial iOACAM.** The default backend here is **YOLOX**, which is
Apache-2.0 for both code and weights, and which the repo README already names
in the target pipeline.

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

Record the licence, the repo and the commit hash for every model you adopt, at
the moment you adopt it.

## The rest of the pipeline has the same problem

The README's target is `GStreamer -> YOLOX -> ByteTrack -> RTMPose -> ST-GCN`.
Worth clearing now rather than in month four:

| Component | Licence | Watch for |
|---|---|---|
| GStreamer core | LGPL-2.1 | Fine when dynamically linked. **But plugin sets differ**: `gst-plugins-ugly` and parts of `-bad` pull in GPL code, and **x264 is GPL**. An H.264 encode path can quietly make the whole appliance GPL. Audit the exact plugin set you ship, not "GStreamer". |
| ByteTrack | MIT | Clean. |
| RTMPose (MMPose) | Apache-2.0 | Clean. |
| ST-GCN | varies by implementation | The original and the several forks are not all licensed alike. Check the specific repo. |
| OpenCV | Apache-2.0 (4.5+) | Clean. Note `opencv-contrib` includes non-free patented algorithms in some builds. |
| ONNX Runtime | MIT | Clean, if you use it instead of OpenCV DNN. |

## How this directory enforces it

- `detectors/` is a seam. The backend is a config string; nothing else in the
  codebase knows which model is running.
- The default is `yolox-onnx`, Apache-2.0 end to end, with no dependency beyond
  the OpenCV already needed for capture.
- `detectors/ultralytics_yolo.py` raises `PermissionError` unless the config
  explicitly sets `acknowledge_agpl: true`, and prints a banner to stderr on
  every run. It exists for bench comparison and must not be present in a
  shipped build.
- Every backend carries a `licence` attribute, and `perceive.py` logs it at
  startup. The licence of the running model appears in the log of every session.
