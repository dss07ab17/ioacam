"""PoseC3D backend, with the optional RGBPose-Conv3D second branch.

RUNS. The published pose-only checkpoint is exported by
`perception/tools/export_posec3d.py` and this backend drives it through
onnxruntime, at ~157 ms per tube on a laptop CPU. `test_posec3d.py` covers the
chain and `preview_pose.py --actions` runs it live off a camera.

What it does NOT do yet is anything useful for a factory. The stock head is
NTU-60 -- a daily-living vocabulary recorded in a lab -- so on real footage it
abstains, correctly, nearly all of the time. Treat the working chain as proof
the plumbing is right, not as a classifier anyone should act on. The head has
to be retrained on site classes before it means anything.

    +--------------------------------------------------------------------+
    |  LICENCE: PoseC3D checkpoints are commonly trained on NTU RGB+D,    |
    |  which is research-only under an academic use agreement, and a      |
    |  fine-tuned model INHERITS that restriction. Check before building  |
    |  on it. See LICENCE-NOTES.md. Same class of problem as the AGPL     |
    |  detector backend: free to resolve now, expensive once a classifier |
    |  is trained and embedded.                                           |
    +--------------------------------------------------------------------+

Why this model for live surveillance:

  Fast. It convolves over small heatmap volumes (17 x 24 x 56 x 56) rather than
  raw pixels, so it is roughly an order of magnitude cheaper than a video
  transformer and can plausibly stay on the device.

  Robust to bad pose. Heatmaps encode uncertainty as blob amplitude, so a
  hallucinated wrist is faint rather than indistinguishable from a real one.
  That is the property that matters at 8 metres with helmets, and it is where
  PoseC3D beats a skeleton-graph model like ST-GCN.

  Extensible to objects. A skeleton cannot see what is in the hand -- phone
  versus spanner is invisible. Enabling the RGB branch (RGBPose-Conv3D) adds a
  pixel pathway with cross-connections to the pose pathway, in the same
  framework and the same training pipeline. Start pose-only for live body
  motion; turn the branch on when object-involving classes need training.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

# The stock head's vocabulary, in the checkpoint's own class order (NTU A1-A60).
# Order is not cosmetic: it is the only thing mapping logit index to meaning,
# and a shuffled list turns every prediction into a different, plausible,
# wrong label with no error anywhere.
#
# Read this list before planning around it. It is a *daily living* vocabulary
# recorded in a lab -- drinking, brushing teeth, taking a selfie -- and the
# last eleven classes need two people. For a factory floor almost none of it
# applies, and the handful that might (falling, staggering, walking) are the
# argument for keeping the backbone and retraining the head rather than for
# shipping these sixty.
NTU60_CLASSES = (
    "drink water", "eat meal/snack", "brushing teeth", "brushing hair",
    "drop", "pickup", "throw", "sitting down",
    "standing up (from sitting position)", "clapping", "reading", "writing",
    "tear up paper", "wear jacket", "take off jacket", "wear a shoe",
    "take off a shoe", "wear on glasses", "take off glasses",
    "put on a hat/cap", "take off a hat/cap", "cheer up", "hand waving",
    "kicking something", "reach into pocket", "hopping (one foot jumping)",
    "jump up", "make a phone call/answer phone",
    "playing with phone/tablet", "typing on a keyboard",
    "pointing to something with finger", "taking a selfie",
    "check time (from watch)", "rub two hands together", "nod head/bow",
    "shake head", "wipe face", "salute", "put the palms together",
    "cross hands in front (say stop)", "sneeze/cough", "staggering",
    "falling", "touch head (headache)",
    "touch chest (stomachache/heart pain)", "touch back (backache)",
    "touch neck (neckache)", "nausea or vomiting condition",
    "use a fan (with hand or paper)/feeling warm",
    "punching/slapping other person", "kicking other person",
    "pushing other person", "pat on back of other person",
    "point finger at the other person", "hugging other person",
    "giving something to other person", "touch other person's pocket",
    "handshaking", "walking towards each other",
    "walking apart from each other",
)


class PoseC3DRecognizer:
    name = "posec3d"
    licence = "CHECK BEFORE SHIPPING -- NTU-pretrained weights are research-only"

    def __init__(
        self,
        model_path: str,
        classes: Sequence[str],
        num_frames: int = 32,
        heatmap_size: int = 56,
        num_keypoints: int = 17,
        device: str = "cuda",
        rgb_branch: bool = False,
        input_size: int = 224,
    ) -> None:
        self.model_path = model_path
        self.classes = list(classes)
        self.num_frames = num_frames
        self.heatmap_size = heatmap_size
        self.num_keypoints = num_keypoints
        self.device = device
        self.rgb_branch = rgb_branch
        self.input_size = input_size
        self._session = None

    def _load(self):
        if self._session is not None:
            return self._session
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "PoseC3D backend needs onnxruntime. Export the MMAction2 "
                "checkpoint to ONNX first. Staying in ONNX is also what keeps "
                "the RKNN path open, which matters if this is to run on the "
                "device rather than a server."
            ) from exc
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.device == "cuda"
            else ["CPUExecutionProvider"]
        )
        self._session = ort.InferenceSession(self.model_path, providers=providers)
        return self._session

    def infer(self, tube) -> dict[str, float]:
        heat = getattr(tube, "heatmaps", None)
        if heat is None:
            raise TypeError(
                "PoseC3D needs a PoseTube (heatmap volume). Build it with "
                "PoseTubeExtractor."
            )

        # Fail here with a readable message rather than three layers into the
        # network. The published pose_only checkpoint has 17 input channels on
        # backbone.conv1 and was trained at clip_len=32, scale=(56,56); a
        # mismatched volume either crashes or, worse, runs and quietly returns
        # nonsense.
        k, t, h, w = heat.shape
        if (k, t, h, w) != (
            self.num_keypoints, self.num_frames, self.heatmap_size, self.heatmap_size
        ):
            raise ValueError(
                f"volume is {(k, t, h, w)} but this backend expects "
                f"{(self.num_keypoints, self.num_frames, self.heatmap_size, self.heatmap_size)}. "
                f"Match PoseTubeExtractor's num_keypoints/num_frames/heatmap_size "
                f"to the checkpoint the model was exported from."
            )

        session = self._load()


        # N, C, T, H, W -- one person per call, so N = 1.
        x = heat[None].astype(np.float32)
        feeds = {session.get_inputs()[0].name: x}

        if self.rgb_branch:
            if tube.frames is None:
                raise ValueError(
                    "rgb_branch is enabled but the tube carries no frames. Set "
                    "keep_rgb=True on the PoseTubeExtractor."
                )
            rgb = tube.frames.astype(np.float32) / 255.0
            rgb = np.transpose(rgb, (3, 0, 1, 2))[None]   # 1,C,T,H,W
            if len(session.get_inputs()) < 2:
                raise ValueError(
                    "rgb_branch is enabled but the exported model has one "
                    "input; export the RGBPose-Conv3D variant."
                )
            feeds[session.get_inputs()[1].name] = rgb.astype(np.float32)

        logits = session.run(None, feeds)[0][0]

        # zip() would truncate silently. The example config carries six site
        # class names while the stock head emits sixty NTU logits, so without
        # this the first six logits get site labels and every prediction is
        # confidently, invisibly wrong -- the exact failure this file's shape
        # validation exists to prevent, one line further on.
        if len(self.classes) != len(logits):
            raise ValueError(
                f"model emits {len(logits)} logits but {len(self.classes)} "
                f"class names were configured. The stock checkpoint is "
                f"NTU-60: pass posec3d.NTU60_CLASSES, or fine-tune "
                f"cls_head.fc_cls to your own class count and export again. "
                f"Pairing them off in order would relabel someone else's "
                f"classes with your names."
            )
        return {c: float(v) for c, v in zip(self.classes, logits)}
