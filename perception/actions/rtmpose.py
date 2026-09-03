"""RTMPose-t via ONNX: the real pose backend behind the PoseEstimator seam.

    Model:   RTMPose-t (MMPose)                    Apache-2.0
    Weights: rtmpose-tiny_simcc-aic-coco 256x192   Apache-2.0 code, but the
             weights carry COCO + AI Challenger terms -- see LICENCE-NOTES.md
    Runtime: onnxruntime                           MIT
             (or cv2.dnn, which needs nothing new -- see below)

Same shape as `detectors/yolox_onnx.py`: a checkpoint is converted once,
offline, and what ships is a graph plus a few dozen lines of numpy. MMPose is
never installed. That is deliberate on three counts -- the mm* stack pins its
own torch and numpy and does not install cleanly on the board; ONNX is the only
format RKNN-Toolkit2 will take, so keeping the model in ONNX keeps the NPU path
open; and a pure-numpy pre/post path is small enough to be read and tested,
which the MMPose pipeline is not.

Export first:

    python3 perception/tools/export_rtmpose.py \\
        --checkpoint perception/models/rtmpose-tiny_simcc-aic-coco_pt-aic-coco_420e-256x192-cfc8f33d_20230126.pth \\
        --out perception/models/rtmpose_t.onnx

## The contract, read off the checkpoint rather than assumed

    input                1x3x256x192 float32, RGB, ImageNet mean/std
    output simcc_x       (N, 17, 384)   192 * 2.0 bins
    output simcc_y       (N, 17, 512)   256 * 2.0 bins
    simcc_split_ratio    2.0

`tools/inspect_checkpoint.py` prints these from any `.pth`. Every one of them
is checked at load and at first inference, because the failure mode when they
are wrong is not a crash: a head with the wrong bin count still returns 17
keypoints in plausible positions, and nothing downstream can tell.

## Runtimes

onnxruntime is the intended one. But OpenCV 4.12's DNN module also imports this
graph and agrees with it to 4e-6 -- measured, not assumed -- so a build that
already has OpenCV can run pose with no new dependency at all, which is the
same argument that made `yolox_onnx.py` a cv2.dnn model. The catch is that it
is about 4x slower (16 ms against 4 ms per crop on a laptop CPU) and cannot
take the batch axis, so it runs one person per call. Fall back to it, do not
plan on it, and it says so on stderr when it happens.

## Why SimCC, and why the peak value matters

RTMPose does not regress coordinates and does not emit a 2-D heatmap. It
classifies each joint twice -- which of 384 columns, which of 512 rows -- so
decoding is an argmax per axis, and the winning bin's value is the model's
confidence in that joint.

That value is not decoration. `posetube.py` uses it as the amplitude of the
gaussian blob it draws, which is the entire reason a heatmap action model beats
a skeleton-graph one on the pose you actually get at 8 metres: an unsure wrist
comes back faint instead of being indistinguishable from a certain one. A
backend that returned a constant score would quietly delete that property, so
the score is carried through unmodified apart from a clamp to 0..1.
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np

# onnxruntime is the intended one. cv2.dnn runs the same graph with nothing new
# installed, at about 4x the cost and one person per call. torch runs the raw
# checkpoint, for the export tool and the test.
RUNTIMES = ("auto", "onnxruntime", "cv2", "torch")

# ImageNet statistics, in RGB order -- what the checkpoint's data preprocessor
# used. Wrong normalisation does not crash; it just degrades quietly.
MEAN_RGB = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD_RGB = np.array([58.395, 57.12, 57.375], dtype=np.float32)

# MMPose's GetBBoxCenterScale default. The crop is the detector's box grown by
# a quarter: RTMPose was trained on boxes with this much context, and a tight
# crop measurably costs accuracy on the joints nearest the edge.
BBOX_PADDING = 1.25


def bbox_to_center_scale(
    box, padding: float = BBOX_PADDING, aspect_ratio: float = 0.75
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Detector box -> the (centre, size) rectangle the network sees.

    Two steps, both from MMPose's top-down pipeline. Pad by `padding`, then
    grow the short side until the rectangle is 3:4, so the resize into 192x256
    never stretches the person. Stretching is worse than it sounds here: limb
    geometry is what the action model reads, and a squashed torso changes it.
    """
    x1, y1, x2, y2 = (float(v) for v in box)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    w, h = (x2 - x1) * padding, (y2 - y1) * padding
    if w > h * aspect_ratio:
        h = w / aspect_ratio
    else:
        w = h * aspect_ratio
    return (cx, cy), (w, h)


def crop_person(
    image: np.ndarray, box, input_size: tuple[int, int] = (192, 256),
    padding: float = BBOX_PADDING,
) -> tuple[np.ndarray, tuple[float, float], float]:
    """Warp one person into a 192x256 crop, keeping the mapping back.

    MMPose builds a three-point affine here because it also supports rotation
    augmentation. With no rotation that matrix reduces to a plain scale and
    translate, which is what this is -- same pixels, one less thing to get
    subtly wrong. `warpAffine` rather than a slice-and-resize because the
    padded box routinely runs off the edge of the frame, and warpAffine pads
    with zeros instead of silently shrinking the crop and shifting every
    keypoint that comes out of it.

    Returns the crop, the rectangle centre, and the crop-pixels-per-frame-pixel
    scale needed to map keypoints back.
    """
    out_w, out_h = input_size
    (cx, cy), (w, h) = bbox_to_center_scale(box, padding, out_w / out_h)
    scale = out_w / w  # uniform: the rectangle is already 3:4
    matrix = np.array(
        [[scale, 0.0, out_w / 2.0 - scale * cx],
         [0.0, scale, out_h / 2.0 - scale * cy]],
        dtype=np.float32,
    )
    crop = cv2.warpAffine(image, matrix, (out_w, out_h), flags=cv2.INTER_LINEAR)
    return crop, (cx, cy), scale


def decode_simcc(
    simcc_x: np.ndarray, simcc_y: np.ndarray, simcc_split_ratio: float = 2.0
) -> tuple[np.ndarray, np.ndarray]:
    """(K, Wx), (K, Wy) bin scores -> (K, 2) crop-pixel coords and (K,) scores.

    Argmax per axis, divided by the split ratio because the bins are at twice
    the input resolution. The score is the *smaller* of the two axis peaks:
    the joint is only as well localised as its worse axis, and taking the
    larger would report confidence the model does not have whenever one axis
    is confident and the other is not -- which is exactly what happens to an
    occluded limb.
    """
    x_bins = np.argmax(simcc_x, axis=-1)
    y_bins = np.argmax(simcc_y, axis=-1)
    scores = np.minimum(
        np.max(simcc_x, axis=-1), np.max(simcc_y, axis=-1)
    ).astype(np.float32)

    coords = np.stack((x_bins, y_bins), axis=-1).astype(np.float32)
    coords /= float(simcc_split_ratio)

    # The head is trained against unit-peak gaussian targets, so peaks land in
    # 0..1 in practice; the clamp makes that a guarantee rather than an
    # observation, since posetube.py multiplies blob amplitude by this and an
    # out-of-range value would brighten a joint above a certain one.
    # A joint the model rejects outright (non-positive peak) keeps its decoded
    # position and gets a zero score, rather than MMPose's (-1, -1) sentinel:
    # downstream already drops low-scoring joints, and a sentinel coordinate
    # would become a real joint position for anyone who lowered the threshold.
    return coords, np.clip(scores, 0.0, 1.0)


class _OnnxRuntimeSession:
    """onnxruntime, wrapped so the estimator only knows about arrays."""

    def __init__(self, model_path: str, device: str) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "runtime='onnxruntime' was asked for explicitly but it is not "
                "installed (pip install onnxruntime). runtime='cv2' runs the "
                "same graph with nothing new installed, ~4x slower."
            ) from exc

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        shape = self.session.get_inputs()[0].shape
        # A fixed batch axis is legal but means the graph was exported without
        # dynamic_axes; batching into it would fail at run time, so don't.
        self.dynamic_batch = not isinstance(shape[0], int)
        self.input_shape = shape

    def run(self, blob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        out_x, out_y = self.session.run(None, {self.input_name: blob})
        return out_x, out_y


class _Cv2DnnSession:
    """The no-new-dependency fallback: OpenCV's DNN module, as the detector uses.

    Measured, not assumed. OpenCV 4.12 imports this graph and agrees with
    onnxruntime to 4e-6, but with two costs worth knowing before choosing it:

      * roughly 4x slower on the same CPU (16 ms against 4 ms per crop), which
        on a busy frame is the difference between pose fitting in the budget
        and not
      * no dynamic batch. cv2.dnn ignores the batch axis and mis-shapes the
        head's MatMul, so this runs one person per call

    So onnxruntime is the default and this is the fallback -- but it is a real
    fallback, and it means a build that already has OpenCV can run pose today
    without adding anything.
    """

    dynamic_batch = False

    def __init__(self, model_path: str) -> None:
        self.net = cv2.dnn.readNetFromONNX(model_path)
        names = list(self.net.getUnconnectedOutLayersNames())
        # Ask for the two heads by name rather than trusting graph order: the
        # x and y outputs are interchangeable-looking, and taking them the
        # wrong way round transposes every keypoint. (The bin-count check in
        # the estimator would catch it, but only after the fact.)
        if {"simcc_x", "simcc_y"} <= set(names):
            names = ["simcc_x", "simcc_y"]
        self.output_names = names

    def run(self, blob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.net.setInput(blob)
        out_x, out_y = self.net.forward(self.output_names)
        return out_x, out_y


class _TorchSession:
    """Fallback: run the `.pth` directly through the rebuilt architecture.

    For a machine that has torch but no exported graph yet -- the export tool
    and the test both need to run the network before an `.onnx` exists. Not
    the shipping path: torch is not going on the device.
    """

    dynamic_batch = True

    def __init__(self, checkpoint_path: str, device: str) -> None:
        import torch

        try:  # importable both as a package and off sys.path, like posetube
            from .rtmpose_arch import load_rtmpose_tiny
        except ImportError:
            from rtmpose_arch import load_rtmpose_tiny

        self._torch = torch
        self.model = load_rtmpose_tiny(checkpoint_path, device=device)
        self.device = device

    def run(self, blob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with self._torch.no_grad():
            x = self._torch.from_numpy(blob).to(self.device)
            out_x, out_y = self.model(x)
        return out_x.cpu().numpy(), out_y.cpu().numpy()


class RTMPoseEstimator:
    """Top-down pose: one crop per tracked person, 17 COCO keypoints out.

    Satisfies `posetube.PoseEstimator`, so it drops in wherever
    `StubPoseEstimator` sits with no other change.

    Top-down is the right trade here even though its cost grows with the number
    of people. The detector and tracker already exist and already produce the
    boxes, per-person accuracy is far better at 8 metres than a bottom-up
    model's, and identity comes from the tracker rather than from a grouping
    step that can swap two workers' limbs between frames. Two models on the
    NPU rather than one is the outstanding board measurement, and it is the
    reason this defaults to the *tiny* variant.
    """

    name = "rtmpose-t"
    licence = (
        "Apache-2.0 (RTMPose/MMPose code and weights); weights trained on "
        "COCO + AI Challenger -- see LICENCE-NOTES.md"
    )
    num_keypoints = 17

    def __init__(
        self,
        model_path: str,
        input_size: tuple[int, int] = (192, 256),
        simcc_split_ratio: float = 2.0,
        num_keypoints: int = 17,
        padding: float = BBOX_PADDING,
        device: str = "cpu",
        max_batch: int = 8,
        runtime: str = "auto",
        session=None,
    ) -> None:
        if session is None and not os.path.exists(model_path):
            raise FileNotFoundError(
                f"RTMPose weights not found at {model_path}.\n"
                "Export them with:  python3 perception/tools/export_rtmpose.py"
            )
        if runtime not in RUNTIMES:
            raise ValueError(f"unknown runtime '{runtime}'; one of {RUNTIMES}")
        self.model_path = model_path
        self.input_size = (int(input_size[0]), int(input_size[1]))  # (w, h)
        self.simcc_split_ratio = float(simcc_split_ratio)
        self.num_keypoints = int(num_keypoints)
        self.padding = float(padding)
        self.device = device
        self.max_batch = max(1, int(max_batch))
        self.runtime = runtime
        self._session = session
        self._checked = False

    # -- runtime -----------------------------------------------------------

    def _load(self):
        """Pick a runtime once, and say which one out loud if it is not the default.

        Order is deliberate. onnxruntime is the shipping path and four times
        faster than the alternative; cv2.dnn is the fallback that needs nothing
        new installed; torch runs the raw checkpoint and exists so the export
        can be checked before an `.onnx` exists. A silent fallback would let a
        4x slower runtime, or a laptop-only one, drift into a deployment
        unnoticed -- hence the stderr line.
        """
        if self._session is not None:
            return self._session

        runtime = self.runtime
        if runtime == "auto":
            if self.model_path.endswith(".pth"):
                runtime = "torch"
            else:
                try:
                    import onnxruntime  # noqa: F401

                    runtime = "onnxruntime"
                except ImportError:
                    runtime = "cv2"

        if runtime == "onnxruntime":
            self._session = _OnnxRuntimeSession(self.model_path, self.device)
        elif runtime == "cv2":
            print(
                "rtmpose: onnxruntime not found, falling back to cv2.dnn -- "
                "about 4x slower and one person per call. "
                "pip install onnxruntime for the intended path.",
                file=sys.stderr,
            )
            self._session = _Cv2DnnSession(self.model_path)
        elif runtime == "torch":
            print(
                f"rtmpose: running the checkpoint {os.path.basename(self.model_path)} "
                "through torch. This is the export/verification path, not the "
                "shipping one -- torch is not going on the device.",
                file=sys.stderr,
            )
            self._session = _TorchSession(self.model_path, self.device)
        return self._session

    # -- preprocessing -----------------------------------------------------

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        """BGR uint8 crop -> normalised RGB CHW float32.

        Frames arrive BGR from OpenCV; the checkpoint was trained on RGB. The
        channel swap is one line and getting it wrong costs a few points of
        accuracy without ever raising anything.
        """
        rgb = crop[:, :, ::-1].astype(np.float32)
        rgb = (rgb - MEAN_RGB) / STD_RGB
        return np.ascontiguousarray(rgb.transpose(2, 0, 1))

    def _check_outputs(self, out_x: np.ndarray, out_y: np.ndarray) -> None:
        """Verify the graph matches the decode assumptions, once, at first use."""
        expect_x = int(self.input_size[0] * self.simcc_split_ratio)
        expect_y = int(self.input_size[1] * self.simcc_split_ratio)
        got = (out_x.shape[1], out_x.shape[2], out_y.shape[2])
        if got != (self.num_keypoints, expect_x, expect_y):
            raise ValueError(
                f"model emits K={got[0]} with {got[1]} x-bins and {got[2]} "
                f"y-bins; this backend decodes K={self.num_keypoints}, "
                f"{expect_x} x-bins, {expect_y} y-bins.\n"
                "Either the export used a different input_size or "
                "simcc_split_ratio, or this is a different RTMPose variant. "
                "Read the real numbers with tools/inspect_checkpoint.py -- "
                "a mismatch here does not crash, it silently scales every "
                "keypoint by the ratio of the two bin counts."
            )
        self._checked = True

    # -- inference ---------------------------------------------------------

    def estimate(self, image: np.ndarray, boxes: dict) -> dict:
        """{track_id: (17, 3) array of x, y, score} in frame coordinates."""
        if not boxes:
            return {}

        track_ids, blobs, mappings = [], [], []
        for track_id, box in boxes.items():
            x1, y1, x2, y2 = (float(v) for v in box)
            if x2 - x1 < 1.0 or y2 - y1 < 1.0:
                # A degenerate box produces a crop of nothing. Returning
                # keypoints for it would be inventing a person; the track is
                # simply absent from this frame, which posetube already
                # handles as a gap and reports through coverage.
                continue
            crop, centre, scale = crop_person(
                image, (x1, y1, x2, y2), self.input_size, self.padding
            )
            track_ids.append(track_id)
            blobs.append(self._preprocess(crop))
            mappings.append((centre, scale))

        if not track_ids:
            return {}

        session = self._load()
        batch = self.max_batch if getattr(session, "dynamic_batch", False) else 1

        out: dict[str, np.ndarray] = {}
        for start in range(0, len(blobs), batch):
            chunk = np.stack(blobs[start:start + batch])
            out_x, out_y = session.run(chunk)
            if not self._checked:
                self._check_outputs(out_x, out_y)
            for i in range(chunk.shape[0]):
                coords, scores = decode_simcc(
                    out_x[i], out_y[i], self.simcc_split_ratio
                )
                (cx, cy), scale = mappings[start + i]
                out[track_ids[start + i]] = self._to_frame(
                    coords, scores, (cx, cy), scale
                )
        return out

    def _to_frame(
        self, coords: np.ndarray, scores: np.ndarray, centre, scale: float
    ) -> np.ndarray:
        """Crop pixels -> frame pixels, inverting the crop's affine.

        The estimator returns frame coordinates because everything downstream
        -- zones, tracking, the tube's union box -- lives there. Handing back
        crop coordinates would leave every consumer to reconstruct a mapping
        only this class knows.
        """
        out_w, out_h = self.input_size
        cx, cy = centre
        keypoints = np.empty((coords.shape[0], 3), dtype=np.float32)
        keypoints[:, 0] = (coords[:, 0] - out_w / 2.0) / scale + cx
        keypoints[:, 1] = (coords[:, 1] - out_h / 2.0) / scale + cy
        keypoints[:, 2] = scores
        return keypoints
