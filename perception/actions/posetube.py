"""Pose tubes: turning tracked keypoints into the volume PoseC3D expects.

PoseC3D does not consume a skeleton graph the way ST-GCN does. It consumes a
stack of joint heatmaps -- one channel per keypoint, one slice per frame -- and
runs an ordinary 3D convolution over it. That indirection is exactly why it is
the right choice at 8 metres with helmets on:

  * A graph model treats every joint coordinate as a fact. A hallucinated wrist
    is indistinguishable from a real one, so pose noise propagates straight
    into the classification.

  * A heatmap model can encode UNCERTAINTY. A low-confidence keypoint becomes a
    faint, diffuse blob rather than a crisp one, so the convolution naturally
    weights it less. Nothing has to be special-cased; the representation does
    the work.

That property is the whole reason this file spends effort on confidence
weighting and sigma rather than just plotting dots.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence

import numpy as np

US_PER_S = 1_000_000

# COCO-17, the layout RTMPose and most pose models emit.
COCO_KEYPOINTS = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)


class PoseEstimator(Protocol):
    """Seam for the pose model. RTMPose in production, a stub in tests."""

    name: str
    licence: str
    num_keypoints: int

    def estimate(self, image, boxes: dict) -> dict:
        """{track_id: (K,3) array of x, y, score}."""
        ...


@dataclass
class PoseFrameRecord:
    """One frame's keypoints, per track."""

    timestamp_us: int
    keypoints: dict[str, np.ndarray]   # track_id -> (K,3) x, y, score
    boxes: dict[str, tuple[float, float, float, float]]
    image: Optional[np.ndarray] = None  # only needed if the RGB branch is on


@dataclass
class PoseTube:
    track_id: str
    heatmaps: np.ndarray               # K x T x H x W, float32 in 0..1
    window_start_us: int
    window_end_us: int
    coverage: float
    mean_keypoint_score: float
    frames: Optional[np.ndarray] = None  # T x S x S x 3, for the RGB branch

    @property
    def duration_s(self) -> float:
        return (self.window_end_us - self.window_start_us) / US_PER_S


class PoseTubeExtractor:
    """Rolling pose buffer, and heatmap volume construction.

    Defaults are read off the published pose_only checkpoint rather than
    guessed: `clip_len=32`, `scale=(56, 56)`, `hw_ratio=1.0`, and 17 input
    channels on `backbone.conv1` (one per COCO keypoint, `with_kp=True` and
    `with_limb=False`). Changing num_frames or heatmap_size without retraining
    will silently degrade accuracy -- the network was fitted to these.

    Note the RGBPose variant uses different numbers (RGB clip_len 8 against
    pose 32, and 64x64 heatmaps). Do not carry those over to the pose-only
    path.

    Three decisions here shape how well the model generalises.

    **Normalise to the person's box, not the frame.** A worker three metres away
    and one fifteen metres away perform the same action; normalising to the box
    makes both produce the same volume. Normalising to the frame instead would
    make the model learn camera distance, and it would then fail on the first
    site with a different mounting height.

    **Use one box for the whole window, not one per frame.** Per-frame
    normalisation would centre the person in every slice and delete their
    motion across the scene -- walking and standing-still would look identical.
    The union box over the window keeps that motion while staying scale
    invariant.

    **Scale each blob by its keypoint confidence.** This is the property that
    makes heatmaps beat graphs on noisy pose. A wrist the estimator was unsure
    about becomes faint, and the convolution downweights it without anything
    being special-cased.
    """

    def __init__(
        self,
        window_s: float = 1.5,
        num_frames: int = 32,
        heatmap_size: int = 56,
        num_keypoints: int = 17,
        sigma: float = 0.6,
        box_margin: float = 0.10,
        min_keypoint_score: float = 0.15,
        buffer_seconds: float = 4.0,
        fps_hint: float = 15.0,
        keep_rgb: bool = False,
        rgb_size: int = 224,
    ) -> None:
        # A shorter window than the RGB pipeline on purpose. Every action model
        # is inherently late by its window length -- you cannot classify an
        # action before it has happened -- so for live use the window is a
        # latency budget, not just an accuracy knob.
        self.window_us = int(window_s * US_PER_S)
        self.num_frames = num_frames
        self.heatmap_size = heatmap_size
        self.num_keypoints = num_keypoints
        self.sigma = sigma
        self.box_margin = box_margin
        self.min_keypoint_score = min_keypoint_score
        self.keep_rgb = keep_rgb
        self.rgb_size = rgb_size
        self.buffer = deque(maxlen=max(8, int(buffer_seconds * fps_hint)))

        # Precomputed grid; the gaussian is evaluated against it per keypoint.
        grid = np.arange(heatmap_size, dtype=np.float32)
        self._gy, self._gx = np.meshgrid(grid, grid, indexing="ij")

    # ------------------------------------------------------------------

    def push(self, record: PoseFrameRecord) -> None:
        self.buffer.append(record)

    def _window(self, now_us: int) -> list[PoseFrameRecord]:
        w = [r for r in self.buffer if now_us - r.timestamp_us <= self.window_us]
        w.sort(key=lambda r: r.timestamp_us)
        return w

    def tracks_ready(self, now_us: int) -> list[str]:
        window = self._window(now_us)
        if len(window) < 2:
            return []
        counts: dict[str, int] = {}
        for r in window:
            for tid in r.keypoints:
                counts[tid] = counts.get(tid, 0) + 1
        need = max(2, len(window) // 2)
        return [tid for tid, n in counts.items() if n >= need]

    def extract(self, track_id: str, now_us: int) -> Optional[PoseTube]:
        window = self._window(now_us)
        if len(window) < 2:
            return None
        present = [r for r in window if track_id in r.keypoints]
        if not present:
            return None

        box = self._union_box(present, track_id)
        if box is None:
            return None

        start_us, end_us = window[0].timestamp_us, window[-1].timestamp_us
        targets = np.linspace(start_us, end_us, self.num_frames)

        volume = np.zeros(
            (self.num_keypoints, self.num_frames, self.heatmap_size, self.heatmap_size),
            dtype=np.float32,
        )
        crops: list[np.ndarray] = []
        hits = 0
        scores: list[float] = []
        last_kpts: Optional[np.ndarray] = None

        for t_idx, t in enumerate(targets):
            record = min(window, key=lambda r: abs(r.timestamp_us - t))
            kpts = record.keypoints.get(track_id)
            if kpts is not None:
                last_kpts = kpts
                hits += 1
            elif last_kpts is None:
                last_kpts = present[0].keypoints[track_id]

            volume[:, t_idx] = self._heatmap(last_kpts, box)
            scores.append(float(np.mean(last_kpts[:, 2])))

            if self.keep_rgb and record.image is not None:
                crops.append(self._rgb_crop(record.image, box))

        return PoseTube(
            track_id=track_id,
            heatmaps=volume,
            window_start_us=start_us,
            window_end_us=end_us,
            coverage=round(hits / self.num_frames, 4),
            mean_keypoint_score=round(float(np.mean(scores)), 4),
            frames=np.stack(crops) if crops else None,
        )

    # ------------------------------------------------------------------

    def _union_box(
        self, records: Sequence[PoseFrameRecord], track_id: str
    ) -> Optional[tuple[float, float, float, float]]:
        """One box covering the track across the whole window.

        Keeps the person's movement across the scene inside the volume, which a
        per-frame box would centre away and delete.
        """
        xs1, ys1, xs2, ys2 = [], [], [], []
        for r in records:
            b = r.boxes.get(track_id)
            if b is None:
                k = r.keypoints[track_id]
                good = k[k[:, 2] >= self.min_keypoint_score]
                if len(good) == 0:
                    continue
                b = (
                    float(good[:, 0].min()), float(good[:, 1].min()),
                    float(good[:, 0].max()), float(good[:, 1].max()),
                )
            xs1.append(b[0]); ys1.append(b[1]); xs2.append(b[2]); ys2.append(b[3])

        if not xs1:
            return None

        x1, y1, x2, y2 = min(xs1), min(ys1), max(xs2), max(ys2)
        w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
        mx, my = w * self.box_margin, h * self.box_margin
        x1, y1, x2, y2 = x1 - mx, y1 - my, x2 + mx, y2 + my

        # Square it, so the heatmap does not stretch limb geometry -- the same
        # reason the RGB tube pads rather than resizes.
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        side = max(x2 - x1, y2 - y1)
        return (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)

    def _heatmap(self, kpts: np.ndarray, box) -> np.ndarray:
        """One frame of the volume: a gaussian per keypoint, scaled by score."""
        x1, y1, x2, y2 = box
        bw, bh = max(x2 - x1, 1e-6), max(y2 - y1, 1e-6)
        out = np.zeros(
            (self.num_keypoints, self.heatmap_size, self.heatmap_size), dtype=np.float32
        )
        two_sigma_sq = 2.0 * self.sigma * self.sigma

        for k in range(min(self.num_keypoints, kpts.shape[0])):
            x, y, s = float(kpts[k, 0]), float(kpts[k, 1]), float(kpts[k, 2])
            if not np.isfinite(x) or not np.isfinite(y) or s < self.min_keypoint_score:
                # A keypoint the estimator did not find contributes nothing.
                # Plotting it at its guessed position would assert a joint
                # location the model has no evidence for.
                continue

            hx = (x - x1) / bw * (self.heatmap_size - 1)
            hy = (y - y1) / bh * (self.heatmap_size - 1)
            if not (-2 <= hx <= self.heatmap_size + 1 and -2 <= hy <= self.heatmap_size + 1):
                continue

            d2 = (self._gx - hx) ** 2 + (self._gy - hy) ** 2
            # Amplitude carries the estimator's confidence, so an uncertain
            # joint is faint rather than indistinguishable from a certain one.
            out[k] = np.exp(-d2 / two_sigma_sq) * min(1.0, max(0.0, s))

        return out

    def _rgb_crop(self, image: np.ndarray, box) -> np.ndarray:
        h, w = image.shape[:2]
        x1, y1, x2, y2 = box
        xi1, yi1 = int(max(0, x1)), int(max(0, y1))
        xi2, yi2 = int(min(w, x2)), int(min(h, y2))
        if xi2 <= xi1 or yi2 <= yi1:
            return np.zeros((self.rgb_size, self.rgb_size, 3), dtype=np.uint8)
        patch = image[yi1:yi2, xi1:xi2]
        ph, pw = patch.shape[:2]
        side = max(ph, pw)
        canvas = np.zeros((side, side, 3), dtype=patch.dtype)
        oy, ox = (side - ph) // 2, (side - pw) // 2
        canvas[oy:oy + ph, ox:ox + pw] = patch
        try:
            import cv2

            return cv2.resize(canvas, (self.rgb_size, self.rgb_size))
        except ImportError:
            idx = np.arange(self.rgb_size) * side // self.rgb_size
            return canvas[np.ix_(idx, idx)]


class StubPoseEstimator:
    """Deterministic pose for tests. Never ships."""

    name = "stub-pose"
    licence = "n/a (test stub)"
    num_keypoints = 17

    def __init__(self, score: float = 0.9) -> None:
        self.score = score

    def estimate(self, image, boxes: dict) -> dict:
        out = {}
        for tid, (x1, y1, x2, y2) in boxes.items():
            w, h = x2 - x1, y2 - y1
            k = np.zeros((self.num_keypoints, 3), dtype=np.float32)
            for i in range(self.num_keypoints):
                k[i] = (
                    x1 + w * (0.3 + 0.4 * ((i % 3) / 2)),
                    y1 + h * (i / max(1, self.num_keypoints - 1)),
                    self.score,
                )
            out[tid] = k
        return out
