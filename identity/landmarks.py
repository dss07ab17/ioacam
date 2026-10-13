"""A landmark source that needs no new model file and no download.

RFB-320 gives boxes and nothing else, so alignment has no points to work from.
This supplies the two eyes using the Haar eye cascade that ships inside the
opencv-python wheel already in requirements.txt. No extra dependency, no extra
weights, nothing to fetch at commissioning.

WHAT THIS IS AND IS NOT FOR

It is honest about its own quality. A Haar cascade is a 2001 detector: on a
frontal, evenly lit, reasonably large face it finds both eyes reliably, and on
a face that is turned, shadowed, small, or wearing glasses it often finds one
eye, three eyes, or none. That profile makes it right for one job and wrong for
the other:

  enrolment   Right. Enrolment photos are posed, close and controlled, and a
              failure is visible immediately -- the operator is standing there
              and can retake the photo.

  live        Wrong as a permanent answer. At monitoring range it will fail
              often, and each failure is a face that cannot be aligned. That
              is survivable (the resolver reports unverified, which is true)
              but it caps identification rate at whatever fraction of frames
              the cascade happens to succeed on.

The real fix for live is a detector that emits landmarks with the box -- SCRFD
or RetinaFace. Then `FaceBox.landmarks` is populated, `landmarks_from_box`
picks it up, and this module stops being used without anything else changing.
Treat this as the thing that unblocks alignment now, not the end state.

WHY NOT JUST GUESS THE EYE POSITIONS

Because assuming eyes sit at fixed fractions of the detector box reintroduces
the problem alignment exists to solve: the box position varies with the
detector, the head angle and the crop margin, so the "landmarks" would encode
detector jitter rather than face geometry, and every embedding would inherit
it. Better to fail and say so.
"""

from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np

from .alignment import FaceLandmarks

# The cascade ships inside the opencv-python package.
_CASCADE_FILENAME = "haarcascade_eye.xml"


def _cascade_path() -> str:
    data_dir = getattr(cv2, "data", None)
    if data_dir is None or not hasattr(data_dir, "haarcascades"):
        raise FileNotFoundError(
            "This OpenCV build does not ship the Haar cascade data files. "
            "Install opencv-python (not opencv-python-headless-minimal), or "
            "supply landmarks from a detector that emits them."
        )
    path = os.path.join(data_dir.haarcascades, _CASCADE_FILENAME)
    if not os.path.exists(path):
        raise FileNotFoundError(f"eye cascade not found at {path}")
    return path


class CascadeEyeLandmarks:
    """Finds two eyes inside a face box.

    Stateless apart from the loaded cascade, so one instance can be shared by
    the enrolment tool and the live loop.
    """

    name = "haar-eye-cascade"
    licence = "OpenCV (Apache-2.0); cascade data BSD-3-Clause"

    def __init__(
        self,
        scale_factor: float = 1.1,
        min_neighbors: int = 6,
        min_eye_fraction: float = 0.08,
    ) -> None:
        self._cascade = cv2.CascadeClassifier(_cascade_path())
        if self._cascade.empty():
            raise RuntimeError("eye cascade failed to load")
        self.scale_factor = scale_factor
        # Higher than the OpenCV default of 3. False eyes are far more harmful
        # here than missed ones: a spurious detection produces a confidently
        # wrong alignment, whereas a miss produces an honest refusal.
        self.min_neighbors = min_neighbors
        # Reject detections too small to be an eye on a face this size.
        self.min_eye_fraction = min_eye_fraction

    def find(self, frame: np.ndarray, box) -> Optional[FaceLandmarks]:
        """Return both eyes in FRAME coordinates, or None.

        None is a normal outcome and means "do not align this face", not
        "something broke".
        """
        crop = box.crop(frame)
        if crop is None or crop.size == 0:
            return None

        crop_h, crop_w = crop.shape[:2]
        if crop_h < 20 or crop_w < 20:
            # Below this the cascade returns noise.
            return None

        grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        grey = cv2.equalizeHist(grey)

        min_side = int(min(crop_h, crop_w) * self.min_eye_fraction)
        detections = self._cascade.detectMultiScale(
            grey,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(max(4, min_side), max(4, min_side)),
        )

        if len(detections) < 2:
            return None

        # Eyes live in the upper part of the face. Anything in the lower half
        # is a nostril or a mouth corner the cascade has mistaken for an eye.
        upper = [d for d in detections if (d[1] + d[3] / 2.0) < crop_h * 0.6]
        if len(upper) < 2:
            return None

        # Largest two: the true eyes are usually the strongest responses, and
        # size is the only ranking signal detectMultiScale gives us without
        # switching to the weighted variant.
        upper.sort(key=lambda d: d[2] * d[3], reverse=True)
        first, second = upper[0], upper[1]

        centres = [
            (float(first[0] + first[2] / 2.0), float(first[1] + first[3] / 2.0)),
            (float(second[0] + second[2] / 2.0), float(second[1] + second[3] / 2.0)),
        ]
        # Smaller x is the image-left eye, which is template point 0.
        centres.sort(key=lambda p: p[0])
        left, right = centres

        separation = right[0] - left[0]
        if separation < crop_w * 0.15:
            # Two detections on the same eye. Aligning on these would scale
            # the face up enormously and produce a confident nonsense crop.
            return None

        vertical_offset = abs(right[1] - left[1])
        if vertical_offset > separation * 0.6:
            # Implausible roll for a working face; more likely one eye and one
            # eyebrow or a shadow.
            return None

        # Back to frame coordinates.
        offset_x, offset_y = float(box.x1), float(box.y1)
        return FaceLandmarks(
            left_eye=(left[0] + offset_x, left[1] + offset_y),
            right_eye=(right[0] + offset_x, right[1] + offset_y),
        )


def build_landmark_source(cfg: Optional[dict] = None):
    """Config-selected landmark source, same shape as build_detector.

    backend: "cascade" (default) or "none". "none" means the detector is
    expected to supply landmarks itself, and faces without them are refused.
    """
    cfg = cfg or {}
    backend = cfg.get("backend", "cascade")

    if backend == "none":
        return None

    if backend == "cascade":
        return CascadeEyeLandmarks(
            scale_factor=float(cfg.get("scale_factor", 1.1)),
            min_neighbors=int(cfg.get("min_neighbors", 6)),
            min_eye_fraction=float(cfg.get("min_eye_fraction", 0.08)),
        )

    raise ValueError(f"Unknown landmark backend {backend!r}")
