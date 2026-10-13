"""Face alignment: put the face where the recognition model expects it.

ArcFace was trained on faces warped onto a fixed 112x112 template -- eyes on a
particular line, nose at a particular height. Feeding it a raw detector box
resized to 112x112 is not the same picture, and the embedding it returns is
correspondingly worse. The failure is quiet: you get a plausible 512-d vector
that simply scores lower against the right person, so identification degrades
into "unverified" rather than into an error anyone can see.

The transform is a similarity: rotation, uniform scale and translation, no
shear. That is what the reference implementations use, and it is the right
family -- it removes head roll and camera distance while leaving the face's
proportions alone.

WHAT THIS NEEDS THAT THE DETECTOR DOES NOT PROVIDE

Alignment needs landmarks, and RFB-320 emits boxes only. So a landmark source
has to come from somewhere:

  5-point (best)   A detector that emits landmarks with the box -- SCRFD or
                   RetinaFace. This is the destination; `FaceBox.landmarks`
                   exists so such a detector can drop in without touching
                   anything here.

  2-point (good)   Both eyes. Two point pairs fully determine a similarity
                   transform, so this removes roll and normalises scale
                   exactly. It cannot correct yaw, which 5-point partially
                   does. `landmarks.py` supplies this from the OpenCV eye
                   cascade, which is adequate for controlled enrolment photos.

  none (refuse)    No landmarks means no alignment. The honest response is to
                   decline the comparison rather than embed a crop the model
                   was not trained on -- the same reasoning as the resolver
                   returning `unverified` instead of guessing. `naive_crop`
                   exists for measuring how much alignment is worth, and is
                   not a production path.

THIS INVALIDATES EXISTING TEMPLATES

An embedding of an aligned face and an embedding of a raw crop are not
comparable. Templates therefore record which method produced them, exactly as
they already record `model_sha256`, and the matcher refuses to compare across
methods. Anyone enrolled before this change has to be enrolled again.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np

# Method names, recorded on every template.
ALIGNMENT_5POINT = "arcface-5pt"
ALIGNMENT_2POINT = "eyes-2pt"
ALIGNMENT_NONE = "none"

OUTPUT_SIZE = 112

# The canonical ArcFace 5-point template for a 112x112 crop, in
# (x, y) pixel coordinates: left eye, right eye, nose tip, left mouth corner,
# right mouth corner. "Left" means left in the IMAGE, i.e. smaller x -- which
# is the subject's right side. These are the reference numbers the model was
# trained against; they are not tuneable.
ARCFACE_TEMPLATE_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float64,
)


class AlignmentError(RuntimeError):
    """The face could not be aligned, so no embedding should be taken."""


@dataclass(frozen=True)
class FaceLandmarks:
    """Landmark points in frame pixel coordinates.

    Only the eyes are required. The other three are present when the source
    can supply them, and their presence is what selects 5-point alignment.
    """

    left_eye: tuple[float, float]
    right_eye: tuple[float, float]
    nose: Optional[tuple[float, float]] = None
    mouth_left: Optional[tuple[float, float]] = None
    mouth_right: Optional[tuple[float, float]] = None

    @property
    def is_five_point(self) -> bool:
        return (
            self.nose is not None
            and self.mouth_left is not None
            and self.mouth_right is not None
        )

    def as_array(self) -> np.ndarray:
        """Return (5,2) if complete, else (2,2) with just the eyes."""
        if self.is_five_point:
            return np.array(
                [
                    self.left_eye,
                    self.right_eye,
                    self.nose,
                    self.mouth_left,
                    self.mouth_right,
                ],
                dtype=np.float64,
            )
        return np.array([self.left_eye, self.right_eye], dtype=np.float64)


# ----------------------------------------------------------------------
# Transform estimation
# ----------------------------------------------------------------------


def similarity_from_two_points(
    src: np.ndarray, dst: np.ndarray
) -> np.ndarray:
    """Exact 2x3 similarity mapping two source points onto two destinations.

    Closed form rather than least squares: with two pairs the similarity is
    uniquely determined (4 degrees of freedom, 4 equations), so there is
    nothing to fit and no residual to worry about.
    """
    if src.shape != (2, 2) or dst.shape != (2, 2):
        raise AlignmentError("two-point alignment needs exactly two pairs")

    src_vec = src[1] - src[0]
    dst_vec = dst[1] - dst[0]

    src_len = float(np.linalg.norm(src_vec))
    dst_len = float(np.linalg.norm(dst_vec))

    if src_len < 1e-6:
        # Both eyes at the same pixel: the detector has failed, and scaling by
        # 1/0 would produce a garbage warp rather than an obvious one.
        raise AlignmentError("landmark points coincide; cannot estimate scale")

    scale = dst_len / src_len
    angle = math.atan2(dst_vec[1], dst_vec[0]) - math.atan2(src_vec[1], src_vec[0])

    cos_a = math.cos(angle) * scale
    sin_a = math.sin(angle) * scale

    rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float64)
    translation = dst[0] - rotation @ src[0]

    matrix = np.zeros((2, 3), dtype=np.float64)
    matrix[:, :2] = rotation
    matrix[:, 2] = translation
    return matrix


def similarity_umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity for three or more point pairs (Umeyama 1991).

    With five landmarks the pairs will not agree exactly -- a real face is not
    a rigid copy of the template -- so this minimises the total squared error
    instead of honouring any one point.
    """
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2:
        raise AlignmentError("point sets must be matching (N,2) arrays")
    n = src.shape[0]
    if n < 3:
        raise AlignmentError("umeyama needs at least three pairs")

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    covariance = dst_demean.T @ src_demean / n

    d = np.ones(2, dtype=np.float64)
    if np.linalg.det(covariance) < 0:
        # Reflection. Faces do not mirror, so forbid it: allowing a flip here
        # would let a badly ordered landmark list produce a confident,
        # completely wrong alignment.
        d[1] = -1.0

    u, s, vt = np.linalg.svd(covariance)
    rank = np.linalg.matrix_rank(covariance)
    if rank == 0:
        raise AlignmentError("degenerate landmark configuration")

    if rank == 1:
        if np.linalg.det(u) * np.linalg.det(vt) > 0:
            rotation = u @ vt
        else:
            saved = d[1]
            d[1] = -1.0
            rotation = u @ np.diag(d) @ vt
            d[1] = saved
    else:
        rotation = u @ np.diag(d) @ vt

    variance = src_demean.var(axis=0).sum()
    if variance < 1e-12:
        raise AlignmentError("landmark points coincide; cannot estimate scale")

    scale = float((s * d).sum() / variance)

    matrix = np.zeros((2, 3), dtype=np.float64)
    matrix[:, :2] = scale * rotation
    matrix[:, 2] = dst_mean - scale * rotation @ src_mean
    return matrix


# ----------------------------------------------------------------------
# The alignment call
# ----------------------------------------------------------------------


def align_face(
    frame: np.ndarray,
    landmarks: FaceLandmarks,
    size: int = OUTPUT_SIZE,
) -> tuple[np.ndarray, str]:
    """Warp a face onto the canonical template.

    Returns (aligned BGR crop of shape (size, size, 3), method name). The
    method name goes onto the template so that probes and enrolments can be
    checked for compatibility later.
    """
    if frame is None or frame.size == 0:
        raise AlignmentError("empty frame passed to align_face")

    scale = size / float(OUTPUT_SIZE)
    template = ARCFACE_TEMPLATE_112 * scale

    src = landmarks.as_array()

    if landmarks.is_five_point:
        matrix = similarity_umeyama(src, template)
        method = ALIGNMENT_5POINT
    else:
        matrix = similarity_from_two_points(src, template[:2])
        method = ALIGNMENT_2POINT

    aligned = cv2.warpAffine(
        frame,
        matrix.astype(np.float32),
        (size, size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return aligned, method


def naive_crop(
    frame: np.ndarray,
    box,
    size: int = OUTPUT_SIZE,
) -> tuple[np.ndarray, str]:
    """The pre-alignment behaviour: crop the box, squash it to a square.

    Kept so the cost of not aligning can be measured on your own footage
    rather than argued about. Templates produced this way are tagged
    ALIGNMENT_NONE and the matcher will not compare them against aligned
    probes.
    """
    crop = box.crop(frame)
    if crop is None or crop.size == 0:
        raise AlignmentError("face box produced an empty crop")
    return cv2.resize(crop, (size, size)), ALIGNMENT_NONE


def landmarks_from_box(box) -> Optional[FaceLandmarks]:
    """Lift landmarks off a FaceBox if the detector supplied them.

    RFB-320 does not. SCRFD and RetinaFace do, and this is the seam that lets
    one replace the other without any change here or in the matcher.
    """
    pts: Optional[Sequence] = getattr(box, "landmarks", None)
    if not pts:
        return None
    if len(pts) == 5:
        return FaceLandmarks(
            left_eye=(float(pts[0][0]), float(pts[0][1])),
            right_eye=(float(pts[1][0]), float(pts[1][1])),
            nose=(float(pts[2][0]), float(pts[2][1])),
            mouth_left=(float(pts[3][0]), float(pts[3][1])),
            mouth_right=(float(pts[4][0]), float(pts[4][1])),
        )
    if len(pts) == 2:
        return FaceLandmarks(
            left_eye=(float(pts[0][0]), float(pts[0][1])),
            right_eye=(float(pts[1][0]), float(pts[1][1])),
        )
    return None
