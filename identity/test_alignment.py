"""Tests for face alignment.

Run with:  python3 identity/test_alignment.py

No model and no camera. What is under test is the geometry, because the
geometry is where a silent error lives: a wrong transform does not raise, it
returns a plausible 112x112 picture of a face in the wrong place, and the only
symptom is a recognition rate that is worse than it should be for reasons
nobody can point at.

So every test here asserts where the landmarks LAND, not that the call
succeeded.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from identity.alignment import (  # noqa: E402
    ALIGNMENT_2POINT,
    ALIGNMENT_5POINT,
    ALIGNMENT_NONE,
    ARCFACE_TEMPLATE_112,
    AlignmentError,
    FaceLandmarks,
    align_face,
    landmarks_from_box,
    similarity_from_two_points,
    similarity_umeyama,
)

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"[{'PASS' if condition else 'FAIL'}] {name}")
    if not condition:
        failures.append(f"{name}: {detail}")


def apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 2x3 affine to an (N,2) point set."""
    pts = np.asarray(points, dtype=np.float64)
    return (matrix[:, :2] @ pts.T).T + matrix[:, 2]


# ----------------------------------------------------------------------
# Two-point transform
# ----------------------------------------------------------------------

src = np.array([[100.0, 200.0], [160.0, 210.0]])
dst = ARCFACE_TEMPLATE_112[:2]
m = similarity_from_two_points(src, dst)
landed = apply(m, src)

check(
    "two-point transform puts both eyes exactly on the template",
    np.allclose(landed, dst, atol=1e-9),
    f"landed at {landed.tolist()}, template is {dst.tolist()}",
)

# A similarity has no shear, so a square must stay a square. This is the
# property that separates a similarity from a general affine, and getting it
# wrong would stretch faces in a way that is hard to see by eye.
square = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
warped = apply(m, square)
side_a = np.linalg.norm(warped[1] - warped[0])
side_b = np.linalg.norm(warped[2] - warped[1])
diag_a = np.linalg.norm(warped[2] - warped[0])
diag_b = np.linalg.norm(warped[3] - warped[1])
check(
    "the transform is a similarity: no shear, no anisotropic scale",
    math.isclose(side_a, side_b, rel_tol=1e-9)
    and math.isclose(diag_a, diag_b, rel_tol=1e-9),
    f"sides {side_a:.6f} vs {side_b:.6f}, diagonals {diag_a:.6f} vs {diag_b:.6f}",
)

# Head roll is the thing two-point alignment is for. A rolled face and an
# upright one must produce the same aligned geometry.
angle = math.radians(23.0)
rot = np.array(
    [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
)
rolled = (rot @ src.T).T + np.array([37.0, -12.0])
m_rolled = similarity_from_two_points(rolled, dst)
check(
    "a rolled face aligns to the same place as an upright one",
    np.allclose(apply(m_rolled, rolled), dst, atol=1e-9),
    "roll is exactly what a similarity removes; if this fails, alignment is "
    "doing nothing for tilted heads",
)

# Scale invariance: the same face twice as far from the camera.
m_far = similarity_from_two_points(src * 0.5, dst)
check(
    "camera distance does not change the aligned result",
    np.allclose(apply(m_far, src * 0.5), dst, atol=1e-9),
)

try:
    similarity_from_two_points(np.array([[5.0, 5.0], [5.0, 5.0]]), dst)
    check("coincident eyes are refused", False, "no AlignmentError raised")
except AlignmentError:
    check(
        "coincident eyes are refused",
        True,
        "dividing by a zero eye separation would produce a confident "
        "nonsense crop rather than an obvious failure",
    )


# ----------------------------------------------------------------------
# Five-point transform
# ----------------------------------------------------------------------

# An exact similarity of the template must be recovered exactly.
angle = math.radians(-15.0)
rot = np.array(
    [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
)
transformed = (1.7 * rot @ ARCFACE_TEMPLATE_112.T).T + np.array([60.0, 25.0])
m5 = similarity_umeyama(transformed, ARCFACE_TEMPLATE_112)
check(
    "five-point recovers an exact similarity exactly",
    np.allclose(apply(m5, transformed), ARCFACE_TEMPLATE_112, atol=1e-6),
    str(apply(m5, transformed).tolist()),
)

# A real face is not a rigid copy of the template, so with noise the fit is a
# compromise. What matters is that it stays a compromise -- bounded, not wild.
rng = np.random.default_rng(0)
noisy = transformed + rng.normal(0.0, 1.5, transformed.shape)
m_noisy = similarity_umeyama(noisy, ARCFACE_TEMPLATE_112)
residual = np.linalg.norm(
    apply(m_noisy, noisy) - ARCFACE_TEMPLATE_112, axis=1
).max()
check(
    "five-point degrades gracefully under landmark noise",
    residual < 6.0,
    f"worst point off by {residual:.2f}px; least squares should spread the "
    f"error, not concentrate it",
)

# Reflection must never be produced: a mirrored alignment would be a
# confidently wrong crop.
mirrored = ARCFACE_TEMPLATE_112.copy()
mirrored[:, 0] = 112.0 - mirrored[:, 0]
m_mirror = similarity_umeyama(mirrored, ARCFACE_TEMPLATE_112)
det = np.linalg.det(m_mirror[:, :2])
check(
    "the fit never mirrors the face",
    det > 0,
    f"determinant {det:.4f}; a negative determinant is a reflection",
)

try:
    similarity_umeyama(np.zeros((5, 2)), ARCFACE_TEMPLATE_112)
    check("degenerate landmarks are refused", False, "no AlignmentError")
except AlignmentError:
    check("degenerate landmarks are refused", True)


# ----------------------------------------------------------------------
# align_face end to end
# ----------------------------------------------------------------------

frame = np.zeros((480, 640, 3), dtype=np.uint8)

two = FaceLandmarks(left_eye=(280.0, 200.0), right_eye=(340.0, 214.0))
aligned, method = align_face(frame, two)
check(
    "two-point alignment reports its method and the right shape",
    aligned.shape == (112, 112, 3) and method == ALIGNMENT_2POINT,
    f"{aligned.shape}, {method}",
)

five = FaceLandmarks(
    left_eye=(280.0, 200.0),
    right_eye=(340.0, 214.0),
    nose=(308.0, 236.0),
    mouth_left=(286.0, 262.0),
    mouth_right=(334.0, 274.0),
)
aligned5, method5 = align_face(frame, five)
check(
    "a complete landmark set selects five-point alignment",
    aligned5.shape == (112, 112, 3) and method5 == ALIGNMENT_5POINT,
    f"{aligned5.shape}, {method5}",
)

aligned_big, _ = align_face(frame, five, size=224)
check(
    "the template scales with the requested output size",
    aligned_big.shape == (224, 224, 3),
    str(aligned_big.shape),
)


# ----------------------------------------------------------------------
# Landmarks carried on the detection
# ----------------------------------------------------------------------


class Box:
    def __init__(self, landmarks=None):
        self.x1, self.y1, self.x2, self.y2 = 10.0, 20.0, 90.0, 120.0
        self.landmarks = landmarks


check(
    "a detector with no landmarks yields none",
    landmarks_from_box(Box()) is None,
    "RFB-320 is this case; the caller must find landmarks elsewhere",
)

five_pts = tuple((float(x), float(y)) for x, y in ARCFACE_TEMPLATE_112)
lm = landmarks_from_box(Box(five_pts))
check(
    "a detector emitting five points is used directly",
    lm is not None and lm.is_five_point,
    "this is the seam that lets SCRFD or RetinaFace replace RFB-320 without "
    "touching the identity layer",
)

lm2 = landmarks_from_box(Box(five_pts[:2]))
check(
    "a detector emitting two points falls back to eye alignment",
    lm2 is not None and not lm2.is_five_point,
)


# ----------------------------------------------------------------------
# Template compatibility
# ----------------------------------------------------------------------
# Imported here rather than at the top: arcface_matcher pulls in onnxruntime,
# and the geometry above must be testable without it.

try:
    from identity.arcface_matcher import ArcFaceMatcher  # noqa: E402

    compatible = ArcFaceMatcher.alignment_compatible
    check(
        "aligned and unaligned templates are not comparable",
        not compatible(ALIGNMENT_5POINT, ALIGNMENT_NONE)
        and not compatible(ALIGNMENT_NONE, ALIGNMENT_2POINT),
        "an embedding of a warped face and an embedding of a raw crop are "
        "pictures of different things; comparing them produces a low score "
        "for the right person",
    )
    check(
        "two-point and five-point templates are comparable",
        compatible(ALIGNMENT_2POINT, ALIGNMENT_5POINT),
        "both land on the same canonical template; they differ in how well "
        "they correct yaw, not in where the face ends up",
    )
    check(
        "unaligned matches unaligned",
        compatible(ALIGNMENT_NONE, ALIGNMENT_NONE),
        "the pre-alignment baseline must remain measurable",
    )
except ImportError as exc:  # pragma: no cover
    print(f"[SKIP] template compatibility ({exc})")


print()
if failures:
    print(f"{len(failures)} alignment test(s) failed")
    for f in failures:
        print(f"  !! {f}")
    raise SystemExit(1)
print("all alignment tests passed")
