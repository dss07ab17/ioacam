"""Raw model score -> calibrated confidence, with the components kept.

The README leaves one question open: "calibration.temperature is loaded and
carried but not applied; confidence arrives pre-calibrated from perception.
Where that correction lives is a perception-layer decision." This module is
that decision. Temperature scaling is applied HERE, once, and the engine keeps
receiving confidence it can take at face value.

Everything below is arithmetic and geometry. Nothing is learned, nothing is
inferred by a second model, which is what the schema means by "Computed
without AI" -- a quality score that is itself a model output would need its own
calibration, and the regress has to stop somewhere.
"""

from __future__ import annotations

import math
from typing import Sequence

from detectors.base import Detection

EPS = 1e-6


def _logit(p: float) -> float:
    p = min(max(p, EPS), 1.0 - EPS)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def temperature_scale(raw_score: float, temperature: float) -> float:
    """Divide the logit by T. T > 1 softens overconfidence toward 0.5.

    Detectors are systematically overconfident; a raw 0.95 is not a 95% chance
    of being right. T comes from the site's fitted calibration and MUST match
    the value in the policy file, or the engine's confidence thresholds are
    being compared against a differently-scaled number than they were tuned on.
    """
    if temperature <= 0:
        raise ValueError(f"Calibration temperature must be positive, got {temperature}")
    return _sigmoid(_logit(raw_score) / temperature)


def quality(
    detection: Detection,
    frame_w: int,
    frame_h: int,
    others: Sequence[Detection] = (),
    blur_score: float = 1.0,
    luminance_score: float = 1.0,
    reference_height_px: float = 220.0,
    min_height_px: float = 40.0,
) -> float:
    """Viewing-condition factor in 0..1. Multiplicative, all-must-hold.

    A product, not a mean, because these are independent ways of being
    unusable: a tack-sharp 12-pixel-tall person is still not identifiable, and
    averaging would let good lighting hide that.
    """
    # Pixel height: the dominant term. Below min_height_px a person is not
    # reliably a person, let alone in the right zone.
    h = detection.height
    if h <= min_height_px:
        size = 0.0
    else:
        size = min(1.0, (h - min_height_px) / max(EPS, reference_height_px - min_height_px))
        size = 0.3 + 0.7 * size  # a small-but-usable subject is not worthless

    # Truncation: a box against the frame edge is a partial view of a subject
    # whose feet -- the ground point zone membership depends on -- may be
    # outside the image entirely.
    margin = 2.0
    truncated = sum(
        1
        for touching in (
            detection.x1 <= margin,
            detection.y1 <= margin,
            detection.x2 >= frame_w - margin,
            detection.y2 >= frame_h - margin,
        )
        if touching
    )
    truncation = max(0.4, 1.0 - 0.2 * truncated)

    # Occlusion, approximated by overlap with other person boxes. Without a
    # depth cue we cannot tell who is in front, so both parties are discounted.
    overlap = 0.0
    for other in others:
        if other is detection:
            continue
        ix = max(0.0, min(detection.x2, other.x2) - max(detection.x1, other.x1))
        iy = max(0.0, min(detection.y2, other.y2) - max(detection.y1, other.y1))
        if detection.area > 0:
            overlap = max(overlap, (ix * iy) / detection.area)
    occlusion = max(0.3, 1.0 - overlap)

    return max(0.0, min(1.0, size * truncation * occlusion * blur_score * luminance_score))


def blur_and_luminance(frame) -> tuple[float, float]:
    """Scene-level sharpness and exposure, both mapped into 0..1.

    Computed once per frame rather than per detection: motion blur and
    luminance are properties of the capture, and doing it per box at 30 fps on
    a laptop CPU costs more than the detector.
    """
    import cv2
    import numpy as np

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(grey, (160, 120), interpolation=cv2.INTER_AREA)

    # Variance of the Laplacian. ~100+ is sharp for a webcam; below ~20 the
    # frame is smeared and box edges are not trustworthy.
    sharpness = float(cv2.Laplacian(small, cv2.CV_64F).var())
    blur_score = max(0.4, min(1.0, sharpness / 100.0))

    # Penalise both crushed blacks and blown highlights, symmetric about mid-grey.
    mean = float(np.mean(small)) / 255.0
    luminance_score = max(0.4, min(1.0, 1.0 - 2.0 * abs(mean - 0.5)))
    return blur_score, luminance_score


def compose(
    raw_score: float,
    temperature: float,
    quality_score: float,
    persistence: float,
    frames_observed: int,
    agreement: float = 1.0,
) -> tuple[float, dict]:
    """Combine into the final confidence plus the schema's components block.

    Returned together because the schema requires them to be consistent, and
    the one thing a site engineer must be able to do is read the components and
    see WHY a confidence was low -- a genuinely ambiguous detection and a good
    detection in bad light are different problems with different fixes.
    """
    agreement = max(0.5, min(1.2, agreement))  # schema bounds; not independent sensors
    calibrated = temperature_scale(raw_score, temperature)
    confidence = max(0.0, min(1.0, calibrated * quality_score * persistence * agreement))
    return confidence, {
        "raw_score": round(min(1.0, max(0.0, raw_score)), 4),
        "calibrated_score": round(calibrated, 4),
        "quality": round(quality_score, 4),
        "persistence": round(persistence, 4),
        "agreement": round(agreement, 4),
        "frames_observed": max(1, int(frames_observed)),
    }
