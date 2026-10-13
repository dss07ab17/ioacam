"""The PPE model: its ten classes, its floors, and the backend that reads them.

Model:   YOLO26n fine-tuned on Construction-Site-Safety   AGPL-3.0 (Ultralytics)
Weights: perception/models/ppe_yolo26.onnx                same terms
Runtime: onnxruntime                                      MIT

This is a *second detector*, not a second class list on the first one. The
person detector keeps running exactly as it did -- it owns tracking, zones,
occupancy and pose -- and this one runs alongside it purely to say whether the
people it already found are wearing a hard hat and a vest. Nothing here mints a
track id.

## Why a separate model rather than more classes on the COCO one

Because COCO has no hard hat and no high-vis vest, and no threshold on a
COCO-trained graph will conjure one. The classes below come from a different
training set, so they need a different checkpoint, and a different checkpoint
is a different forward pass. That cost is real and is reported in the README;
it is why the whole stage is off by default.

## Why the decode is `yolo26.py`'s, unchanged

The fine-tune changed the training data and the number of output channels. It
did not change the head: `end2end=True`, `nms=False`, output `(1, 300, 6)` of
`[x1, y1, x2, y2, score, class_id]`, 640x640 letterboxed input. Every word of
`yolo26.py`'s note on why that decode is not YOLOX's with the NMS deleted
applies here identically, so `PpeDetector` is that decode with this
vocabulary bolted on and nothing else.

## The class list is not COCO's, and that is the only thing that differs

Ten classes, in training order (the order is the class-id mapping and is not
editable). Six of them describe PPE; the other four are scene furniture the
fine-tune happened to be trained on.

Note `Person` here, capitalised, is this model's own person class and is NOT
the one the pipeline tracks. It is off by default and should stay off: two
detectors both emitting people would produce two sets of boxes for the same
bodies, and the tracker has no way to know they are the same person.

## Only two of the six PPE classes have somewhere to go

`schema/event.schema.json` has `ppe_helmet`, `ppe_vest`, `ppe_glasses` and
`ppe_gloves`, and the observation list is closed on purpose. Hardhat and Safety
Vest map onto the first two; the model's Mask classes map onto nothing, because
a mask is not glasses and is not gloves. They are detectable and are left
un-emitted rather than squeezed into the nearest-looking slot -- an engine rule
written against `ppe_glasses` must not fire on a face mask. Adding
`ppe_mask` is a schema change with a feasibility assessment and an ATP test
behind it, which is what that list's own description asks for.

## The floors are asymmetric on purpose

Measured recall on the NO-* classes on site footage is 0.55-0.62, against
noticeably better numbers on the positive classes. That asymmetry is not
noise: an unworn hard hat is a smaller, less distinctive visual event than a
worn one -- there is no object to find, only the absence of one on a head that
may be turned away, in shadow, or half occluded.

The costs are asymmetric too, and in the opposite direction. A false
`ppe_helmet: false` costs somebody walking over to look. A missed one costs an
unprotected head on a live floor. So the NO-* classes are floored at 0.20 and
the positives at 0.40, and the resulting over-flagging is the intended trade
rather than a defect to tune away. A site that wants the other trade raises
them in `class_thresholds`; nothing here stops it.
"""

from __future__ import annotations

from .vocabulary import Vocabulary
from .yolo26 import Yolo26Detector

DEFAULT_PPE_MODEL = "perception/models/ppe_yolo26.onnx"

# Training order. This IS the class-id mapping -- index 2 is what the head
# emits for a bare head -- so it is not editable, sortable or de-duplicated.
PPE_CLASSES = (
    "Hardhat",          # 0
    "Mask",             # 1
    "NO-Hardhat",       # 2
    "NO-Mask",          # 3
    "NO-Safety Vest",   # 4
    "Person",           # 5  this model's person, NOT the tracked one
    "Safety Cone",      # 6
    "Safety Vest",      # 7
    "machinery",        # 8
    "vehicle",          # 9
)

# class name -> (schema observation, the boolean value it asserts).
#
# A closed map rather than a naming convention: deriving the observation by
# stripping a "NO-" prefix would silently invent an observation for every
# future class the fine-tune gains, and the schema's list is closed.
PPE_OBSERVATIONS = {
    "Hardhat": ("ppe_helmet", True),
    "NO-Hardhat": ("ppe_helmet", False),
    "Safety Vest": ("ppe_vest", True),
    "NO-Safety Vest": ("ppe_vest", False),
}

# The four classes that map onto an observation. Enabling the other six by
# default would cost the same forward pass and emit nothing.
DEFAULT_PPE_CLASSES = tuple(PPE_OBSERVATIONS)

# See the module note: recall on the NO-* classes is 0.55-0.62 and a miss is
# worse than a false alarm, so they start low and the positives do not.
PPE_DEFAULT_THRESHOLDS = {
    "Hardhat": 0.40,
    "Safety Vest": 0.40,
    "NO-Hardhat": 0.20,
    "NO-Safety Vest": 0.20,
    # Detectable, not emittable (no schema observation). Floored consistently
    # with their counterparts so that turning them on for a diagnostic run
    # measures the same thing the emitted classes are measured at.
    "Mask": 0.40,
    "NO-Mask": 0.20,
}

PPE = Vocabulary(
    name="PPE",
    names=PPE_CLASSES,
    unknown_hint=(
        "This is the 10-class construction-site-safety fine-tune, not COCO -- "
        "its names are case-sensitive and spelled exactly: "
        + ", ".join(PPE_CLASSES)
        + ". COCO names ('person', 'helmet') are rejected here rather than "
        "silently returning nothing. Note 'Person' is this model's own person "
        "class and must not be enabled to track people: the person detector "
        "configured under `detector` does that."
    ),
    default_thresholds=PPE_DEFAULT_THRESHOLDS,
)


class PpeDetector(Yolo26Detector):
    """`Yolo26Detector` over `PPE` instead of `COCO`. Nothing else differs."""

    name = "yolo26-ppe"
    licence = (
        "AGPL-3.0 (Ultralytics YOLO26 code, and weights fine-tuned from an "
        "Ultralytics checkpoint) -- NOT SHIPPABLE without a commercial licence "
        "from Ultralytics; see perception/LICENCE-NOTES.md"
    )
    agpl_note = (
        "Internal evaluation only. Unlike the person detector, this one has no\n"
        "shippable alternative: there is no non-AGPL PPE model here yet."
    )
    weights_hint = (
        "These weights are a site fine-tune, not a download: they are not\n"
        "produced by export_yolo26.py and there is no --download for them."
    )

    def __init__(
        self,
        model_path: str = DEFAULT_PPE_MODEL,
        classes=DEFAULT_PPE_CLASSES,
        **kwargs,
    ) -> None:
        super().__init__(
            model_path=model_path, classes=classes, vocabulary=PPE, **kwargs
        )


def observation_for(label: str):
    """(observation, value) for a PPE class, or None if it maps to nothing."""
    return PPE_OBSERVATIONS.get(label)


__all__ = [
    "DEFAULT_PPE_CLASSES",
    "DEFAULT_PPE_MODEL",
    "PPE",
    "PPE_CLASSES",
    "PPE_DEFAULT_THRESHOLDS",
    "PPE_OBSERVATIONS",
    "PpeDetector",
    "observation_for",
]
