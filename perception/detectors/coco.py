"""The COCO-80 class list, shared by every COCO-trained backend.

It lives here rather than in one backend because there is now more than one
backend reading the same 80 columns, and two copies of an order-sensitive list
is a defect waiting to happen: nothing would fail loudly if they drifted, the
detector would simply start labelling cups as bowls.

Order is the only thing mapping a column (YOLOX) or a class id (YOLO26) to a
meaning, so it is not editable.

The validation and threshold-expansion that used to live here are now generic,
in `vocabulary.py`, because a second model with a different class list arrived
(`ppe.py`). `resolve_classes` is kept as the COCO-bound spelling of it so the
existing call sites and their tests read unchanged.
"""

from __future__ import annotations

from .vocabulary import Vocabulary

COCO_PERSON_CLASS = 0

COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)

COCO = Vocabulary(
    name="COCO",
    names=COCO_CLASSES,
    unknown_hint=(
        "This model knows only its 80 training classes -- a site-specific "
        "object (a torque wrench, a specific tool) needs a detector trained "
        "on it, and asking for it here would silently return nothing."
    ),
    # No per-class defaults: nothing has been measured across the 80 that would
    # justify one class starting lower than another. A site that has measured
    # its own puts them in `class_thresholds`.
)

CLASS_INDEX = COCO.index


def resolve_classes(
    classes, score_threshold: float, class_thresholds: dict | None = None
) -> tuple[tuple[str, ...], dict[str, float]]:
    """COCO's `Vocabulary.resolve`. See `vocabulary.py` for what it enforces."""
    return COCO.resolve(classes, score_threshold, class_thresholds)


__all__ = [
    "CLASS_INDEX",
    "COCO",
    "COCO_CLASSES",
    "COCO_PERSON_CLASS",
    "resolve_classes",
]
