"""The COCO-80 class list, shared by every COCO-trained backend.

It lives here rather than in one backend because there is now more than one
backend reading the same 80 columns, and two copies of an order-sensitive list
is a defect waiting to happen: nothing would fail loudly if they drifted, the
detector would simply start labelling cups as bowls.

Order is the only thing mapping a column (YOLOX) or a class id (YOLO26) to a
meaning, so it is not editable.
"""

from __future__ import annotations

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
CLASS_INDEX = {name: i for i, name in enumerate(COCO_CLASSES)}


def resolve_classes(
    classes, score_threshold: float, class_thresholds: dict | None
) -> tuple[tuple[str, ...], dict[str, float]]:
    """Validate a site's class list and expand its per-class thresholds.

    Shared because the validation is a licence-independent property of "the
    model was trained on COCO", and because both failure modes below must
    behave identically whichever backend is configured -- a site that swaps
    backends and finds its class list silently accepted by one and rejected by
    the other has been handed a debugging session, not a config option.

    Per class, because one threshold cannot serve both. A person fills the
    frame and scores 0.95; a phone is 40 pixels across and scores 0.45 when it
    is really there. One number either floods the log with phantom objects or
    never sees a real one.
    """
    classes = tuple(classes)
    if not classes:
        raise ValueError("classes is empty: the detector would return nothing")
    unknown = [c for c in classes if c not in CLASS_INDEX]
    if unknown:
        raise ValueError(
            f"not COCO classes: {unknown}. This model knows only its 80 "
            f"training classes -- a site-specific object (a torque wrench, "
            f"a specific tool) needs a detector trained on it, and asking "
            f"for it here would silently return nothing."
        )
    thresholds = {
        name: float((class_thresholds or {}).get(name, score_threshold))
        for name in classes
    }
    return classes, thresholds


__all__ = [
    "CLASS_INDEX",
    "COCO_CLASSES",
    "COCO_PERSON_CLASS",
    "resolve_classes",
]
