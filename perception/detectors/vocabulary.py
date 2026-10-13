"""A model's class list, and what a site is allowed to ask of it.

Two checkpoints now run the same YOLO26 decode over different training
vocabularies: the stock COCO-80 one, and a 10-class construction-site-safety
fine-tune. The decode is genuinely identical -- the end-to-end head emits
`[x1, y1, x2, y2, score, class_id]` either way -- but everything that turns a
`class_id` into a name, validates a site's `classes` list, or picks a floor for
it is a property of the training set and not of the graph.

Bundling those together is the whole point of the type. A names tuple and a
validator that disagree raise nothing: the validator accepts a name the tuple
cannot label, and the detector then returns nothing at all for a class the
config asked for, forever, silently. Passing one object makes that pairing
unrepresentable at the call site.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence


@dataclass(frozen=True)
class Vocabulary:
    """The class list of one trained model, plus its validation and floors."""

    name: str
    """Human name for the training set, used in error messages: 'COCO'."""

    names: tuple[str, ...]
    """Class names in TRAINING ORDER. Order is the only thing mapping a class
    id (YOLO26) or a column (YOLOX) to a meaning, so it is not editable."""

    unknown_hint: str
    """Appended to the error raised when a site asks for a name this model was
    never trained on. Written per-vocabulary because the useful next step
    differs: for COCO it is 'that needs a different model', for a small
    fine-tune it is usually 'you meant one of these ten'."""

    default_thresholds: Mapping[str, float] = field(default_factory=dict)
    """Per-class score floors that apply when the config does not override
    them. Empty for COCO, whose classes have no measured asymmetry between
    them. Non-empty where recall has actually been measured per class and the
    cost of a miss is not the same for every class -- see `ppe.py`."""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "index", {name: i for i, name in enumerate(self.names)}
        )
        unknown = [k for k in self.default_thresholds if k not in self.index]
        if unknown:
            # A typo here is silent otherwise: the default never applies and
            # the class quietly takes the flat score_threshold instead.
            raise ValueError(
                f"{self.name}: default_thresholds names classes that do not "
                f"exist in this vocabulary: {unknown}"
            )

    def resolve(
        self,
        classes: Sequence[str],
        score_threshold: float,
        class_thresholds: Mapping[str, float] | None = None,
    ) -> tuple[tuple[str, ...], dict[str, float]]:
        """Validate a site's class list and expand its per-class thresholds.

        Shared across backends because the validation is a property of the
        training set rather than of the runtime, and because both failure modes
        below must behave identically whichever backend is configured -- a site
        that swaps backends and finds its class list silently accepted by one
        and rejected by the other has been handed a debugging session, not a
        config option.

        Per class, because one threshold cannot serve every class. A person
        fills the frame and scores 0.95; a phone is 40 pixels across and scores
        0.45 when it is really there. One number either floods the log with
        phantom objects or never sees a real one.

        Three layers, most specific first: the site's `class_thresholds`, then
        the vocabulary's own measured `default_thresholds`, then the flat
        `score_threshold`. The middle layer exists so that a model whose recall
        is known to differ by class carries that knowledge in code rather than
        depending on every site config repeating it correctly.
        """
        classes = tuple(classes)
        if not classes:
            raise ValueError("classes is empty: the detector would return nothing")
        unknown = [c for c in classes if c not in self.index]
        if unknown:
            raise ValueError(
                f"not {self.name} classes: {unknown}. {self.unknown_hint}"
            )

        overrides = dict(class_thresholds or {})
        stray = [c for c in overrides if c not in self.index]
        if stray:
            # Not fatal to the run, but always a mistake: a threshold written
            # against a name the model does not have is a threshold that is not
            # being applied, and nothing else would ever say so.
            raise ValueError(
                f"class_thresholds names classes {stray}, which are not "
                f"{self.name} classes and would never be applied. Known "
                f"names: {', '.join(self.names)}"
            )

        thresholds = {
            name: float(
                overrides.get(
                    name, self.default_thresholds.get(name, score_threshold)
                )
            )
            for name in classes
        }
        return classes, thresholds


__all__ = ["Vocabulary"]
