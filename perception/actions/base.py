"""Action recognition: the contract, and the two rules that make it safe to use.

An action recogniser answers one question -- what is this person doing -- and it
must be able to decline to answer.

That second part is not a nicety. A classifier trained on N classes does not say
"I have not seen this before"; it returns the nearest class it knows with a
mediocre score. So an action nobody trained it on comes back as "standing, 0.4"
and silently disappears. If a prohibited-action list is built on top of that,
the list only catches what the model was already taught, and everything else
reads as normal. Abstention is what turns that silent miss into an `unknown`
that a human sees.

The seam follows `perception/detectors/`: the backend is a config choice, and
its licence is a property of the backend rather than of the pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Protocol, Sequence


@dataclass
class ActionScore:
    """One recogniser's opinion about one person over one time window."""

    track_id: str
    label: Optional[str]              # None when the recogniser abstained
    confidence: float
    raw_scores: dict[str, float] = field(default_factory=dict)
    abstained: bool = False
    reason: str = ""
    backend: str = ""
    window_start_us: int = 0
    window_end_us: int = 0

    @property
    def decided(self) -> bool:
        return self.label is not None and not self.abstained


class ActionRecognizer(Protocol):
    """Backends implement this and nothing else knows which one is loaded."""

    name: str
    licence: str
    classes: Sequence[str]
    # Frames the backend expects in a tube, and at what side length.
    num_frames: int
    input_size: int

    def infer(self, tube) -> dict[str, float]:
        """Return {class: score} for one person's tube. Scores need not be
        normalised; the abstention policy handles that."""
        ...


class AbstentionPolicy:
    """Decides whether a set of class scores constitutes an answer.

    Two independent tests, and the score must pass both.

    `min_confidence` is the obvious one: the top class has to clear a bar.

    `min_margin` is the one people forget. A model that scores its top two
    classes at 0.41 and 0.39 has not recognised anything -- it is undecided
    between two options, and picking the higher one records a coin flip as a
    fact. On a factory floor the confusable pairs are exactly the ones that
    matter (tightening vs inspecting, reaching vs reaching-with-tool), so the
    margin test rejects more real errors than the threshold test does.

    Thresholds are per class, because a model's reliability is not uniform
    across its vocabulary: coarse whole-body actions are far more separable
    than fine-grained manipulations, and holding them to one number either
    admits noise on the hard classes or discards good detections on the easy
    ones.
    """

    def __init__(
        self,
        min_confidence: float = 0.55,
        min_margin: float = 0.15,
        per_class: Optional[dict[str, float]] = None,
        temperature: float = 1.0,
    ) -> None:
        self.min_confidence = min_confidence
        self.min_margin = min_margin
        self.per_class = per_class or {}
        # Same correction as everywhere else in this system: a raw model score
        # is not a probability, and the number that drives a response has to
        # mean what it says. Fitted on held-out site data at commissioning.
        self.temperature = temperature

    # ------------------------------------------------------------------

    def softmax(self, scores: dict[str, float]) -> dict[str, float]:
        if not scores:
            return {}
        t = max(self.temperature, 1e-6)
        vals = {k: v / t for k, v in scores.items()}
        hi = max(vals.values())
        exp = {k: math.exp(v - hi) for k, v in vals.items()}
        total = sum(exp.values()) or 1.0
        return {k: v / total for k, v in exp.items()}

    def decide(
        self, track_id: str, scores: dict[str, float], backend: str = ""
    ) -> ActionScore:
        if not scores:
            return ActionScore(
                track_id=track_id,
                label=None,
                confidence=0.0,
                abstained=True,
                reason="backend returned no scores",
                backend=backend,
            )

        probs = self.softmax(scores)
        ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        top_label, top = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0

        threshold = self.per_class.get(top_label, self.min_confidence)

        if top < threshold:
            return ActionScore(
                track_id=track_id,
                label=None,
                confidence=round(top, 4),
                raw_scores=probs,
                abstained=True,
                reason=(
                    f"best class '{top_label}' at {top:.3f} below threshold "
                    f"{threshold:.3f}"
                ),
                backend=backend,
            )

        if top - runner_up < self.min_margin:
            return ActionScore(
                track_id=track_id,
                label=None,
                confidence=round(top, 4),
                raw_scores=probs,
                abstained=True,
                reason=(
                    f"undecided between '{top_label}' ({top:.3f}) and "
                    f"'{ranked[1][0]}' ({runner_up:.3f}); margin "
                    f"{top - runner_up:.3f} below {self.min_margin:.3f}"
                ),
                backend=backend,
            )

        return ActionScore(
            track_id=track_id,
            label=top_label,
            confidence=round(top, 4),
            raw_scores=probs,
            abstained=False,
            reason=(
                f"'{top_label}' at {top:.3f}, margin {top - runner_up:.3f} "
                f"over '{ranked[1][0] if len(ranked) > 1 else '-'}'"
            ),
            backend=backend,
        )

    @staticmethod
    def from_config(d: dict) -> "AbstentionPolicy":
        return AbstentionPolicy(
            min_confidence=float(d.get("min_confidence", 0.55)),
            min_margin=float(d.get("min_margin", 0.15)),
            per_class=d.get("per_class_confidence") or {},
            temperature=float(d.get("temperature", 1.0)),
        )
