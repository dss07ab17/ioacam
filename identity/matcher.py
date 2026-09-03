"""The face matching seam, and the threshold policy that sits in front of it.

Two separate concerns here, kept apart deliberately.

`FaceMatcher` is the seam: an interface with a stub for testing. The real
implementation wraps whatever embedding model the door device already uses,
and nothing else in the codebase knows which one that is -- the same pattern as
`perception/detectors/`.

`ThresholdPolicy` is the part worth reading. It answers one question: given
that we are comparing this face against N candidates, how good does the best
match have to be before we believe it?
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Sequence


class FaceMatcher(Protocol):
    """Compare a probe against enrolled templates.

    Scores are similarities in 0..1, higher is more alike. The interface takes
    the whole candidate list at once because a real implementation batches the
    comparison, and because the policy needs the runner-up score, not just the
    winner.
    """

    licence: str

    def compare(self, probe, identities: Sequence[str]) -> dict[str, float]:
        """Return {identity: similarity} for every candidate."""
        ...


class StubMatcher:
    """Deterministic matcher for tests. Never ships.

    Scores come from a dict supplied by the test, so the resolver's logic can
    be exercised without a model, a camera, or enrolment data.
    """

    licence = "n/a (test stub)"

    def __init__(self, scores: Optional[dict] = None, default: float = 0.10) -> None:
        # {probe_key: {identity: score}}
        self.scores = scores or {}
        self.default = default

    def compare(self, probe, identities: Sequence[str]) -> dict[str, float]:
        table = self.scores.get(probe, {})
        return {ident: table.get(ident, self.default) for ident in identities}


@dataclass
class RocPoint:
    """One operating point from the model's measured ROC curve."""

    threshold: float
    false_match_rate: float


class ThresholdPolicy:
    """Picks a similarity threshold from the size of the candidate set.

    The reason thresholds must tighten as the candidate list grows is not
    intuition, it is arithmetic. If a single comparison has false match rate p,
    then comparing against N candidates gives roughly N*p chance that at least
    one wrong person scores above threshold. Holding one threshold fixed while
    N goes from 4 to 200 multiplies the false match rate by fifty.

    So the policy is expressed the other way round. Declare the false match
    rate you are willing to accept per DECISION, and derive the per-comparison
    threshold from the candidate count:

        required per-comparison FMR <= target / N

    This is the same principle as the engine's confidence calibration: the
    number that goes into the decision has to mean what it says, and what it
    means is measured at commissioning rather than assumed.

    The ROC table comes from the site's own enrolment data. A curve measured on
    the vendor's benchmark will not describe your cameras, your lighting or your
    population.
    """

    def __init__(
        self,
        roc: Sequence[RocPoint],
        target_false_match_rate: float = 1e-4,
        min_margin: float = 0.05,
        floor: float = 0.35,
    ) -> None:
        if not roc:
            raise ValueError("ThresholdPolicy needs a measured ROC curve")
        self.roc = sorted(roc, key=lambda p: p.threshold)
        self.target = target_false_match_rate
        # The best match must beat the runner-up by this much. With four
        # candidates scoring 0.71 and 0.70, the winner is noise, and binding an
        # identity on noise is how the wrong person's name ends up attached to
        # a violation.
        self.min_margin = min_margin
        # Below this, no threshold is trusted regardless of what the ROC says.
        self.floor = floor

    def threshold_for(self, n_candidates: int) -> Optional[float]:
        """Lowest threshold meeting the target for this candidate count.

        Returns None when no operating point is good enough. That is a real
        outcome, not an error: with a large enough candidate set and a weak
        enough model, the honest answer is that this face cannot be identified
        to the required confidence, and the caller must report unverified
        rather than guess.
        """
        n = max(1, n_candidates)
        required = self.target / n
        for point in self.roc:
            if point.false_match_rate <= required:
                return max(point.threshold, self.floor)
        return None

    def max_candidates(self) -> int:
        """Largest candidate set this model can decide over at the target.

        Worth computing at commissioning and putting in front of whoever owns
        the site, because it is a hard ceiling and it is easy to be surprised
        by. With a strongest measured operating point of FMR p, the arithmetic
        allows at most target/p candidates; beyond that every comparison is
        refused and tier 2 stops working entirely.

        The consequence is operational, not algorithmic: the site roster has to
        be kept small. That is what badge-out readers, the presence TTL and the
        end-of-day reset are actually for. They are not hygiene, they are what
        keeps whole-site identification functional.
        """
        strongest = min(point.false_match_rate for point in self.roc)
        return max(1, int(self.target / strongest))

    @staticmethod
    def from_config(d: dict) -> "ThresholdPolicy":
        return ThresholdPolicy(
            roc=[
                RocPoint(float(p["threshold"]), float(p["false_match_rate"]))
                for p in d["roc"]
            ],
            target_false_match_rate=float(d.get("target_false_match_rate", 1e-4)),
            min_margin=float(d.get("min_margin", 0.05)),
            floor=float(d.get("floor", 0.35)),
        )
