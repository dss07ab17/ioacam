"""Fusing an RGB stream with a skeleton stream.

Running both is worth the cost because they fail differently. The RGB model is
the only one that can see objects, so it is the one that separates "reaching"
from "reaching with a spanner". The skeleton model is largely immune to
lighting, clothing colour and site-to-site appearance shift, so it is the one
that still works when the RGB model has quietly gone out of distribution.

The rule below is deliberately conservative: agreement raises confidence a
little, disagreement collapses it. That asymmetry is the point. Two streams
disagreeing means at least one is wrong and we do not know which, and acting on
the louder one is how a two-stream system ends up less reliable than either
stream alone.
"""

from __future__ import annotations

from typing import Optional

from base import ActionScore


class StreamFusion:
    """Combine per-track scores from two recognisers."""

    def __init__(
        self,
        rgb_weight: float = 0.65,
        skeleton_weight: float = 0.35,
        agreement_bonus: float = 1.10,
        disagreement_penalty: float = 0.55,
    ) -> None:
        total = rgb_weight + skeleton_weight
        self.rgb_weight = rgb_weight / total
        self.skeleton_weight = skeleton_weight / total
        self.agreement_bonus = agreement_bonus
        self.disagreement_penalty = disagreement_penalty

    def fuse(
        self, rgb: Optional[ActionScore], skeleton: Optional[ActionScore]
    ) -> ActionScore:
        # Only one stream available: pass it through unchanged. Inventing a
        # fusion bonus from a single opinion would be manufacturing confidence.
        if rgb is None and skeleton is None:
            return ActionScore(
                track_id="",
                label=None,
                confidence=0.0,
                abstained=True,
                reason="no stream produced a score",
                backend="fusion",
            )
        if skeleton is None:
            return rgb
        if rgb is None:
            return skeleton

        track_id = rgb.track_id or skeleton.track_id

        # Both abstained: nothing to fuse, and the reasons are worth keeping
        # together because they usually differ.
        if rgb.abstained and skeleton.abstained:
            return ActionScore(
                track_id=track_id,
                label=None,
                confidence=max(rgb.confidence, skeleton.confidence),
                abstained=True,
                reason=f"both streams abstained (rgb: {rgb.reason}; skel: {skeleton.reason})",
                backend="fusion",
                window_start_us=rgb.window_start_us,
                window_end_us=rgb.window_end_us,
            )

        # One abstained. Take the other, but discount it: a stream declining to
        # answer is evidence against the one that did, not neutral.
        if rgb.abstained or skeleton.abstained:
            decided = skeleton if rgb.abstained else rgb
            other = rgb if rgb.abstained else skeleton
            return ActionScore(
                track_id=track_id,
                label=decided.label,
                confidence=round(decided.confidence * self.disagreement_penalty, 4),
                raw_scores=decided.raw_scores,
                abstained=False,
                reason=(
                    f"only {decided.backend} decided ('{decided.label}'); "
                    f"{other.backend} abstained -- discounted"
                ),
                backend="fusion",
                window_start_us=decided.window_start_us,
                window_end_us=decided.window_end_us,
            )

        if rgb.label == skeleton.label:
            blended = (
                rgb.confidence * self.rgb_weight
                + skeleton.confidence * self.skeleton_weight
            )
            return ActionScore(
                track_id=track_id,
                label=rgb.label,
                confidence=round(min(1.0, blended * self.agreement_bonus), 4),
                raw_scores=rgb.raw_scores,
                abstained=False,
                reason=(
                    f"both streams agree on '{rgb.label}' "
                    f"(rgb {rgb.confidence:.2f}, skel {skeleton.confidence:.2f})"
                ),
                backend="fusion",
                window_start_us=rgb.window_start_us,
                window_end_us=rgb.window_end_us,
            )

        # Outright disagreement. Abstain rather than pick a winner: at least one
        # stream is wrong and nothing here says which.
        return ActionScore(
            track_id=track_id,
            label=None,
            confidence=round(
                max(rgb.confidence, skeleton.confidence) * self.disagreement_penalty, 4
            ),
            raw_scores={},
            abstained=True,
            reason=(
                f"streams disagree: rgb '{rgb.label}' ({rgb.confidence:.2f}) vs "
                f"skeleton '{skeleton.label}' ({skeleton.confidence:.2f})"
            ),
            backend="fusion",
            window_start_us=rgb.window_start_us,
            window_end_us=rgb.window_end_us,
        )

    @staticmethod
    def from_config(d: dict) -> "StreamFusion":
        return StreamFusion(
            rgb_weight=float(d.get("rgb_weight", 0.65)),
            skeleton_weight=float(d.get("skeleton_weight", 0.35)),
            agreement_bonus=float(d.get("agreement_bonus", 1.10)),
            disagreement_penalty=float(d.get("disagreement_penalty", 0.55)),
        )
