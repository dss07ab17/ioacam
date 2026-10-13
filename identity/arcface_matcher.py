"""ArcFace face matcher with multi-view enrollment support.

A probe face is embedded once, then compared against every enrolled sample of
each candidate identity.

The identity's score is the BEST similarity among its samples.

Example:

    rahul
        front embedding
        left embedding
        right embedding

A live face is compared against all three, and the highest similarity becomes
Rahul's final candidate score.
"""

from __future__ import annotations

import hashlib
import os

from typing import Sequence


import cv2
import numpy as np
import onnxruntime as ort

from .alignment import (
    ALIGNMENT_2POINT,
    ALIGNMENT_5POINT,
    ALIGNMENT_NONE,
)


REC_INPUT_SIZE = 112
EMBEDDING_SIZE = 512


def sha256_of(
    path: str
) -> str:

    h = hashlib.sha256()

    with open(
        path,
        "rb"
    ) as f:

        for chunk in iter(
            lambda: f.read(
                1 << 20
            ),
            b""
        ):

            h.update(
                chunk
            )

    return h.hexdigest()


class ModelIntegrityError(
    RuntimeError
):
    """
    Weights on disk don't match the hash
    recorded in configuration.
    """


class TemplateMismatchError(
    RuntimeError
):
    """
    Enrollment template was generated
    using a different recognition model.
    """


class ArcFaceMatcher:
    """
    ArcFace ResNet100 ONNX face matcher.

    templates normally has this structure:

        {
            "rahul": [
                front_template,
                left_template,
                right_template
            ],

            "user2": [
                front_template,
                left_template,
                right_template
            ]
        }

    Old single-template values are also accepted.
    """

    def __init__(
        self,
        model_path: str,
        templates,
        expected_sha256: str | None = None,
        session: (
            ort.InferenceSession
            | None
        ) = None,
        require_alignment: bool = True,
        on_incompatible_template=None,
    ) -> None:

        self.licence = (
            "arcfaceresnet100-11-int8.onnx "
            "(ONNX Model Zoo), "
            "onnxruntime"
        )

        self.model_path = (
            model_path
        )

        self.templates = (
            templates
        )

        # -----------------------------------------------------
        # Alignment contract
        #
        # True means this matcher refuses to embed anything
        # that has not been warped onto the ArcFace template,
        # and refuses to compare against templates that were
        # enrolled without alignment.
        #
        # Set False only to measure what alignment is worth on
        # your own footage. Do not ship it False: the failure
        # is silent, and shows up as a quietly poor recognition
        # rate rather than as an error.
        # -----------------------------------------------------

        self.require_alignment = (
            require_alignment
        )

        # Called with (identity, reason) when a template has to
        # be skipped, so the caller can log it. Skipping is
        # per-identity on purpose: one person enrolled against
        # old weights or an old alignment must not blind
        # identification for everybody else in the zone.

        self._on_incompatible = (
            on_incompatible_template
        )

        # -----------------------------------------------------
        # Normal production model loading
        # -----------------------------------------------------

        if session is None:

            if not os.path.exists(
                model_path
            ):

                raise FileNotFoundError(
                    f"ArcFace recognizer weights "
                    f"not found at {model_path}.\n"
                    f"Copy "
                    f"arcfaceresnet100-11-int8.onnx "
                    f"into identity/models/."
                )

            actual_sha256 = sha256_of(
                model_path
            )

            # -------------------------------------------------
            # Verify model integrity
            # -------------------------------------------------

            if (
                expected_sha256
                and actual_sha256
                != expected_sha256
            ):

                raise ModelIntegrityError(

                    f"{model_path} hash "
                    f"{actual_sha256} "
                    f"does not match expected "
                    f"{expected_sha256}. "

                    f"A changed recognition model "
                    f"invalidates existing thresholds "
                    f"and templates."
                )

            self.model_sha256 = (
                actual_sha256
            )

            # -------------------------------------------------
            # ONNX Runtime session
            # -------------------------------------------------

            session = (
                ort.InferenceSession(

                    model_path,

                    providers=[
                        "CPUExecutionProvider"
                    ],
                )
            )

        else:

            # Test seam
            self.model_sha256 = (
                expected_sha256
                or "test-session"
            )

        self.session = (
            session
        )

        self._input_name = (
            session
            .get_inputs()[0]
            .name
        )

        self._output_name = (
            session
            .get_outputs()[0]
            .name
        )

    # =========================================================
    # EMBEDDING
    # =========================================================

    def _embed(
        self,
        face_crop: np.ndarray
    ) -> np.ndarray:

        """
        Embed an ALIGNED 112x112 face.

        The input is expected to come from
        alignment.align_face, not straight from
        FaceBox.crop. Resizing a detector box to
        112x112 puts the eyes, nose and mouth in
        the wrong places, and ArcFace returns a
        worse embedding without complaining.
        """

        if (
            face_crop is None
            or face_crop.size == 0
        ):

            raise ValueError(
                "empty face crop passed "
                "to ArcFaceMatcher"
            )

        # -----------------------------------------------------
        # Refuse an unaligned crop rather than resize it
        #
        # The old code resized whatever it was handed. That is
        # exactly how an unaligned face gets embedded silently:
        # the shape is made to fit and nothing reports that the
        # geometry is wrong.
        # -----------------------------------------------------

        if self.require_alignment:

            h, w = face_crop.shape[:2]

            if (
                h != REC_INPUT_SIZE
                or w != REC_INPUT_SIZE
            ):

                raise ValueError(

                    f"ArcFaceMatcher expects an aligned "
                    f"{REC_INPUT_SIZE}x{REC_INPUT_SIZE} crop "
                    f"from alignment.align_face, "

                    f"got {w}x{h}. "

                    f"Pass require_alignment=False only to "
                    f"measure the unaligned baseline."
                )

        # -----------------------------------------------------
        # BGR → RGB
        # -----------------------------------------------------

        rgb = cv2.cvtColor(

            face_crop,

            cv2.COLOR_BGR2RGB
        )

        # -----------------------------------------------------
        # ArcFace input = 112 × 112
        # -----------------------------------------------------

        resized = cv2.resize(

            rgb,

            (
                REC_INPUT_SIZE,
                REC_INPUT_SIZE
            )
        )

        # -----------------------------------------------------
        # HWC → CHW
        #
        # Result:
        #
        #   1 x 3 x 112 x 112
        # -----------------------------------------------------

        chw = (
            resized
            .astype(
                np.float32
            )
            .transpose(
                2,
                0,
                1
            )[None]
        )

        blob = (
            np.ascontiguousarray(
                chw
            )
        )

        # -----------------------------------------------------
        # Run ArcFace
        # -----------------------------------------------------

        raw = (
            self.session.run(

                [
                    self._output_name
                ],

                {
                    self._input_name:
                        blob
                },

            )[0][0]
        )

        raw = (
            np.asarray(
                raw,
                dtype=np.float32
            )
            .reshape(-1)
        )

        # -----------------------------------------------------
        # Validate output
        # -----------------------------------------------------

        if (
            raw.size
            != EMBEDDING_SIZE
        ):

            raise ValueError(

                f"ArcFace returned "
                f"embedding size "
                f"{raw.size}; "

                f"expected "
                f"{EMBEDDING_SIZE}"
            )

        # -----------------------------------------------------
        # L2 normalization
        # -----------------------------------------------------

        norm = float(
            np.linalg.norm(
                raw
            )
        )

        if norm <= 1e-12:

            raise ValueError(
                "ArcFace returned "
                "a zero embedding"
            )

        embedding = (
            raw / norm
        )

        return embedding

    # =========================================================
    # TEMPLATE HELPER
    # =========================================================

    # =========================================================
    # COMPATIBILITY
    # =========================================================

    @staticmethod
    def alignment_compatible(
        a: str,
        b: str
    ) -> bool:

        """
        Can an embedding made with alignment `a` be
        compared against one made with `b`?

        Aligned-vs-unaligned is the split that
        matters: those two are different pictures of
        the same face and their cosine similarity is
        meaningless.

        2-point and 5-point are treated as
        compatible. Both warp onto the same canonical
        template, so they differ in how well they
        correct yaw, not in where the face ends up.
        Scores across them are slightly noisier and
        still meaningful.
        """

        aligned = {
            ALIGNMENT_2POINT,
            ALIGNMENT_5POINT,
        }

        if a in aligned and b in aligned:
            return True

        return a == b

    def _skip(
        self,
        ident: str,
        reason: str
    ) -> None:

        """
        Record that one identity could not be
        compared, without stopping the comparison for
        anyone else.
        """

        if self._on_incompatible is not None:

            self._on_incompatible(
                ident,
                reason
            )

    @staticmethod
    def _as_sample_list(
        value
    ) -> list:
        """
        Support both:

            identity -> template

        and:

            identity -> [template1, template2, ...]
        """

        if value is None:

            return []

        if isinstance(
            value,
            (
                list,
                tuple
            )
        ):

            return list(
                value
            )

        # Old format
        return [
            value
        ]

    # =========================================================
    # FACE MATCHING
    # =========================================================

    def compare(
        self,
        probe,
        identities: Sequence[str],
        probe_alignment: str | None = None,
    ) -> dict[str, float]:

        """
        Compare the probe face against
        every supplied candidate identity.

        For a person with multiple templates:

            front
            left
            right

        all samples are compared and the
        highest score becomes that person's
        final score.
        """

        # -----------------------------------------------------
        # Generate probe embedding ONCE
        # -----------------------------------------------------

        probe_embedding = (
            self._embed(
                probe
            )
        )

        # How the probe was aligned. Callers that use
        # alignment.align_face get the method name back
        # from it and should pass it here. Absent, we
        # assume the matcher's own contract: aligned if
        # alignment is required, raw crop if not.

        if probe_alignment is None:

            probe_alignment = (
                ALIGNMENT_5POINT
                if self.require_alignment
                else ALIGNMENT_NONE
            )

        scores: dict[
            str,
            float
        ] = {}

        # -----------------------------------------------------
        # Each candidate identity
        # -----------------------------------------------------

        for ident in identities:

            samples = (
                self._as_sample_list(

                    self.templates.get(
                        ident
                    )
                )
            )

            # -------------------------------------------------
            # No enrollment data
            #
            # Score 0.0 rather than omitting the identity.
            #
            # A roster is built from badge transactions, so it
            # will always contain people who have not been
            # enrolled yet. Omitting them hands the resolver a
            # shorter list than the one its threshold was
            # derived from, and an entirely unenrolled
            # candidate set hands it an empty one.
            #
            # 0.0 is the honest score: no template means no
            # evidence, and no evidence never clears a
            # threshold.
            # -------------------------------------------------

            if not samples:

                scores[ident] = 0.0

                continue

            sample_scores: list[
                float
            ] = []

            # -------------------------------------------------
            # Compare against every enrollment view
            # -------------------------------------------------

            for sample in samples:

                # ---------------------------------------------
                # Verify same ArcFace model
                # ---------------------------------------------

                if (
                    sample.model_sha256
                    != self.model_sha256
                ):

                    self._skip(

                        ident,

                        f"enrolled with model "
                        f"{sample.model_sha256}, "
                        f"matcher runs "
                        f"{self.model_sha256}; "
                        f"re-enroll this identity"
                    )

                    continue

                # ---------------------------------------------
                # Verify same alignment method
                #
                # A template made from a raw crop cannot be
                # compared against an aligned probe. Skipping
                # is the safe direction: the person reads as
                # unverified until re-enrolled, rather than
                # scoring badly and being confused with
                # somebody else.
                # ---------------------------------------------

                if not self.alignment_compatible(
                    sample.alignment,
                    probe_alignment
                ):

                    self._skip(

                        ident,

                        f"enrolled with alignment "
                        f"{sample.alignment!r}, "
                        f"probe is "
                        f"{probe_alignment!r}; "
                        f"re-enroll this identity"
                    )

                    continue

                # ---------------------------------------------
                # Enrollment embedding
                # ---------------------------------------------

                enrolled_embedding = (

                    np.asarray(

                        sample.embedding,

                        dtype=np.float32
                    )

                    .reshape(-1)
                )

                # ---------------------------------------------
                # Dimension validation
                # ---------------------------------------------

                if (
                    enrolled_embedding.size
                    != EMBEDDING_SIZE
                ):

                    self._skip(

                        ident,

                        f"embedding size "
                        f"{enrolled_embedding.size}, "
                        f"expected {EMBEDDING_SIZE}; "
                        f"re-enroll this identity"
                    )

                    continue

                # ---------------------------------------------
                # Normalize enrolled embedding
                # ---------------------------------------------

                norm = float(

                    np.linalg.norm(
                        enrolled_embedding
                    )
                )

                if norm <= 1e-12:

                    self._skip(

                        ident,

                        "contains a zero embedding; "
                        "re-enroll this identity"
                    )

                    continue

                enrolled_embedding = (
                    enrolled_embedding
                    / norm
                )

                # ---------------------------------------------
                # Cosine similarity
                #
                # Both embeddings are normalized.
                # Therefore dot-product == cosine similarity.
                # ---------------------------------------------

                similarity = float(

                    np.dot(

                        probe_embedding,

                        enrolled_embedding
                    )
                )

                sample_scores.append(
                    similarity
                )

            # -------------------------------------------------
            # BEST enrollment view becomes person's score
            # -------------------------------------------------

            if sample_scores:

                scores[
                    ident
                ] = max(
                    sample_scores
                )

            else:

                # Every sample was skipped. Same reasoning as
                # an identity with no templates at all: return
                # 0.0 so the resolver sees a full candidate
                # list, rather than dropping the identity and
                # shortening the list its threshold was derived
                # from.

                scores[
                    ident
                ] = 0.0

        return scores