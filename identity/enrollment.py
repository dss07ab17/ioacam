"""Multi-template enrolled face store.

Each identity can own multiple ArcFace embeddings, for example front, left,
right, and additional controlled enrollment views.

The key is a PERSON ID, not a name
----------------------------------

`identity` is the store's key, and it is the site's person identifier --
``EMP-4471`` -- not ``Rajesh``. Three reasons, and none of them is style:

  * Names are not unique. Two people share one and the store silently merges
    two faces under one key.
  * Names change. Marriage, correction, transliteration. The key must not.
  * The ID is already what the badge reader emits. Roster, door and workflow
    then speak one language, with no translation step between them, and a
    translation step is exactly where the wrong person gets named.

It also means the event store holds no personal names on its face, which
materially reduces what is being retained.

A human-readable `display_name` may travel alongside, for showing in a UI. It
is never a key and never takes part in matching.

The loader is backward-compatible with the original JSON format where the
same identity could appear only once. New saves use one JSON entry per
identity with a ``samples`` list. It is also tolerant of keys enrolled before
the ID rule existed: those load and warn rather than being dropped or
rewritten, because silently discarding somebody's enrolment is worse than
carrying a bad key until they are re-enrolled.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------
# Person ID format
# ----------------------------------------------------------------------

# Two to five letters, a hyphen, three to six digits: EMP-4471, CONTR-100234.
# Configurable because site schemes differ; the shape is what matters, which
# is that it cannot be mistaken for something a person typed as their name.
DEFAULT_PERSON_ID_PATTERN = r"[A-Z]{2,5}-[0-9]{3,6}"


def compile_person_id_pattern(pattern: str | None = None) -> re.Pattern:
    """Anchor a person-ID pattern so it must match the whole string.

    Without the anchors ``EMP-4471 Rajesh`` would pass, which defeats the
    point of validating at all.
    """
    return re.compile(r"\A(?:" + (pattern or DEFAULT_PERSON_ID_PATTERN) + r")\Z")


def is_person_id(value: str, pattern: str | None = None) -> bool:
    """Does this look like a person ID rather than a name?"""
    return bool(compile_person_id_pattern(pattern).fullmatch(str(value)))


def warn_legacy_identity(identity: str, pattern: str | None = None) -> str:
    """The stderr text for a key that predates the person-ID rule."""
    return (
        f"[identity] WARNING enrolled key {identity!r} is not a person ID "
        f"(expected {pattern or DEFAULT_PERSON_ID_PATTERN}). It still loads "
        f"and still matches, but events carry {identity!r} as the subject "
        f"identity, which nothing downstream can resolve to a person. "
        f"Re-enrol under the badge ID: python3 identity/enroll.py"
    )


def display_name_for(
    templates: "TemplateStore",
    identity: str,
) -> str:
    """What to show a human for this identity.

    The first display name any of the person's samples carries, or the ID
    itself when none do. Callers that write events or logs must use the ID
    directly and never this.
    """
    for sample in templates.get(identity) or []:
        name = getattr(sample, "display_name", "")
        if name:
            return name
    return identity


@dataclass(frozen=True)
class EnrolledTemplate:
    identity: str                  # the person ID, and the key everywhere
    embedding: np.ndarray          # (512,) float32, L2-normalised
    model_sha256: str              # weights that produced this embedding
    enrolled_at: str = ""          # ISO timestamp, informational
    source_note: str = ""          # photo/session information
    # How the face was placed in the 112x112 crop before embedding. An
    # aligned face and a raw crop of the same person do not produce
    # comparable embeddings, so this has to travel with the template for
    # the same reason model_sha256 does: it is part of what the number
    # means. Templates written before alignment existed carry no such key,
    # and default to "none" on load, which is what they in fact were.
    alignment: str = "none"
    # For showing in a UI, and for nothing else. Never a key, never compared,
    # never written to an event. Empty when the enroller did not supply one,
    # and omitted from the JSON entirely in that case so the absence is
    # visible in the file rather than encoded as an empty string.
    display_name: str = ""

    def to_json(self) -> dict:
        d = {
            "embedding": self.embedding.astype(float).tolist(),
            "model_sha256": self.model_sha256,
            "enrolled_at": self.enrolled_at,
            "source_note": self.source_note,
            "alignment": self.alignment,
        }
        if self.display_name:
            d["display_name"] = self.display_name
        return d

    @staticmethod
    def from_json(identity: str, d: dict) -> "EnrolledTemplate":
        emb = np.asarray(d["embedding"], dtype=np.float32)

        if emb.ndim != 1:
            raise ValueError(
                f"template for {identity!r} is not a 1-D embedding"
            )

        norm = float(np.linalg.norm(emb))

        if norm <= 1e-12:
            raise ValueError(
                f"template for {identity!r} has a zero embedding"
            )

        # Normalize again while loading to avoid small serialization drift.
        emb = emb / norm

        return EnrolledTemplate(
            identity=identity,
            embedding=emb.astype(np.float32),
            model_sha256=d["model_sha256"],
            enrolled_at=d.get("enrolled_at", ""),
            source_note=d.get("source_note", ""),
            # Absent means this template predates alignment.
            alignment=d.get("alignment", "none"),
            # Absent means nobody supplied one. Not an error: the ID is
            # always a sufficient label.
            display_name=d.get("display_name", ""),
        )

TemplateStore = dict[str, list[EnrolledTemplate]]


def load_templates(
    path: str,
    id_pattern: str | None = None,
    warn=None,
) -> TemplateStore:
    """
    Load:

        {
            identity: [
                EnrolledTemplate,
                EnrolledTemplate,
                ...
            ]
        }

    Supports both the new multi-template format and your old
    single-template format.

    Keys that are not person IDs -- anything enrolled before the ID rule, of
    which ``"user"`` in the shipped templates.json is one -- load normally
    and warn on stderr, naming the key and how to re-enrol. They are
    deliberately not rewritten and not dropped: guessing somebody's badge
    number would be inventing data, and discarding their enrolment would
    quietly stop identifying a real person.

    `warn` takes one line of text; the default writes it to stderr.
    """

    if warn is None:
        def warn(msg: str) -> None:
            print(msg, file=sys.stderr)

    p = Path(path)

    if not p.exists():
        return {}

    raw = json.loads(p.read_text())

    store: TemplateStore = {}

    # The old format repeats the same identity once per sample, so warn per
    # key rather than per entry.
    warned: set[str] = set()

    for entry in raw.get("templates", []):

        identity = str(entry["identity"])

        if identity not in warned and not is_person_id(identity, id_pattern):
            warned.add(identity)
            warn(warn_legacy_identity(identity, id_pattern))

        # -------------------------------------------------------------
        # New multi-template format
        # -------------------------------------------------------------
        if "samples" in entry:

            samples = entry.get("samples", [])

            store[identity] = [
                EnrolledTemplate.from_json(
                    identity,
                    sample
                )
                for sample in samples
            ]

        # -------------------------------------------------------------
        # Old single-template format
        # -------------------------------------------------------------
        else:

            store.setdefault(identity, []).append(
                EnrolledTemplate.from_json(
                    identity,
                    entry
                )
            )

    return store


def save_templates(
    path: str,
    templates: TemplateStore
) -> None:
    """
    Save every identity and all its enrollment samples.
    """

    p = Path(path)

    p.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    payload = {
        "format_version": 3,

        "templates": [

            {
                "identity": identity,

                "samples": [
                    sample.to_json()
                    for sample in samples
                ],
            }

            for identity, samples
            in sorted(templates.items())
        ],
    }

    # Write temporary file first.
    tmp = p.with_suffix(
        p.suffix + ".tmp"
    )

    tmp.write_text(
        json.dumps(
            payload,
            indent=2
        )
    )

    tmp.replace(p)