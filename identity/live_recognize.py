from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2


# ============================================================
# PROJECT PATH SETUP
# ============================================================

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# Only the project root goes on the path. Adding identity/ as well would let
# the same module be imported under two names (matcher and identity.matcher),
# which gives two copies of every class.
sys.path.insert(0, str(ROOT))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from identity.alignment import (  # noqa: E402
    AlignmentError,
    align_face,
    landmarks_from_box,
)
from identity.arcface_matcher import ArcFaceMatcher  # noqa: E402
from identity.landmarks import build_landmark_source  # noqa: E402
from identity.directory import (  # noqa: E402
    DEFAULT_DIRECTORY_PATH,
    EmployeeDirectory,
)
from identity.enrollment import display_name_for, load_templates  # noqa: E402
from identity.matcher import ThresholdPolicy  # noqa: E402
from perception.face import build_face_detector  # noqa: E402


# ============================================================
# DEFAULT PATHS
# ============================================================

DEFAULT_CONFIG = "identity/config/identity.example.json"

DEFAULT_DETECTOR_MODEL = (
    "perception/models/version-RFB-320-int8.onnx"
)

DEFAULT_RECOGNIZER_MODEL = (
    "identity/models/arcfaceresnet100-11-int8.onnx"
)

DEFAULT_TEMPLATES = (
    "identity/config/templates.json"
)

DEFAULT_DIRECTORY = (
    DEFAULT_DIRECTORY_PATH
)


# ============================================================
# CONFIG
# ============================================================

def load_config(path: str) -> dict:
    """
    Load identity configuration.
    """

    config_path = Path(path)

    if not config_path.is_file():

        raise FileNotFoundError(
            f"Identity config not found:\n"
            f"{path}"
        )

    return json.loads(
        config_path.read_text()
    )


# ============================================================
# IDENTITY DECISION
# ============================================================

def decide_identity(
    scores: dict[str, float],
    candidate_count: int,
    policy: ThresholdPolicy,
):
    """
    Apply the same security decision idea used by IdentityResolver.

    Requirements:

        1. Best score must pass candidate-count threshold.
        2. Best score must beat runner-up by min_margin.

    Returns:

        identity
        best_score
        runner_up
        threshold
        reason
    """

    if not scores:

        return (
            None,
            0.0,
            0.0,
            None,
            "no enrolled candidates",
        )

    # --------------------------------------------------------
    # Highest similarity first
    # --------------------------------------------------------

    ranked = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    best_identity = ranked[0][0]
    best_score = float(
        ranked[0][1]
    )

    # --------------------------------------------------------
    # Runner-up
    # --------------------------------------------------------

    if len(ranked) > 1:

        runner_up = float(
            ranked[1][1]
        )

    else:

        runner_up = 0.0

    # --------------------------------------------------------
    # Threshold depends on number of candidates
    # --------------------------------------------------------

    threshold = policy.threshold_for(
        candidate_count
    )

    if threshold is None:

        return (
            None,
            best_score,
            runner_up,
            None,
            "candidate set exceeds calibrated policy",
        )

    # --------------------------------------------------------
    # Threshold check
    # --------------------------------------------------------

    if best_score < threshold:

        return (
            None,
            best_score,
            runner_up,
            threshold,
            (
                f"score below threshold "
                f"{threshold:.3f}"
            ),
        )

    # --------------------------------------------------------
    # Runner-up margin check
    #
    # For only one enrolled identity there is no runner-up.
    # --------------------------------------------------------

    if len(ranked) > 1:

        margin = (
            best_score
            - runner_up
        )

        if margin < policy.min_margin:

            return (
                None,
                best_score,
                runner_up,
                threshold,
                (
                    f"margin below "
                    f"{policy.min_margin:.3f}"
                ),
            )

    # --------------------------------------------------------
    # Recognition accepted
    # --------------------------------------------------------

    return (
        best_identity,
        best_score,
        runner_up,
        threshold,
        "matched",
    )


# ============================================================
# MAIN
# ============================================================

def shown_name(
    directory: EmployeeDirectory,
    templates,
    identity: str,
) -> str:
    """What to put on screen for this ID.

    Three sources, in order, and the order is the point.

      1. the employee directory, which HR maintains and which is current;
      2. the `display_name` on the enrolment, which was typed once by
         whoever enrolled them and may be years old;
      3. the ID itself, which is always a sufficient label.

    The directory wins because it is the one that gets corrected. A name
    typed at enrolment is a snapshot; a name in the directory is maintained.
    Neither ever reaches an event -- see identity/directory.py.
    """
    from_directory = directory.display_name(identity)

    if from_directory != identity:
        return from_directory

    return display_name_for(
        templates,
        identity
    )


def screen_label(
    directory: EmployeeDirectory,
    templates,
    identity: str,
) -> str:
    """`shown_name`, marked when the directory says the person has left.

    A leaver matching is not a normal recognition and must not be drawn as
    one. Marking the label is the minimum; the warning line on stderr is
    what an operator actually reviews.
    """
    label = shown_name(
        directory,
        templates,
        identity
    )

    if directory.is_inactive(identity):

        return f"{label} [INACTIVE]"

    return label


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Standalone live face recognition "
            "using RFB-320 + ArcFace."
        )
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="OpenCV camera index. Default: 0",
    )

    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=(
            "Identity configuration file."
        ),
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=0.50,
        help=(
            "Seconds between ArcFace recognition "
            "operations. Default: 0.50"
        ),
    )

    parser.add_argument(
        "--directory",
        default=None,
        help=(
            "Employee directory CSV, for putting "
            "names on screen instead of badge IDs. "
            "Default: the identity config's "
            "directory.path, or "
            f"{DEFAULT_DIRECTORY}. An absent file "
            "is fine -- boxes are then labelled "
            "with the ID."
        ),
    )

    args = parser.parse_args()

    # ========================================================
    # STARTUP
    # ========================================================

    print()
    print("=" * 65)
    print("LIVE FACE RECOGNITION")
    print("=" * 65)

    # ========================================================
    # LOAD CONFIG
    # ========================================================

    print()
    print(
        f"Loading configuration: "
        f"{args.config}"
    )

    config = load_config(
        args.config
    )

    face_config = config.get(
        "face",
        {}
    )

    threshold_config = config.get(
        "threshold_policy"
    )

    if not threshold_config:

        raise RuntimeError(
            "threshold_policy is missing "
            "from identity configuration."
        )

    # ========================================================
    # THRESHOLD POLICY
    # ========================================================

    policy = ThresholdPolicy.from_config(
        threshold_config
    )

    # ========================================================
    # MODEL PATHS
    # ========================================================

    detector_model = face_config.get(
        "detector_model_path",
        DEFAULT_DETECTOR_MODEL,
    )

    recognizer_model = face_config.get(
        "recognizer_model_path",
        DEFAULT_RECOGNIZER_MODEL,
    )

    templates_path = face_config.get(
        "templates_path",
        DEFAULT_TEMPLATES,
    )

    expected_sha256 = face_config.get(
        "recognizer_sha256"
    )

    # ========================================================
    # LOAD TEMPLATES
    # ========================================================

    print(
        f"Loading templates: "
        f"{templates_path}"
    )

    templates = load_templates(
        templates_path,
        id_pattern=face_config.get(
            "person_id_pattern"
        ),
    )

    if not templates:

        raise RuntimeError(
            "\nNo enrolled face templates found.\n\n"
            "Enroll a person first using:\n\n"
            "    python3 identity/enroll.py"
        )

    candidates = sorted(
        templates.keys()
    )

    # ========================================================
    # LOAD EMPLOYEE DIRECTORY
    # ========================================================

    # ID -> name, for this window and nothing else. Loaded once, here, and
    # then cached: a lookup is a dict hit and the file is re-stat'ed at most
    # every few seconds, never per frame. An absent file is the normal case
    # on a laptop and leaves every box labelled with its badge ID.

    directory_config = config.get(
        "directory",
        {}
    )

    directory_path = (
        args.directory
        or directory_config.get(
            "path",
            DEFAULT_DIRECTORY,
        )
    )

    print(
        f"Loading directory: "
        f"{directory_path}"
    )

    directory = EmployeeDirectory(
        directory_path,
        auto_reload=bool(
            directory_config.get(
                "auto_reload",
                True,
            )
        ),
        stat_interval_s=float(
            directory_config.get(
                "stat_interval_s",
                5.0,
            )
        ),
        # Nothing here feeds the roster, so the role column is unused by
        # this program. It is a viewer.
        supply_roles=False,
        log=lambda msg: print(
            msg,
            file=sys.stderr,
        ),
    )

    print()
    print(
        f"Enrolled identities: "
        f"{len(candidates)}"
    )

    for identity in candidates:

        samples = templates[
            identity
        ]

        # New multi-template format
        if isinstance(
            samples,
            (
                list,
                tuple,
            ),
        ):

            count = len(
                samples
            )

        # Backward-compatible old format
        else:

            count = 1

        shown = shown_name(
            directory,
            templates,
            identity
        )

        label = (
            identity
            if shown == identity
            else f"{identity} ({shown})"
        )

        # An enrolled template for somebody who has left is a revocation
        # that never ran. Saying so at startup is cheaper than discovering
        # it from a match six weeks later.
        if directory.is_inactive(
            identity
        ):

            label = (
                f"{label} "
                f"-- INACTIVE in the directory, "
                f"this enrolment should be removed"
            )

        print(
            f"  {label}: "
            f"{count} sample(s)"
        )

    # ========================================================
    # LOAD FACE DETECTOR
    # ========================================================

    print()
    print(
        "Loading RFB-320 face detector..."
    )

    detector = build_face_detector(
        {
            "backend":
                "rfb-320-onnx",

            "model_path":
                detector_model,
        }
    )

    print(
        "Face detector loaded."
    )

    # --------------------------------------------------------
    # Landmark source
    #
    # RFB-320 emits boxes only, so alignment has nothing to
    # work from without this. The config block is
    # face.landmarks; backend "none" means the detector is
    # expected to supply landmarks itself, and faces without
    # them are refused rather than matched unaligned.
    # --------------------------------------------------------

    landmark_source = build_landmark_source(
        config.get(
            "face",
            {}
        ).get(
            "landmarks",
            {}
        )
    )

    if landmark_source is not None:

        print(
            f"Landmark source: "
            f"{landmark_source.name}"
        )

    # ========================================================
    # LOAD ARCFACE
    #
    # Passing expected_sha256 preserves your model-integrity
    # verification.
    # ========================================================

    print(
        "Loading ArcFace recognizer..."
    )

    matcher = ArcFaceMatcher(

        model_path=
            recognizer_model,

        templates=
            templates,

        expected_sha256=
            expected_sha256,
    )

    print(
        "ArcFace loaded."
    )

    print(
        f"ArcFace model SHA-256:"
    )

    print(
        f"  {matcher.model_sha256}"
    )

    # ========================================================
    # SHOW CURRENT DECISION POLICY
    # ========================================================

    threshold = policy.threshold_for(
        len(candidates)
    )

    print()
    print("=" * 65)
    print("RECOGNITION POLICY")
    print("=" * 65)

    print(
        f"Candidates       : "
        f"{len(candidates)}"
    )

    if threshold is None:

        print(
            "Threshold        : "
            "NO VALID THRESHOLD"
        )

    else:

        print(
            f"Threshold        : "
            f"{threshold:.3f}"
        )

    print(
        f"Minimum margin   : "
        f"{policy.min_margin:.3f}"
    )

    print(
        f"Recognition every: "
        f"{args.interval:.2f} seconds"
    )

    print("=" * 65)

    # ========================================================
    # OPEN CAMERA
    # ========================================================

    print()
    print(
        f"Opening camera "
        f"{args.camera}..."
    )

    camera = cv2.VideoCapture(
        args.camera
    )

    if not camera.isOpened():

        raise RuntimeError(
            f"Could not open camera "
            f"{args.camera}."
        )

    print(
        "Camera opened successfully."
    )

    print()
    print(
        "Look at the camera."
    )

    print(
        "Turn your head left/right "
        "to test the multi-view enrollment."
    )

    print()
    print(
        "Press Q or ESC to exit."
    )

    print()

    # ========================================================
    # RECOGNITION STATE
    # ========================================================

    last_recognition_time = 0.0

    # Stores recognition results for currently
    # detected faces.
    last_results = []

    # IDs already reported as matching a departed employee, so the warning
    # is said once per person rather than once per recognition pass.
    reported_inactive: set[str] = set()

    try:

        # ====================================================
        # LIVE LOOP
        # ====================================================

        while True:

            ok, frame = camera.read()

            if not ok:

                print(
                    "Failed to read camera frame."
                )

                break

            if frame is None:

                continue

            # ================================================
            # FACE DETECTION
            #
            # Detection can run continuously.
            # ================================================

            faces = list(
                detector.detect(
                    frame
                )
            )

            current_time = (
                time.monotonic()
            )

            # ================================================
            # ARCFACE RECOGNITION
            #
            # Run only at configured interval.
            # ================================================

            if (
                current_time
                - last_recognition_time
                >= max(
                    0.05,
                    args.interval,
                )
            ):

                new_results = []

                for face_index, face in enumerate(
                    faces
                ):

                    # ----------------------------------------
                    # Align the face
                    #
                    # A face with no locatable eyes cannot be
                    # aligned, and an unaligned crop is not
                    # comparable with an aligned template. So
                    # this reports unverified rather than
                    # matching on a crop the model was not
                    # trained for.
                    #
                    # Expect this to fire often at range with
                    # the cascade landmark source. That is the
                    # cascade's limit showing honestly, and the
                    # fix is a detector that emits landmarks,
                    # not a lower bar here.
                    # ----------------------------------------

                    marks = landmarks_from_box(
                        face
                    )

                    if (
                        marks is None
                        and landmark_source is not None
                    ):

                        marks = landmark_source.find(
                            frame,
                            face
                        )

                    if marks is None:

                        new_results.append(
                            (
                                None,
                                0.0,
                                0.0,
                                None,
                                "eyes not located; "
                                "cannot align",
                            )
                        )

                        continue

                    try:

                        (
                            face_crop,
                            probe_alignment,
                        ) = align_face(
                            frame,
                            marks
                        )

                    except AlignmentError as exc:

                        new_results.append(
                            (
                                None,
                                0.0,
                                0.0,
                                None,
                                f"alignment failed: {exc}",
                            )
                        )

                        continue

                    try:

                        # ====================================
                        # MULTI-TEMPLATE COMPARISON
                        #
                        # arcface_matcher.py compares the live
                        # embedding against every enrolled
                        # view for every identity.
                        # ====================================

                        scores = matcher.compare(
                            face_crop,
                            candidates,
                            probe_alignment=
                                probe_alignment,
                        )

                        (
                            identity,
                            best_score,
                            runner_up,
                            threshold_used,
                            reason,
                        ) = decide_identity(

                            scores=
                                scores,

                            candidate_count=
                                len(candidates),

                            policy=
                                policy,
                        )

                        new_results.append(
                            (
                                identity,
                                best_score,
                                runner_up,
                                threshold_used,
                                reason,
                            )
                        )

                        # ====================================
                        # TERMINAL OUTPUT
                        # ====================================

                        # The ID, not the display name. This line is the
                        # log of what the matcher concluded, and the log
                        # carries the key. The name goes on the window.
                        if identity:

                            print(
                                "[RECOGNIZED] "
                                f"{identity} "
                                f"| score={best_score:.3f} "
                                f"| runner-up={runner_up:.3f}"
                            )

                        else:

                            print(
                                "[UNVERIFIED] "
                                f"best={best_score:.3f} "
                                f"| runner-up={runner_up:.3f} "
                                f"| {reason}"
                            )

                    except Exception as exc:

                        # ------------------------------------
                        # Do not silently accept recognition
                        # when an integrity/template check
                        # fails.
                        # ------------------------------------

                        print(
                            "[RECOGNITION ERROR] "
                            f"{exc}"
                        )

                        new_results.append(
                            (
                                None,
                                0.0,
                                0.0,
                                None,
                                f"ERROR: {exc}",
                            )
                        )

                last_results = (
                    new_results
                )

                last_recognition_time = (
                    current_time
                )

            # ================================================
            # DRAW FACE RESULTS
            # ================================================

            for face_index, face in enumerate(
                faces
            ):

                x1 = int(
                    face.x1
                )

                y1 = int(
                    face.y1
                )

                x2 = int(
                    face.x2
                )

                y2 = int(
                    face.y2
                )

                # --------------------------------------------
                # Use latest recognition result
                # --------------------------------------------

                if face_index < len(
                    last_results
                ):

                    (
                        identity,
                        best_score,
                        runner_up,
                        threshold_used,
                        reason,
                    ) = last_results[
                        face_index
                    ]

                else:

                    identity = None
                    best_score = 0.0
                    runner_up = 0.0
                    threshold_used = None
                    reason = "scanning"

                # --------------------------------------------
                # Recognized
                # --------------------------------------------

                if identity:

                    box_color = (
                        0,
                        255,
                        0,
                    )

                    # On screen a person reads a name; the store is keyed
                    # on the ID. screen_label falls back through the
                    # employee directory, then the enrolment's own
                    # display_name, then the ID, so an unnamed enrolment on
                    # a machine with no directory still labels its box.
                    label = (
                        f"{screen_label(directory, templates, identity)} "
                        f"score={best_score:.3f}"
                    )

                    # A leaver who still matches is not a normal
                    # recognition. Amber, not green, and a line on stderr.
                    flag = directory.inactive_match(
                        identity,
                        best_score,
                    )

                    if flag is not None:

                        box_color = (
                            0,
                            165,
                            255,
                        )

                        # Once per ID. The recognition runs every
                        # --interval seconds, so a leaver standing in
                        # frame for a minute would otherwise produce two
                        # hundred identical lines.
                        if (
                            flag.employee_id
                            not in reported_inactive
                        ):

                            reported_inactive.add(
                                flag.employee_id
                            )

                            print(
                                f"[identity] INACTIVE MATCH "
                                f"{flag.detail}",
                                file=sys.stderr,
                            )

                # --------------------------------------------
                # Unverified
                # --------------------------------------------

                else:

                    box_color = (
                        0,
                        0,
                        255,
                    )

                    label = (
                        f"UNVERIFIED "
                        f"score={best_score:.3f}"
                    )

                # --------------------------------------------
                # Bounding box
                # --------------------------------------------

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    box_color,
                    2,
                )

                # --------------------------------------------
                # Name / score
                # --------------------------------------------

                text_y = max(
                    25,
                    y1 - 10,
                )

                cv2.putText(

                    frame,

                    label,

                    (
                        x1,
                        text_y,
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.65,

                    box_color,

                    2,

                    cv2.LINE_AA,
                )

                # --------------------------------------------
                # Face detector confidence
                # --------------------------------------------

                detector_text = (
                    f"face={face.score:.2f}"
                )

                cv2.putText(

                    frame,

                    detector_text,

                    (
                        x1,
                        min(
                            frame.shape[0] - 10,
                            text_y + 23,
                        ),
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.45,

                    (
                        255,
                        255,
                        255,
                    ),

                    1,

                    cv2.LINE_AA,
                )

            # ================================================
            # TOP STATUS
            # ================================================

            status = (
                f"Enrolled identities: "
                f"{len(candidates)}"
            )

            cv2.putText(

                frame,

                status,

                (
                    10,
                    25,
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.55,

                (
                    255,
                    255,
                    255,
                ),

                1,

                cv2.LINE_AA,
            )

            cv2.putText(

                frame,

                "Q / ESC = Exit",

                (
                    10,
                    50,
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.50,

                (
                    255,
                    255,
                    255,
                ),

                1,

                cv2.LINE_AA,
            )

            # ================================================
            # DISPLAY
            # ================================================

            cv2.imshow(
                "Live Face Recognition",
                frame,
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key in (
                ord("q"),
                ord("Q"),
                27,
            ):

                break

    finally:

        camera.release()

        cv2.destroyAllWindows()

        print()
        print(
            "Camera released."
        )

        print(
            "Live recognition stopped."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()