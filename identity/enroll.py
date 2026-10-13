"""Interactive multi-image face enrollment.

Run:

    python3 identity/enroll.py

Workflow:
    1. Ask for the person ID (the badge number, not a name).
    2. Optionally ask for a display name, stored beside the ID.
    3. Ask for multiple image paths.
    4. User pastes one image path per line.
    5. Empty line finishes image input.
    6. Validate every image.
    7. Require exactly one face per image.
    8. Generate ArcFace embedding for every valid image.
    9. Save all embeddings under the same person ID.

Important:
    All images are processed successfully BEFORE templates.json is modified.

Therefore:
    If one enrollment image fails, no new samples are saved.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path


# ============================================================
# PROJECT PATH SETUP
# ============================================================

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# Only the project root goes on the path. Adding identity/ as well would let
# the same module be imported under two names (arcface_matcher and
# identity.arcface_matcher), which gives two copies of every class.

sys.path.insert(
    0,
    str(ROOT)
)


# ============================================================
# IMPORTS
# ============================================================

import cv2  # noqa: E402

from identity.arcface_matcher import (  # noqa: E402
    ArcFaceMatcher,
    sha256_of,
)

from identity.alignment import (  # noqa: E402
    AlignmentError,
    align_face,
    landmarks_from_box,
)

from identity.landmarks import (  # noqa: E402
    build_landmark_source,
)

from identity.directory import (  # noqa: E402
    DEFAULT_DIRECTORY_PATH,
    EmployeeDirectory,
)

from identity.enrollment import (  # noqa: E402
    DEFAULT_PERSON_ID_PATTERN,
    EnrolledTemplate,
    display_name_for,
    is_person_id,
    load_templates,
    save_templates,
)

from perception.face import (  # noqa: E402
    build_face_detector,
)


# ============================================================
# DEFAULT PATHS
# ============================================================

DEFAULT_DET_MODEL = (
    "perception/models/"
    "version-RFB-320-int8.onnx"
)

DEFAULT_REC_MODEL = (
    "identity/models/"
    "arcfaceresnet100-11-int8.onnx"
)

DEFAULT_TEMPLATES_PATH = (
    "identity/config/templates.json"
)


# ============================================================
# INPUT HELPERS
# ============================================================

def ask_person_id(
    id_pattern: str | None = None
) -> str:
    """
    Ask for the person ID that this enrolment will be keyed on.

    Not a name. The prompt says why, because a refusal without a reason is
    the kind of thing somebody quietly reverts six months later.
    """

    pattern = (
        id_pattern
        or DEFAULT_PERSON_ID_PATTERN
    )

    print()
    print(
        "Enrolment is keyed on the person's ID, "
        "not their name."
    )

    print(
        "  - names are not unique, and two people "
        "sharing one would share a face template"
    )

    print(
        "  - names change; the key must not"
    )

    print(
        "  - the ID is what the badge reader emits, "
        "so the roster, the door and the workflow"
    )

    print(
        "    engine all say the same thing about "
        "the same person, with nothing in between"
    )

    print(
        "    them to get it wrong"
    )

    print()
    print(
        f"Expected format: {pattern}   "
        f"(for example EMP-4471)"
    )

    print(
        "A display name can be added on the next "
        "question, for showing on screen."
    )

    while True:

        person_id = input(
            "\nEnter person ID: "
        ).strip()

        if not person_id:

            print(
                "Person ID cannot be empty."
            )

            continue

        if is_person_id(
            person_id,
            id_pattern
        ):

            return person_id

        print()
        print(
            f"{person_id!r} is not a person ID "
            f"(expected {pattern})."
        )

        print(
            "If that is the person's name, enter "
            "their badge ID here instead and give"
        )

        print(
            "the name at the display-name prompt."
        )


def check_directory(
    identity: str,
    directory: EmployeeDirectory,
) -> bool:
    """
    Show what the employee directory knows about this ID, and refuse to
    enrol somebody who has left.

    Returns True when enrolment may go ahead.

    Three outcomes, and the middle one is the reason this exists:

      active      print the name as confirmation. Enrolling the wrong badge
                  number is an easy mistake and this is where it is caught,
                  before a face is attached to it.

      inactive    refuse. A template for somebody who has left keeps
                  matching, and every match then reads as a current
                  employee standing in the building.

      not listed  allow, and say so. Contractors, new starters and sites
                  with no directory at all are all this case. The directory
                  is a convenience for the screen, not a gate on enrolment,
                  and making it a prerequisite would be a worse system.
    """

    if not directory.available:

        return True

    refusal = directory.enrolment_refusal(
        identity
    )

    if refusal:

        print()
        print("-" * 60)
        print("REFUSED")
        print("-" * 60)
        print(refusal)
        print("-" * 60)

        return False

    entry = directory.get(
        identity
    )

    if entry is None:

        print()
        print(
            f"{identity} is not in "
            f"{directory.path}."
        )

        print(
            "Enrolling anyway -- the directory is "
            "for showing names on screen, not a"
        )

        print(
            "gate on enrolment. Add the row later "
            "if they should have one."
        )

        return True

    print()
    print(
        f"Directory     : {entry.display_name}"
        + (
            f", {entry.role}"
            if entry.role
            else ""
        )
        + (
            f", {entry.department}"
            if entry.department
            else ""
        )
    )

    return True


def ask_display_name() -> str:
    """
    Ask for an optional human-readable name.

    Stored as a separate field on the template. It is shown on screen and
    never used for matching, never used as a key, and never written to an
    event.
    """

    print()
    print(
        "Display name (optional). Shown on screen "
        "only; never used for matching, and never"
    )

    print(
        "written to an event -- events carry the ID. "
        "Press ENTER to skip."
    )

    return input(
        "\nEnter display name: "
    ).strip()


def ask_images() -> list[str]:
    """
    Ask user to paste multiple enrollment image paths.

    No comma or separator is required.

    Example:

        /home/rahul/front.jpg
        /home/rahul/left.jpg
        /home/rahul/right.jpg

    User presses ENTER on an empty line after the final path.
    """

    print()
    print(
        "Enter enrollment image paths."
    )

    print(
        "Paste all image paths, "
        "one path per line."
    )

    print(
        "No commas or separators are required."
    )

    print()
    print(
        "After the final image, "
        "press ENTER on an empty line."
    )

    print()

    photo_paths: list[str] = []

    while True:

        try:

            line = input()

        except EOFError:

            break

        line = line.strip()

        # ----------------------------------------------------
        # Empty line = finish image entry
        # ----------------------------------------------------

        if not line:

            if photo_paths:
                break

            print(
                "Please enter at least "
                "one image path."
            )

            continue

        # ----------------------------------------------------
        # Remove optional surrounding quotes
        # ----------------------------------------------------

        line = (
            line
            .strip('"')
            .strip("'")
        )

        photo_paths.append(
            line
        )

    return photo_paths


def validate_image_paths(
    photo_paths: list[str]
) -> bool:
    """
    Validate that every entered path exists and is a file.
    """

    all_valid = True

    for photo_path in photo_paths:

        path = Path(
            photo_path
        )

        if not path.exists():

            print()
            print(
                "ERROR: image does not exist:"
            )

            print(
                f"  {photo_path}"
            )

            all_valid = False

            continue

        if not path.is_file():

            print()
            print(
                "ERROR: path is not a file:"
            )

            print(
                f"  {photo_path}"
            )

            all_valid = False

    return all_valid


# ============================================================
# ENROLLMENT FUNCTION
# ============================================================

def enroll(
    identity: str,
    photo_paths: list[str],
    replace: bool = False,
    display_name: str = "",
) -> None:
    """
    Enroll one person ID using one or more images.

    `identity` is the person ID and is the key in templates.json.
    `display_name` is optional, travels on every sample written here, and is
    for showing on screen only.

    Every image must:
        - be readable,
        - contain exactly one detectable face,
        - produce a valid ArcFace embedding.

    templates.json is modified only after every image succeeds.
    """

    if not photo_paths:

        raise RuntimeError(
            "No enrollment images supplied."
        )

    print()
    print("=" * 60)
    print("FACE ENROLLMENT")
    print("=" * 60)

    print(
        f"Person ID    : {identity}"
    )

    if display_name:

        print(
            f"Display name : {display_name}"
        )

    print(
        f"Images       : {len(photo_paths)}"
    )

    print("=" * 60)
    print()

    # ========================================================
    # LOAD FACE DETECTOR
    # ========================================================

    print(
        "Loading face detector..."
    )

    detector = build_face_detector(
        {
            "model_path":
                DEFAULT_DET_MODEL
        }
    )

    # --------------------------------------------------------
    # Landmark source
    #
    # Only used when the detector does not supply landmarks
    # itself. RFB-320 does not. Swap in SCRFD or RetinaFace and
    # this falls out of the path automatically.
    # --------------------------------------------------------

    landmark_source = build_landmark_source()

    # ========================================================
    # LOAD ARCFACE
    # ========================================================

    print(
        "Loading ArcFace recognizer..."
    )

    matcher = ArcFaceMatcher(

        model_path=
            DEFAULT_REC_MODEL,

        templates={}
    )

    # ========================================================
    # MODEL HASH
    # ========================================================

    model_sha256 = sha256_of(
        DEFAULT_REC_MODEL
    )

    print(
        "Models loaded successfully."
    )

    print(
        f"Recognition model SHA-256: "
        f"{model_sha256}"
    )

    print()

    # ========================================================
    # CREATE ALL NEW TEMPLATES IN MEMORY FIRST
    #
    # IMPORTANT:
    #
    # Nothing is written to templates.json during this loop.
    #
    # If ANY image fails:
    #
    #     enrollment stops
    #
    # and:
    #
    #     templates.json remains unchanged.
    # ========================================================

    new_samples: list[
        EnrolledTemplate
    ] = []

    # ========================================================
    # PROCESS IMAGES
    # ========================================================

    for index, photo_path in enumerate(
        photo_paths,
        start=1
    ):

        print(
            f"[{index}/{len(photo_paths)}] "
            f"Processing image"
        )

        print(
            f"    {photo_path}"
        )

        # ----------------------------------------------------
        # Read image
        # ----------------------------------------------------

        frame = cv2.imread(
            photo_path
        )

        if frame is None:

            raise RuntimeError(
                "\nCould not read image:\n"
                f"{photo_path}\n\n"
                "Enrollment cancelled."
            )

        if frame.size == 0:

            raise RuntimeError(
                "\nImage is empty:\n"
                f"{photo_path}\n\n"
                "Enrollment cancelled."
            )

        print(
            "    Image loaded."
        )

        # ----------------------------------------------------
        # Detect faces
        # ----------------------------------------------------

        faces = detector.detect(
            frame
        )

        face_count = len(
            faces
        )

        if face_count == 0:

            raise RuntimeError(
                "\nNo face detected in:\n"
                f"{photo_path}\n\n"
                "Each enrollment image must "
                "contain exactly one clearly "
                "detectable face.\n\n"
                "Enrollment cancelled."
            )

        if face_count > 1:

            raise RuntimeError(
                f"\n{face_count} faces detected in:\n"
                f"{photo_path}\n\n"
                "Each enrollment image must "
                "contain exactly one person's face.\n\n"
                "Enrollment cancelled."
            )

        print(
            "    Exactly one face detected."
        )

        # ----------------------------------------------------
        # Align face
        #
        # The detector box is not what ArcFace was trained on.
        # The face has to be rotated and scaled onto the
        # canonical template first, and that needs landmarks.
        #
        # Landmarks come from the detector when it produces
        # them, and otherwise from the eye cascade. Enrollment
        # is the right place for the cascade: the photo is
        # posed and controlled, and a failure here is
        # recoverable -- the operator retakes the picture.
        # ----------------------------------------------------

        face_box = faces[0]

        marks = landmarks_from_box(
            face_box
        )

        if marks is None and landmark_source is not None:

            marks = landmark_source.find(
                frame,
                face_box
            )

        if marks is None:

            raise RuntimeError(
                "\nCould not locate both eyes in:\n"
                f"{photo_path}\n\n"
                "Without eye positions the face cannot be "
                "aligned, and an unaligned enrollment would "
                "quietly weaken every future match against "
                "this person.\n\n"
                "Use a front-facing, evenly lit photo where "
                "both eyes are visible and unobscured.\n\n"
                "Enrollment cancelled."
            )

        try:

            aligned, alignment_method = align_face(
                frame,
                marks
            )

        except AlignmentError as exc:

            raise RuntimeError(
                "\nFace alignment failed for:\n"
                f"{photo_path}\n\n"
                f"{exc}\n\n"
                "Enrollment cancelled."
            )

        print(
            f"    Face aligned "
            f"({alignment_method})."
        )

        # ----------------------------------------------------
        # Generate ArcFace embedding
        # ----------------------------------------------------

        embedding = matcher._embed(
            aligned
        )

        # ----------------------------------------------------
        # Additional embedding validation
        # ----------------------------------------------------

        if embedding is None:

            raise RuntimeError(
                "\nArcFace returned no embedding for:\n"
                f"{photo_path}\n\n"
                "Enrollment cancelled."
            )

        if embedding.ndim != 1:

            raise RuntimeError(
                "\nArcFace embedding has invalid shape for:\n"
                f"{photo_path}\n\n"
                "Enrollment cancelled."
            )

        if embedding.size != 512:

            raise RuntimeError(
                "\nArcFace embedding has invalid size "
                f"{embedding.size} for:\n"
                f"{photo_path}\n\n"
                "Expected 512 dimensions.\n\n"
                "Enrollment cancelled."
            )

        print(
            "    512-D ArcFace embedding generated."
        )

        # ----------------------------------------------------
        # Create template sample
        # ----------------------------------------------------

        sample = EnrolledTemplate(

            identity=identity,

            embedding=embedding,

            model_sha256=
                model_sha256,

            enrolled_at=
                datetime.now(
                    timezone.utc
                ).isoformat(),

            source_note=
                Path(
                    photo_path
                ).name,

            alignment=
                alignment_method,

            display_name=
                display_name,
        )

        new_samples.append(
            sample
        )

        print(
            "    Enrollment sample accepted."
        )

        print()

    # ========================================================
    # ONLY NOW LOAD AND MODIFY TEMPLATE DATABASE
    # ========================================================

    templates = load_templates(
        DEFAULT_TEMPLATES_PATH
    )

    # ========================================================
    # ADD OR REPLACE
    # ========================================================

    if replace:

        templates[
            identity
        ] = list(
            new_samples
        )

    else:

        templates.setdefault(
            identity,
            []
        )

        templates[
            identity
        ].extend(
            new_samples
        )

    # ========================================================
    # SAVE DATABASE
    # ========================================================

    save_templates(
        DEFAULT_TEMPLATES_PATH,
        templates
    )

    # ========================================================
    # SUCCESS
    # ========================================================

    print()
    print("=" * 60)
    print("ENROLLMENT SUCCESSFUL")
    print("=" * 60)

    print(
        f"Person ID      : {identity}"
    )

    if display_name:

        print(
            f"Display name   : {display_name}"
        )

    print(
        f"Added samples  : "
        f"{len(new_samples)}"
    )

    print(
        f"Total samples  : "
        f"{len(templates[identity])}"
    )

    print(
        f"Template file  : "
        f"{DEFAULT_TEMPLATES_PATH}"
    )

    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print()
    print("=" * 60)
    print("MULTI-IMAGE FACE ENROLLMENT")
    print("=" * 60)

    # ========================================================
    # STEP 1
    # ASK PERSON ID
    # ========================================================

    # ID -> name, for confirming that this is the badge number somebody
    # meant and for refusing an ID the directory marks as departed. An
    # absent file means no checks and no obstruction: see check_directory.

    directory = EmployeeDirectory(
        DEFAULT_DIRECTORY_PATH,
        log=lambda msg: print(
            msg,
            file=sys.stderr,
        ),
    )

    while True:

        identity = ask_person_id()

        if check_directory(
            identity,
            directory,
        ):

            break

        print()
        print(
            "Enter a different person ID, or "
            "press Ctrl+C to stop."
        )

    # ========================================================
    # STEP 2
    # ASK DISPLAY NAME (OPTIONAL)
    # ========================================================

    display_name = ask_display_name()

    # The directory already knows the name, and it is the copy that gets
    # corrected. Taking it when the operator skipped the prompt saves
    # typing it a second time and saves typing it differently.

    if not display_name:

        from_directory = directory.display_name(
            identity
        )

        if from_directory != identity:

            display_name = from_directory

            print(
                f"Using the directory's name: "
                f"{display_name}"
            )

    # ========================================================
    # STEP 3
    # ASK IMAGE PATHS
    # ========================================================

    while True:

        photo_paths = ask_images()

        if validate_image_paths(
            photo_paths
        ):

            break

        print()
        print(
            "One or more image paths "
            "are invalid."
        )

        print(
            "Please enter all image paths again."
        )

    # ========================================================
    # SHOW SELECTED IMAGES
    # ========================================================

    print()
    print("=" * 60)
    print("ENROLLMENT SELECTION")
    print("=" * 60)

    print(
        f"Person ID    : {identity}"
    )

    if display_name:

        print(
            f"Display name : {display_name}"
        )

    print(
        f"Images       : {len(photo_paths)}"
    )

    print()

    for index, photo_path in enumerate(
        photo_paths,
        start=1
    ):

        print(
            f"  {index}. {photo_path}"
        )

    print("=" * 60)

    # ========================================================
    # CHECK EXISTING IDENTITY
    # ========================================================

    templates = load_templates(
        DEFAULT_TEMPLATES_PATH
    )

    replace = False

    if (
        identity in templates
        and templates[identity]
    ):

        existing_count = len(
            templates[identity]
        )

        existing_name = display_name_for(
            templates,
            identity
        )

        print()
        print(
            f"Person ID '{identity}' is "
            f"already enrolled."
        )

        if existing_name != identity:

            print(
                f"Enrolled display name: "
                f"{existing_name}"
            )

        print(
            f"Existing samples: "
            f"{existing_count}"
        )

        # Carry the stored name forward when this run did not supply one,
        # so adding a fourth photo does not quietly drop the name that the
        # first three carry.
        if not display_name and existing_name != identity:

            display_name = existing_name

        print()
        print(
            "Choose an action:"
        )

        print(
            "  1 = Add these images "
            "to existing enrollment"
        )

        print(
            "  2 = Replace all existing "
            "samples with these images"
        )

        print(
            "  3 = Cancel"
        )

        while True:

            choice = input(
                "\nChoice [1/2/3]: "
            ).strip()

            if choice == "1":

                replace = False
                break

            if choice == "2":

                replace = True
                break

            if choice == "3":

                print()
                print(
                    "Enrollment cancelled."
                )

                return

            print(
                "Invalid choice. "
                "Enter 1, 2, or 3."
            )

    # ========================================================
    # FINAL CONFIRMATION
    # ========================================================

    print()

    confirm = input(
        "Proceed with enrollment? [Y/n]: "
    ).strip().lower()

    if confirm not in (
        "",
        "y",
        "yes"
    ):

        print()
        print(
            "Enrollment cancelled."
        )

        return

    # ========================================================
    # PERFORM ENROLLMENT
    # ========================================================

    try:

        enroll(
            identity=identity,
            photo_paths=photo_paths,
            replace=replace,
            display_name=display_name,
        )

    except Exception as exc:

        print()
        print("=" * 60)
        print("ENROLLMENT FAILED")
        print("=" * 60)

        print(
            str(exc)
        )

        print()
        print(
            "No new enrollment samples "
            "were saved."
        )

        print(
            "Existing templates remain unchanged."
        )

        print("=" * 60)

        sys.exit(1)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()