"""Action recognition backends, selected by config.

Same seam as perception/detectors/: the pipeline never names a model, so the
licence and the hardware requirement are properties of the backend rather than
of the system.
"""

from __future__ import annotations

BACKENDS = ("stub", "posec3d")
POSE_BACKENDS = ("stub", "rtmpose")


def build_recognizer(cfg: dict):
    backend = cfg.get("backend", "stub")

    if backend == "stub":
        from stub import StubRecognizer

        return StubRecognizer(
            classes=cfg.get("classes", ["standing", "walking", "reaching"]),
            num_frames=int(cfg.get("num_frames", 16)),
            input_size=int(cfg.get("input_size", 224)),
        )

    if backend == "posec3d":
        from posec3d import NTU60_CLASSES, PoseC3DRecognizer

        # The stock checkpoint's head is 60 NTU classes. Naming that vocabulary
        # by a sentinel rather than pasting sixty strings into every config
        # keeps them from drifting out of the checkpoint's own order, which is
        # the one thing that must not happen: a reordered list relabels every
        # prediction with no error anywhere.
        classes = cfg["classes"]
        if isinstance(classes, str):
            if classes != "ntu60":
                raise ValueError(
                    f"unknown class vocabulary '{classes}'; use 'ntu60' for the "
                    f"stock checkpoint, or list your own class names once the "
                    f"head is retrained"
                )
            classes = NTU60_CLASSES

        return PoseC3DRecognizer(
            model_path=cfg["model_path"],
            classes=classes,
            num_frames=int(cfg.get("num_frames", 24)),
            heatmap_size=int(cfg.get("heatmap_size", 56)),
            num_keypoints=int(cfg.get("num_keypoints", 17)),
            device=cfg.get("device", "cuda"),
            rgb_branch=bool(cfg.get("rgb_branch", False)),
            input_size=int(cfg.get("input_size", 224)),
        )


    raise ValueError(f"unknown action backend '{backend}'; one of {BACKENDS}")


def build_pose_estimator(cfg: dict):
    """The pose seam, from the `pose` block of the actions config.

    Separate from the recognizer because they are separate models with separate
    licences and separate hardware costs -- pose runs per person, ahead of the
    action model, and that second inference is the outstanding board
    measurement.
    """
    backend = cfg.get("backend", "stub")

    if backend == "stub":
        from posetube import StubPoseEstimator

        return StubPoseEstimator(score=float(cfg.get("score", 0.9)))

    if backend == "rtmpose":
        from rtmpose import RTMPoseEstimator

        return RTMPoseEstimator(
            model_path=cfg["model_path"],
            input_size=tuple(cfg.get("input_size", (192, 256))),
            simcc_split_ratio=float(cfg.get("simcc_split_ratio", 2.0)),
            num_keypoints=int(cfg.get("num_keypoints", 17)),
            device=cfg.get("device", "cpu"),
            max_batch=int(cfg.get("max_batch", 8)),
        )

    raise ValueError(f"unknown pose backend '{backend}'; one of {POSE_BACKENDS}")
