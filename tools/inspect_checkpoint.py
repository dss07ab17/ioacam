"""Read a PyTorch checkpoint's input contract without installing torch.

Why this exists: the numbers a pose model expects -- keypoint count, clip
length, heatmap size -- are not documentation, they are baked into the weights.
Guessing them wrong does not always crash. A volume of the wrong temporal
length can run and quietly return nonsense, which is far worse than an error.

This reads the shapes straight out of the pickle, so it works on any machine
and needs nothing installed. Use it whenever a new checkpoint arrives, before
wiring it up.

    python3 tools/inspect_checkpoint.py path/to/model.pth

Tested against pose_only_20230228-fa40054e.pth, from which the defaults in
perception/actions/posetube.py were derived rather than guessed.
"""

from __future__ import annotations

import io
import pickle
import re
import sys
import zipfile
from pathlib import Path


class _OD(dict):
    pass


class _Stub:
    """Stands in for any class we deliberately do not import.

    It has to tolerate everything the unpickler might do to it. Returning a
    bare None instead fails on BUILD with "state is not a dictionary" as soon
    as a checkpoint contains an object whose __reduce__ produces a non-dict
    state -- which RTMPose checkpoints do.
    """

    def __init__(self, *args, **kwargs):
        self._args = args

    def __setstate__(self, state):
        self._state = state

    def __setitem__(self, key, value):
        pass

    def __getitem__(self, key):
        return None

    def append(self, value):
        pass

    def extend(self, values):
        pass

    def __repr__(self):
        return "<stub>"


def _rebuild(*args, **kwargs):
    """Stand-in for torch._utils._rebuild_tensor_v2 and its siblings.

    We only want the shape, which is the third positional argument for
    _rebuild_tensor_v2. _rebuild_parameter wraps a tensor, so unwrap it.
    The storage itself is a persistent-id reference we never resolve.
    """
    if args and isinstance(args[0], dict) and "shape" in args[0]:
        return args[0]                      # _rebuild_parameter(tensor, ...)
    try:
        return {"shape": tuple(args[2])}
    except Exception:
        return {"shape": None}


class _Unpickler(pickle.Unpickler):
    """Resolves torch classes to harmless stubs so nothing needs importing."""

    def find_class(self, module, name):
        if name in ("_rebuild_tensor_v2", "_rebuild_tensor", "_rebuild_parameter"):
            return _rebuild
        if name == "OrderedDict":
            return _OD
        return _Stub

    def persistent_load(self, pid):
        return None


def _diagnose(path: Path) -> None:
    """Explain what the file actually is before failing.

    By far the most common cause of an unreadable checkpoint is a download that
    returned an HTML error or login page and got saved with a .pth extension.
    Saying so is more useful than a format error.
    """
    size = path.stat().st_size
    head = path.open("rb").read(512)

    if head[:1] in (b"<", b"\n", b"\r") or b"<!DOCTYPE" in head or b"<html" in head:
        raise ValueError(
            f"{path.name} is an HTML page, not a checkpoint ({size / 1024:.0f} KB).\n"
            f"The download returned an error or redirect page. Re-fetch with:\n"
            f"  curl -L -o <file> <url>\n"
            f"or in PowerShell:\n"
            f"  Invoke-WebRequest -Uri <url> -OutFile <file>"
        )

    if size < 100_000:
        raise ValueError(
            f"{path.name} is only {size / 1024:.0f} KB, which is too small for a "
            f"pose or action checkpoint. The download was probably truncated."
        )


def load(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise ValueError(f"{path} does not exist")

    if zipfile.is_zipfile(path):
        z = zipfile.ZipFile(path)
        entry = next((n for n in z.namelist() if n.endswith("data.pkl")), None)
        if entry is None:
            raise ValueError("no data.pkl inside the archive")
        return _Unpickler(io.BytesIO(z.read(entry))).load()

    _diagnose(path)

    # Legacy torch format (pre-1.6, and still emitted by some export paths):
    # four pickles written back to back -- magic number, protocol version,
    # system info, then the object itself.
    with path.open("rb") as fh:
        try:
            for _ in range(3):
                _Unpickler(fh).load()
            return _Unpickler(fh).load()
        except Exception as exc:
            raise ValueError(
                f"{path.name} is neither a zip-format nor a legacy-pickle "
                f"checkpoint ({path.stat().st_size / 1e6:.1f} MB). It may be "
                f"corrupt, or a format this tool does not read. Underlying "
                f"error: {exc}"
            ) from exc


def report(path: str | Path) -> None:
    obj = load(path)
    size_mb = Path(path).stat().st_size / 1e6
    print(f"{Path(path).name}  ({size_mb:.1f} MB)")
    print("=" * 70)

    meta = obj.get("meta") if isinstance(obj, dict) else None
    state = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    tensors = {k: v for k, v in state.items() if isinstance(v, dict) and "shape" in v}
    print(f"tensors: {len(tensors)}")

    if isinstance(meta, dict):
        for k in ("experiment_name", "epoch", "iter", "time", "mmengine_version"):
            if k in meta:
                print(f"{k}: {meta[k]}")

    # --- the input contract -------------------------------------------
    print("\ninput contract")
    print("-" * 70)
    first = next(
        (k for k in tensors if re.search(r"(conv1|stem|patch_embed).*weight$", k)), None
    )
    if first and tensors[first]["shape"]:
        shape = tensors[first]["shape"]
        print(f"  first layer     {first}  {shape}")
        if len(shape) >= 2:
            print(
                f"  input channels  {shape[1]}"
                + ("   (17 = COCO keypoints, one heatmap each)" if shape[1] == 17 else "")
                + ("   (3 = RGB)" if shape[1] == 3 else "")
            )

    # Classification heads, across the naming conventions actually in use.
    head = [
        k for k in tensors
        if re.search(
            r"(cls_head|fc_cls|head\.fc|classifier|final_layer|cls_x|cls_y|"
            r"out_layer|keypoint_head).*weight$",
            k,
        )
    ]
    for k in head:
        shape = tensors[k]["shape"]
        if not shape:
            continue
        print(f"  head            {k}  {shape}")
        if re.search(r"cls_x|cls_y", k):
            # SimCC predicts each coordinate as a classification over bins, so
            # the head is two layers (x and y) rather than one. The bin count
            # is input_size * simcc_split_ratio, not a class count.
            print(f"  simcc bins      {shape[0]}   (coordinate bins, not classes)")
        elif len(shape) == 2:
            print(f"  classes         {shape[0]}")
            print(f"  feature dim     {shape[1]}   (what you keep when fine-tuning)")
        elif len(shape) >= 2:
            print(f"  output channels {shape[0]}")

    # Keypoint count, found by shape rather than by name. Pose models disagree
    # wildly about layer naming, but a 17 in the leading dimension of a head
    # tensor is unambiguous.
    kp_candidates = sorted({
        shape[0]
        for k, v in tensors.items()
        for shape in [v["shape"]]
        if shape
        and re.search(r"(head|final_layer|out|keypoint)", k)
        and shape[0] in (17, 21, 26, 133, 134)
    })
    if kp_candidates:
        print(f"  keypoints       {', '.join(str(c) for c in kp_candidates)}")
        if 17 in kp_candidates:
            print("                  17 = COCO, which is what PoseC3D expects")
        else:
            print("                  NOT 17 -- needs remapping before PoseC3D")

    # --- pipeline numbers, if the training cfg was embedded ------------
    cfg = meta.get("cfg") if isinstance(meta, dict) else None
    if isinstance(cfg, str):
        print("\nfrom the embedded training config")
        print("-" * 70)
        patterns = {
            "clip_len": r"clip_len=[^,\)]+",
            "heatmap scale": r"scale=\([^\)]*\)",
            "hw_ratio": r"hw_ratio=[^,\)]+",
            "sigma": r"sigma=\([^\)]*\)|sigma=[\d.]+",
            "input_size": r"input_size=\([^\)]*\)",
            "simcc_split_ratio": r"simcc_split_ratio=[\d.]+",
            "out_channels": r"out_channels=\d+",
            "num_keypoints": r"num_keypoints=\d+",
            "with_kp / with_limb": r"with_(?:kp|limb)=\w+",
            "use_score": r"use_score=\w+",
            "backbone": r"type='ResNet3d\w*'|type='\w*ViT\w*'",
            "num_classes": r"num_classes=\d+",
        }
        for label, pat in patterns.items():
            hits = sorted(set(re.findall(pat, cfg)))
            if hits:
                print(f"  {label:<20} {', '.join(hits[:6])}")

        print(
            "\n  These are what the network was fitted to. Changing them without\n"
            "  retraining does not always fail loudly -- a wrong temporal length\n"
            "  can run and return nonsense."
        )

    if not head:
        print("\n  no classification head matched; this may be a backbone-only "
              "checkpoint, or use a naming convention not covered here")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    for p in sys.argv[1:]:
        report(p)
        print()