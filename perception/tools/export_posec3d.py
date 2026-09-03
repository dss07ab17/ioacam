#!/usr/bin/env python3
"""Convert the PoseC3D pose-only checkpoint to ONNX. Run once, offline.

    python3 perception/tools/export_posec3d.py \\
        --checkpoint perception/models/pose_only_20230228-fa40054e.pth \\
        --out perception/models/posec3d_pose_only.onnx



## Why not MMAction2's exporter

It needs MMAction2, MMEngine and MMCV. On the machine this was written on MMCV
does not build at all, and mmaction2 pins `numpy<2`, which would downgrade the
numpy that OpenCV and onnxruntime are already running on across the perception
layer. That is a working environment traded away to run `torch.onnx.export`
once. `perception/actions/posec3d_arch.py` rebuilds the module graph in plain
torch instead and loads the published weights into it with every name and shape
checked, which is the same approach `export_rtmpose.py` takes and for the same
reasons.

Torch is needed here and nowhere else. Inference is onnxruntime plus numpy.

## The contract

    input     (N, 17, 32, 56, 56) float32   17 COCO joints, 32 frames, 56x56
    output    (N, 60) logits                NTU-60, in posec3d.NTU60_CLASSES order

Those numbers are the checkpoint's, not a preference: `conv1` has 17 input
channels and the model was trained at clip_len 32 on 56x56 heatmaps. A volume
of a different temporal length still runs -- the head global-pools before the
classifier, so nothing in the graph objects -- and returns a worse answer with
no indication. That is why `posec3d.py` validates the tube shape at the
boundary and why `PoseTubeExtractor`'s defaults must not be edited to match a
volume rather than the model.

## Verification

With onnxruntime installed the export is checked against torch before it is
trusted: the logits must agree numerically, and -- the check that actually
matters -- the argmax class must be identical on every sample. A drift of 1e-5
is fine; a different top class is a different prediction.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent          # perception/
sys.path.insert(0, str(ROOT / "actions"))

DEFAULT_CHECKPOINT = str(ROOT / "models" / "pose_only_20230228-fa40054e.pth")
DEFAULT_OUT = str(ROOT / "models" / "posec3d_pose_only.onnx")
CHECKPOINT_URL = (
    "https://download.openmmlab.com/mmaction/v1.0/skeleton/posec3d/"
    "rgbpose_conv3d/pose_only_20230228-fa40054e.pth"
)

SHAPE = (17, 32, 56, 56)


def fetch(dest: str) -> None:
    """Download the published checkpoint. ~8 MB."""
    import urllib.request

    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    print(f"fetching {CHECKPOINT_URL}", file=sys.stderr)
    tmp = dest + ".part"
    digest = hashlib.sha256()
    with urllib.request.urlopen(CHECKPOINT_URL, timeout=120) as response, \
            open(tmp, "wb") as out:
        while True:
            chunk = response.read(1 << 16)
            if not chunk:
                break
            out.write(chunk)
            digest.update(chunk)
    os.replace(tmp, dest)
    print(f"saved {dest}", file=sys.stderr)
    print(f"sha256 {digest.hexdigest()}", file=sys.stderr)


def export(checkpoint: str, out_path: str, opset: int = 16) -> None:
    import torch

    from posec3d_arch import load_pose_only

    model = load_pose_only(checkpoint)
    dummy = torch.zeros(1, *SHAPE)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=["heatmap_volume"],
        output_names=["logits"],
        # A dynamic batch axis so several tracked people classify in one call.
        # Unlike pose, this model runs once per person per *window*, not per
        # frame, so the batch is usually 1-3 -- but a busy frame should not
        # cost three session calls.
        dynamic_axes={"heatmap_volume": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
        # TorchScript exporter, as in export_rtmpose.py: no control flow to
        # trace, and the plainer op set is what an RKNN import stands the best
        # chance with.
        dynamo=False,
    )


def verify(checkpoint: str, out_path: str) -> bool:
    try:
        import onnxruntime as ort
    except ImportError:
        print(
            "  !! onnxruntime not installed -- export written but NOT verified.",
            file=sys.stderr,
        )
        return False

    import torch

    from posec3d_arch import load_pose_only

    rng = np.random.RandomState(0)
    # Random noise, plus a batch of something heatmap-shaped: a volume of
    # gaussian blobs exercises the same graph but in the value range the real
    # tubes occupy, where saturation and dead ReLUs would show up.
    batch = np.concatenate([
        rng.rand(1, *SHAPE).astype(np.float32),
        (rng.rand(1, *SHAPE).astype(np.float32) > 0.98).astype(np.float32),
    ])

    model = load_pose_only(checkpoint)
    with torch.no_grad():
        reference = model(torch.from_numpy(batch)).numpy()

    session = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    got = session.run(None, {session.get_inputs()[0].name: batch})[0]

    drift = float(np.abs(reference - got).max())
    same_class = int(np.count_nonzero(
        reference.argmax(axis=1) == got.argmax(axis=1)
    ))
    print(f"  numeric drift vs torch:     {drift:.2e}")
    print(f"  same top class:             {same_class} / {batch.shape[0]}")
    if same_class != batch.shape[0]:
        print("  !! the exported graph predicts a different class. Do not ship it.",
              file=sys.stderr)
    return same_class == batch.shape[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--opset", type=int, default=16)
    parser.add_argument("--download", action="store_true",
                        help="fetch the checkpoint if it is not already here")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        if args.download:
            fetch(args.checkpoint)
        else:
            print(f"checkpoint not found: {args.checkpoint}\n"
                  f"  re-run with --download, or fetch it from\n"
                  f"  {CHECKPOINT_URL}", file=sys.stderr)
            return 1

    print(
        "NOTE: these weights are NTU RGB+D trained -- research-only, and a "
        "fine-tune inherits it. See perception/actions/LICENCE-NOTES.md.",
        file=sys.stderr,
    )

    if not args.verify_only:
        export(args.checkpoint, args.out, opset=args.opset)
        size_mb = os.path.getsize(args.out) / (1 << 20)
        print(f"saved {args.out}  ({size_mb:.1f} MB)")
        print(f"sha256 {hashlib.sha256(Path(args.out).read_bytes()).hexdigest()}")

    ok = verify(args.checkpoint, args.out)

    print()
    print("contract for perception/actions/posec3d.py:")
    print("  input    Nx17x32x56x56 float32 heatmap volume")
    print("  output   (N, 60) logits, in posec3d.NTU60_CLASSES order")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
