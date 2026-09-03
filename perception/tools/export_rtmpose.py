#!/usr/bin/env python3
"""Convert the RTMPose-t checkpoint to ONNX. Run once, offline, on a laptop.

    python3 perception/tools/export_rtmpose.py \\
        --checkpoint perception/models/rtmpose-tiny_simcc-aic-coco_pt-aic-coco_420e-256x192-cfc8f33d_20230126.pth \\
        --out perception/models/rtmpose_t.onnx

Why not MMPose's own `tools/deployment` exporter: it needs MMPose, MMEngine,
MMCV and MMDetection, which pin their own torch and numpy and do not install
cleanly on the board. Dragging that stack in to run `torch.onnx.export` once
would put it in the dependency list forever. `perception/actions/rtmpose_arch.py`
rebuilds the same module graph in plain torch instead, and loads the published
weights into it with every name and shape checked.

Torch is needed *here* and nowhere else. Nothing on the inference path imports
it: the runtime is `perception/actions/rtmpose.py`, which is onnxruntime plus
numpy.

## Why ONNX rather than shipping the .pth

Same reason as the detector. RKNN-Toolkit2 takes ONNX, so an ONNX model is the
only one that can be tried on the RK3568 NPU without a second conversion
project. Whether it converts is the board measurement still outstanding, and
the point of staying in ONNX is that finding out stays cheap.

For that conversion, note the normalisation is *not* folded into the graph. The
graph takes a normalised float tensor and `rtmpose.py` does the mean/std in
numpy, which is what RKNN wants: pass the same numbers to `rknn.config()` as
`mean_values`/`std_values` and the NPU does it on quantised uint8 input.

## Verification

With onnxruntime installed the export is checked against torch on random input
before it is written -- both the raw bin scores and, more to the point, the
decoded keypoints. That second check is the one that matters: a small numeric
drift is fine, an argmax that lands in a different bin is a moved joint.
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

DEFAULT_CHECKPOINT = str(
    ROOT / "models"
    / "rtmpose-tiny_simcc-aic-coco_pt-aic-coco_420e-256x192-cfc8f33d_20230126.pth"
)
DEFAULT_OUT = str(ROOT / "models" / "rtmpose_t.onnx")


def export(
    checkpoint: str, out_path: str, input_size=(192, 256), opset: int = 16
) -> None:
    import torch

    from rtmpose_arch import load_rtmpose_tiny

    model = load_rtmpose_tiny(checkpoint)
    width, height = input_size
    dummy = torch.zeros(1, 3, height, width)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=["input"],
        output_names=["simcc_x", "simcc_y"],
        # A dynamic batch axis so one call covers every person in the frame.
        # Top-down pose costs one forward pass per person, so this is the
        # difference between one session call and eight on a busy frame.
        dynamic_axes={
            "input": {0: "batch"},
            "simcc_x": {0: "batch"},
            "simcc_y": {0: "batch"},
        },
        opset_version=opset,
        do_constant_folding=True,
        # The TorchScript exporter, not torch 2.9's new dynamo one, which warns
        # about it. Deliberate: this graph has no control flow to trace, and
        # the older exporter emits the plainer op set -- which is what a
        # downstream RKNN or cv2.dnn import has the best chance of accepting.
        # Revisit if it is ever actually removed.
        dynamo=False,
    )


def verify(checkpoint: str, out_path: str, input_size=(192, 256)) -> bool:
    """Compare torch and onnxruntime on random input. Returns False if it drifted."""
    try:
        import onnxruntime as ort
    except ImportError:
        print(
            "  !! onnxruntime not installed -- export written but NOT verified.\n"
            "     Install it and re-run with --verify-only before trusting this "
            "graph. An unverified export is exactly the kind of thing that runs "
            "and returns plausible nonsense.",
            file=sys.stderr,
        )
        return False

    import torch

    from rtmpose import decode_simcc
    from rtmpose_arch import load_rtmpose_tiny

    width, height = input_size
    rng = np.random.RandomState(0)
    batch = rng.randn(2, 3, height, width).astype(np.float32)

    model = load_rtmpose_tiny(checkpoint)
    with torch.no_grad():
        ref_x, ref_y = model(torch.from_numpy(batch))
    ref_x, ref_y = ref_x.numpy(), ref_y.numpy()

    session = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    got_x, got_y = session.run(None, {session.get_inputs()[0].name: batch})

    drift = max(
        float(np.abs(ref_x - got_x).max()), float(np.abs(ref_y - got_y).max())
    )
    moved = 0
    for i in range(batch.shape[0]):
        a, _ = decode_simcc(ref_x[i], ref_y[i])
        b, _ = decode_simcc(got_x[i], got_y[i])
        moved += int(np.count_nonzero(np.any(a != b, axis=1)))

    print(f"  numeric drift vs torch:  {drift:.2e}")
    print(f"  keypoints in a different bin: {moved} / {batch.shape[0] * 17}")
    if moved:
        print(
            "  !! the exported graph decodes to different joint positions. "
            "Do not ship it.",
            file=sys.stderr,
        )
    return moved == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--opset", type=int, default=16)
    parser.add_argument(
        "--verify-only", action="store_true",
        help="check an existing .onnx against the checkpoint, exporting nothing",
    )
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 1

    if not args.verify_only:
        print(f"exporting {args.checkpoint}", file=sys.stderr)
        export(args.checkpoint, args.out, opset=args.opset)
        size_mb = os.path.getsize(args.out) / (1 << 20)
        digest = hashlib.sha256(Path(args.out).read_bytes()).hexdigest()
        print(f"saved {args.out}  ({size_mb:.1f} MB)")
        print(f"sha256 {digest}")

    ok = verify(args.checkpoint, args.out)

    print()
    print("contract for perception/actions/rtmpose.py:")
    print("  input      Nx3x256x192 float32, RGB, ImageNet mean/std")
    print("  simcc_x    (N, 17, 384)")
    print("  simcc_y    (N, 17, 512)")
    print("  decode     argmax per axis / 2.0 -> crop pixels; peak value = score")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
