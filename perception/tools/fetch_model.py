#!/usr/bin/env python3
"""Download the Apache-2.0 YOLOX weights. No AGPL code is involved.

    python perception/tools/fetch_model.py            # yolox_s, the default
    python perception/tools/fetch_model.py --model yolox_tiny

Weights are ~36 MB for yolox_s and are gitignored: they are a build artefact,
not source.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request

BASE = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0"

MODELS = {
    # name: (filename, approximate size in MB, note)
    "yolox_nano": ("yolox_nano.onnx", 4, "fastest, but measurably misses people; avoid"),
    "yolox_tiny": ("yolox_tiny.onnx", 20, "default: yolox_s recall at 2.6x the speed"),
    "yolox_s": ("yolox_s.onnx", 36, "accuracy reference; ~2 fps on a laptop CPU"),
    "yolox_m": ("yolox_m.onnx", 97, "better recall on small/distant subjects; slower again"),
}


def download(url: str, dest: str) -> None:
    print(f"fetching {url}", file=sys.stderr)
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    tmp = dest + ".part"
    with urllib.request.urlopen(url) as response, open(tmp, "wb") as out:
        total = int(response.headers.get("Content-Length", 0))
        read = 0
        digest = hashlib.sha256()
        while True:
            chunk = response.read(1 << 16)
            if not chunk:
                break
            out.write(chunk)
            digest.update(chunk)
            read += len(chunk)
            if total:
                print(f"\r  {100 * read // total:3d}%  {read >> 20} MB", end="", file=sys.stderr)
    print(file=sys.stderr)
    os.replace(tmp, dest)
    print(f"saved {dest}", file=sys.stderr)
    print(f"sha256 {digest.hexdigest()}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="yolox_tiny", choices=sorted(MODELS))
    parser.add_argument("--out-dir", default="perception/models")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    filename, size_mb, note = MODELS[args.model]
    dest = os.path.join(args.out_dir, filename)
    if os.path.exists(dest) and not args.force:
        print(f"{dest} already present (--force to re-download)", file=sys.stderr)
        return 0

    print(f"{args.model}: ~{size_mb} MB -- {note}", file=sys.stderr)
    try:
        download(f"{BASE}/{filename}", dest)
    except Exception as exc:
        print(f"download failed: {exc}", file=sys.stderr)
        print(f"Fetch {filename} manually from {BASE} and put it in {args.out_dir}/",
              file=sys.stderr)
        return 1

    if args.model != "yolox_tiny":
        print("\nRemember to point detector.model_path at this file in your config.",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
