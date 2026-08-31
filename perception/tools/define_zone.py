#!/usr/bin/env python3
"""Click a zone polygon on the live camera and print it as config JSON.

    python perception/tools/define_zone.py --zone-id zone-assembly-4

    left click   add a point
    u            undo the last point
    s            print the JSON for this zone
    q / Esc      quit

Output is normalised (0..1) so the polygon survives a resolution change or a
different camera later.

Click where the floor is, not where the subject's head will be: membership is
tested at the subject's ground-contact point.
"""

from __future__ import annotations

import argparse
import json
import sys

import cv2
import numpy as np

points: list[tuple[int, int]] = []


def on_mouse(event, x, y, flags, _param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))


def as_config(zone_id: str, width: int, height: int) -> str:
    polygon = [[round(x / width, 4), round(y / height, 4)] for x, y in points]
    return json.dumps(
        {"zone_id": zone_id, "coordinates": "normalized", "polygon": polygon}, indent=2
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--zone-id", default="zone-assembly-4")
    parser.add_argument("--source", default="0")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    if isinstance(source, int) and sys.platform == "win32":
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(source)
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        print(f"could not open {source!r}", file=sys.stderr)
        return 1

    window = "define zone -- click corners, s to print, q to quit"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            height, width = frame.shape[:2]

            if len(points) >= 3:
                overlay = frame.copy()
                cv2.fillPoly(overlay, [np.array(points, dtype=np.int32)], (0, 140, 255))
                cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
            if len(points) >= 2:
                cv2.polylines(frame, [np.array(points, dtype=np.int32)],
                              len(points) >= 3, (0, 140, 255), 2)
            for i, point in enumerate(points):
                cv2.circle(frame, point, 5, (0, 220, 0), -1)
                cv2.putText(frame, str(i), (point[0] + 8, point[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, f"{args.zone_id}: {len(points)} points", (10, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("u") and points:
                points.pop()
            if key == ord("s"):
                if len(points) < 3:
                    print("need at least 3 points", file=sys.stderr)
                else:
                    print(as_config(args.zone_id, width, height))
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
