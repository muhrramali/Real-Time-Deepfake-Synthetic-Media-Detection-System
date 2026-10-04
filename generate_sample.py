"""Generate a self-contained demo video from a photo.

Usage:
  python generate_sample.py path/to/photo.jpg        -> samples/fake_demo.mp4

Creates a video with simulated deepfake-style artifacts (warping, flicker)
plus a manipulated still frame so the dashboard can be demoed end-to-end
without external data.
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from deepfake.face import FaceDetector
from deepfake.augment import make_fake_image, make_fake_video

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    src = sys.argv[1]
    img = cv2.imread(src)
    if img is None:
        print(f"Could not read: {src}")
        return 1

    os.makedirs(OUT_DIR, exist_ok=True)
    faces = FaceDetector().detect(img)
    if faces:
        fake = make_fake_image(img, faces[0].bbox)
    else:
        print("WARNING: no face detected - writing unchanged frame.")
        fake = img

    fake_path = os.path.join(OUT_DIR, "fake_still.png")
    cv2.imwrite(fake_path, fake)

    # build ~4 s clip at 25 fps from the fake still with warping
    frames = [fake.copy() for _ in range(100)]
    video_path = os.path.join(OUT_DIR, "fake_demo.mp4")
    make_fake_video(frames, video_path, fps=25.0)

    print(f"Wrote:\n  {fake_path}\n  {video_path}")
    print("Now upload these files to the dashboard or API.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())