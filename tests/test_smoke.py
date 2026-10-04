"""Smoke test: verifies the full pipeline runs on image + video inputs."""
import os
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepfake.detector import DeepfakeDetector
from deepfake.augment import make_fake_video


def _synthetic_face_color_photo():
    """Small easy-to-detect BGR test image with a drawn face-ish blob.
    MediaPipe may not find a real face here, so we fall back to a smooth
    region test (pipeline must not crash)."""
    img = np.full((240, 320, 3), 210, np.uint8)
    cv2.ellipse(img, (160, 120), (80, 110), 0, 0, 360, (160, 140, 130), -1)
    cv2.ellipse(img, (140, 95), (12, 14), 0, 0, 360, (40, 30, 25), -1)
    cv2.ellipse(img, (180, 95), (12, 14), 0, 0, 360, (40, 30, 25), -1)
    cv2.ellipse(img, (160, 145), (30, 14), 0, 0, 360, (90, 70, 60), -1)
    return img


def main():
    detector = DeepfakeDetector()

    # ---- image path
    img = _synthetic_face_color_photo()
    r_img = detector.analyze_image(img)
    assert r_img["media_type"] == "image"
    assert "overall" in r_img and "faces" in r_img
    assert r_img["annotated"].shape == img.shape
    print("[OK] image analysis ->", r_img["overall"], "|", r_img["summary"])

    # ---- video path (list of frames)
    frames = [_synthetic_face_color_photo() for _ in range(6)]
    tmp = os.path.join(tempfile.gettempdir(), "deepfake_smoke.mp4")
    make_fake_video(frames, tmp, fps=5.0)
    r_vid = detector.analyze_video(tmp)
    assert r_vid["media_type"] == "video"
    assert r_vid["num_frames_sampled"] >= 1
    print("[OK] video analysis ->", r_vid["overall"], "|", r_vid["summary"])

    # ---- API import check
    import api  # noqa: F401
    print("[OK] api module imports cleanly")

    print("\nAll smoke tests passed ✅")


if __name__ == "__main__":
    main()