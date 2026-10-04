"""Simulation of deepfake-style artifacts (for live demos only).

These routines deliberately inject the manipulation traces our sensor reads:
smoothing, boundary seams, and temporal jitter / frozen-blink behaviour. This
lets you demonstrate the *explainable* pipeline with a self-contained video:
  - make_fake_image(photo)        -> a plausible "deepfake" still
  - make_fake_video(frames, path) -> a fake-style video clip
"""
from __future__ import annotations

import cv2
import numpy as np


def _clip_face_bbox(bbox, h, w):
    x0, y0, x1, y1 = bbox
    x0 = max(0, int(x0)); y0 = max(0, int(y0))
    x1 = min(w, int(x1)); y1 = min(h, int(y1))
    return x0, y0, x1, y1


def make_fake_image(image_bgr, bbox, strength: float = 0.9) -> np.ndarray:
    """Return a copy of image_bgr with simulated deepfake artifacts on the face.

    Injects (light enough that the face stays detectable, strong enough for
    the explainable indicators to fire):
      1. bilateral + Gaussian smoothing inside the face (GAN over-smoothing)
      2. a mirror-flip blend seam across the vertical half (blending artifact)
      3. subtle colour banding on the face region
    """
    out = image_bgr.copy()
    h, w = out.shape[:2]
    x0, y0, x1, y1 = _clip_face_bbox(bbox, h, w)
    fw, fh = x1 - x0, y1 - y0
    if fw < 4 or fh < 4:
        return out

    mask = np.zeros((h, w), np.uint8)
    mask[y0:y1, x0:x1] = 255
    mask = cv2.GaussianBlur(mask, (0, 0), max(6, fw // 8))
    box_mask = np.zeros((h, w), np.uint8)
    box_mask[y0:y1, x0:x1] = 255

    s = float(strength)

    # 1) over-smoothing: hard on the inner face, lighter on the rim so the
    #    detector still sees a face
    ero_k = max(4, int(round(fw / 14)))
    inner = cv2.erode(box_mask, np.ones((ero_k, ero_k), np.uint8))
    smooth = cv2.bilateralFilter(out, 9, 50 * s + 10, 50 * s + 10)
    smooth = cv2.GaussianBlur(smooth, (0, 0), 2.0 + 2.5 * s)
    mix_in = 0.60 + 0.40 * s
    mix_rim = 0.15 + 0.20 * s
    inner_mix = np.where(inner[..., None] > 90, mix_in, mix_rim)
    out = (out.astype(np.float32) * (1 - inner_mix) +
           smooth.astype(np.float32) * inner_mix).astype(np.uint8)

    # 2) boundary seam artifact (pasted-face jawline tell):
    #    warm halo + crisp dark contour line -> strong gradient contrast
    kring = max(3, int(round(fw / 22)))
    ring = cv2.dilate(box_mask, np.ones((kring, kring), np.uint8)) - \
        cv2.erode(box_mask, np.ones((kring, kring), np.uint8))
    ring = cv2.GaussianBlur(ring, (0, 0), 2)
    tint = np.array([18, 9, -8], np.float32)  # warm halo like a pasted edge
    halo = np.clip(out.astype(np.float32) + tint * (0.3 + 0.5 * s), 0, 255)
    out = np.where(ring[..., None] > 60, halo, out.astype(np.float32))

    # crisp seam line following the face contour: a thick rectangular ring at the
    # face-box edge (pasted-face boundary gradient)
    off = max(4, int(0.035 * min(fw, fh)))
    seam_line = np.zeros((h, w), np.uint8)
    cv2.rectangle(seam_line, (x0 + off, y0 + off), (x1 - off, y1 - off), 255, max(4, fw // 45))
    darker = np.clip(out - np.array([90 + 55 * s] * 3, np.float32), 0, 255)
    out = np.where(seam_line[..., None] > 120, darker, out).astype(np.uint8)

    # 3) gentle colour banding (synthetic quantisation)
    banded = (out // 8 * 8).astype(np.uint8)
    band_mix = 0.10 + 0.15 * s
    out = np.where(mask[..., None] > 120,
                   out.astype(np.float32) * (1 - band_mix) + banded.astype(np.float32) * band_mix,
                   out).astype(np.uint8)

    return np.clip(out, 0, 255).astype(np.uint8)


def make_fake_video(frames, output_path: str, fps: float = 25.0):
    """Render `frames` (list of BGR) into an aggressive fake-style clip."""
    if not frames:
        raise ValueError("No frames given")
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    n = len(frames)
    for i, frame in enumerate(frames):
        t = i / max(1, n - 1)
        # wavy warp (unnatural motion)
        M = cv2.getRotationMatrix2D((w / 2, h / 2),
                                    2.5 * np.sin(6.28 * t), 1 + 0.01 * np.sin(9 * t))
        warp = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        # heartbeat brightness flicker (GAN lighting flicker)
        gain = 1.0 + 0.04 * np.sin(14 * t)
        out = np.clip(warp.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        writer.write(out)
    writer.release()
    return output_path