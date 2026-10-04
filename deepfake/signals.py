"""Per-frame feature extraction used as deepfake "manipulation indicators".

Indicators implemented (all explainable):
  * eye aspect ratio (EAR)            -> natural blinking behaviour
  * sharpness (Laplacian variance)    -> GAN over-smoothing
  * high-frequency FFT energy         -> loss of natural texture detail
  * facial symmetry error             -> blending / mirror artifacts
  * boundary seam gradient contrast   -> face cut-out / paste seams
All functions are pure numpy/cv2 so they run fast and CPU-only.
"""
from __future__ import annotations

import numpy as np
import cv2

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 387, 385, 263, 380, 373]


def eye_aspect_ratio(landmarks: np.ndarray, idxs) -> float:
    """Eye Aspect Ratio (Soukupova & Cech). Ratio ~0.25 open, drops on blink."""
    p = landmarks[idxs]
    a = np.linalg.norm(p[1] - p[5])
    b = np.linalg.norm(p[2] - p[4])
    c = np.linalg.norm(p[0] - p[3])
    return float((a + b) / (2.0 * c + 1e-6))


def both_ears(landmarks: np.ndarray) -> float:
    return float(
        np.mean(
            [
                eye_aspect_ratio(landmarks, LEFT_EYE),
                eye_aspect_ratio(landmarks, RIGHT_EYE),
            ]
        )
    )


def laplacian_sharpness(gray_face: np.ndarray) -> float:
    """Variance of Laplacian measured on the inner face region (ignores the
    padded crop border which is dominated by hair/hat edges). Real photos:
    hundreds-to-thousands; smoothed GAN faces drop below ~150."""
    if gray_face is None or gray_face.size == 0:
        return 0.0
    h, w = gray_face.shape[:2]
    hh, ww = int(h * 0.30), int(w * 0.30)
    inner = gray_face[hh : h - hh, ww : w - ww]
    if inner.size == 0:
        inner = gray_face
    inner = cv2.resize(inner, (96, 96))
    return float(cv2.Laplacian(inner, cv2.CV_64F).var())


def high_frequency_energy(gray_face: np.ndarray) -> float:
    """Fraction of 2D FFT energy above a radius cut-off inside the inner face
    region -> natural texture detail. GAN/smoothed faces show very low values."""
    if gray_face is None or gray_face.size == 0:
        return 0.5
    h, w = gray_face.shape[:2]
    hh, ww = int(h * 0.35), int(w * 0.35)
    inner = gray_face[hh : h - hh, ww : w - ww]
    if inner.size == 0:
        inner = gray_face
    g = cv2.resize(inner, (128, 128)).astype(np.float32)
    g = g - g.mean()
    f = np.fft.fftshift(np.fft.fft2(g))
    mag = np.abs(f)
    total = mag.sum() + 1e-9
    cy = cx = 64
    yy, xx = np.mgrid[0:128, 0:128]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    high = mag[r > 34].sum()
    return float(high / total)


def face_symmetry_error(gray_face: np.ndarray) -> float:
    """Mean |face - horizontal flip| normalised. Deepfake blends / mirrored GAN
    faces tend to show a higher structural asymmetry."""
    if gray_face is None or gray_face.size == 0:
        return 0.5
    g = cv2.resize(gray_face, (128, 128))
    flip = cv2.flip(g, 1)
    return float(np.abs(g.astype(np.float32) - flip.astype(np.float32)).mean() / 255.0)


def boundary_seam_score(gray_full, hull_pts, bbox) -> float:
    """Gradient contrast between the face-box border band and the face interior.

    A swapped/pasted face usually shows a strong gradient ring at its pasted
    edge together with an unnaturally smooth interior (the GAN blend), i.e.
    boundary_gradient >> interior_gradient -> high seam score.
    """
    x0, y0, x1, y1 = bbox
    crop = gray_full[y0:y1, x0:x1]
    if crop.size == 0:
        return 0.0
    hh, ww = crop.shape[:2]
    border = max(3, int(0.08 * min(hh, ww)))
    if hh <= 2 * border + 4 or ww <= 2 * border + 4:
        return 0.0
    inner = np.zeros((hh, ww), np.uint8)
    inner[border : hh - border, border : ww - border] = 255
    ring_mask = 255 - inner
    gy = np.abs(cv2.Sobel(crop, cv2.CV_64F, 1, 0, ksize=3))
    gx = np.abs(cv2.Sobel(crop, cv2.CV_64F, 0, 1, ksize=3))
    grad = gx + gy
    boundary = float(grad[ring_mask > 0].mean())
    interior = float(grad[inner > 0].mean()) + 1e-6
    ratio = boundary / interior
    return float(np.clip((ratio - 1.0) / 2.0, 0.0, 1.0))


def compute_spatial(gray_full, face) -> dict:
    """All single-frame spatial features for one detected face."""
    ear = both_ears(face.landmarks)
    lap = laplacian_sharpness(face.gray_face)
    hf = high_frequency_energy(face.gray_face)
    sym = face_symmetry_error(face.gray_face)
    seam = boundary_seam_score(gray_full, face.hull, face.bbox)
    return {
        "ear": ear,
        "sharpness": lap,
        "hf_frac": hf,
        "sym": sym,
        "seam": seam,
    }