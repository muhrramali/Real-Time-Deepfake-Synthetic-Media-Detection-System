"""Visualisation helpers: annotated frames, montages and timeline charts."""
from __future__ import annotations

import numpy as np
import cv2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

GREEN = (76, 175, 80)
RED = (244, 67, 54)
AMBER = (255, 193, 7)


def _verdict_color(prob: float):
    if prob >= 0.65:
        return RED
    if prob >= 0.4:
        return AMBER
    return GREEN


def draw_face_result(bgr, bbox, landmarks, prob, indicators) -> np.ndarray:
    """Draw the bounding box, score tag and landmark mesh on a BGR frame."""
    x0, y0, x1, y1 = bbox
    color = _verdict_color(prob)
    cv2.rectangle(bgr, (x0, y0), (x1, y1), color, 2)

    lab = f"{prob*100:.0f}% synth"
    (tw, th), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(bgr, (x0, y0 - th - 14), (x0 + tw + 12, y0), color, -1)
    cv2.putText(
        bgr, lab, (x0 + 6, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2
    )

    # landmark dots (thin, semi-transparent)
    overlay = bgr.copy()
    for (px, py) in landmarks[::4]:
        cv2.circle(overlay, (int(px), int(py)), 1, color, -1)
    cv2.addWeighted(overlay, 0.35, bgr, 0.65, 0, bgr)

    # top-right indicator tag
    if indicators:
        first = indicators[0][:38]
        (tw2, th2), _ = cv2.getTextSize(first, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        ty = max(y1 + th2 + 8, 16)
        cv2.putText(
            bgr, "!" + first, (x0, ty),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA,
        )
    return bgr


def build_montage(images, probs, max_cols: int = 5, max_w: int = 280) -> np.ndarray:
    """Horizontal strip of the top suspicious frames (RGB images)."""
    if not images:
        return None
    imgs = []
    for img, p in zip(images, probs):
        h, w = img.shape[:2]
        scale = min(1.0, max_w / w)
        th, tw = int(h * scale), int(w * scale)
        small = cv2.resize(img, (tw, th))
        color = _verdict_color(p)
        small = cv2.copyMakeBorder(
            small, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=color
        )
        imgs.append(small)
    return np.hstack([np.ones_like(imgs[0]) * 255] + [i for i in imgs] if len(imgs) > 1 else imgs)


def verdict_color_rgb(prob: float):
    """CSS-style (r,g,b) tuple for the verdict colour."""
    if prob >= 0.65:
        return (244, 67, 54)
    if prob >= 0.4:
        return (255, 193, 7)
    return (76, 175, 80)


def timeline_chart(timeline, title: str = "Frame-level synthetic probability") -> np.ndarray:
    """Return a chart PNG (as RGB ndarray) of the per-frame timeline."""
    if not timeline:
        return None
    times = [t["time_sec"] for t in timeline]
    probs = [t["prob"] * 100 for t in timeline]
    fig, ax = plt.subplots(figsize=(9, 3.2), dpi=110)
    ax.plot(times, probs, color="#e53935", lw=1.6, marker="o", ms=3)
    ax.axhline(65, color="#e53935", ls="--", lw=1, alpha=0.6)
    ax.axhline(40, color="#fdd835", ls="--", lw=1, alpha=0.6)
    ax.text(times[-1], 68, "fake >", fontsize=8, color="#e53935", ha="right")
    ax.text(times[-1], 42, "uncertain >", fontsize=8, color="#f9a825", ha="right")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Synthetic probability (%)")
    ax.set_xlabel("Time (seconds)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)
    return buf