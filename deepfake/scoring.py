"""Explainable scoring: fuse raw signals into a probability, verdict, and a
human-readable list of manipulation indicators.

Design: every indicator maps to a suspicion in [0, 1]. A weighted sum produces
the final deepfake probability. Weights follow the literature intuition that
temporal signals (blinking, motion jitter) are the strongest video tells, while
spatial artifacts (seams, over-smoothing, symmetry) drive image analysis.
"""
from __future__ import annotations

import statistics
from typing import Optional

import numpy as np

INDICATOR_LABELS = {
    "blink": "Irregular blinking / missing natural eye blinks",
    "jitter": "Unstable landmark motion (temporal jitter)",
    "sym": "Facial asymmetry / mirrored blending artifacts",
    "smooth": "Over-smoothed texture (Gaussian/GAN smoothing)",
    "seam": "Face boundary blending seam / cut-out artifacts",
    "hf": "Missing natural high-frequency detail",
}


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, x)))


def _rising(value: float, onset: float, width: float) -> float:
    """suspicion increases 0->1 as value moves from onset to onset+width."""
    return _clamp((value - onset) / width)


def _falling(value: float, onset: float, width: float) -> float:
    """suspicion increases 0->1 as value moves from onset down to onset-width."""
    return _clamp((onset - value) / width)


def _blink_suspicion(ear_series, frame_step: int, fps: int) -> float:
    """No/slow blinking (deepfake tell), or unnaturally fast blinking."""
    if len(ear_series) < 8:
        return 0.5
    base = float(statistics.mean(ear_series))
    thr = base * 0.75
    blinks = 0
    low = False
    refractory = 0
    for v in ear_series:
        if refractory > 0:
            refractory -= 1
            continue
        if v < thr:
            if not low:
                blinks += 1
            low = True
            refractory = 2
        else:
            low = False
    dur = len(ear_series) * frame_step / fps
    if dur <= 0:
        return 0.5
    bps = blinks / dur
    if bps < 0.12:
        return _clamp(0.8 + (0.8 - dur) * 0.1)  # short clips -> slightly less sure
    if bps > 1.2:
        return _rising(bps, 1.2, 1.0)
    return 0.15


def _jitter_suspicion(centers, landmarks_list, fps: int) -> float:
    """Track smoothness: mean frame-to-frame landmark speed & its std (jerk)."""
    if len(landmarks_list) < 5:
        return 0.5
    speeds = []
    for i in range(1, len(landmarks_list)):
        d = landmarks_list[i] - landmarks_list[i - 1]
        speed = float(np.linalg.norm(d, axis=1).mean())
        speeds.append(speed)
    speed_std = float(statistics.pstdev(speeds)) if len(speeds) > 1 else 0.5
    return _clamp(_rising(speed_std, 1.0, 2.5) * 0.7 + _rising(float(np.mean(speeds)), 2.0, 4.0) * 0.3)


def _mean(items) -> Optional[float]:
    if not items:
        return None
    return float(statistics.mean(items))


def score_face(spatial: dict, temporal: Optional[dict] = None) -> dict:
    """Combine spatial + optional temporal signals for one face track.

    spatial:  dict from signals.compute_spatial
    temporal: dict with keys {
        'ear_series', 'blinks_per_sec'?, 'centers', 'landmarks_list',
        'frame_step', 'fps'
    }
    """
    s = {"blink": 0.5, "jitter": 0.5}
    if temporal:
        if temporal.get("ear_series"):
            s["blink"] = _blink_suspicion(
                temporal["ear_series"], temporal.get("frame_step", 2), temporal.get("fps", 25)
            )
        if temporal.get("landmarks_list"):
            s["jitter"] = _jitter_suspicion(
                temporal.get("centers", []),
                temporal["landmarks_list"],
                temporal.get("fps", 25),
            )

    # spatial suspicions
    lap = spatial.get("sharpness", 0.0)
    s["smooth"] = _falling(lap, 150.0, 130.0)
    s["hf"] = _falling(spatial.get("hf_frac", 0.3), 0.30, 0.20)
    s["sym"] = _rising(spatial.get("sym", 0.05), 0.16, 0.10)
    s["seam"] = spatial.get("seam", 0.0)

    has_temporal = bool(temporal and temporal.get("ear_series") and len(temporal["ear_series"]) >= 8)

    if has_temporal:
        weights = {
            "blink": 0.22, "jitter": 0.14, "sym": 0.16,
            "smooth": 0.14, "seam": 0.20, "hf": 0.14,
        }
    else:
        weights = {
            "blink": 0.0, "jitter": 0.0, "sym": 0.16,
            "smooth": 0.34, "seam": 0.28, "hf": 0.22,
        }

    total_w = sum(w for k, w in weights.items() if k in s)
    prob = sum(s[k] * weights[k] for k in weights if k in s) / max(total_w, 1e-9)
    prob = _clamp(prob)

    indicators = [
        INDICATOR_LABELS[k] for k, v in s.items() if weights.get(k, 0) > 0 and v >= 0.6
    ]
    return {
        "probability": prob,
        "indicators": indicators,
        "signals": {k: round(v, 3) for k, v in s.items() if weights.get(k, 0) > 0},
    }


def verdict_of(prob: float) -> str:
    if prob >= 0.65:
        return "Fake / Manipulated"
    if prob >= 0.4:
        return "Uncertain"
    return "Likely Real"


def confidence_of(prob: float, n_frames: int = 1) -> float:
    """Confidence grows with extremity of score and amount of evidence."""
    how_extreme = abs(prob - 0.5) * 2.0
    evidence = min(1.0, n_frames / 120.0)
    conf = 0.35 + 0.45 * how_extreme + 0.2 * evidence
    return _clamp(conf) * 100.0