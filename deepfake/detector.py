"""Detection pipelines for images and videos.

The DeepfakeDetector class decouples signal extraction from scoring so new
backbones / trained models can be swapped into the "model slots" without
redesigning the application (see README "Model Slots").
"""
from __future__ import annotations

import math
import os

import cv2
import numpy as np

from .face import FaceDetector
from . import signals, scoring
from .visualization import draw_face_result, build_montage

MAX_SAMPLED_FRAMES = 360  # keep the hackathon demo fast


class DeepfakeDetector:
    def __init__(self, max_faces: int = 4, min_detection_confidence: float = 0.5):
        self._fd = FaceDetector(
            max_faces=max_faces, min_detection_confidence=min_detection_confidence
        )

    # ------------------------------------------------------------------ image
    def analyze_image(self, image_bgr: np.ndarray) -> dict:
        """Analyze any BGR image. Returns scores, indicators, annotated frame."""
        faces = self._fd.detect(image_bgr)
        per_face = []
        areas = []
        for f in faces:
            spatial = signals.compute_spatial(image_bgr, f)
            res = scoring.score_face(spatial, temporal=None)
            w = float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            per_face.append(
                {
                    "bbox": list(f.bbox),
                    **res,
                    "confidence": scoring.confidence_of(res["probability"], 1),
                }
            )
            areas.append(w)

        if per_face:
            prob = float(np.average([p["probability"] for p in per_face], weights=areas))
        else:
            prob = 0.5  # no face -> neutral
        overall = {
            "probability": round(prob, 4),
            "verdict": scoring.verdict_of(prob),
            "confidence": round(scoring.confidence_of(prob, 1), 2),
        }

        annotated = image_bgr.copy()
        for fi, f in enumerate(faces):
            res = per_face[fi]
            annotated = draw_face_result(
                annotated, f.bbox, f.landmarks, res["probability"], res["indicators"]
            )

        return {
            "media_type": "image",
            "num_faces": len(faces),
            "overall": overall,
            "faces": per_face,
            "annotated": annotated,
            "summary": build_summary(overall, "image", len(faces)),
        }

    # ------------------------------------------------------------------ video
    def analyze_video(self, path_or_bgr_list, fps: float = 25.0) -> dict:
        """Analyze a video file (or list of BGR frames).

        Returns a frame-level timeline, annotated previews of the most
        suspicious frames and an aggregate verdict fused from per-face
        *track* statistics (blinking, motion jitter) + spatial signals.
        """
        frames, reported_fps = _load_frames(path_or_bgr_list, fps)
        if not frames:
            return {"error": "Could not read video", "media_type": "video"}

        frame_step = (
            max(1, int(round(frames[1][0] - frames[0][0]))) if len(frames) > 1 else 1
        )

        # 1. spatial features per sampled frame
        frame_results = []  # (frame_idx, tsec, list[(FaceInfo, spatial)])
        for idx, bgr in frames:
            detected = self._fd.detect(bgr)
            frame_results.append(
                (idx, idx / reported_fps,
                 [(f, signals.compute_spatial(bgr, f)) for f in detected])
            )

        # 2. nearest-centre lightweight tracker across sampled frames
        track_data: dict[int, dict] = {}
        next_id = [0]
        prev_centers: list[tuple[int, tuple]] = []
        for _idx, _tsec, fr in frame_results:
            assigned = []
            for f, spatial in fr:
                best_id, best_d = None, float("inf")
                for tid, c in prev_centers:
                    d = math.hypot(f.center[0] - c[0], f.center[1] - c[1])
                    if d < best_d and d < 120.0:
                        best_id, best_d = tid, d
                if best_id is None:
                    best_id = next_id[0]
                    next_id[0] += 1
                    track_data[best_id] = {
                        "ear_series": [], "landmarks_list": [], "centers": [],
                        "spatials": [], "frames": [],
                    }
                t = track_data[best_id]
                t["ear_series"].append(spatial["ear"])
                t["landmarks_list"].append(f.landmarks.copy())
                t["centers"].append(f.center)
                t["spatials"].append(spatial)
                t["frames"].append(_idx)
                assigned.append(best_id)
            prev_centers = [(tid, track_data[tid]["centers"][-1]) for tid in assigned]

        # 3. per-track temporal+spatial fusion
        per_track = []
        for tid, t in track_data.items():
            mean_spatial = {
                "sharpness": float(np.mean([s["sharpness"] for s in t["spatials"]])),
                "hf_frac": float(np.mean([s["hf_frac"] for s in t["spatials"]])),
                "sym": float(np.mean([s["sym"] for s in t["spatials"]])),
                "seam": float(np.mean([s["seam"] for s in t["spatials"]])),
            }
            temporal = {
                "ear_series": t["ear_series"],
                "centers": t["centers"],
                "landmarks_list": t["landmarks_list"],
                "frame_step": frame_step,
                "fps": reported_fps,
            }
            res = scoring.score_face(mean_spatial, temporal)
            per_track.append({"track_id": tid, "n_frames": len(t["frames"]), **res})

        if not per_track:
            overall = {"probability": 0.5, "verdict": "No face detected", "confidence": 20.0}
        else:
            wts = [t["n_frames"] for t in per_track]
            prob = float(np.average([t["probability"] for t in per_track], weights=wts))
            overall = {
                "probability": round(prob, 4),
                "verdict": scoring.verdict_of(prob),
                "confidence": round(scoring.confidence_of(prob, len(frames)), 2),
            }

        # 4. frame-level timeline (spatial-only probability)
        timeline = []
        for idx, tsec, fr in frame_results:
            if not fr:
                timeline.append({"frame": idx, "time_sec": tsec, "prob": 0.5, "nr_faces": 0})
            else:
                ps = [scoring.score_face(s)["probability"] for (_, s) in fr]
                timeline.append({
                    "frame": idx,
                    "time_sec": tsec,
                    "prob": round(float(np.mean(ps)), 4),
                    "nr_faces": len(fr),
                })

        # 5. most suspicious frames -> annotated previews
        top = sorted([t for t in timeline if t["nr_faces"] > 0],
                     key=lambda x: x["prob"], reverse=True)[:5]
        top.reverse()
        frame_by_idx = {idx: bgr for idx, bgr in frames}
        suspicious_crops = []
        for t in top:
            bgr = frame_by_idx[t["frame"]].copy()
            for f, spatial in frame_results[timeline.index(t)][2]:
                res = scoring.score_face(spatial, None)
                bgr = draw_face_result(bgr, f.bbox, f.landmarks,
                                       res["probability"], res["indicators"])
            suspicious_crops.append({
                "frame": t["frame"],
                "time_sec": round(t["time_sec"], 2),
                "prob": t["prob"],
                "image": cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
            })

        return {
            "media_type": "video",
            "fps": reported_fps,
            "num_frames_sampled": len(frames),
            "duration_sec": round(len(frames) / reported_fps, 2),
            "overall": overall,
            "tracks": [
                {"track_id": t["track_id"], "probability": t["probability"],
                 "indicators": t["indicators"], "signals": t["signals"]}
                for t in per_track
            ],
            "timeline": timeline,
            "top_frames": suspicious_crops,
            "montage": (
                build_montage([c["image"] for c in suspicious_crops],
                              [c["prob"] for c in suspicious_crops])
                if suspicious_crops else None
            ),
            "summary": build_summary(overall, "video", len(frames)),
        }


def _load_frames(path_or_list, fps: float = 25.0):
    """Sample a video (or list of BGR frames) into <= MAX_SAMPLED_FRAMES frames."""
    if isinstance(path_or_list, (list, tuple)):
        frames = [(i, np.asarray(b)) for i, b in enumerate(path_or_list) if b is not None]
        if not frames:
            return [], fps
        total = len(frames)
        step = 1 if total <= MAX_SAMPLED_FRAMES else math.ceil(total / MAX_SAMPLED_FRAMES)
        return frames[::step], fps

    if not os.path.exists(path_or_list):
        raise FileNotFoundError(path_or_list)
    cap = cv2.VideoCapture(path_or_list)
    if not cap.isOpened():
        return [], fps
    real_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        total = 999999
    step = 1 if total <= MAX_SAMPLED_FRAMES else math.ceil(total / MAX_SAMPLED_FRAMES)
    frames = []
    idx = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if idx % step == 0:
            frames.append((idx, bgr))
        idx += 1
        if len(frames) >= MAX_SAMPLED_FRAMES:
            break
    cap.release()
    return frames, real_fps


def build_summary(overall: dict, media_type: str, n: int) -> str:
    v = overall["verdict"]
    p = round(overall["probability"] * 100)
    if media_type == "image":
        return (
            f"Image analysis: {v} with a synthetic-manipulation probability of "
            f"{p}% (confidence {overall['confidence']:.0f}%)."
        )
    return (
        f"Video analysis of {n} sampled frames: {v} with a synthetic-manipulation "
        f"probability of {p}% (confidence {overall['confidence']:.0f}%)."
    )
