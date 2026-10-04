"""FastAPI backend for the Deepfake Detection System.

Run:  uvicorn api:app --reload --port 8000
Docs: http://127.0.0.1:8000/docs

Endpoints:
  GET  /health
  POST /analyze-image        (multipart file "file") -> JSON + base64 annotated
  POST /analyze-video        (multipart file "file") -> JSON (timeline etc.)
"""
from __future__ import annotations

import base64
import io
import os
import sys

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from deepfake.detector import DeepfakeDetector  # noqa: E402

app = FastAPI(title="Deepfake Detection API", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)
detector = DeepfakeDetector()


def _b64(rgb: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf.tobytes()).decode()


def _to_bgr(upload: UploadFile) -> np.ndarray:
    data = np.frombuffer(upload.file.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image - send a valid JPG/PNG.")
    return img


@app.get("/health")
def health():
    return {"status": "ok", "service": "deepfake-detection"}


@app.post("/analyze-image")
def analyze_image(file: UploadFile = File(...)):
    try:
        bgr = _to_bgr(file)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    result = detector.analyze_image(bgr)
    result["annotated_b64"] = _b64(result["annotated"])
    result.pop("annotated", None)
    result["ok"] = True
    return result


@app.post("/analyze-video")
def analyze_video(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1] or ".mp4"
    tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_upload" + suffix)
    with open(tmp, "wb") as f:
        f.write(file.file.read())
    result = detector.analyze_video(tmp)
    result["ok"] = "error" not in result
    if result.get("montage") is not None:
        result["montage_b64"] = _b64(result["montage"])
    result.pop("montage", None)
    for t in result.get("top_frames", []):
        t["image_b64"] = _b64(t["image"])
        t.pop("image", None)
    return result