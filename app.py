"""Real-Time Deepfake & Synthetic Media Detection System - Dashboard.

Run:  streamlit run app.py
"""
from __future__ import annotations

import os
import sys
import io

import cv2
import numpy as np
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from deepfake.detector import DeepfakeDetector  # noqa: E402
from deepfake.augment import make_fake_image  # noqa: E402
from deepfake.visualization import timeline_chart  # noqa: E402

st.set_page_config(page_title="Deepfake Detection System",
                   page_icon="🛡️", layout="wide")

_ACCENT = "#e53935"
_GOOD = "#43a047"
_WARN = "#f9a825"


@st.cache_resource
def get_detector():
    return DeepfakeDetector()


def to_png(rgb: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return buf.tobytes()


def show_verdict(overall: dict):
    prob = overall["probability"] * 100
    verdict = overall["verdict"]
    conf = overall["confidence"]
    if prob >= 65:
        color = _ACCENT
    elif prob >= 40:
        color = _WARN
    else:
        color = _GOOD
    st.markdown(
        f"""
        <div style="border:3px solid {color};border-radius:14px;padding:18px 22px;
                    background:#0f1115;">
          <h3 style="color:{color};margin:0;">Verdict: {verdict}</h3>
          <p style="margin:6px 0 0;font-size:1.05rem;">
            Synthetic-manipulation probability: <b>{prob:.0f}%</b>
            &nbsp;&nbsp;•&nbsp;&nbsp; Confidence: {conf:.0f}%</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_indicators(face_results):
    st.subheader("🔍 Detected manipulation indicators")
    shown = set()
    for face in face_results:
        for ind in face.get("indicators", []):
            tag = ind.split("(")[0].strip()
            if tag in shown:
                continue
            shown.add(tag)
            st.markdown(f"- :red[**!**] {ind}")
    if not shown:
        st.markdown(f"- :green[No strong manipulation indicators detected for this media.]")


# --------------------------------------------------------------- sidebar
st.sidebar.title("🛡️ Deepfake Detection System")
st.sidebar.caption("Real-Time Deepfake & Synthetic Media Detection")
st.sidebar.markdown(
    "Explainable • Multimodal • Real-time\n\n"
    "**Signals used:** eye-blink regularity, motion jitter, facial symmetry, "
    "texture smoothing, boundary blending seams, high-frequency detail."
)
mode = st.sidebar.radio(
    "Analysis mode",
    ["Image Upload", "Video Upload", "Live Webcam", "Simulate Deepfake"],
)

detector = get_detector()

if mode in ("Image Upload", "Live Webcam", "Simulate Deepfake"):
    st.header("🖼️ Image & Still Analysis")
uploaded = None
if mode == "Image Upload":
    uploaded = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])
elif mode == "Live Webcam":
    uploaded = st.camera_input("Take a photo with your webcam")
elif mode == "Simulate Deepfake":
    uploaded = st.file_uploader("Upload a real photo first", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    raw = np.frombuffer(uploaded.getvalue(), np.uint8)
    image_bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)

    if mode == "Simulate Deepfake":
        faces = detector._fd.detect(image_bgr)
        if not faces:
            st.error("No face found in the photo - cannot simulate.")
            st.stop()
        image_bgr = make_fake_image(image_bgr, faces[0].bbox, strength=0.9)
        st.info("Applied simulated deepfake artifacts (smoothing + blending seam).")

    with st.spinner("Running deep-learning signal analysis..."):
        result = detector.analyze_image(image_bgr)
    st.image(cv2.cvtColor(result["annotated"], cv2.COLOR_BGR2RGB),
             caption="Annotated analysis", use_container_width=True)
    c1, c2 = st.columns([1, 1])
    with c1:
        show_verdict(result["overall"])
        st.markdown(result["summary"])
    with c2:
        show_indicators(result["faces"])
        if result["faces"]:
            st.markdown("**Raw signal breakdown**")
            st.json({k: v for k, v in result["faces"][0]["signals"].items()})
else:
    st.info("Upload an image or capture a webcam shot to begin.")

st.markdown("---")
st.markdown("##### 🔬 Explainability")
st.markdown(
    "Results are driven by interpretable computer-vision signals, not a black box. "
    "Each indicator listed on the left can be traced to a measured feature "
    "(see `deepfake/signals.py`), and trained CNN backbones plug into the same "
    "pipeline via the `DeepfakeDetector` model slots."
)

if mode == "Video Upload":
    st.header("🎥 Temporal Video Analysis")
    vid = st.file_uploader("Upload a video (MP4/AVI/MOV)", type=["mp4", "avi", "mov", "mkv"])
    if vid is not None:
        tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_upload.mp4")
        with open(tmp, "wb") as f:
            f.write(vid.getbuffer())
        with st.spinner("Analyzing frames (blinking, motion jitter, seams)..."):
            result = detector.analyze_video(tmp)
        if "error" in result:
            st.error(result["error"])
        else:
            show_verdict(result["overall"])
            st.markdown(result["summary"])
            st.caption(
                f"{result['num_frames_sampled']} sampled frames • "
                f"{result['duration_sec']} s analyzed at {result['fps']:.0f} fps"
            )
            colA, colB = st.columns([2, 1])
            with colA:
                st.subheader("📈 Frame-level probability timeline")
                chart = timeline_chart(result["timeline"])
                if chart is not None:
                    st.image(chart, use_container_width=True)
            with colB:
                st.subheader("🎯 Most suspicious frames")
                for top in result["top_frames"]:
                    st.image(top["image"], caption=f"t={top['time_sec']}s  {top['prob']*100:.0f}% synthetic")

            if result["montage"] is not None:
                st.subheader("🗂️ Suspicious-frame montage")
                st.image(result["montage"], use_container_width=True)

            st.subheader("👥 Face-track evidence")
            seen = set()
            for tr in result["tracks"]:
                for ind in tr["indicators"]:
                    if ind in seen:
                        continue
                    seen.add(ind)
                    st.markdown(f"- :red[**!**] {ind}")
            if not seen:
                st.markdown(f"- :green[No strong manipulation indicators across the video track.]")
    else:
        st.info("Upload a video file to run temporal consistency analysis.")

st.sidebar.markdown("---")
st.sidebar.caption(
    "Hackathon build • PyTorch-style modular pipeline (CPU-first), "
    "OpenCV + MediaPipe signals, FastAPI backend ready (`api.py`)."
)
