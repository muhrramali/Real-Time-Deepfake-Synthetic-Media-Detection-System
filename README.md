# 🛡️ Real-Time Deepfake & Synthetic Media Detection System

An AI-powered, explainable platform for detecting manipulated, AI-generated and
synthetic digital media — images and videos — in (near) real time.

> Built for the hackathon: fully self-contained, CPU-first, no external model
> downloads required. Every result ships with *why* it was flagged.

---

## ✨ Key Features

| Feature | Where |
|---|---|
| Real-time / near-real-time image analysis | `DeepfakeDetector.analyze_image` |
| Temporal video analysis (frame timeline) | `DeepfakeDetector.analyze_video` |
| Explainable confidence score (0–100%) | `deepfake/scoring.py` |
| Suspicious-frame visualization + highlighting | `deepfake/visualization.py` |
| Manipulation indicator list (natural-language) | `scoring.INDICATOR_LABELS` |
| Interactive dashboard (Streamlit) | `app.py` |
| REST API (FastAPI) | `api.py` |
| Modular model slots (swap backbones) | `deepfake/detector.py` |

### Detected manipulation indicators
- **Irregular blinking / missing natural eye blinks** — eye aspect ratio (EAR) curve
- **Unstable landmark motion (temporal jitter)** — per-track landmark speed std
- **Facial asymmetry / mirrored blending artifacts** — horizontal flip symmetry error
- **Over-smoothed texture** — Laplacian sharpness collapse
- **Face boundary blending seam** — gradient contrast on the face contour
- **Missing high-frequency detail** — FFT band-energy analysis

---

## 🚀 Quick Start

```bash
cd Hackathon2

# 1. install dependencies (venv recommended)
python -m pip install -r requirements.txt

# 2. launch the dashboard
streamlit run app.py
```

Windows one-click: double-click **`run.bat`**.

Video clip is analyzed in seconds on CPU (up to 360 sampled frames).

### Live demo without a model
```bash
python generate_sample.py samples/lena_real.jpg   # creates samples/fake_demo.mp4
```
Or in the dashboard, use the **"Simulate Deepfake"** tab: it injects realistic
artifacts into your own photo and shows the detector catching them.

Sample assets already included in `samples/`:
- `lena_real.jpg` — a real photo (scores **Likely Real**, ~18%)
- `fake_demo.mp4` — a simulated fake clip (timeline + suspicious frames)
You can reproduce the fake-still yourself with the `Simulate Deepfake` tab.

> ⚠️ Environment note: `mediapipe==0.10.21` requires `protobuf==4.25.3`
> (newer protobuf 5/6.x crashes MediaPipe with `MessageFactory.GetPrototype`
> errors). This pair is pinned in `requirements.txt`.

### REST API
```bash
uvicorn api:app --reload --port 8000
# interactive docs at http://127.0.0.1:8000/docs
# POST /analyze-image  (multipart 'file')
# POST /analyze-video  (multipart 'file')
```

---

## 🏗️ Architecture

```
                    ┌──────────────────────────────────────────┐
 upload/webcam ───► │  deepfake/                                │
                    │  face.py        MediaPipe+OpenCV face/468 │
                    │  signals.py     EAR, jitter, symmetry,    │
                    │                 smoothing, seam, FFT HF   │
                    │  scoring.py     weighted suspicion fusion │
                    │  detector.py    image & video pipelines   │
                    │  visualization  boxes, montage, timeline  │
                    └──────────────┬───────────────────────────┘
                                   │
                   ┌───────────────┴───────────────┐
                   ▼                               ▼
            app.py (Streamlit UI)         api.py (FastAPI)
            explainable dashboard         JSON + annotated PNG
```

**Model slots** — `DeepfakeDetector` separates *feature extraction* from
*scoring*; a trained CNN/ViT backbone (e.g. fine-tuned EfficientNet on
FaceForensics++) can be dropped into `scoring.score_face` as an additional
signal without redesigning the app.

## ✅ Hackathon deliverables map
- Project proposal → this folder's proposal PDF
- Working prototype → `deepfake/` pipeline
- Interactive dashboard → `app.py`
- Source code / modular model slots → `deepfake/`, `api.py`
- README / user guide → this file
- Evaluation smoke tests → `python tests/test_smoke.py`

## 📁 Structure
```
Hackathon2/
├── app.py                  Streamlit dashboard (main entry)
├── api.py                  FastAPI backend
├── deepfake/               detection package (modular)
│   ├── face.py  signals.py scoring.py detector.py
│   ├── visualization.py augment.py
├── generate_sample.py      creates sample fake media for demos
├── tests/test_smoke.py     pipeline smoke test
├── requirements.txt  run.bat  README.md
```

## ⚠️ Honest limitations (demo scope)
Heuristics catch *classic* generation/blending artifacts; adversarial or
high-quality commercial deepfakes need the trained-backbone model slots. This
build targets a self-contained, explainable hackathon demo.