"""Real-Time Deepfake & Synthetic Media Detection System.

Modular detection package:
  face.py          -> face detection + 468-point landmark extraction (MediaPipe + OpenCV)
  signals.py       -> per-frame spatial / ocular feature extraction
  scoring.py       -> explainable probability + verdict + indicator fusion
  detector.py      -> image / video analysis pipelines
  visualization.py -> annotated frames, montages, timeline charts
  augment.py       -> simulate deepfake artifacts for live demos
"""

from .detector import DeepfakeDetector

__all__ = ["DeepfakeDetector"]
__version__ = "1.0.0"