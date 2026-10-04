import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from deepfake.detector import DeepfakeDetector

D = DeepfakeDetector()
real = D.analyze_image(cv2.imread(os.path.join("samples", "lena_real.jpg")))
fake = D.analyze_image(cv2.imread(os.path.join("samples", "fake_still.png")))
print("REAL PHOTO ->", real["overall"]["verdict"], str(round(real["overall"]["probability"] * 100)) + "%")
print("FAKE STILL ->", fake["overall"]["verdict"], str(round(fake["overall"]["probability"] * 100)) + "%")
print("  indicators:", sorted({i for f in fake["faces"] for i in f["indicators"]}))