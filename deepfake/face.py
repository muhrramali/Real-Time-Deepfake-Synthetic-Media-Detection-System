"""Face detection and 468-point landmark extraction.

Backed by MediaPipe FaceMesh and OpenCV. Each detected face is wrapped in a
FaceInfo object holding the bounding box, landmarks (pixels), the cropped face
image and a convex-hull mask (used by the boundary seam / blending analysis).
"""
from __future__ import annotations

import cv2
import numpy as np
import mediapipe as mp


class FaceInfo:
    __slots__ = ("bbox", "landmarks", "face_img", "gray_face", "hull", "center")

    def __init__(self, bbox, landmarks, face_img, gray_face, hull):
        # bbox = (x0, y0, x1, y1) integers
        self.bbox = bbox
        # landmarks: (468, 2) float32 pixels
        self.landmarks = landmarks
        self.face_img = face_img
        self.gray_face = gray_face
        self.hull = hull
        self.center = (
            (bbox[0] + bbox[2]) / 2.0,
            (bbox[1] + bbox[3]) / 2.0,
        )


def _to_pixels(landmarks, w, h):
    pts = np.array(
        [(lm.x * w, lm.y * h) for lm in landmarks], dtype=np.float32
    )
    return pts


class FaceDetector:
    """Detects faces and returns FaceInfo instances."""

    def __init__(self, max_faces: int = 4, min_detection_confidence: float = 0.5):
        self.max_faces = max_faces
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=max_faces,
            refine_landmarks=False,
            min_detection_confidence=min_detection_confidence,
        )

    def detect(self, bgr: np.ndarray) -> list[FaceInfo]:
        """Detect up to max_faces faces in a BGR image."""
        h, w = bgr.shape[:2]
        if h == 0 or w == 0:
            return []
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        results = self._mesh.process(rgb)
        faces: list[FaceInfo] = []
        if not results or not results.multi_face_landmarks:
            return faces

        for lm in results.multi_face_landmarks:
            pts = _to_pixels(lm.landmark, w, h)
            xs, ys = pts[:, 0], pts[:, 1]
            x0, x1 = float(xs.min()), float(xs.max())
            y0, y1 = float(ys.min()), float(ys.max())
            pad_x = (x1 - x0) * 0.25
            pad_y = (y1 - y0) * 0.25
            bb = (
                int(max(0, x0 - pad_x)),
                int(max(0, y0 - pad_y)),
                int(min(w, x1 + pad_x)),
                int(min(h, y1 + pad_y)),
            )
            if bb[2] - bb[0] < 8 or bb[3] - bb[1] < 8:
                continue
            face_img = bgr[bb[1] : bb[3], bb[0] : bb[2]]
            gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            hull = cv2.convexHull(pts.astype(np.int32))
            faces.append(FaceInfo(bb, pts, face_img, gray_face, hull))
        return faces