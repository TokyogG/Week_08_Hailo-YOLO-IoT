from __future__ import annotations

from typing import Optional
import numpy as np

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None


class Camera:
    def __init__(self, index: int = 0):
        self.index = index
        self.cap = None
        if cv2 is not None:
            self.cap = cv2.VideoCapture(index)

    def read(self) -> Optional[np.ndarray]:
        if cv2 is None or self.cap is None:
            return None
        ok, frame = self.cap.read()
        if not ok:
            return None
        return frame

    def preprocess(self, frame: np.ndarray, out_w: int = 640, out_h: int = 640) -> np.ndarray:
        """
        Basic preprocessing:
        - Resize to out_w/out_h
        - Convert to float32
        - Normalize to [0,1]

        NOTE: Real YOLO pipelines often do letterboxing + RGB conversion.
        Keep this simple for now; refine once you confirm expected model preprocessing.
        """
        if cv2 is None:
            return frame.astype(np.float32) / 255.0

        resized = cv2.resize(frame, (out_w, out_h))
        # BGR -> RGB (often expected by YOLO exports; verify for your specific model)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        x = rgb.astype(np.float32) / 255.0
        # Add batch dim if needed (1, H, W, C) - depends on model
        x = np.expand_dims(x, axis=0)
        return x
