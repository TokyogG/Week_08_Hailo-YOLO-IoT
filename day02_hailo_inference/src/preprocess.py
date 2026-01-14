# day02_hailo_inference/src/preprocess.py
from __future__ import annotations
from typing import Tuple
import numpy as np
import cv2


def letterbox_640(
    bgr: np.ndarray,
    new_size: int = 640,
    color=(114, 114, 114),
) -> Tuple[np.ndarray, float, float, float]:
    """
    Letterbox an image to (new_size, new_size) keeping aspect ratio.

    Returns:
      img640 (BGR uint8 640x640),
      scale (float),
      pad_x (float),
      pad_y (float)
    """
    h, w = bgr.shape[:2]
    scale = min(new_size / w, new_size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))

    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((new_size, new_size, 3), color, dtype=np.uint8)
    pad_x = (new_size - nw) / 2
    pad_y = (new_size - nh) / 2

    x1 = int(round(pad_x))
    y1 = int(round(pad_y))
    canvas[y1:y1 + nh, x1:x1 + nw] = resized
    return canvas, scale, pad_x, pad_y


def bgr_to_hailo_input_flat_uint8(img640_bgr: np.ndarray) -> np.ndarray:
    """
    HEF expects UINT8 640x640x3. We will send as flat (1, 640*640*3) like Week06.
    Convert BGR->RGB then flatten.
    """
    rgb = cv2.cvtColor(img640_bgr, cv2.COLOR_BGR2RGB)
    flat = np.ascontiguousarray(rgb, dtype=np.uint8).reshape(1, -1)
    flat.setflags(write=1)
    return flat