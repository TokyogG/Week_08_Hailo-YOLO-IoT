# day02_hailo_inference/src/postprocess.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class Detection:
    class_id: int
    score: float
    # xyxy in original image pixel coordinates
    x1: float
    y1: float
    x2: float
    y2: float


def decode_hailo_nms_by_class(raw: np.ndarray,
                             num_classes: int = 80,
                             max_bboxes_per_class: int = 100,
                             score_thresh: float = 0.35):
    raw = np.asarray(raw, dtype=np.float32).reshape(-1)

    stride = 1 + 5 * max_bboxes_per_class
    expected = num_classes * stride
    if raw.size != expected:
        raise ValueError(f"Unexpected output size {raw.size}, expected {expected}")

    dets = []

    for c in range(num_classes):
        base = c * stride
        count_f = float(raw[base])
        if not np.isfinite(count_f):
            count = 0
        else:
            count = int(round(count_f))
            # If it's insane, treat as invalid rather than clamp
            if count < 0 or count > max_bboxes_per_class:
                count = 0
                
        boxes_start = base + 1
        for i in range(count):
            off = boxes_start + i * 5

            # Hailo NMS-by-class layout (common): score first
            score = float(raw[off + 0])
            ymin  = float(raw[off + 1])
            xmin  = float(raw[off + 2])
            ymax  = float(raw[off + 3])
            xmax  = float(raw[off + 4])

            if score < score_thresh:
                continue

            dets.append(Detection(
                class_id=c,
                score=score,
                x1=xmin, y1=ymin, x2=xmax, y2=ymax
            ))

    return dets


def clip_box_xyxy(x1, y1, x2, y2, w, h):
    x1 = max(0.0, min(float(w - 1), x1))
    y1 = max(0.0, min(float(h - 1), y1))
    x2 = max(0.0, min(float(w - 1), x2))
    y2 = max(0.0, min(float(h - 1), y2))
    return x1, y1, x2, y2


def scale_boxes_back_from_letterbox(
    dets: List[Detection],
    orig_w: int,
    orig_h: int,
    scale: float,
    pad_x: float,
    pad_y: float,
) -> List[Detection]:
    """
    Convert boxes from 640x640 letterboxed pixel coordinates back to original image coordinates.
    Assumes dets are in pixel coords for the 640x640 input.
    """
    out: List[Detection] = []
    for d in dets:
        # Undo padding then scaling
        x1 = (d.x1 - pad_x) / scale
        y1 = (d.y1 - pad_y) / scale
        x2 = (d.x2 - pad_x) / scale
        y2 = (d.y2 - pad_y) / scale
        x1, y1, x2, y2 = clip_box_xyxy(x1, y1, x2, y2, orig_w, orig_h)
        out.append(Detection(d.class_id, d.score, x1, y1, x2, y2))
    return out