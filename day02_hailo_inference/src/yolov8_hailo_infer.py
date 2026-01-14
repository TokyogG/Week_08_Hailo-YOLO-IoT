#!/usr/bin/env python3
# day02_hailo_inference/src/yolov8_hailo_infer.py
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import cv2
import signal

from hailo_platform import (
    HEF,
    VDevice,
    HailoStreamInterface,
    ConfigureParams,
    InputVStreamParams,
    OutputVStreamParams,
    FormatType,
)

from preprocess import letterbox_640, bgr_to_hailo_input_flat_uint8
from postprocess import (
    decode_hailo_nms_by_class,
    scale_boxes_back_from_letterbox,
    Detection,
)

INPUT_NAME = "yolov8s/input_layer1"
OUTPUT_NAME = "yolov8s/yolov8_nms_postprocess"

NUM_CLASSES = 80
MAX_BBOXES_PER_CLASS = 100
INPUT_SIZE = 640


def load_labels(path: Optional[str]) -> Optional[List[str]]:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

def draw_dets(
    bgr: np.ndarray,
    dets: List[Detection],
    labels: Optional[List[str]] = None,
    max_draw: int = 50,
) -> np.ndarray:
    out = bgr.copy()
    dets_sorted = sorted(dets, key=lambda d: d.score, reverse=True)[:max_draw]
    for d in dets_sorted:
        x1, y1, x2, y2 = map(int, [d.x1, d.y1, d.x2, d.y2])
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        name = labels[d.class_id] if labels and d.class_id < len(labels) else f"class_{d.class_id}"
        txt = f"{name} {d.score:.2f}"
        cv2.putText(out, txt, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return out

def safe_recv(output_stream, timeout_ms: int = 5000):
    """
    Try to recv with a timeout. If timeout occurs, raise TimeoutError.
    This prevents hanging forever and reduces the need for Ctrl+C mid-recv.
    """
    t0 = time.time()
    while True:
        try:
            return output_stream.recv()
        except Exception as e:
            # If HailoRT exposes timeout exceptions differently, we just loop briefly.
            if (time.time() - t0) * 1000 > timeout_ms:
                raise TimeoutError(f"recv() timed out after {timeout_ms} ms") from e
            time.sleep(0.01)

class HailoYoloRunner:
    def __init__(self, hef_path: str, debug: bool = False):
        self.hef_path = hef_path
        self.hef = HEF(hef_path)
        self.debug = debug

        self.vdevice = None
        self.configured = None
        self.in_vs = None
        self.out_vs = None
        self.input_stream = None
        self.output_stream = None
        self._activation_ctx = None

        self.in_params = None
        self.out_params = None

    def __enter__(self):
        self.vdevice = VDevice().__enter__()
        cfg = ConfigureParams.create_from_hef(self.hef, interface=HailoStreamInterface.PCIe)
        self.configured = self.vdevice.configure(self.hef, cfg)[0]

        if self.debug:
            print("\nConfiguredNetwork API:")
            
        for name in dir(self.configured):
            if "stream" in name.lower() or "activate" in name.lower():
                print(" ", name)
        print()

        self.in_params = InputVStreamParams.make_from_network_group(
            self.configured, quantized=True, format_type=FormatType.UINT8
        )
        self.out_params = OutputVStreamParams.make_from_network_group(
            self.configured, quantized=False, format_type=FormatType.FLOAT32
        )

        # Create vstreams ONCE
        self.in_vs = self.configured._create_input_vstreams(self.in_params)
        self.out_vs = self.configured._create_output_vstreams(self.out_params)

        self.input_stream = self.in_vs.get_input_by_name(INPUT_NAME)
        self.output_stream = self.out_vs.get_output_by_name(OUTPUT_NAME)

        # Activate ONCE (for batch loops)
        self._activation_ctx = self.configured.activate()
        self._activation_ctx.__enter__()

        if self.debug:
            print("HEF:", self.hef_path)
            print("Input stream:", INPUT_NAME)
            print("Output stream:", OUTPUT_NAME)

        return self

    def __exit__(self, exc_type, exc, tb):
        # Close streams first
        try:
            if self.out_vs is not None:
                self.out_vs.close()
        except Exception:
            pass
        try:
            if self.in_vs is not None:
                self.in_vs.close()
        except Exception:
            pass

        # Then deactivate (if you entered activation)
        try:
            if self._activation_ctx is not None:
                self._activation_ctx.__exit__(None, None, None)
        except Exception:
            pass

        # Then close vdevice LAST
        try:
            if self.vdevice is not None:
                self.vdevice.__exit__(None, None, None)
        except Exception:
            pass

        self.vdevice = None
        self.configured = None
        self.in_vs = None
        self.out_vs = None
        self.input_stream = None
        self.output_stream = None
        self._activation_ctx = None

    def infer_bgr(self, frame_bgr: np.ndarray, score_thresh: float) -> Tuple[List[Detection], dict]:
        orig_h, orig_w = frame_bgr.shape[:2]

        img640, scale, pad_x, pad_y = letterbox_640(frame_bgr, new_size=INPUT_SIZE)
        inp = bgr_to_hailo_input_flat_uint8(img640)

        self.input_stream.send(inp)
        raw = safe_recv(self.output_stream, timeout_ms=5000)
        raw_np = np.asarray(raw, dtype=np.float32).reshape(-1)

        if self.debug and not getattr(self, "_did_probe", False):
            self._did_probe = True

            num_classes = NUM_CLASSES
            max_bboxes = MAX_BBOXES_PER_CLASS
            stride = 1 + 5 * max_bboxes

            if raw_np.size == num_classes * stride:
                arr = raw_np.reshape(num_classes, stride)
                counts_f = arr[:, 0]
                boxes = arr[:, 1:].reshape(num_classes, max_bboxes, 5)

                # Use nan-safe stats because unused slots may contain NaN
                print("counts(min/max):",
                    float(np.nanmin(counts_f)), float(np.nanmax(counts_f)),
                    "sum:", float(np.nansum(counts_f)))

                for j in range(5):
                    vals = boxes[:, :, j].reshape(-1)
                    print(f"box[{j}] min/max:",
                        float(np.nanmin(vals)), float(np.nanmax(vals)))
            else:
                print("Probe skipped: unexpected output size:", raw_np.size)

        if self.debug:
            print("RAW:", type(raw_np), "len:", raw_np.size, "dtype:", raw_np.dtype)
            print("RAW head:", raw_np[:20])

        dets_640 = decode_hailo_nms_by_class(
            raw_np,
            num_classes=NUM_CLASSES,
            max_bboxes_per_class=MAX_BBOXES_PER_CLASS,
            score_thresh=score_thresh,
        )

        normalized = False
        if dets_640:
            coords_max = max(max(d.x1, d.y1, d.x2, d.y2) for d in dets_640)
            if coords_max <= 1.5:
                normalized = True
                for d in dets_640:
                    d.x1 *= INPUT_SIZE
                    d.y1 *= INPUT_SIZE
                    d.x2 *= INPUT_SIZE
                    d.y2 *= INPUT_SIZE

        dets_orig = scale_boxes_back_from_letterbox(dets_640, orig_w, orig_h, scale, pad_x, pad_y)

        meta = {"orig_w": orig_w, "orig_h": orig_h, "scale": scale, "pad_x": pad_x, "pad_y": pad_y, "normalized": normalized}
        return dets_orig, meta

def iter_images(path: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if path.is_file():
        return [path]
    files = [p for p in sorted(path.glob("*")) if p.suffix.lower() in exts]
    return files


def run_on_images(args) -> int:
    src = Path(args.source)
    imgs = iter_images(src)
    if not imgs:
        print(f"No images found in {src}")
        return 2

    labels = load_labels(args.labels)
    outdir = Path(args.save_dir) if args.save_dir else None
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)

    with HailoYoloRunner(args.hef, debug=args.debug) as runner:
        t0 = time.time()
        for i, img_path in enumerate(imgs, 1):
            bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if bgr is None:
                print(f"Skipping unreadable image: {img_path}")
                continue

            dets, meta = runner.infer_bgr(bgr, score_thresh=args.score_thresh)

            if args.debug:
                print(f"[{i}/{len(imgs)}] {img_path.name}: {len(dets)} dets (normalized={meta['normalized']})")

            vis = draw_dets(bgr, dets, labels=labels, max_draw=args.max_draw)

            if args.display:
                cv2.imshow("YOLOv8s + Hailo", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if outdir:
                out_path = outdir / f"{img_path.stem}_det.jpg"
                cv2.imwrite(str(out_path), vis)

        dt = time.time() - t0
        print(f"Done. Processed {len(imgs)} images in {dt:.2f}s")
    return 0


def run_on_camera(args) -> int:
    labels = load_labels(args.labels)
    outdir = Path(args.save_dir) if args.save_dir else None
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)

    # Camera: default /dev/video0. For Pi Camera (libcamera), many setups expose via v4l2.
    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print("Could not open camera. Try --camera-index 0/1 or use --video /path/to.mp4")
        return 2

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_height)

    frame_count = 0
    t0 = time.time()

    with HailoYoloRunner(args.hef, debug=args.debug) as runner:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_count += 1
            dets, _ = runner.infer_bgr(frame, score_thresh=args.score_thresh)
            vis = draw_dets(frame, dets, labels=labels, max_draw=args.max_draw)

            if args.display:
                cv2.imshow("YOLOv8s + Hailo (Camera)", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if outdir and (frame_count % args.save_every_n == 0):
                out_path = outdir / f"frame_{frame_count:06d}.jpg"
                cv2.imwrite(str(out_path), vis)

            if args.fps_every > 0 and (frame_count % args.fps_every == 0):
                dt = time.time() - t0
                fps = frame_count / max(dt, 1e-6)
                print(f"Frames: {frame_count} | FPS: {fps:.2f} | last dets: {len(dets)}")

    cap.release()
    cv2.destroyAllWindows()
    return 0


def run_on_video(args) -> int:
    labels = load_labels(args.labels)
    outdir = Path(args.save_dir) if args.save_dir else None
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Could not open video: {args.video}")
        return 2

    frame_count = 0
    t0 = time.time()

    with HailoYoloRunner(args.hef, debug=args.debug) as runner:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_count += 1

            dets, _ = runner.infer_bgr(frame, score_thresh=args.score_thresh)
            vis = draw_dets(frame, dets, labels=labels, max_draw=args.max_draw)

            if args.display:
                cv2.imshow("YOLOv8s + Hailo (Video)", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if outdir and (frame_count % args.save_every_n == 0):
                out_path = outdir / f"frame_{frame_count:06d}.jpg"
                cv2.imwrite(str(out_path), vis)

            if args.fps_every > 0 and (frame_count % args.fps_every == 0):
                dt = time.time() - t0
                fps = frame_count / max(dt, 1e-6)
                print(f"Frames: {frame_count} | FPS: {fps:.2f} | last dets: {len(dets)}")

    cap.release()
    cv2.destroyAllWindows()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hef", required=True, help="Path to yolov8s.hef")
    ap.add_argument("--labels", default=None, help="Optional labels file (COCO names)")
    ap.add_argument("--score-thresh", type=float, default=0.25, help="Score threshold")
    ap.add_argument("--max-draw", type=int, default=50, help="Max boxes to draw")
    ap.add_argument("--display", action="store_true", help="Display window (requires GUI)")
    ap.add_argument("--save-dir", default=None, help="Save annotated outputs here")

    # Mode selection
    ap.add_argument("--source", default=None, help="Image file or directory (Mode A)")
    ap.add_argument("--camera", action="store_true", help="Use camera (Mode B)")
    ap.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index")
    ap.add_argument("--cam-width", type=int, default=1280)
    ap.add_argument("--cam-height", type=int, default=720)
    ap.add_argument("--video", default=None, help="Video file path (optional Mode B alt)")
    ap.add_argument("--save-every-n", type=int, default=30, help="Save every N frames (camera/video)")
    ap.add_argument("--fps-every", type=int, default=60, help="Print FPS every N frames (0 disables)")

    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    if args.video:
        return run_on_video(args)

    if args.camera:
        return run_on_camera(args)

    if args.source:
        return run_on_images(args)

    ap.error("Choose one mode: --source <image_or_dir> OR --camera OR --video <path>")

    import gc, time
    gc.collect()
    time.sleep(0.1)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
