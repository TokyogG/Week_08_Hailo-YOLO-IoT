#!/usr/bin/env python3
"""
Day 02 — Hailo YOLO Inference Runtime (Stub)

Goals:
- Camera capture loop
- Preprocess to model input (640x640)
- Run inference (Hailo path or dry-run)
- Postprocess to detections
- Measure FPS and latency
- Log to CSV in ../outputs/fps_logs.csv

This file is intentionally scaffolded to support:
- real HailoRT inference (when integrated)
- dry-run mode for pipeline development
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from camera import Camera
from postprocess import postprocess_yolo, Detection


@dataclass
class PerfStats:
    frame_id: int
    fps: float
    preprocess_ms: float
    infer_ms: float
    postprocess_ms: float
    total_ms: float
    num_dets: int


def now_ms() -> float:
    return time.perf_counter() * 1000.0


def try_import_hailo():
    """
    Hailo APIs differ depending on your environment (HailoRT Python bindings, SDK, examples).
    We keep this flexible: integrate your known-good Week06 patterns here.
    """
    try:
        import hailo_platform  # type: ignore
        return hailo_platform
    except Exception:
        return None


def run_inference_stub(input_tensor: np.ndarray, dry_run: bool) -> np.ndarray:
    """
    Returns raw model outputs.

    In dry_run mode, we return a fake tensor with the right “shape-ish” structure so the
    rest of the pipeline can be built and timed.
    """
    if dry_run:
        # Fake output: [N, 85] style (x,y,w,h,obj,80 cls) - not exact, but enough for pipeline dev
        n = 200
        fake = np.random.rand(n, 85).astype(np.float32)
        # Boost some "objectness"
        fake[:, 4] = np.random.uniform(0.0, 1.0, size=(n,))
        return fake

    # TODO: Replace with real Hailo inference call.
    # Integrate the working approach you used in Week06 (hef + configured network group).
    raise NotImplementedError("Real Hailo inference not wired yet. Use --dry-run for now.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hef", type=str, default="", help="Path to compiled .hef (required for real mode)")
    ap.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--max-frames", type=int, default=300, help="Stop after N frames (0 = unlimited)")
    ap.add_argument("--dry-run", action="store_true", help="Run without Hailo (fake outputs) to validate pipeline")
    ap.add_argument("--csv-out", type=str, default="../outputs/fps_logs.csv")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.csv_out), exist_ok=True)

    hailo_mod = try_import_hailo()
    if not args.dry_run and hailo_mod is None:
        print("[WARN] Hailo Python module not found. Falling back to --dry-run behavior.")
        args.dry_run = True

    cam = Camera(index=args.camera)

    # CSV logging
    header = ["frame_id", "fps", "preprocess_ms", "infer_ms", "postprocess_ms", "total_ms", "num_dets"]
    with open(args.csv_out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        t_start = time.perf_counter()
        frame_id = 0
        last_fps_ts = time.perf_counter()

        while True:
            if args.max_frames and frame_id >= args.max_frames:
                break

            t0 = now_ms()
            frame = cam.read()
            if frame is None:
                print("[ERROR] Camera frame is None. Exiting.")
                break

            # Preprocess (basic): resize to 640x640 and normalize to [0,1]
            tp0 = now_ms()
            input_tensor = cam.preprocess(frame, out_w=args.width, out_h=args.height)
            tp1 = now_ms()

            # Inference
            ti0 = now_ms()
            raw = run_inference_stub(input_tensor, dry_run=args.dry_run)
            ti1 = now_ms()

            # Postprocess
            tpp0 = now_ms()
            dets: List[Detection] = postprocess_yolo(raw, conf_th=args.conf, iou_th=args.iou)
            tpp1 = now_ms()

            t1 = now_ms()

            # FPS (smoothed-ish)
            dt = time.perf_counter() - last_fps_ts
            if dt <= 0:
                fps = 0.0
            else:
                fps = 1.0 / dt
            last_fps_ts = time.perf_counter()

            stats = PerfStats(
                frame_id=frame_id,
                fps=fps,
                preprocess_ms=tp1 - tp0,
                infer_ms=ti1 - ti0,
                postprocess_ms=tpp1 - tpp0,
                total_ms=t1 - t0,
                num_dets=len(dets),
            )

            writer.writerow([
                stats.frame_id,
                f"{stats.fps:.2f}",
                f"{stats.preprocess_ms:.2f}",
                f"{stats.infer_ms:.2f}",
                f"{stats.postprocess_ms:.2f}",
                f"{stats.total_ms:.2f}",
                stats.num_dets,
            ])

            # Minimal console output
            if frame_id % 30 == 0:
                print(
                    f"[frame {frame_id}] fps={stats.fps:.1f} "
                    f"pre={stats.preprocess_ms:.1f}ms infer={stats.infer_ms:.1f}ms "
                    f"post={stats.postprocess_ms:.1f}ms dets={stats.num_dets}"
                )

            frame_id += 1

    elapsed = time.perf_counter() - t_start
    print(f"Done. Logged to {args.csv_out}. Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
