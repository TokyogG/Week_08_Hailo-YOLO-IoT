#!/usr/bin/env python3
from __future__ import annotations

import argparse
import socket
import time
from typing import Any, Optional

import cv2


# Import Day02 runner + helpers (package-style)
from day02_hailo_inference.src.yolov8_hailo_infer import (
    HailoYoloRunner,
    load_labels,
    draw_dets,
)

from .mqtt_client import MqttConfig, MqttPublisher
from .event_gate import GateConfig, EventGate
from .topics import build_topics


def device_id_default() -> str:
    return socket.gethostname()


def det_to_dict(d, labels=None) -> dict[str, Any]:
    name = labels[d.class_id] if labels and d.class_id < len(labels) else f"class_{d.class_id}"
    return {
        "class_id": int(d.class_id),
        "class": name,
        "confidence": float(d.score),
        "bbox_xyxy": [float(d.x1), float(d.y1), float(d.x2), float(d.y2)],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hef", required=True)
    ap.add_argument("--labels", default=None)

    ap.add_argument("--broker", default="192.168.0.85")
    ap.add_argument("--port", type=int, default=1883)
    ap.add_argument("--device-id", default=device_id_default())
    ap.add_argument("--topic-events", default=None)
    ap.add_argument("--topic-telemetry", default=None)

    ap.add_argument("--score-thresh", type=float, default=0.25)
    ap.add_argument("--min-event-score", type=float, default=0.50)
    ap.add_argument("--min-interval-ms", type=int, default=250)

    ap.add_argument("--cam-width", type=int, default=1280)
    ap.add_argument("--cam-height", type=int, default=720)
    ap.add_argument("--camera-index", type=int, default=0)
    ap.add_argument("--display", action="store_true")

    ap.add_argument("--picamera2", action="store_true", help="Use Pi Camera via Picamera2 (recommended)")
    ap.add_argument("--camera", action="store_true", help="Use OpenCV VideoCapture (USB cam / V4L2)")
    ap.add_argument("--debug", action="store_true", help="Extra prints")

    args = ap.parse_args()

    labels = load_labels(args.labels)

    # Topics (allow overrides)
    default_topics = build_topics(args.device_id)
    topic_events = args.topic_events or default_topics["events"]
    topic_telemetry = args.topic_telemetry or default_topics["telemetry"]

    # MQTT
    pub = MqttPublisher(
        MqttConfig(
            host=args.broker,
            port=args.port,
            client_id=f"{args.device_id}-yolo",
            verbose=False,  # set True if you want publish spam
        )
    )
    pub.connect()

    # One-shot startup ping (proves publish path)
    print("[MQTT] sending startup ping...")
    ok = pub.publish_json("test/ping", {"msg": "publisher_from_yolo started"})
    print("[MQTT] startup ping published:", ok)

    # Event gating
    gate = EventGate(GateConfig(min_interval_ms=args.min_interval_ms, min_score=args.min_event_score))

    # Camera selection
    use_picam2 = args.picamera2 or (not args.camera)  # default to Picamera2 unless explicitly --camera
    cap: Optional[cv2.VideoCapture] = None
    picam2 = None

    # Shared counters
    frame_id = 0
    t0 = time.time()
    last_tel = 0.0

    try:
        # Init camera
        if use_picam2:
            from picamera2 import Picamera2

            picam2 = Picamera2()
            config = picam2.create_video_configuration(
                main={"format": "RGB888", "size": (args.cam_width, args.cam_height)}
            )
            picam2.configure(config)
            picam2.start()
            print("[CAM] Picamera2 started")
        else:
            cap = cv2.VideoCapture(args.camera_index)
            if not cap.isOpened():
                print("[CAM] Could not open OpenCV camera. Try --picamera2 or another --camera-index.")
                return 2
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cam_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_height)
            print(f"[CAM] OpenCV VideoCapture started index={args.camera_index}")

        # Init runner
        if args.debug:
            print("[RUNNER] about to init HailoYoloRunner")

        with HailoYoloRunner(args.hef, debug=False) as runner:
            if args.debug:
                print("[RUNNER] initialized, entering loop")

            while True:
                # Grab frame
                if use_picam2:
                    rgb = picam2.capture_array()
                    if rgb is None:
                        print("[CAM] capture_array returned None")
                        continue
                    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                else:
                    ok_cam, frame = cap.read()
                    if not ok_cam:
                        print("[CAM] cap.read() failed")
                        break

                frame_id += 1

                # Inference
                dets, _meta = runner.infer_bgr(frame, score_thresh=args.score_thresh)

                # Publish events only when gate says yes
                gate_in = [(d.class_id, d.score) for d in dets]
                if gate.should_publish(gate_in):
                    payload = {
                        "device_id": args.device_id,
                        "ts_ms": int(time.time() * 1000),
                        "frame_id": frame_id,
                        "num_detections": len(dets),
                        "detections": [det_to_dict(d, labels) for d in dets],
                    }
                    ok_evt = pub.publish_json(topic_events, payload, qos=0, retain=False)
                    if args.debug:
                        print("[MQTT] event publish:", ok_evt, "dets:", len(dets))

                # Telemetry every ~2 seconds (always, even with no dets)
                now = time.time()
                if now - last_tel > 2.0:
                    dt = now - t0
                    fps = frame_id / max(dt, 1e-6)
                    tel = {
                        "device_id": args.device_id,
                        "ts_ms": int(now * 1000),
                        "fps": float(fps),
                        "frame_id": int(frame_id),
                        "topic_events": topic_events,
                        "topic_telemetry": topic_telemetry,
                    }
                    ok_tel = pub.publish_json(topic_telemetry, tel, qos=0, retain=False)
                    if args.debug:
                        print("[MQTT] telemetry publish:", ok_tel, "fps:", fps)
                    last_tel = now

                # Optional display
                if args.display:
                    vis = draw_dets(frame, dets, labels=labels, max_draw=50)
                    cv2.imshow("YOLOv8 + MQTT", vis)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break

    except KeyboardInterrupt:
        print("\n[SYS] Ctrl+C received, shutting down...")
    finally:
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        try:
            if picam2 is not None:
                picam2.stop()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        pub.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
