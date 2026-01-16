#!/usr/bin/env python3
from __future__ import annotations

import argparse
import socket
import time
import signal
from typing import Any, Optional

import cv2

from day02_hailo_inference.src.yolov8_hailo_infer import (
    HailoYoloRunner,
    load_labels,
    draw_dets,
)

from .mqtt_client import MqttConfig, MqttPublisher
from .event_gate import GateConfig, EventGate
from .topics import build_topics


# ---------- graceful shutdown ----------
_STOP = False


def _handle_sigint(signum, frame):
    global _STOP
    print("\n[SYS] SIGINT received — stopping loop cleanly...")
    _STOP = True


signal.signal(signal.SIGINT, _handle_sigint)
# --------------------------------------


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

    ap.add_argument("--picamera2", action="store_true")
    ap.add_argument("--camera", action="store_true")
    ap.add_argument("--debug", action="store_true")

    args = ap.parse_args()

    labels = load_labels(args.labels)

    topics = build_topics(args.device_id)
    topic_events = args.topic_events or topics["events"]
    topic_telemetry = args.topic_telemetry or topics["telemetry"]

    # MQTT
    pub = MqttPublisher(
        MqttConfig(
            host=args.broker,
            port=args.port,
            client_id=f"{args.device_id}-yolo",
        )
    )
    pub.connect()
    pub.publish_json("test/ping", {"msg": "publisher_from_yolo started"})

    gate = EventGate(
        GateConfig(
            min_interval_ms=args.min_interval_ms,
            min_score=args.min_event_score,
        )
    )

    use_picam2 = args.picamera2 or not args.camera
    cap: Optional[cv2.VideoCapture] = None
    picam2 = None

    frame_id = 0
    t0 = time.time()
    last_tel = 0.0

    try:
        # Camera init
        if use_picam2:
            from picamera2 import Picamera2

            picam2 = Picamera2()
            cfg = picam2.create_video_configuration(
                main={"format": "RGB888", "size": (args.cam_width, args.cam_height)}
            )
            picam2.configure(cfg)
            picam2.start()
            print("[CAM] Picamera2 started")
        else:
            cap = cv2.VideoCapture(args.camera_index)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cam_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_height)

        print("[RUNNER] initializing HailoYoloRunner")

        with HailoYoloRunner(args.hef, debug=False) as runner:
            print("[RUNNER] ready")

            while not _STOP:
                if use_picam2:
                    rgb = picam2.capture_array()
                    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                else:
                    ok, frame = cap.read()
                    if not ok:
                        break

                frame_id += 1
                dets, _ = runner.infer_bgr(frame, score_thresh=args.score_thresh)

                gate_in = [(d.class_id, d.score) for d in dets]
                if gate.should_publish(gate_in):
                    pub.publish_json(
                        topic_events,
                        {
                            "device_id": args.device_id,
                            "ts_ms": int(time.time() * 1000),
                            "frame_id": frame_id,
                            "num_detections": len(dets),
                            "detections": [det_to_dict(d, labels) for d in dets],
                        },
                    )

                now = time.time()
                if now - last_tel > 2.0:
                    fps = frame_id / max(now - t0, 1e-6)
                    pub.publish_json(
                        topic_telemetry,
                        {
                            "device_id": args.device_id,
                            "ts_ms": int(now * 1000),
                            "fps": fps,
                            "frame_id": frame_id,
                        },
                    )
                    last_tel = now

    finally:
        print("[SYS] shutting down resources...")
        if cap:
            cap.release()
        if picam2:
            picam2.stop()
        cv2.destroyAllWindows()
        pub.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
