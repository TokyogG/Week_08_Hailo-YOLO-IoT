#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
from collections import Counter
from typing import Optional

from .subscriber_base import SubConfig, MqttSubscriber, build_common_argparser
from .utils import RollingRate, fmt_ts_ms, now_ms


def clear():
    # Basic terminal clear (portable-ish)
    os.system("clear")


def main() -> int:
    ap = build_common_argparser("Simple Console Dashboard (telemetry + events)")
    ap.add_argument("--topic-telemetry", default="edge/+/telemetry")
    ap.add_argument("--topic-events", default="edge/+/events")
    ap.add_argument("--refresh-ms", type=int, default=500)
    args = ap.parse_args()

    state = {
        "last_tel": None,
        "last_evt": None,
        "last_evt_counts": Counter(),
        "last_evt_ts": None,
    }

    evt_rate = RollingRate(window_s=5.0)

    def handler(topic: str, obj: Optional[dict], raw: bytes) -> None:
        ts = now_ms()
        if obj is None:
            return

        if "/telemetry" in topic:
            state["last_tel"] = (ts, topic, obj)
        elif "/events" in topic:
            dets = obj.get("detections", [])
            names = []
            for d in dets:
                if isinstance(d, dict):
                    names.append(d.get("class", f"id_{d.get('class_id', 'x')}"))
            state["last_evt_counts"] = Counter(names)
            state["last_evt"] = (ts, topic, obj)
            state["last_evt_ts"] = ts
            evt_rate.tick()

    # We subscribe to BOTH topics with one client
    topics = (args.topic_telemetry, args.topic_events)

    sub = MqttSubscriber(
        SubConfig(
            host=args.host,
            port=args.port,
            client_id="simple-dashboard",
            topics=topics,
            qos=args.qos,
            verbose=args.verbose,
        ),
        on_message=handler,
    )

    # Run subscriber in background loop thread; we render in main thread
    # We'll use the same loop_start pattern from subscriber
    import threading

    def run_sub():
        sub.run_forever()

    t = threading.Thread(target=run_sub, daemon=True)
    t.start()

    try:
        while True:
            # clear()  # disabled: can interfere with Ctrl+C in some terminals
            print("=== Day04 Simple MQTT Dashboard ===")
            print(f"Broker: {args.host}:{args.port}")
            print(f"Topics: {args.topic_telemetry} | {args.topic_events}")
            print("")

            # Telemetry panel
            if state["last_tel"] is None:
                print("[Telemetry] (no messages yet)")
            else:
                ts, topic, obj = state["last_tel"]
                device = obj.get("device_id", "unknown")
                fps = obj.get("fps", "?")
                frame_id = obj.get("frame_id", "?")
                print("[Telemetry]")
                print(f"  time     : {fmt_ts_ms(ts)}")
                print(f"  device   : {device}")
                print(f"  fps      : {fps}")
                print(f"  frame_id : {frame_id}")
                print(f"  topic    : {topic}")

            print("")

            # Events panel
            if state["last_evt"] is None:
                print("[Events] (no events yet)")
            else:
                ts, topic, obj = state["last_evt"]
                device = obj.get("device_id", "unknown")
                num = obj.get("num_detections", len(obj.get("detections", [])))
                counts = state["last_evt_counts"]
                counts_str = ", ".join([f"{k}:{v}" for k, v in counts.items()]) if counts else "none"
                print("[Events]")
                print(f"  time       : {fmt_ts_ms(ts)}")
                print(f"  device     : {device}")
                print(f"  detections : {num}")
                print(f"  counts     : {counts_str}")
                print(f"  rate(5s)   : {evt_rate.rate_per_sec():.2f} events/sec")
                print(f"  topic      : {topic}")

            print("")
            print("Ctrl+C to exit")

            time.sleep(max(args.refresh_ms / 1000.0, 0.1))

    except (KeyboardInterrupt, SystemExit):
        print("\n[SYS] exiting dashboard...")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())