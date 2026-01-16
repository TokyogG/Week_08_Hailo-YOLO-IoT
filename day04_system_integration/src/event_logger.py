#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Optional

from .subscriber_base import SubConfig, MqttSubscriber, build_common_argparser
from .utils import fmt_ts_ms, now_ms


def main() -> int:
    ap = build_common_argparser("MQTT Event Logger (edge/+/events)")
    ap.add_argument("--topic", default="edge/+/events")
    ap.add_argument("--out", default="day04_system_integration/outputs/events.log")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def handler(topic: str, obj: Optional[dict], raw: bytes) -> None:
        ts = now_ms()
        if obj is None:
            line = f"{fmt_ts_ms(ts)} topic={topic} raw={raw[:200]!r}\n"
            print(line, end="")
            out_path.open("a", encoding="utf-8").write(line)
            return

        device = obj.get("device_id", "unknown")
        dets = obj.get("detections", [])
        names = []
        for d in dets:
            if isinstance(d, dict):
                names.append(d.get("class", f"id_{d.get('class_id', 'x')}"))

        counts = Counter(names)
        counts_str = ", ".join([f"{k}:{v}" for k, v in counts.items()]) if counts else "none"

        line = (
            f"{fmt_ts_ms(ts)} device={device} num={len(dets)} "
            f"counts=[{counts_str}] topic={topic}\n"
        )

        print(line, end="")
        out_path.open("a", encoding="utf-8").write(line)

    sub = MqttSubscriber(
        SubConfig(
            host=args.host,
            port=args.port,
            client_id="event-logger",
            topics=(args.topic,),
            qos=args.qos,
            verbose=args.verbose,
        ),
        on_message=handler,
    )

    print(f"[EVENTS] writing to {out_path}")
    return sub.run_forever()


if __name__ == "__main__":
    raise SystemExit(main())