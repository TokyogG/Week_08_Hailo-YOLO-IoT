#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from .subscriber_base import SubConfig, MqttSubscriber, build_common_argparser
from .utils import fmt_ts_ms, now_ms


def main() -> int:
    ap = build_common_argparser("MQTT Telemetry Logger (edge/+/telemetry)")
    ap.add_argument("--topic", default="edge/+/telemetry")
    ap.add_argument("--out", default="day04_system_integration/outputs/telemetry.log")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def handler(topic: str, obj: Optional[dict], raw: bytes) -> None:
        # Always log something; JSON decode may fail if someone publishes non-json
        ts = now_ms()
        if obj is None:
            line = f"{fmt_ts_ms(ts)} topic={topic} raw={raw[:200]!r}\n"
        else:
            device = obj.get("device_id", "unknown")
            fps = obj.get("fps", None)
            frame_id = obj.get("frame_id", None)
            line = f"{fmt_ts_ms(ts)} device={device} fps={fps} frame_id={frame_id} topic={topic}\n"

        print(line, end="")
        out_path.open("a", encoding="utf-8").write(line)

    sub = MqttSubscriber(
        SubConfig(
            host=args.host,
            port=args.port,
            client_id="telemetry-logger",
            topics=(args.topic,),
            qos=args.qos,
            verbose=args.verbose,
        ),
        on_message=handler,
    )

    print(f"[TELEMETRY] writing to {out_path}")
    return sub.run_forever()


if __name__ == "__main__":
    raise SystemExit(main())
