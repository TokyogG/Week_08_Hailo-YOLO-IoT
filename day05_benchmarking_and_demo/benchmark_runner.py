#!/usr/bin/env python3
"""
Day05 Benchmark Runner — Week_08_Hailo-YOLO-IoT

What it does
- Subscribes to MQTT topics (detections + optional telemetry)
- Runs an optional publisher command (e.g., your Hailo YOLO publisher)
- Collects:
  - FPS (overall + post-warmup)
  - End-to-end latency (if message contains a producer timestamp)
  - CPU/RAM stats of the local machine during the run
- Writes:
  - day05_benchmarking_and_demo/benchmarks.csv (appends one row per run)
  - day05_benchmarking_and_demo/run_metadata.json
  - day05_benchmarking_and_demo/raw_messages_<run_id>.jsonl

Assumptions / Message schema
- Messages are JSON (string payload)
- For latency, any of these fields will be used if present:
  - "ts" or "timestamp" or "t" (producer timestamp; seconds or milliseconds)
  - If milliseconds detected (> 1e12), auto-converts to seconds
- FPS is computed from message arrival times on the detection topic.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional, Dict, List, Tuple

try:
    import psutil  # type: ignore
except ImportError:
    psutil = None

try:
    import paho.mqtt.client as mqtt  # type: ignore
except ImportError:
    mqtt = None


# -----------------------------
# Helpers
# -----------------------------

def utc_now_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def parse_json_payload(payload: bytes) -> Optional[Dict[str, Any]]:
    try:
        txt = payload.decode("utf-8", errors="replace").strip()
        if not txt:
            return None
        return json.loads(txt)
    except Exception:
        return None


def extract_producer_ts_seconds(msg: dict) -> Optional[float]:
    # Prefer explicit ms field if present
    if "ts_ms" in msg:
        try:
            return float(msg["ts_ms"]) / 1000.0
        except Exception:
            pass

    # Otherwise accept seconds fields
    for k in ("ts", "timestamp", "t", "time"):
        if k in msg:
            try:
                v = float(msg[k])
                # Only convert if it's clearly ms
                if v > 1e12:
                    v = v / 1000.0
                return v
            except Exception:
                continue
    return None


def percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    xs = sorted(values)
    if p <= 0:
        return xs[0]
    if p >= 100:
        return xs[-1]
    k = (len(xs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    d0 = xs[f] * (c - k)
    d1 = xs[c] * (k - f)
    return d0 + d1


# -----------------------------
# Data model
# -----------------------------

@dataclass
class RunConfig:
    broker_host: str
    broker_port: int
    topic_detect: str
    topic_telemetry: Optional[str]
    duration_s: int
    warmup_s: int
    qos: int
    publisher_cmd: Optional[str]
    output_dir: str
    model_name: str
    hef_path: str
    notes: str


@dataclass
class RunResults:
    run_id: str
    started_utc: str
    finished_utc: str
    duration_s: int
    warmup_s: int

    messages_total: int
    messages_post_warmup: int

    fps_overall: Optional[float]
    fps_post_warmup: Optional[float]

    latency_ms_avg_post_warmup: Optional[float]
    latency_ms_p50_post_warmup: Optional[float]
    latency_ms_p95_post_warmup: Optional[float]
    latency_ms_p99_post_warmup: Optional[float]

    cpu_percent_avg: Optional[float]
    ram_mb_avg: Optional[float]

    broker_host: str
    broker_port: int
    topic_detect: str
    topic_telemetry: str

    model_name: str
    hef_path: str
    publisher_cmd: str
    notes: str


# -----------------------------
# Benchmark runner
# -----------------------------

class BenchmarkRunner:
    def __init__(self, cfg: RunConfig):
        if mqtt is None:
            raise RuntimeError(
                "Missing dependency: paho-mqtt. Install with:\n"
                "  pip install paho-mqtt"
            )
        self.cfg = cfg
        self.run_id = uuid.uuid4().hex[:10]
        self.started_utc = utc_now_iso()

        self._stop = False

        # Data collected
        self.arrival_times: List[float] = []
        self.latency_ms_post_warmup: List[float] = []
        self._first_arrival: Optional[float] = None

        # Sys stats
        self.cpu_samples: List[float] = []
        self.ram_samples_mb: List[float] = []

        self.raw_path = Path(cfg.output_dir) / f"raw_messages_{self.run_id}.jsonl"
        safe_mkdir(Path(cfg.output_dir))

        # Publisher process
        self.publisher_proc: Optional[subprocess.Popen] = None

        # MQTT client
        self.client = mqtt.Client()
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message

    def _on_connect(self, client, userdata, flags, rc):
        if rc != 0:
            print(f"[ERR] MQTT connect failed rc={rc}", file=sys.stderr)
            return
        client.subscribe(self.cfg.topic_detect, qos=self.cfg.qos)
        if self.cfg.topic_telemetry:
            client.subscribe(self.cfg.topic_telemetry, qos=self.cfg.qos)

    def _on_message(self, client, userdata, message):
        now = time.time()
        if message.topic != self.cfg.topic_detect:
            return

        if self._first_arrival is None:
            self._first_arrival = now

        self.arrival_times.append(now)

        payload_obj = parse_json_payload(message.payload)

        # Raw write (best-effort)
        try:
            rec = {
                "run_id": self.run_id,
                "arrival_ts": now,
                "topic": message.topic,
                "payload": payload_obj if payload_obj is not None else message.payload.decode("utf-8", errors="replace"),
            }
            with self.raw_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception:
            pass

        # Latency (only after warmup & only if producer ts exists)
        if payload_obj is None:
            return

        producer_ts = extract_producer_ts_seconds(payload_obj)
        if producer_ts is None:
            return

        # compute post-warmup window using arrival time
        if self._first_arrival is None:
            return
        if now < (self._first_arrival + self.cfg.warmup_s):
            return

        self.latency_ms_post_warmup.append((now - producer_ts) * 1000.0)

    def _start_publisher(self) -> None:
        if not self.cfg.publisher_cmd:
            return
        # Start publisher as a shell command so user can pass env vars easily
        self.publisher_proc = subprocess.Popen(
            self.cfg.publisher_cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        print(f"[INFO] Started publisher PID={self.publisher_proc.pid}")

    def _stop_publisher(self) -> None:
        if not self.publisher_proc:
            return
        try:
            self.publisher_proc.send_signal(signal.SIGINT)
        except Exception:
            pass

        # Give it a moment to exit
        try:
            self.publisher_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                self.publisher_proc.kill()
            except Exception:
                pass

        # Dump last lines of publisher output (optional)
        try:
            if self.publisher_proc.stdout:
                out = self.publisher_proc.stdout.read()
                if out:
                    tail = out.splitlines()[-20:]
                    print("[INFO] Publisher output (tail):")
                    for line in tail:
                        print("  " + line)
        except Exception:
            pass

    def _sample_system(self) -> None:
        if psutil is None:
            return
        # cpu_percent with interval=None returns last computed; call once first
        self.cpu_samples.append(psutil.cpu_percent(interval=None))
        mem = psutil.virtual_memory()
        self.ram_samples_mb.append((mem.total - mem.available) / (1024 * 1024))

    def run(self) -> RunResults:
        if psutil is not None:
            # prime cpu_percent
            psutil.cpu_percent(interval=None)

        # MQTT connect + loop
        self.client.connect(self.cfg.broker_host, self.cfg.broker_port, keepalive=60)
        self.client.loop_start()

        # Start publisher (optional)
        self._start_publisher()

        t0 = time.time()
        end_t = t0 + self.cfg.duration_s

        try:
            while time.time() < end_t and not self._stop:
                self._sample_system()
                time.sleep(0.5)
        finally:
            self._stop_publisher()
            self.client.loop_stop()
            self.client.disconnect()

        finished_utc = utc_now_iso()

        # Compute stats
        messages_total = len(self.arrival_times)

        # Determine post-warmup messages by arrival timestamp
        messages_post = 0
        fps_overall = None
        fps_post = None

        if messages_total >= 2:
            span = self.arrival_times[-1] - self.arrival_times[0]
            if span > 0:
                fps_overall = messages_total / span

        if self._first_arrival is not None:
            warm_cut = self._first_arrival + self.cfg.warmup_s
            post_times = [t for t in self.arrival_times if t >= warm_cut]
            messages_post = len(post_times)
            if len(post_times) >= 2:
                span_post = post_times[-1] - post_times[0]
                if span_post > 0:
                    fps_post = messages_post / span_post

        lat_avg = (sum(self.latency_ms_post_warmup) / len(self.latency_ms_post_warmup)) if self.latency_ms_post_warmup else None
        lat_p50 = percentile(self.latency_ms_post_warmup, 50) if self.latency_ms_post_warmup else None
        lat_p95 = percentile(self.latency_ms_post_warmup, 95) if self.latency_ms_post_warmup else None
        lat_p99 = percentile(self.latency_ms_post_warmup, 99) if self.latency_ms_post_warmup else None

        cpu_avg = (sum(self.cpu_samples) / len(self.cpu_samples)) if self.cpu_samples else None
        ram_avg = (sum(self.ram_samples_mb) / len(self.ram_samples_mb)) if self.ram_samples_mb else None

        res = RunResults(
            run_id=self.run_id,
            started_utc=self.started_utc,
            finished_utc=finished_utc,
            duration_s=self.cfg.duration_s,
            warmup_s=self.cfg.warmup_s,
            messages_total=messages_total,
            messages_post_warmup=messages_post,
            fps_overall=fps_overall,
            fps_post_warmup=fps_post,
            latency_ms_avg_post_warmup=lat_avg,
            latency_ms_p50_post_warmup=lat_p50,
            latency_ms_p95_post_warmup=lat_p95,
            latency_ms_p99_post_warmup=lat_p99,
            cpu_percent_avg=cpu_avg,
            ram_mb_avg=ram_avg,
            broker_host=self.cfg.broker_host,
            broker_port=self.cfg.broker_port,
            topic_detect=self.cfg.topic_detect,
            topic_telemetry=self.cfg.topic_telemetry or "",
            model_name=self.cfg.model_name,
            hef_path=self.cfg.hef_path,
            publisher_cmd=self.cfg.publisher_cmd or "",
            notes=self.cfg.notes,
        )
        return res


# -----------------------------
# Output writers
# -----------------------------

def write_metadata(out_dir: Path, res: RunResults, cfg: RunConfig) -> None:
    meta = {
        "run_id": res.run_id,
        "started_utc": res.started_utc,
        "finished_utc": res.finished_utc,
        "config": asdict(cfg),
        "results": asdict(res),
    }
    path = out_dir / "run_metadata.json"
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def append_benchmarks_csv(out_dir: Path, res: RunResults) -> None:
    path = out_dir / "benchmarks.csv"
    row = asdict(res)

    # Create file with header if missing
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


# -----------------------------
# CLI
# -----------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Week08 Day05 Benchmark Runner (MQTT + optional publisher)")
    p.add_argument("--broker-host", default="localhost")
    p.add_argument("--broker-port", type=int, default=1883)

    p.add_argument("--topic-detect", default="edgeai/detections")
    p.add_argument("--topic-telemetry", default=None)

    p.add_argument("--duration-s", type=int, default=60)
    p.add_argument("--warmup-s", type=int, default=5)
    p.add_argument("--qos", type=int, default=0, choices=[0, 1, 2])

    p.add_argument(
        "--publisher-cmd",
        default=None,
        help='Shell command to run your publisher during the benchmark. Example: '
             '"python3 day02_hailo_inference/src/yolov8_hailo_infer.py --mqtt ..."',
    )

    p.add_argument("--output-dir", default="day05_benchmarking_and_demo")
    p.add_argument("--model-name", default="yolo")
    p.add_argument("--hef-path", default="")
    p.add_argument("--notes", default="")

    return p


def main() -> int:
    args = build_argparser().parse_args()

    cfg = RunConfig(
        broker_host=args.broker_host,
        broker_port=args.broker_port,
        topic_detect=args.topic_detect,
        topic_telemetry=args.topic_telemetry,
        duration_s=args.duration_s,
        warmup_s=args.warmup_s,
        qos=args.qos,
        publisher_cmd=args.publisher_cmd,
        output_dir=args.output_dir,
        model_name=args.model_name,
        hef_path=args.hef_path,
        notes=args.notes,
    )

    out_dir = Path(cfg.output_dir)
    safe_mkdir(out_dir)

    print("[INFO] Starting benchmark")
    print(f"       run_id      : {uuid.uuid4().hex[:0]} (generated internally)")
    print(f"       broker      : {cfg.broker_host}:{cfg.broker_port}")
    print(f"       topic_detect: {cfg.topic_detect}")
    print(f"       duration    : {cfg.duration_s}s (warmup {cfg.warmup_s}s)")
    if cfg.publisher_cmd:
        print(f"       publisher   : {cfg.publisher_cmd}")

    runner = BenchmarkRunner(cfg)
    res = runner.run()

    # Persist
    write_metadata(out_dir, res, cfg)
    append_benchmarks_csv(out_dir, res)

    # Console summary
    print("\n=== Benchmark Summary ===")
    print(f"run_id: {res.run_id}")
    print(f"messages_total: {res.messages_total}")
    print(f"messages_post_warmup: {res.messages_post_warmup}")
    print(f"fps_overall: {res.fps_overall}")
    print(f"fps_post_warmup: {res.fps_post_warmup}")
    if res.latency_ms_avg_post_warmup is not None:
        print(f"latency_ms_avg_post_warmup: {res.latency_ms_avg_post_warmup:.2f}")
        print(f"latency_ms_p50_post_warmup: {res.latency_ms_p50_post_warmup:.2f}")
        print(f"latency_ms_p95_post_warmup: {res.latency_ms_p95_post_warmup:.2f}")
        print(f"latency_ms_p99_post_warmup: {res.latency_ms_p99_post_warmup:.2f}")
    else:
        print("latency: (no producer timestamp found in messages)")

    if res.cpu_percent_avg is not None:
        print(f"cpu_percent_avg: {res.cpu_percent_avg:.1f}")
    else:
        print("cpu_percent_avg: (psutil not installed)")

    if res.ram_mb_avg is not None:
        print(f"ram_mb_avg: {res.ram_mb_avg:.1f}")
    else:
        print("ram_mb_avg: (psutil not installed)")

    print("\nWrote:")
    print(f"- {out_dir / 'benchmarks.csv'}")
    print(f"- {out_dir / 'run_metadata.json'}")
    print(f"- {out_dir / f'raw_messages_{res.run_id}.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())