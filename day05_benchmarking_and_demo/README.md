# Day05 — Persistence, Benchmarking, and Demo Prep

## Objective

Stabilize the Week 08 edge AI system and produce **reproducible, interpretable performance metrics** for a real end-to-end pipeline (camera → inference → MQTT → subscriber).

This day focuses on **what the system actually emits**, not raw model throughput.

---

## System Snapshot

* **Edge hardware:** Raspberry Pi 5 + Hailo-8L
* **Model:** YOLO (compiled to Hailo HEF)
* **Camera:** Pi CSI camera (Picamera2)
* **Messaging:** MQTT (event + telemetry topics)
* **Publisher:** Day03 `publisher_from_yolo.py`
* **Benchmark host:** Desktop (subscriber-side measurement)

---

## Benchmark Methodology

Two MQTT streams are produced by the edge system:

* **Telemetry** (`edge/<device_id>/telemetry`)

  * Periodic heartbeat (~every 2s)
  * Reports edge-side FPS and system status
  * Control-plane signal

* **Events** (`edge/<device_id>/events`)

  * Semantic detections only
  * Gated by confidence threshold and minimum interval
  * Represents actionable outputs

### Benchmark Scope (Day05)

* **Primary benchmark target:** `events` topic
* **Duration:** 60 seconds
* **Warm-up:** 5 seconds
* **Metrics collected (subscriber-side):**

  * Event throughput (events/sec)
  * Message timing (wall-clock delta, see notes)
  * CPU and RAM usage on subscriber host

Raw inference FPS is **not** measured here; it is reported by the edge via telemetry.

---

## Results

### Events Benchmark (Semantic Detections)

| Metric                   | Value                  |
| ------------------------ | ---------------------- |
| Messages (total)         | 12                     |
| Messages (post-warmup)   | 10                     |
| Event rate (post-warmup) | **0.26 events/sec**    |
| Avg CPU (subscriber)     | **~3.1%**              |
| Avg RAM (subscriber)     | **~7.0 GB**            |
| Reported latency (avg)   | **-20 ms** (see notes) |

Interpretation:

* ~1 semantic detection every **3–4 seconds**
* Low event rate is **expected and intentional**, due to event gating
* Subscriber load is minimal; system scales comfortably on desktop side

---

### Telemetry Observations (Contextual)

* Telemetry published at a fixed interval (~2s)
* Telemetry-reported inference FPS observed via MQTT:

  * **~24–25 FPS** on the Pi 5 + Hailo-8L
* Telemetry throughput (~0.5 msg/sec) matches design

---

## Notes on Latency Measurement

* Producer timestamps are generated on the Pi
* Arrival timestamps are recorded on the Desktop
* Even with NTP enabled on both machines, small clock offsets (10–30 ms) can result in slightly negative wall-clock deltas

For this reason:

* Cross-device latency is **not treated as a reliable KPI** in Day05
* Latency is retained in logs for completeness but not highlighted as a performance metric

---

## Demo Instructions

1. **Desktop:** start MQTT broker

   ```bash
   mosquitto -p 1883 -v
   ```

2. **Pi 5:** run YOLO publisher

   ```bash
   python3 -m day03_messaging_mqtt.src.publisher_from_yolo \
     --hef day01_model_and_compile/outputs/yolov8s.hef \
     --broker <DESKTOP_IP> \
     --picamera2
   ```

3. **Desktop:** verify messages (optional)

   ```bash
   mosquitto_sub -h <DESKTOP_IP> -t "edge/#" -v
   ```

4. **Desktop:** run benchmark

   ```bash
   python3 day05_benchmarking_and_demo/benchmark_runner.py \
     --broker-host <DESKTOP_IP> \
     --topic-detect edge/raspberrypi/events \
     --duration-s 60 \
     --warmup-s 5
   ```

5. Inspect outputs:

   * `benchmarks.csv`
   * `run_metadata.json`
   * `raw_messages_<run_id>.jsonl`

---

## Lessons Learned

* **Benchmarking must match semantics**

  * Event streams and telemetry streams answer different questions
  * Measuring the wrong topic leads to misleading conclusions

* **Low FPS does not imply slow inference**

  * Event gating intentionally reduces output rate
  * Raw inference FPS is better surfaced via telemetry

* **Cross-machine latency is subtle**

  * Wall-clock deltas require clock-offset calibration to be meaningful
  * Subscriber-side metrics are still valuable for throughput and load

* **System architecture held up**

  * Clean separation between inference (Day02), messaging (Day03), integration (Day04), and benchmarking (Day05)
  * No need to modify the publisher to benchmark it
