# Edge AI Bootcamp — Week 08 Context (Authoritative)

Bootcamp:
- 16-week Edge AI Engineering Bootcamp
- Current week: Week 08 (YOLO + Hailo + MQTT + benchmarking)

Goal of Week 08:
- Build and validate an end-to-end edge vision system:
  Camera → Hailo inference → MQTT → subscribers → benchmarks

Hardware:
- Edge: Raspberry Pi 5 + Hailo-8L + Pi CSI camera (Picamera2)
- Desktop: Ubuntu PC (MQTT broker + subscriber benchmarks)

Canonical Architecture (DO NOT DUPLICATE):
- Inference: day02_hailo_inference/
- YOLO → MQTT publisher (canonical):
  day03_messaging_mqtt/src/publisher_from_yolo.py
- Topic definitions:
  day03_messaging_mqtt/src/topics.py
- Subscribers / dashboards:
  day04_system_integration/
- Benchmarking (subscriber-side only):
  day05_benchmarking_and_demo/

  Optional Day Intent:

- Day06 (Optional):
  Purpose: Compare YOLO model variants
  Scope: New ONNX → new HEF → same runtime + benchmarks
  Constraint: One variable change only

- Day07 (Optional):
  Purpose: Stress / stability testing
  Scope: Same HEF, longer runtime, more subscribers/logging
  Constraint: No inference or messaging changes


Messaging Semantics:
- Telemetry: edge/<device_id>/telemetry
  - Heartbeat + edge-reported FPS
  - Published ~every 2s
- Events: edge/<device_id>/events
  - Gated semantic detections
  - Primary benchmark signal

Benchmarking Rules:
- Benchmarks run on Desktop only
- Events benchmark ≠ inference FPS
- Telemetry FPS (~24–25 FPS) reported by edge
- Cross-machine latency logged but not treated as KPI

HEF Rules:
- .hef = model compiled + quantized + scheduled for Hailo
- Baseline HEF exists and must NOT be overwritten
- Optional days may introduce new HEFs as new files only

Baseline HEF:
- day01_model_and_compile/outputs/yolov8s.hef
- Verified end-to-end (camera → MQTT → benchmarks)


Current Status:
- Day05 complete: system benchmarked and documented
- System is demo-ready
- Optional Day06/Day07 remain

Current Day / Question:
- Optional Day06/Day07 .hef creation
- Optional Day06/Day07 Yolo variant benchmarking


---

# Week 08 — File Manifest (Authoritative)

> Purpose: Define ownership, intent, and stability of all files in Week 08
> This prevents duplication, accidental refactors, and context loss across chats.

---

## Root

* `README.md`
  **Week 08 overview** — goals, architecture, and day-by-day summary

* `assets/`
  Shared static assets (images, diagrams, etc.)

* `hailort.log`
  Hailo runtime log (debug / diagnostic only)

---

## Day 01 — Model Compilation (`day01_model_and_compile/`)

**Scope:** Model selection, compilation, and HEF generation
**Stability:** High (baseline artifacts should not be overwritten)

* `model_info.md`
  Model details (YOLO variant, input size, assumptions)

* `notes.md`
  Compilation notes, issues, and observations

* `outputs/`

  * `yolov8s.hef`
    ✅ **Baseline HEF (known-good, canonical)**

    * Compiled YOLOv8s model for Hailo-8L
    * Must not be overwritten
    * Used by all downstream days unless explicitly testing variants

---

## Day 02 — Hailo Inference (`day02_hailo_inference/`)

**Scope:** Camera → preprocess → Hailo inference → postprocess
**Stability:** High (core compute pipeline)

### `src/` (core inference code)

* `camera.py`
  Camera abstraction (Picamera2 / OpenCV)

* `preprocess.py`
  Image resize, normalization, tensor formatting

* `postprocess.py`
  YOLO output decoding → bounding boxes

* `yolo_infer.py`
  Local inference loop (no MQTT)

* `yolov8_hailo_infer.py`
  Main Hailo inference entry point (used by publisher)

* `camera_test.py`, `camera_test_picamera2.py`
  Camera validation scripts (debug only)

* `hailort.log`
  Runtime debug log (local to inference)

### `assets/`

* `coco.names`
  Class label definitions

### `outputs/`

* `annotated/`
  Saved detection images

* `test_images/`
  Static test inputs

* `live_demo/`
  Captured frames during live runs

* `fps_logs.csv`
  Edge-side FPS logs (historical)

---

## Day 03 — Messaging & Events (`day03_messaging_mqtt/`)

**Scope:** YOLO → MQTT publishing and topic semantics
**Stability:** **Very High (canonical system boundary)**

### `src/`

* `publisher_from_yolo.py`
  ✅ **Canonical YOLO → MQTT publisher**

  * Publishes telemetry and events
  * Must not be duplicated elsewhere

* `event_gate.py`
  Gating logic (confidence + minimum interval)

* `mqtt_client.py`
  Shared MQTT client wrapper

* `topics.py`
  Centralized topic definitions

  * `edge/<device_id>/telemetry`
  * `edge/<device_id>/events`

* `payload_schema.json`
  Reference schema for published messages

### Docs

* `README.md`
  Messaging design and usage

* `notes.md`
  Design notes and experiments

---

## Day 04 — System Integration (`day04_system_integration/`)

**Scope:** Subscribers, logging, dashboards
**Stability:** Medium (integration layer)

### `src/`

* `subscriber_base.py`
  Common MQTT subscriber abstraction

* `event_logger.py`
  Logs semantic detection events

* `telemetry_logger.py`
  Logs telemetry stream

* `simple_dashboard.py`
  Lightweight visualization

* `utils.py`
  Shared helpers

⚠️ Note:

* `yolo_publisher.py` is **not canonical**
  (Deprecated / experimental artifact from debugging; should not be used)

### Other

* `README.md`
  Integration overview

* `diagrams/`, `outputs/`, `notes.md`
  Supporting materials

---

## Day 05 — Benchmarking & Demo (`day05_benchmarking_and_demo/`)

**Scope:** Subscriber-side performance measurement
**Stability:** High (system evaluation)

* `benchmark_runner.py`
  MQTT subscriber benchmark tool

  * Measures event throughput
  * Logs CPU / RAM usage
  * Persists results

* `benchmarks.csv`
  Aggregated benchmark results

* `run_metadata.json`
  Per-run metadata

* `README.md`
  Benchmark methodology and interpretation

* `demo.md`
  Demo instructions

* `power.md`
  Power considerations (estimates / notes)

---

## Day 06 — Optional Model Benchmark (`day06_optional_yolo5_benchmark/`)

**Scope:** Optional model comparison
**Stability:** Experimental

* `benchmarks.csv`
  Comparative results (if run)

* `notes.md`
  Observations, trade-offs

⚠️ May introduce additional `.hef` files (kept separate from baseline)

---

## Day 07 — Optional Stress Test (`day07_optional_stress_test/`)

**Scope:** Long-run stability and reliability
**Stability:** Experimental

* `benchmarks.csv`
  Stress test metrics

* `notes.md`
  Runtime observations (memory drift, drops, etc.)

---

## Raw Tree
```

.
├── assets
├── day01_model_and_compile
│   ├── model_info.md
│   ├── notes.md
│   └── outputs
│       └── yolov8s.hef
├── day02_hailo_inference
│   ├── assets
│   │   └── coco.names
│   ├── __init__.py
│   ├── notes.md
│   ├── outputs
│   │   ├── annotated
│   │   │   ├── bus_det.jpg
│   │   │   └── zidane_det.jpg
│   │   ├── fps_logs.csv
│   │   ├── live_demo
│   │   │   ├── frame_000570.jpg
│   │   │   ├── frame_000600.jpg
│   │   │   └── frame_000630.jpg
│   │   └── test_images
│   │       ├── bus.jpg
│   │       └── zidane.jpg
│   ├── __pycache__
│   │   └── __init__.cpython-311.pyc
│   ├── README.md
│   └── src
│       ├── camera.py
│       ├── camera_test_picamera2.py
│       ├── camera_test.py
│       ├── hailort.log
│       ├── __init__.py
│       ├── postprocess.py
│       ├── preprocess.py
│       ├── __pycache__
│       │   ├── camera.cpython-311.pyc
│       │   ├── __init__.cpython-311.pyc
│       │   ├── postprocess.cpython-311.pyc
│       │   ├── preprocess.cpython-311.pyc
│       │   └── yolov8_hailo_infer.cpython-311.pyc
│       ├── yolo_infer.py
│       └── yolov8_hailo_infer.py
├── day03_messaging_mqtt
│   ├── __init__.py
│   ├── notes.md
│   ├── __pycache__
│   │   └── __init__.cpython-311.pyc
│   ├── README.md
│   └── src
│       ├── event_gate.py
│       ├── __init__.py
│       ├── mqtt_client.py
│       ├── payload_schema.json
│       ├── publisher_from_yolo.py
│       ├── __pycache__
│       │   ├── event_gate.cpython-311.pyc
│       │   ├── __init__.cpython-311.pyc
│       │   ├── mqtt_client.cpython-311.pyc
│       │   ├── publisher_from_yolo.cpython-311.pyc
│       │   └── topics.cpython-311.pyc
│       └── topics.py
├── day04_system_integration
│   ├── diagrams
│   ├── notes.md
│   ├── outputs
│   ├── README.md
│   └── src
│       ├── event_logger.py
│       ├── __init__.py
│       ├── __pycache__
│       │   ├── __init__.cpython-311.pyc
│       │   └── yolo_publisher.cpython-311.pyc
│       ├── simple_dashboard.py
│       ├── subscriber_base.py
│       ├── telemetry_logger.py
│       └── utils.py
├── day05_benchmarking_and_demo
│   ├── benchmark_runner.py
│   ├── benchmarks.csv
│   ├── demo.md
│   ├── power.md
│   ├── README.md
│   └── run_metadata.json
├── day06_optional_yolo5_benchmark
│   ├── benchmarks.csv
│   └── notes.md
├── day07_optional_stress_test
│   ├── benchmarks.csv
│   └── notes.md
├── hailort.log
└── README.md
```

## Notes
- day03 publisher is canonical
- baseline HEF must not be overwritten
- benchmarking is subscriber-side only

Documentation Convention:
- Each day uses README.md as the authoritative record
- notes.md files are deprecated and no longer used
