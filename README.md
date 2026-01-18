# Week 08 — YOLO Benchmarking & Edge Vision Deployment (Hailo)

Part of the **16-Week Edge AI Engineering Bootcamp**

---

## 🎯 Week Objective

Build a **production-style edge vision system** using:
- Raspberry Pi 5
- Hailo-8L accelerator
- YOLO object detection
- Event-based IoT messaging (MQTT)

This week emphasizes **engineering judgment**, not just model execution.

---

## 🧠 Core Learning Goals

- Understand YOLO model trade-offs under hardware constraints
- Deploy INT8 object detection on an edge accelerator
- Measure FPS, latency, power, and accuracy proxies
- Convert vision output into **events**, not video streams
- Make justified deployment decisions

---

## 📊 Target Metrics (Core)

| Metric | Target |
|------|------|
| YOLO FPS (Hailo) | ≥30 FPS |
| End-to-end latency | <100 ms |
| MQTT publish latency | <50 ms |
| Total system power | <5 W |
| Deliverable | Demo video + benchmark table |

---

## 📁 Folder Structure

```text
Week_08_Hailo-YOLO-IoT/
├── day01_model_and_compile/
├── day02_hailo_inference/
├── day03_messaging_mqtt/
├── day04_system_integration/
├── day05_benchmarking_and_demo/
├── day06_optional_yolo5_benchmark/
├── day07_optional_stress_test/
└── README.md

---

## ✅ Day 02 — Hailo Inference Validation (Images → Live Camera)

**Day 02 validates end-to-end YOLOv8 inference on Hailo-8L**, progressing from static image inference to real-time camera input on Raspberry Pi 5.

### What Was Accomplished

* Successfully ran **INT8 YOLOv8 inference** using a compiled `.hef` on Hailo-8L
* Implemented and debugged **Hailo NMS-by-class output decoding**
* Verified correct bounding boxes and confidence scores on test images
* Transitioned inference from images to **live Pi Camera (Picamera2)**
* Achieved **~15–25 FPS real-time object detection** on-device
* Confirmed HailoRT streaming, preprocessing, inference, and rendering pipeline

### Key Engineering Challenges (and Resolutions)

* **Excessive detections (1000+ boxes)**
  → Root cause: misinterpreted Hailo NMS output layout
  → Fixed by correct stride/count parsing and score filtering

* **Segmentation faults on exit**
  → Caused by HailoRT Python teardown order
  → Inference results valid; accepted as a known SDK limitation for now

* **Camera integration issues**
  → Resolved by switching from legacy `libcamera-hello` to `rpicam` + `Picamera2`

### Outcome

Day 02 confirms a **production-viable edge vision pipeline**:

> Camera → Preprocess → Hailo NPU → Postprocess → Visual Output

This establishes a solid foundation for **event-driven vision** in Day 03 (MQTT), where detections become messages instead of pixels.

---

### **Day 03 — Event-Driven YOLOv8 Inference via MQTT**

* Integrated live YOLOv8 INT8 inference on Raspberry Pi 5 with Hailo
* Implemented MQTT publisher for edge-to-system messaging
* Separated **telemetry** (FPS, heartbeat) from **events** (detections)
* Added event gating to reduce message volume and noise
* Verified end-to-end flow from Pi camera → inference → desktop broker
* Implemented clean shutdown and SIGINT handling for hardware safety

---

### **Day 04 — System Integration & MQTT Subscribers**

* Implemented desktop-side MQTT subscribers
* Added telemetry logger, event logger, and console dashboard
* Validated wildcard topic subscriptions (`edge/+/*`)
* Demonstrated clean separation between edge and system layers
* Established observability without modifying edge inference code

---

## 🧠 Why today *felt* lighter (and why that’s correct)

Day03 was hard because it crossed:

* hardware
* native runtimes
* Python packaging
* networking

Day04 is where things **snap into place**:

* no hardware debugging
* no segfaults
* no race conditions
* just system composition

That’s exactly how a good architecture feels once it’s right.

---

### **Day 05 — Benchmarking, Persistence, and Demo Readiness**

Day 05 focused on **measuring the system as deployed**, rather than modifying inference or messaging logic. The goal was to validate performance from the **subscriber’s point of view**, using the same MQTT streams consumed by downstream systems.

#### What Was Accomplished

* Implemented a **subscriber-side benchmark runner** to collect reproducible metrics
* Benchmarked **semantic event throughput** using the `edge/<device_id>/events` topic
* Differentiated between:

  * **Telemetry** (periodic heartbeat, system FPS)
  * **Events** (gated, meaningful detections)
* Persisted benchmark artifacts:

  * `benchmarks.csv`
  * `run_metadata.json`
  * raw MQTT message logs (`jsonl`)
* Verified that benchmarking does **not interfere** with edge inference or publishing

#### Key Results (Events Benchmark)

* **Event rate:** ~0.25 events/sec (≈ one event every 3–4 seconds)
* **Inference FPS (edge-reported via telemetry):** ~24–25 FPS
* **Subscriber CPU usage:** ~3%
* **Subscriber RAM usage:** ~7 GB
* **Latency:** Logged but not treated as a KPI due to cross-machine clock domain effects

Low event throughput is **intentional and correct**, reflecting confidence and interval gating rather than model speed.

#### Outcome

Day 05 confirms the system is **demo-ready and benchmarked as a system**, not a model:

> Camera → Hailo inference → MQTT events → subscribers → persisted metrics

This completes the Week 08 objective of building a **production-style, event-driven edge vision pipeline** with defensible performance data.

---
