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

If you want, next we can also add:

* A **one-line Day 02 takeaway** (great for the very top of the README)
* Or a **“Why this matters”** paragraph aimed at interviews / students

You’re exactly where you should be.
