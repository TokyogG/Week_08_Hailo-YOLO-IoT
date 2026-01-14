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
