## **Day 03 — MQTT Event-Driven Inference (Pi + Hailo)**

### Objective

Extend live YOLOv8 inference on Raspberry Pi + Hailo by publishing **event-driven detections** and **periodic telemetry** over MQTT to a desktop broker.

---

### Architecture

```
[ Pi 5 + Hailo ]
  └─ Picamera2
  └─ YOLOv8 INT8 (.hef)
  └─ Event Gate
  └─ MQTT Publisher
         │
         ▼
[ Desktop / Laptop ]
  └─ Mosquitto Broker
  └─ mosquitto_sub / dashboards
```

---

### What Runs Where

#### Raspberry Pi

* YOLOv8 inference via Hailo
* Picamera2 capture
* MQTT **publisher**

#### Desktop

* Mosquitto MQTT broker
* MQTT subscribers / dashboards

---

### Topics

| Topic                     | Description                   |
| ------------------------- | ----------------------------- |
| `edge/<device>/telemetry` | FPS, heartbeat, frame counter |
| `edge/<device>/events`    | Event-based detections        |
| `test/ping`               | Startup health check          |

---

### Installation

**Desktop**

```bash
sudo apt install mosquitto mosquitto-clients
mosquitto
```

**Pi**

```bash
pip install paho-mqtt opencv-python picamera2
```

---

### Run

**Desktop**

```bash
mosquitto_sub -h localhost -t '#' -v
```

**Pi**

```bash
PYTHONPATH=. python3 -m day03_messaging_mqtt.src.publisher_from_yolo \
  --broker <DESKTOP_IP> \
  --hef day01_model_and_compile/outputs/yolov8s.hef \
  --labels day02_hailo_inference/assets/coco.names \
  --picamera2
```

---

### Example Telemetry

```json
{
  "device_id": "raspberrypi",
  "fps": 24.2,
  "frame_id": 557
}
```

### Example Event

```json
{
  "num_detections": 2,
  "detections": [
    {"class": "person", "confidence": 0.55}
  ]
}
```

---

### Key Takeaways

* Event-driven messaging prevents flooding downstream systems
* Telemetry provides health + performance monitoring
* This pattern scales cleanly to cloud ingestion and dashboards

---

# 3️⃣ Concise Entry for **Week08 README.md**

Use this short block in your main Week08 summary:

---

### **Day 03 — MQTT Event-Driven Inference**

* Integrated YOLOv8 (Hailo) with MQTT messaging
* Raspberry Pi publishes:

  * **Telemetry** (FPS, heartbeat)
  * **Events** (object detections)
* Desktop runs Mosquitto broker + subscribers
* Implemented event gating to reduce message volume
* Achieved ~24 FPS sustained inference on Pi 5 + Hailo
