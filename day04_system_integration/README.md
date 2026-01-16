## **Day 04 — System Integration & MQTT Subscribers**

### Objective

Consume MQTT messages produced by the edge inference pipeline (Day03) and build **desktop-side system components** for observability, logging, and integration.

Day04 validates that the edge system can be monitored, extended, and scaled **without modifying the edge device**.

---

### Architecture Overview

```
[ Raspberry Pi 5 + Hailo ]
   └─ Camera (Picamera2)
   └─ YOLOv8 INT8 Inference
   └─ Event Gate
   └─ MQTT Publisher
            │
            ▼
[ Desktop / Laptop ]
   └─ Mosquitto Broker
   └─ MQTT Subscribers
        ├─ Telemetry Logger
        ├─ Event Logger
        └─ Console Dashboard
```

The Pi publishes messages.
The Desktop consumes and reacts to them.

---

### What Runs Where

#### Raspberry Pi

* YOLOv8 inference
* MQTT publisher (Day03)
* Hardware-specific dependencies (Hailo, camera)

#### Desktop

* MQTT broker (Mosquitto)
* Subscribers and dashboards (Day04)
* No hardware dependencies

---

### MQTT Topics

| Topic                        | Purpose                     |
| ---------------------------- | --------------------------- |
| `edge/<device_id>/telemetry` | Health, FPS, frame counters |
| `edge/<device_id>/events`    | Event-driven detections     |
| `test/ping`                  | Startup / health check      |

Wildcard subscriptions (`edge/+/*`) are used on the desktop to support multiple devices.

---

### Components Built

#### `subscriber_base.py`

Reusable MQTT subscriber abstraction:

* Connect / reconnect handling
* Topic subscription
* JSON-safe decoding
* Clean SIGINT shutdown

---

#### `telemetry_logger.py`

Subscribes to telemetry messages and:

* Prints live FPS and frame counters
* Writes structured logs to disk

Used for health monitoring and performance tracking.

---

#### `event_logger.py`

Subscribes to detection events and:

* Logs detection counts
* Aggregates classes per event
* Produces an audit trail of activity

---

#### `simple_dashboard.py`

Lightweight terminal dashboard showing:

* Latest telemetry
* Latest detection event
* Detection class counts
* Rolling event rate

Designed for quick observability without GUI frameworks.

---

### Running Day04 (Desktop)

From the **Week_08_Hailo-YOLO-IoT repo root**:

```bash
PYTHONPATH=. python3 -m day04_system_integration.src.telemetry_logger --host localhost
```

```bash
PYTHONPATH=. python3 -m day04_system_integration.src.event_logger --host localhost
```

```bash
PYTHONPATH=. python3 -m day04_system_integration.src.simple_dashboard --host localhost
```

The Day03 publisher must be running on the Pi.

---

### Key Learnings

* Edge devices should **publish state**, not manage consumers
* MQTT enables fan-out without coupling producers and consumers
* Desktop services can evolve independently of edge firmware
* Observability is a first-class system requirement
* Clean shutdown matters on both publisher and subscriber sides

---

### Completion Criteria

Day04 is complete when:

* Multiple subscribers run simultaneously
* Desktop continues functioning if Pi restarts
* No edge code changes are required to add new consumers