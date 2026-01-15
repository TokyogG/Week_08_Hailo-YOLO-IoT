# Day 01 — Model Selection & Design Rationale

## Objective

Select a YOLO model variant suitable for **real-time edge deployment** on
**Raspberry Pi 5 + Hailo-8L**, balancing:

- Throughput (FPS)
- Latency
- Power consumption
- Detection accuracy
- Industry maturity / tooling support

Target input resolution is **640 × 640**.

---

## Candidate Models Considered

### 1. YOLOv8n (Nano)

| Property | Value |
|-------|------|
| Parameters | ~3.2M |
| Model Size (FP32) | ~6 MB |
| Typical INT8 Size | ~1.6 MB |
| Compute | Very low |
| Edge FPS (Hailo est.) | 45–60 FPS |
| Accuracy (COCO mAP) | Lowest in v8 family |

**Pros**
- Excellent for tight power budgets
- Highest FPS potential on edge NPUs
- Very fast compile and iteration cycles
- Ideal for event-driven detection (presence / absence)

**Cons**
- Lower recall on small objects
- Reduced robustness in cluttered scenes

**Use cases**
- Smart cameras
- Presence detection
- Safety triggers (person / vehicle detection)

---

### 2. YOLOv8s (Small)

| Property | Value |
|-------|------|
| Parameters | ~11.2M |
| Model Size (FP32) | ~22 MB |
| Typical INT8 Size | ~5–6 MB |
| Compute | Moderate |
| Edge FPS (Hailo est.) | 30–40 FPS |
| Accuracy (COCO mAP) | Strong balance |

**Pros**
- Best accuracy / performance trade-off in v8 family
- Much better small-object detection vs v8n
- Widely used in industry deployments
- Stable post-training INT8 behavior

**Cons**
- Lower FPS than v8n
- Slightly higher power draw

**Use cases**
- Industrial vision
- Railway / infrastructure inspection
- Multi-class detection with reliability requirements

---

### 3. YOLOv8m (Medium)

| Property | Value |
|-------|------|
| Parameters | ~25.9M |
| Model Size (FP32) | ~50 MB |
| Typical INT8 Size | ~12 MB |
| Compute | High |
| Edge FPS (Hailo est.) | 15–25 FPS |
| Accuracy (COCO mAP) | High |

**Pros**
- Strong accuracy, especially on small objects
- Suitable for complex scenes

**Cons**
- FPS may fall below real-time target
- Increased power and thermal load
- Marginal gains vs v8s for edge constraints

**Use cases**
- Edge servers
- Higher-power NPUs
- When accuracy > latency

---

## Why YOLOv8 (Not YOLOv11)

Although newer YOLO versions (e.g. v11) exist:

- YOLOv8 has **mature tooling support**
- Stable ONNX export paths
- Well-understood INT8 calibration behavior
- Proven compatibility with Hailo Dataflow Compiler

**Industry adoption typically lags model releases** due to:
- Validation cost
- Certification cycles
- Toolchain stability requirements

For production-style edge systems, **YOLOv8 remains the practical choice**.

---

## Selected Model

### ✅ YOLOv8s @ 640 × 640

**Rationale**
- Meets real-time FPS targets on Hailo-8L
- Strong accuracy for infrastructure / safety detection
- Widely deployed in industrial and embedded systems
- Better future-proofing than v8n with modest compute increase

This choice balances:
> **Accuracy × Latency × Power × Toolchain Maturity**

---

## Expected Deployment Characteristics

| Metric | Estimate |
|-----|-----|
| Input Resolution | 640 × 640 |
| Precision | INT8 |
| FPS (Hailo) | 30–40 FPS |
| End-to-end latency | <100 ms |
| Power Budget | <5 W total system |

---

## Notes

- Model compiled using Hailo Dataflow Compiler
- Calibration dataset required for optimal INT8 accuracy
- Post-processing includes NMS and confidence filtering
- Event-driven messaging (MQTT) minimizes bandwidth usage

This model selection defines the **performance envelope** for all downstream implementation work.

## Railway / Safety Vision Rationale (Why YOLOv8s)

Railway / safety vision is dominated by:
- high consequences for false negatives (missed hazards)
- variable lighting (snow, tunnels, glare)
- motion blur / vibration
- small-object detection (debris, cones, people at distance)

### Why v8s over v8n
YOLOv8n can be faster, but in safety contexts it tends to:
- miss smaller objects at range
- lose recall under blur/clutter
- become sensitive to threshold tuning

YOLOv8s is a safer deployment default because it:
- improves recall and robustness in complex scenes
- retains real-time performance on Pi5 + Hailo
- reduces “threshold babysitting” across environments

### How we will validate (proxy accuracy)
Instead of full COCO mAP, we will use:
- fixed test clips representative of rail/industrial environments
- consistency across lighting + motion conditions
- counts of false positives / missed detections on key classes

Decision rule:
> Choose the smallest model that meets real-time FPS while maintaining reliable recall on safety-critical classes.
