# Day 01 Notes — INT8 Calibration + Hailo Toolchain Quirks

## Goal of Day 01 (No Code)
Lock the “performance envelope” before writing runtime code:
- Choose model variant (core: YOLOv8s)
- Fix input resolution (640×640)
- Plan INT8 calibration expectations
- Compile to `.hef` and capture logs

---

## INT8 Calibration (Practical Notes)

### Why calibration matters
INT8 quantization compresses activations/weights. If your calibration set is unrepresentative, you’ll see:
- confidence collapse (fewer detections)
- biased detections (some classes vanish)
- increased false positives (noisy activations)
- “works on some scenes, fails on others”

### What a “good” calibration set looks like
For object detection, use ~100–1000 images that cover:
- lighting diversity (day/night/shadow)
- motion blur / vibration (realistic edge conditions)
- scale variety (near/far objects)
- background clutter (industrial scenes)

If you only calibrate on clean internet images, your real camera feed will behave worse.

### Calibration: fixed resolution implications
We’re standardizing on **640×640** because:
- simplest to compare across models
- common YOLO default
- balanced compute vs accuracy for edge

But remember:
- Smaller res (e.g., 416) boosts FPS, hurts small-object recall
- Larger res increases compute quickly, may blow the FPS budget

---

## Hailo-Specific “Quirks” / Engineering Notes

### 1) End-to-end FPS ≠ NPU FPS
Your measured FPS will include:
- camera capture / conversion
- pre-processing (resize, normalize)
- host↔device overhead
- post-processing (NMS)

Real-world success is **pipeline FPS**, not pure NPU throughput.

### 2) Post-processing can become your bottleneck
YOLO post-processing (especially NMS) can eat CPU if implemented poorly.
Plan to:
- keep thresholds sane (avoid thousands of candidates)
- prefer vectorized/Numpy operations
- measure postprocess time separately

### 3) Toolchain maturity differences show up as “friction”
Newer model versions can fail in subtle ways:
- export / ONNX graph ops not supported
- calibration instability
- postprocess mismatch (head format changes)

This is why industry adoption lags.

### 4) Always capture compile logs
Compile logs are part of your portfolio evidence:
- what worked
- what failed
- what you changed

Store logs in Day 01 outputs/ or paste key lines into this notes file.

---

## Decision Criteria (What we optimize for)
Core week: **YOLOv8s @ 640×640**
We prioritize:
1) Reliable detections (safety > speed)
2) Stable real-time performance (≥30 FPS)
3) Toolchain stability (repeatable build)

Optional extension days benchmark:
- YOLOv5s vs YOLOv8s (maturity vs modern baseline)
- YOLOv8n/v8m/v11 (edge limits + “new ≠ deployable” lesson)

---

## “Gotchas” Checklist (Before Day 02)
- `.hef` generated successfully
- model input shape confirmed (640×640)
- confirm expected output format for postprocess (boxes/objectness/classes)
- identify required preprocessing: color space (BGR/RGB), normalization, letterbox
- document thresholds you will start with:
  - conf threshold: 0.25 (typical starting point)
  - IoU threshold: 0.45 (typical starting point)

---

## What to record in model_info.md (minimum)
- chosen model + rationale
- expected FPS range on Pi5+Hailo
- why not YOLOv11 for core week
- planned calibration dataset characteristics

---

## Notes / Observations (fill during work)
- Compile command used:
- Compiler version:
- Model source (export method):
- Calibration dataset:
- Any warnings/errors:
- Anything that surprised you: