# Day 02 — Hailo YOLOv8 Inference

**Part A: Images · Part B: Live Pi Camera**

---

## Objective

Day 02 validates **end-to-end YOLOv8 inference on Hailo-8L**, progressing from static images to **real-time camera input** on Raspberry Pi 5.

By the end of this day, we confirm:

* HailoRT runtime is functional on-device
* YOLOv8 NMS-by-class output is decoded correctly
* Bounding boxes are rendered on images and live video
* Real-time FPS is achievable on Pi + Hailo

---

## Folder Structure

```
day02_hailo_inference/
├── src/
│   ├── yolov8_hailo_infer.py
│   ├── preprocess.py
│   ├── postprocess.py
│   ├── camera.py
│   └── yolo_infer.py
├── outputs/
│   ├── test_images/
│   │   ├── bus.jpg
│   │   └── zidane.jpg
│   ├── annotated/
│   └── live_demo/
├── notes.md
└── README.md
```

---

# Part A — Image Inference (Validation)

## Purpose

Part A verifies that:

* The compiled `.hef` runs correctly
* Post-processing (Hailo NMS by class) is decoded properly
* Bounding boxes and scores are sane

This step is **mandatory before live camera inference**.

---

## Test Images

Located in:

```
outputs/test_images/
```

Example files:

* `bus.jpg`
* `zidane.jpg`

---

## Run Command (Part A)

From `src/`:

```bash
python3 yolov8_hailo_infer.py \
  --hef ../../day01_model_and_compile/outputs/yolov8s.hef \
  --source ../outputs/test_images \
  --save-dir ../outputs/annotated \
  --score-thresh 0.35
```

Optional debug mode:

```bash
--debug
```

---

## Expected Output

Terminal:

```
[1/2] bus.jpg: ~20 detections
[2/2] zidane.jpg: ~20 detections
Done. Processed 2 images in ~0.1s
```

Files written to:

```
outputs/annotated/
```

---

## Notes on Detection Counts

* **~15–30 detections** at `score_thresh=0.35` is **correct**
* Earlier runs with thousands of detections were caused by:

  * Incorrect NMS decoding
  * Misinterpreting raw Hailo output buffers

This has now been resolved.

---

## Known Issue (Part A)

### Segmentation Fault After Exit

You may see:

```
Segmentation fault
```

**after inference completes successfully.**

✔ Results are valid
✔ Images are saved
✔ Can be ignored for Day 02

Cause:

* HailoRT Python teardown instability
* Stream/device cleanup order

We accept this for now and move on.

---

# Part B — Live Camera Inference (Pi Camera)

## Purpose

Part B moves from static images to **real-time video inference** using:

* Raspberry Pi Camera (IMX708)
* Picamera2
* Hailo-8L hardware acceleration

This is the **“wow” demo** for students.

---

## Camera Prerequisites

Verify camera is detected:

```bash
rpicam-hello --list-cameras
```

Expected output includes:

```
imx708 [4608x2592]
```

---

## Run Command (Part B)

From `src/`:

```bash
python3 yolov8_hailo_infer.py \
  --hef ../../day01_model_and_compile/outputs/yolov8s.hef \
  --camera \
  --picamera2 \
  --display \
  --score-thresh 0.25
```

Optional saving of frames:

```bash
--save-dir ../outputs/live_demo \
--save-every-n 30
```

---

## Live Output

* Camera window opens
* Bounding boxes rendered in **green**
* Class labels + confidence scores shown
* FPS printed in terminal (~15–20 FPS observed)

📸 Example screenshots (captured during testing):

* Live bounding boxes on face and object
* Stable real-time performance on Pi 5 + Hailo

---

## Notes on Bounding Boxes

* Boxes may appear large or duplicated at lower thresholds
* This is expected behavior for YOLOv8 at low confidence
* Adjust with:

  * `--score-thresh`
  * `--max-draw`

---

## Known Issues (Part B)

### 1. Segmentation Fault on Exit

Same as Part A — occurs **after** successful run.

Safe to ignore.

---

### 2. `q` Key Not Closing Window

Some Picamera2 windows do not capture keyboard focus.

Workaround:

```
CTRL+C
```

---

## Performance Observations

* ~15–20 FPS at 1280×720
* CPU usage remains low
* Hailo-8L is doing the heavy lifting
* Debug logging significantly reduces FPS

---

## What You Learned in Day 02

By completing Day 02, you have:

* Deployed a YOLOv8 model on Hailo-8L
* Performed real INT8 inference on-device
* Decoded Hailo NMS-by-class outputs
* Built a reusable inference pipeline
* Achieved real-time object detection on Raspberry Pi

This is **production-grade edge AI**