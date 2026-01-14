# Day 02 — Hailo YOLOv8 Inference (Part A: Images)

## Objective

In this part, we run **YOLOv8s object detection on static images** using a Hailo-compiled `.hef` model on Raspberry Pi 5 + Hailo-8L.

This validates:

* End-to-end HailoRT inference
* Correct post-processing (NMS by class)
* Bounding box decoding and visualization

Live camera inference (Part B) is covered next.

---

## Folder Structure

```
day02_hailo_inference/
├── src/
│   ├── yolov8_hailo_infer.py
│   ├── preprocess.py
│   └── postprocess.py
├── outputs/
│   ├── test_images/
│   │   ├── bus.jpg
│   │   └── zidane.jpg
│   └── annotated/
└── README.md
```

---

## Inputs

### Test Images

Located in:

```
outputs/test_images/
```

Example files:

* `bus.jpg`
* `zidane.jpg`

These are standard YOLO validation images and allow deterministic testing.

---

## Running Image Inference (Part A)

From the `src/` directory:

```bash
python3 yolov8_hailo_infer.py \
  --hef ../../day01_model_and_compile/outputs/yolov8s.hef \
  --source ../outputs/test_images \
  --save-dir ../outputs/annotated \
  --score-thresh 0.35
```

Optional debug output:

```bash
--debug
```

---

## Expected Output

Terminal output similar to:

```
[1/2] bus.jpg: ~20 detections
[2/2] zidane.jpg: ~20 detections
Done. Processed 2 images in ~0.1s
```

Annotated images are saved to:

```
outputs/annotated/
```

Each image will contain:

* Bounding boxes
* Class labels
* Confidence scores

---

## Notes on Detection Counts

* Detection counts in the **20–25 range** at `score_thresh=0.35` are **normal and correct** for YOLOv8s.
* Earlier runs with thousands of detections indicated **incorrect NMS decoding**, which has now been fixed.

---

## Known Issues (Important)

### Segmentation Fault After Completion

You may see a message like:

```
Segmentation fault
```

**After** inference completes successfully.

✅ This does **not** affect:

* Detection results
* Saved images
* Model correctness

Cause:

* HailoRT Python bindings occasionally crash during teardown
* Related to stream/device cleanup order

Status:

* Safe to ignore for Day 02
* Will be stabilized later
* Live camera inference uses a different execution pattern

---

## What You Learned in Part A

By completing this section, you have:

* Loaded a Hailo `.hef` model on-device
* Created input/output VStreams
* Performed INT8 inference on Hailo-8L
* Decoded **Hailo NMS-by-class output**
* Produced annotated detection images

This confirms your **model + runtime + post-processing pipeline is correct**.

---

## Next Steps

### Part B (Next Session)

* Live inference using **Pi Camera**
* Continuous streaming inference
* FPS measurement
