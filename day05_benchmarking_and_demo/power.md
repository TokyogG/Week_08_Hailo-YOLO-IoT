# Power Measurement Notes (Pi5 + Hailo)

## Goal
Measure total system power while running:
- YOLOv8s @ 640×640
- sustained inference for 5–10 minutes

## Recommended Method (choose one)
1) Inline USB-C power meter (best)
2) Smart plug with wattage (coarse)
3) Software-only estimates (least reliable)

## Measurement Protocol (keep consistent)
- Same camera source
- Same model resolution
- Same inference loop
- Same runtime duration
- Record ambient conditions if possible (thermal can change results)

## Results Table

| Run | Model | Resolution | Avg FPS | Avg Power (W) | Peak Power (W) | Notes |
|---|---|---|---:|---:|---:|---|
| 1 | YOLOv8s | 640×640 |  |  |  |  |

## Observations
- Did FPS degrade over time? (thermal throttling)
- Did power plateau or fluctuate?
- Any dropped frames / camera issues?
