## Benchmark Protocol

1) Warm-up: run 60 seconds (do not record)
2) Record: run 5 minutes and log fps_logs.csv
3) Capture:
   - average FPS
   - p50/p95 latency (from total_ms)
   - CPU and RAM snapshot (top/htop)
   - power meter readings (avg + peak)

Success = stable FPS, stable detections, no pipeline stalls.
