# Phase 3 Progress Report: Dual-Camera Stress Testing & Optimization

---

## ✅ Step 1 — Dual-Camera Source Setup
- **`app.py`**: Both `SentinelStream` instances now point at distinct ~500MB `.MOV` files.
  - **CAM-01** → `videos/vid1-angle1.MOV`
  - **CAM-02** → `videos/vid2-angle2.MOV`
- **Bug Fix**: `cam02-video` in the dashboard HTML was hardcoded to `/cam1_frame`. Fixed to correctly fetch `/cam2_frame`.

---

## ✅ Step 2 — Frontend Frame Delivery Overhaul
- **`templates/index.html`**: Replaced blind `setInterval` polling with an **async image loader**.
  - The browser now waits for each frame to fully download before requesting the next.
  - This eliminated the startup lag and network flood that was causing the initial "frozen" playback.
- **Maximize Modal**: Fixed to dynamically show the correct camera feed (CAM-01 or CAM-02) using a `data-cam` attribute.

---

## ✅ Step 3 — Real-Time Synchronization (Frame Skipping)
- **`vision_engine.py`**: Added dynamic frame-dropping logic to `_process_loop`.
  - If the M1 chip takes longer than 33ms to process a frame, `cap.grab()` is called to skip the backlogged frames.
  - This keeps the video playing at 1x real-world speed instead of slow motion, mimicking a live RTSP stream's natural frame drop behavior.

---

## ✅ Step 4 — Independent Hyperparameter Tuning
- **`vision_engine.py`**: `SentinelStream.__init__` now accepts 5 independent tuning parameters per camera:
  - `mog2_history` — MOG2 background model memory length (frames)
  - `mog2_threshold` — MOG2 pixel variance threshold (lower = more sensitive)
  - `min_blob_area` — Minimum pixel area for a detection to be accepted
  - `ghost_threshold` — Frames a lost track is kept alive before being dropped
  - `max_capacity` — Maximum crowd count for 100% density calculation
- **`app.py`**: Each `SentinelStream` now explicitly declares its own parameter set. This is the foundation for Phase 4 nighttime tuning.
- **Dynamic Masking**: CAM-02 auto-detects `mask_layer2.png` if it exists; falls back to `mask_layer.png`. Drop a `mask_layer2.png` into the project root to activate a unique restricted zone for CAM-02.

---

## ✅ Step 5 — Performance Profiling
- **`vision_engine.py`**: Each processing loop now samples `psutil.Process.cpu_percent()` and `memory_info().rss` every frame, storing them in `latest_stats` as `cpu_percent` and `ram_mb`.
- **`app.py`**: Added new endpoint **`GET /api/system/health`** (login-required) returning:
  - System-wide CPU% and RAM used/total/percent
  - Per-camera FPS and latency_ms

  Example response:
  ```json
  {
    "cpu_percent": 58.3,
    "ram_used_mb": 412.1,
    "ram_total_mb": 8192.0,
    "ram_percent": 50.2,
    "cam1": { "fps": 18, "latency_ms": 55 },
    "cam2": { "fps": 17, "latency_ms": 60 }
  }
  ```
- **`requirements.txt`**: Added `psutil>=5.9`.

---

## 🚀 Ready for Phase 4
All Phase 3 next steps are now complete. The system is validated as a **true dual-pipeline architecture** with:
- Independent processing threads per camera
- Independent hyperparameter sets per camera
- Independent masks per camera
- Real-time synchronization (frame skipping)
- Live hardware performance telemetry

### Phase 4 Preview: Nighttime CV Adaptation
The following hyperparameter changes will be the starting point for nighttime testing:
| Parameter | CAM-01 Day | CAM-01 Night (target) |
|---|---|---|
| `mog2_threshold` | 16 | 8–12 (more sensitive to low contrast) |
| `min_blob_area` | 800 | 500–600 (detect smaller silhouettes) |
| `mog2_history` | 500 | 300 (faster adaptation to dark BG changes) |
