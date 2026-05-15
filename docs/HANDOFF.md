# Barangay Sentinel — Session Handoff

## Goal

Improve the tier classification accuracy of **Barangay Sentinel**, a classical CV nighttime crowd monitoring system for a local barangay hall. The target is **≥80% tier accuracy** evaluated against a hand-annotated ground truth on `videos/calibration.MOV`.

Tiers are: LOW (≤3 people) / MEDIUM (≤6) / HIGH (>6).

---

## Current Best Result

**YOLO + classical pipeline, conf=0.40, iou=0.35**  
*(iou=0.35 was just applied — evaluate.py has NOT been re-run yet with this setting)*

Previous best confirmed run (conf=0.40, iou default=0.45):
```
MAE           : 0.97 people
RMSE          : 1.58 people
Bias          : +0.64
Tier Accuracy : 77.7%  (153/197 correct)

Tier breakdown:
  LOW     n= 68  MAE=1.84  tier_acc=50%
  MEDIUM  n= 87  MAE=0.60  tier_acc=93%
  HIGH    n= 42  MAE=0.36  tier_acc=90%

Phase breakdown (GT peak=8, peak frame~=9060):
  ENTRY   n= 93  MAE=0.70  bias=+0.53
  PEAK    n= 42  MAE=0.36  bias=-0.36
  DEPART  n= 62  MAE=1.81  bias=+1.48
```

**The next run (iou=0.35) has not been executed yet.** That is the immediate next step.

---

## Architecture Overview

The pipeline is a **classical CV + YOLO hybrid**:

1. **MOG2 background subtraction** (every frame) — detects motion
2. **Headlight suppression** — strips bright halos from vehicle lights
3. **Static background reference diff** — detects stationary people MOG2 absorbed
4. **Morphological sculpting** (open → close → close → dilate)
5. **OccupancyMap** — per-pixel persistence tracker (confirm 3 frames, evict after 10 s)
6. **3-stage collaborative fusion**:
   - Stage 1: watershed on raw fused (motion anchor)
   - Stage 2: watershed on occupancy map (absorption correction)
   - Stage 3: area count correction (disabled — R²=0.0097 < 0.70 threshold)
7. **2-second census** (every 60 frames) — shadow-validated blobs + **YOLO** floor
8. **EMA smoothing** (α=0.4) → final prediction

YOLO only runs in step 7 (every 2 seconds), not per-frame. Acts as a `recount_floor` — raises the count when MOG2 misses stationary/seated people.

---

## Files Actively Modified

| File | What changed |
|---|---|
| `evaluate.py` | Full rewrite — mirrors live pipeline, adds YOLO census, phase breakdown, R² guard |
| `vision_engine.py` | MOG2 learning rate freeze at warmup end; static bg reference diff |
| `app.py` | R²≥0.70 guard on calibration loading |
| `calibrate_area.py` | `OCCUPANCY_CONFIRM` 10→3, `OCCUPANCY_EVICT_SEC` 5.0→10.0 to match live pipeline |

### evaluate.py — key config block (top of file)
```python
USE_YOLO      = True
YOLO_MODEL    = "yolov8n.pt"
YOLO_CONF     = 0.4        # raised from 0.3 — reduces FPs at low density
YOLO_IMGSZ    = 640

VIDEO_PATH    = "videos/calibration.MOV"
MASK_PATH     = "mask_layer_calibration.png"
GT_PATH       = "barangay_ground_truth_calibration.json"
CAL_PATH      = "area_calibration.json"

CONFIRM_FRAMES   = 3
EVICT_SEC        = 10.0
WARMUP_FRAMES    = 1000
COUNT_ALPHA      = 0.4
RECOUNT_INTERVAL = 60
FLOOR_DECAY      = 0.80    # only fires when occupancy map is empty (see decay block)
```

### YOLO census call (inside the 60-frame census block):
```python
_yolo_res = yolo_model(
    frame, device="cpu", imgsz=YOLO_IMGSZ,
    conf=YOLO_CONF, iou=0.35, classes=[0], verbose=False   # iou=0.35 is latest change
)[0]
```

### Floor decay (after census, before EMA):
```python
_occ_area = cv2.countNonZero(persistent_fg)
if total < recount_floor and _occ_area < MIN_BLOB_AREA:
    recount_floor = max(float(total), recount_floor * FLOOR_DECAY)
total = max(total, int(round(recount_floor)))
```

---

## Everything Tried & What Happened

### Classical pipeline only (baseline)
- **52.3% tier accuracy**, MAE=2.46
- PEAK phase: ~0% tier accuracy — 7–8 seated people invisible to MOG2 (absorbed into background after staying still)

### MOG2 learning rate freeze (learningRate=0 after warmup)
- Small improvement only — MOG2 still can't see people it absorbed before freeze
- **54.3% tier accuracy**

### Static background reference diff (absdiff vs bg_reference)
- OR'd with MOG2 output as secondary detector
- Slightly worse: **52.3%** — nighttime clothing ≈ floor in grayscale (absdiff < threshold)

### Alternative detectors tested (HOG, MediaPipe EfficientDet-Lite0)
- Tested on 14 peak frames (GT=7–8) where classical pipeline predicts ~0
- HOG: **28%** detection rate, 297ms/frame — not viable
- MediaPipe EfficientDet-Lite0: **34%** detection rate, 18ms/frame — not viable
- YOLOv8-nano conf=0.3: **98%** detection rate, ~50ms/frame — clear winner

### YOLO integration at conf=0.30
- **75.1% tier accuracy**, MAE=0.98
- PEAK fixed: MAE=0.24, tier_acc=95%
- New problem: late DEPART overcounting (GT=0–3, PRED=5–8)

### Floor decay per-frame at rate=0.95 (wrong gate condition)
- **43.1% tier accuracy** — decay fired every frame during PEAK (MOG2 total ≈ 0, floor ≈ 7), killed floor in ~15 frames
- PEAK tier_acc dropped to 0%

### Floor decay with occupancy gate (_occ_area < MIN_BLOB_AREA)
- **75.1% tier accuracy** — identical to no-decay
- Gate never fired: occupancy map held stale pixels throughout late DEPART (eviction lag)
- Confirmed: decay approach cannot fix late DEPART without also solving occupancy eviction

### YOLO conf raised to 0.40
- **77.7% tier accuracy** — new best
- LOW tier improved: 44% → 50%
- PEAK slightly worse: 95% → 90% (expected — fewer detections at higher threshold)
- Late DEPART still identical — FPs persist above conf=0.40

### Visual inspection of late DEPART failures (inspect_yolo_depart.py)
- Ran `inspect_yolo_depart.py` on frames f=11700–12753
- **Empty chairs are NOT causing false positives** — YOLO ignores them
- **f=12300 (GT=1, YOLO=2)**: two boxes on the same seated person — classic double-detection of upper/lower body
- Root cause: NMS IoU threshold too permissive for seated poses
- Fix: `iou=0.35` (tighter NMS) — **applied but not yet evaluated**

---

## Known Limitations (document in thesis)

1. **Calibration failure**: area calibration R²=0.0097 — OccupancyMap area is non-monotonic with person count (moving people create 4–6× more area than seated). Area counting disabled via R²≥0.70 guard.

2. **Same-video calibration and evaluation**: calibration and evaluation both use `videos/calibration.MOV`. Any per-pixel bias in calibration carries into evaluation — MAE is optimistic. Must disclose.

3. **Late DEPART overcounting**: occupancy map eviction (10 s) is slow to clear after crowd leaves. Combined with YOLO FPs from seated-person double-detection, creates +4 to +5 prediction error in final 15% of video.

4. **Classical CV ceiling**: without YOLO, ceiling is ~62–65% for this scene type. Stationary seated people absorbed by MOG2 are undetectable via motion-only methods.

---

## Next Steps (in order)

### 1. Run evaluate.py now (iou=0.35 just applied)
```bash
cd ~/Desktop/THESIS/RETORNSERO_THESIS
python3 evaluate.py
```
Expected: DEPART MAE drops (fewer double-detections), PEAK holds at ~90%.  
If tier accuracy hits 78–80%: stop tuning, move to integration.

### 2. If still below 78% — try conf=0.45 + iou=0.35
One more data point before accepting current result.

### 3. Integrate YOLO into vision_engine.py (live pipeline)
YOLO is currently only in evaluate.py. The live app (app.py / vision_engine.py) doesn't use it yet.  
Integration point: `SentinelStream._process_loop()`, in the 60-frame census block.  
Model path: `yolov8n.pt` (already downloaded in project root).  
Run model on CPU, `device="cpu"`, every 60 frames. Same ROI mask filter.

### 4. Commit clean final state
Files to commit: `evaluate.py`, `vision_engine.py`, `app.py`, `calibrate_area.py`  
Files to clean up (test artifacts, not needed in repo):
- `inspect_yolo_depart.py`
- `test_yolo_peak.py`
- `test_alternatives_peak.py`
- `MobileNetSSD_deploy.prototxt`
- `MobileNetSSD_deploy.caffemodel`
- `efficientdet_lite0.tflite`
- `yolov8n.onnx` (keep `yolov8n.pt`)
- `yolo_inspect/` folder

---

## Hardware Context

Deployment target: barangay office desktop, Intel i3/i5, no GPU.  
YOLO on CPU (i5, 640×360 input): ~100–150ms per inference.  
YOLO runs every 60 frames (~2 s) — acceptable even at 300ms on i3.  
No GPU required.

---

## Ground Truth Info

- **Video**: `videos/calibration.MOV`
- **GT file**: `barangay_ground_truth_calibration.json`
- **Format**: `{"frames": {"frame_idx": [{"x": ..., "y": ...}, ...]}}`
- **Annotated frames**: every 60 frames, f=1020–12753 (197 frames total)
- **Crowd trajectory**: ENTRY (1–6 people, f=1020–6480) → PEAK (7–8 people, f=6540–9060) → DEPART (6→0 people, f=9120–12753)
- **Warmup**: first 1000 frames excluded from evaluation
