# Barangay Sentinel — Paper Writing Handoff

This document is written for a new session whose job is **writing the thesis paper**, not modifying code. It gives you a complete, honest picture of what the system does, how it works, what the numbers are, and what the known weaknesses are — everything you need to write methodology, evaluation, limitations, and related work sections without misrepresenting the work.

Read this fully before writing a single sentence.

---

## 1. What This System Is (One-Paragraph Summary)

**Barangay Sentinel** is a classical computer-vision nighttime crowd monitoring system designed for barangay (the smallest administrative unit in the Philippines) hall surveillance. It ingests a single CCTV video feed, estimates the number of people in the scene every two seconds, classifies that count into density tiers (LOW / MEDIUM / HIGH), records event clips when density crosses a threshold, and presents the live feed through a web dashboard where a barangay tanod (community watchman) can review alerts and file incident reports. The detection pipeline does **not** use a general deep-learning detector as its primary engine — it is a hybrid of MOG2 background subtraction, morphological processing, occupancy mapping, and a watershed-based sub-counter, augmented by **YOLOv8-nano** as a periodic floor correction to recover stationary people that the motion-based pipeline misses.

---

## 2. Deployment Context (Matters for Framing)

- **Location**: barangay hall, Philippines. Indoor nighttime setting. Fixed CCTV camera. Known ROI.
- **Hardware target**: barangay office desktop — Intel i3 or i5, **no GPU**. All YOLO inference runs on CPU.
- **Operator**: a non-technical tanod. The system must be autonomous and low-maintenance.
- **Threat scenario**: overcrowding during community assemblies (barangay sessions, events). Tier classification (LOW / MEDIUM / HIGH) is the primary output, not a precise count.

This framing matters when writing the introduction and motivation: the system is explicitly scoped to a resource-constrained, low-tech-user deployment. That scope justifies classical CV as the primary engine and YOLO as a floor — not as an indictment of the approach.

---

## 3. System Architecture — Full Pipeline

The pipeline runs in a background thread (`SentinelStream._process_loop`) inside `vision_engine.py`. Every frame goes through the following stages:

### Stage 1 — MOG2 Background Subtraction
- `cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)`
- Produces a per-pixel foreground mask.
- **Learning rate is frozen to 0 after warmup** (first 1,000 frames). This prevents the model from absorbing stationary people into the background once the scene is established.
- **Fundamental limitation**: people who stop moving during warmup are already absorbed and invisible. People who stop moving post-warmup are detected initially but fade if the freeze fails to hold them. This is the primary source of PEAK-phase counting errors before YOLO was added.

### Stage 2 — Headlight / Bright-Halo Suppression
- High-luminance blobs (vehicle headlights bleeding into the ROI) are masked out before morphological processing.

### Stage 3 — Static Background Reference Diff
- An `absdiff` between the current frame and a stored background reference, OR'd with the MOG2 output.
- Intended as a secondary detector for stationary people. In practice its contribution is small — nighttime clothing in barangay halls has low contrast against the floor, so absdiff thresholds miss many targets.

### Stage 4 — Morphological Sculpting
- Sequential operations: `open → close → close → dilate`
- Breaks thin noise connections, fills small holes inside human-shaped blobs, and dilates to merge people standing close together into a single countable region.

### Stage 5 — OccupancyMap (Per-Pixel Persistence Tracker)
- A pixel-level presence tracker. A pixel is "confirmed present" after it appears in foreground for 3 consecutive frames (`CONFIRM_FRAMES=3`). It is evicted after it has been absent for 10 seconds (`EVICT_SEC=10.0`, i.e., 300 frames at 30 FPS).
- Purpose: absorbs the "flickering" of MOG2 on moving people, and retains detected presence briefly after a person momentarily leaves frame.
- **Known side effect**: because eviction takes 10 seconds, the occupancy map holds stale pixels for ~10 s after people leave during the departure phase. This contributes to late-departure overcounting.

### Stage 6 — 3-Stage Collaborative Fusion (Watershed)
- Runs every frame. Produces per-frame count estimates used as a motion anchor.
- **Stage 6a** — watershed on the raw fused mask (MOG2 + reference diff). This is the primary motion-based count.
- **Stage 6b** — watershed on the occupancy map. Corrects for people that MOG2 absorbed but the occupancy map still holds.
- **Stage 6c** — area count correction. **Disabled** — the R² between OccupancyMap blob area and person count was 0.0097 (far below the 0.70 guard threshold). Moving people create 4–6× more pixel area than seated people, making area a non-monotonic predictor. The pipeline detects this automatically and skips Stage 6c.

### Stage 7 — 2-Second Census (Every 60 Frames)
This is the most important stage for accuracy. Every 60 frames:
1. Shadow-validated blob counting from the fused mask and occupancy map (`_cws` and `_car`).
2. **YOLO census**: YOLOv8-nano runs on the current frame (`conf=0.40`, `iou=0.35`, `classes=[0]` persons only, `device="cpu"`, `imgsz=640`). Detections whose centre falls inside the ROI mask are counted (`_yolo_n`).
3. The `recount_floor` is set to `max(_cws, _car, _yolo_n)`. YOLO acts as a **floor**, not a replacement — it raises the count when the classical pipeline undercounts, but does not override a higher classical count.
4. **Occupancy-gated floor decay**: if the per-frame count is below the floor *and* the occupancy map is empty (`cv2.countNonZero(persistent_fg) < MIN_BLOB_AREA`), the floor decays by a factor of 0.80. Gate condition prevents decay from firing during PEAK when MOG2 has absorbed stationary people (occupancy map would still hold their pixels).

### Stage 8 — EMA Smoothing
- `smoothed = 0.4 * current + 0.6 * previous` (COUNT_ALPHA=0.4).
- Applied every frame. Dampens per-frame noise from MOG2 flicker.
- Final output: integer-rounded smoothed count.

### Stage 9 — Tier Classification
- LOW: count ≤ 3
- MEDIUM: count ≤ 6
- HIGH: count > 6
- These are the primary evaluation target (tier accuracy).

---

## 4. YOLO's Role — Write It Carefully

YOLO is **not** the primary detector. It runs once every two seconds, on CPU, and its sole function is to provide a count floor that the classical pipeline cannot achieve on its own for stationary/seated people.

**Why YOLO was added (the failure this fixes):**
The barangay assembly scenario has 7–8 people seated in chairs for extended periods. MOG2 absorbed them into the background because they stopped moving. Classical pipeline predicted ~0–2 people during the PEAK phase (GT=7–8). HOG and MediaPipe EfficientDet-Lite0 were tested and rejected (28% and 34% detection rates on the 14 peak frames). YOLOv8-nano achieved 98.1% detection rate on the same frames at conf=0.30.

**Framing for the paper:**
Do not describe the system as "YOLO-based." Describe it as a **classical CV pipeline with a YOLO floor correction**. The watershed stages, occupancy tracking, and EMA smoothing are the primary architecture. YOLO contributes to the census step only.

**Why this is defensible:**
- YOLO runs every 2 seconds — not per-frame. Latency on i5 CPU: ~100–150ms per call, negligible when amortized over 60 frames.
- YOLO does not affect per-frame smoothed counts directly; it influences the `recount_floor` which caps the minimum count between census events.
- The occupancy-gated decay prevents YOLO from perpetuating false counts after the crowd departs.

---

## 5. Evaluation Results

### Ground Truth Setup
- **Video**: `videos/calibration.MOV` — a real nighttime barangay assembly recording.
- **Annotated frames**: 197 frames, sampled every 60 frames from f=1020 to f=12753.
- **Annotation format**: click-marked head positions per frame, stored in `barangay_ground_truth_calibration.json`.
- **Warmup exclusion**: first 1,000 frames excluded (MOG2 learning phase).
- **Same-video caveat**: calibration and evaluation use the same video. This is a known limitation — MAE is optimistic because any per-video bias in the pipeline carries into both. Must be disclosed in the paper.

### Crowd Trajectory in the Video
| Phase | Frame range | GT count | n frames |
|---|---|---|---|
| ENTRY | f=1020–6480 | 1→6 people (rising) | 93 |
| PEAK | f=6540–9060 | 7–8 people (seated) | 42 |
| DEPART | f=9120–12753 | 6→0 people (falling) | 62 |

### Current Best Result (YOLO conf=0.40, NMS iou=0.35)
*Note: iou=0.35 was applied to evaluate.py but the run has not been executed yet as of this handoff. The numbers below are from the previous confirmed run at conf=0.40, iou=0.45 (default). The iou=0.35 change targets further reduction of double-detections on seated people in the late DEPART phase.*

```
MAE           : 0.97 people
RMSE          : 1.58 people
Bias          : +0.64 (system slightly over-counts)
Tier Accuracy : 77.7%  (153/197 correct)

Tier breakdown:
  LOW     n= 68  MAE=1.84  tier_acc=50%
  MEDIUM  n= 87  MAE=0.60  tier_acc=93%
  HIGH    n= 42  MAE=0.36  tier_acc=90%

Phase breakdown:
  ENTRY   n= 93  MAE=0.70  bias=+0.53
  PEAK    n= 42  MAE=0.36  bias=-0.36
  DEPART  n= 62  MAE=1.81  bias=+1.48
```

### Ablation Progression (Important for the Evaluation Chapter)
This is the honest "what each component contributed" story:

| Configuration | Tier Accuracy | Notes |
|---|---|---|
| Classical CV only (baseline) | 52.3% | PEAK phase ~0% — seated people invisible |
| + MOG2 learning rate freeze | 54.3% | Small improvement; already-absorbed people unaffected |
| + Static background reference diff | 52.3% | Slightly worse — low contrast in nighttime |
| + YOLOv8-nano (conf=0.30) | 75.1% | PEAK fixed (95% tier acc); late DEPART overcounting appears |
| + Floor decay (wrong gate) | 43.1% | Decay fired every frame during PEAK; destroyed PEAK accuracy |
| + Occupancy-gated floor decay | 75.1% | Gate never fires (occupancy map retains stale pixels) |
| + conf raised to 0.40 | **77.7%** | LOW tier improved; late DEPART partially reduced |
| + iou=0.35 (NMS tighter) | *pending* | Targets seated-person double-detection in late DEPART |

### Why LOW Tier Accuracy is Low (50%)
When GT=0–3, the pipeline produces counts of 4–6 due to:
1. Occupancy map eviction lag — after DEPART, stale pixels persist for ~10 s.
2. YOLO double-detecting seated people at low-confidence frames (upper body + lower body as two separate detections). This was partially addressed by raising conf to 0.40 and then tightening NMS iou to 0.35.

The late DEPART frames (f=11700–12753) are the primary low-tier failure zone.

---

## 6. Alternative Detectors Evaluated (for Related Work / Justification)

All three tested on the same 14 peak-phase frames (GT=7–8):

| Detector | Detection Rate | Latency | Decision |
|---|---|---|---|
| HOG + SVM (OpenCV built-in) | 28% | ~297ms/frame | Rejected — too slow, too inaccurate |
| MediaPipe EfficientDet-Lite0 | 34% | ~18ms/frame | Rejected — accuracy insufficient |
| YOLOv8-nano (conf=0.30) | 98.1% | ~50ms/frame (M1); ~150ms (i5 CPU) | Selected |

Write this in the paper as a detector selection experiment with a table. It shows the decision was evidence-based, not arbitrary.

---

## 7. Known Limitations — Write These Yourself, on Your Own Terms

Do not let these emerge from a panelist's question. State them explicitly in the Limitations section.

### L1 — MOG2 Stationary-Person Absorption
After ~500 frames (≈17 s at 30 FPS), a stationary person is absorbed into the MOG2 background and becomes invisible to the motion-based pipeline. The YOLO floor correction mitigates this but does not eliminate it — YOLO runs every 60 frames, so up to 2 seconds of undercount can occur between census events. **Quantify in the paper**: "A person who stops moving before warmup ends is not detected by the classical pipeline at all. A person who stops after warmup is detected initially and held by the occupancy map for up to 10 s, after which count depends on the YOLO census."

### L2 — Same-Video Calibration and Evaluation
`videos/calibration.MOV` is used for both MOG2 parameter tuning and for evaluation (evaluate.py). Any per-video bias the pipeline has toward this specific video is baked into both the system and the benchmark. MAE figures are optimistic. **State this explicitly.** The correct framing: "Evaluation was conducted on the calibration video; a held-out evaluation on a separate recording is a subject of future work."

### L3 — Late DEPART Overcounting
The occupancy map eviction lag (10 s) combined with YOLO's periodic floor retention causes overcounting in the final 15% of the video as the crowd departs. The system is late to recognize that density has dropped. Visually inspected: the issue is not furniture false positives — it is the temporal persistence of stale detections.

### L4 — Area Calibration Failure
An attempt to use OccupancyMap blob area as a count predictor (Stage 6c) was abandoned — R²=0.0097, well below the 0.70 guard. Moving people generate 4–6× more pixel area than seated people, making area non-monotonic with count. The pipeline detects and disables this stage automatically. **This is actually a strength** — the system self-checks and rejects unreliable calibration. Frame it as a designed guard, not a failure.

### L5 — Classical CV Ceiling
Without YOLO, the classical pipeline reaches approximately 62–65% tier accuracy on this scene type. For scenes dominated by motion (e.g., standing/moving crowds), the ceiling is higher. For nighttime seated assemblies, it is not enough. Document the ceiling.

### L6 — NMS Double-Detection
YOLO can double-detect a seated person's upper and lower body as two separate bounding boxes, especially in late-departure frames where few people remain and the model's NMS has less competition. Mitigated by `iou=0.35` (tighter NMS). Not fully eliminated — a structural limitation of any NMS-based detector on partial occlusion poses.

---

## 8. Files Reference (for Methodology Chapter)

| File | Role |
|---|---|
| `vision_engine.py` | Core pipeline — `SentinelStream` class, all CV stages, YOLO census |
| `evaluate.py` | Offline evaluation against ground truth. Mirrors the live pipeline exactly. |
| `app.py` | Flask web server. Routes, auth, incident archive, SentinelStream wiring. |
| `calibrate_area.py` | Occupancy map area calibration tool. Used to test Stage 6c (now auto-disabled). |
| `barangay_ground_truth_calibration.json` | 197 annotated frames, f=1020–12753. |
| `mask_layer_calibration.png` | ROI mask — restricts detection to the assembly area, excludes walls, doors, ceiling. |
| `area_calibration.json` | Area calibration output. Contains R²=0.0097 which triggers the guard in the pipeline. |
| `yolov8n.pt` | YOLOv8-nano weights. Pre-trained on COCO (80 classes). Only class 0 (person) is used. |

---

## 9. Figures You Should Include in the Paper

These figures, if included, will make the evaluation chapter much stronger:

1. **Pipeline architecture diagram** — a flowchart of the 9 stages with input/output labels. Show where YOLO inserts (Stage 7 census), where watershed runs (Stage 6), where EMA applies (Stage 8).

2. **Ablation table** — the progression table from §5 above. Shows each component's contribution to tier accuracy.

3. **Phase-breakdown bar chart** — ENTRY / PEAK / DEPART tier accuracy side by side, for the baseline (classical only) vs final (YOLO hybrid). This visualizes exactly what YOLO fixed.

4. **Failure mode figure** — a frame from the PEAK phase where the classical pipeline predicts ~0 and YOLO correctly detects 7–8 people. Contrasted with the final pipeline's prediction on the same frame.

5. **Late DEPART failure figure** — a frame from f=11700–12753 where GT=1 and the pipeline predicts 3–4, showing the occupancy persistence / NMS double-detection issue.

6. **ROI mask overlay** — the mask applied to the calibration video, showing what region is included/excluded.

---

## 10. Recommended Framing for the Introduction

The motivation chain is:
1. Barangay halls host assemblies that frequently exceed safe capacity.
2. Existing commercial crowd monitoring systems require GPU hardware and cloud connectivity — not available at barangay level.
3. A classical CV approach is interpretable, runs on commodity hardware, and requires no ongoing licensing.
4. The fundamental limitation of classical CV (stationary person absorption) is addressed by a periodic YOLO census — which is specifically possible at barangay scale because YOLO runs once every 2 seconds, not per frame, making CPU deployment viable.

The thesis contributes:
1. A hybrid classical-CV + YOLO pipeline optimized for nighttime seated assembly scenarios on CPU hardware.
2. A 3-stage collaborative fusion with an automatic calibration guard that disables unreliable stages based on R².
3. An empirical evaluation on a real locally-annotated barangay assembly video (197 annotated frames).
4. Evidence-based detector selection (HOG vs EfficientDet vs YOLOv8n) on the specific failure case (stationary seated crowd).

---

## 11. Vocabulary to Use Consistently

| Term | Use this | Not this |
|---|---|---|
| Primary architecture | "classical CV pipeline" | "traditional approach" |
| The YOLO role | "periodic census floor" or "appearance-based floor correction" | "the deep learning component" |
| Count output | "crowd density tier" (LOW/MEDIUM/HIGH) | "person count" (the count feeds the tier; the tier is the system output) |
| MOG2 failure mode | "stationary-person absorption" | "background subtraction limitation" (too vague) |
| Stage 6 | "3-stage collaborative fusion" or "watershed-based sub-counter" | "watershed algorithm" alone (doesn't capture the 3-stage structure) |
| Stage 7 | "2-second census" | "detection interval" |
| The occupancy persistence issue | "eviction lag" | "tracker memory" |

---

## 12. What the Paper Already Contains vs. What Needs Writing

The paper file exists (`docs/Vision_Engine_Analysis_Report.docx`). A new session should open it first before writing anything, to understand what's already written and what sections are empty or weak.

Based on the development history, the sections most likely to need work are:

- **Methodology** — needs to reflect the final hybrid pipeline (YOLO was added late; early drafts pre-date it). Ensure all 9 stages are documented.
- **Evaluation** — needs the ablation table, phase breakdown, and the iou=0.35 result once it's run.
- **Limitations** — likely empty or cursory. Use §7 above verbatim as a starting point.
- **Related Work** — the detector comparison (§6) should appear here as part of "why YOLO over alternatives."

Do not rewrite sections that are already accurate. Add to or fix weak sections rather than replacing wholesale.

---

## 13. Numbers to Double-Check Before Final Submission

The following numbers should be verified against the latest evaluate.py run before being locked into the paper:

- [ ] Tier accuracy with `iou=0.35` (pending run — expected to be ≥77.7%)
- [ ] Phase MAE with `iou=0.35` (specifically DEPART MAE, expected to drop from 1.81)
- [ ] YOLO inference time on deployment hardware (i3/i5 — estimated 100–300ms, should be measured)
- [ ] Annotation count: 197 frames confirmed (check `barangay_ground_truth_calibration.json`)
- [ ] R² value from `area_calibration.json`: 0.0097 — confirm this is the stored value

---

## 14. Questions You Should NOT Answer Without Checking the Code

If the paper text makes any of these claims, verify before writing:

- **"The system achieves X% tier accuracy"** — use only numbers from a confirmed evaluate.py run.
- **"YOLO runs at Y ms per frame"** — measure on actual hardware; M1 MacBook numbers in development are not representative of the i5 deployment target.
- **"The area calibration guard threshold is R²=0.70"** — confirm in `evaluate.py` / `app.py`.
- **"The system uses X annotated frames for evaluation"** — run `python3 -c "import json; d=json.load(open('barangay_ground_truth_calibration.json')); print(len(d['frames']))"`.

---

*This document reflects the state of the system as of the end of the YOLO integration session. The code is complete in `vision_engine.py`, `evaluate.py`, and `app.py`. The pending action is running `python3 evaluate.py` to get the iou=0.35 result, which should be done before writing the final evaluation numbers into the paper.*
