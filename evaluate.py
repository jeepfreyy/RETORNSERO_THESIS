#!/usr/bin/env python3
"""
Offline pipeline evaluation against barangay_ground_truth.json.
Mirrors the exact pipeline parameters from app.py and reports:
  - MAE, RMSE, signed bias
  - Tier classification accuracy (LOW / MEDIUM / HIGH)
  - Per-tier breakdown
"""
import json, cv2, numpy as np, os, sys

from vision_engine import (
    OccupancyMap,
    suppress_headlights,
    count_people_watershed_scene,
    count_people_by_area,
)

# ── Pipeline params — must match app.py exactly ───────────────────────────
VIDEO_PATH     = "videos/main_video.mp4"
MASK_PATH      = "mask_layer1.png"
GT_PATH        = "barangay_ground_truth.json"
CAL_PATH       = "area_calibration.json"

MOG2_HISTORY   = 30000
MOG2_THRESH    = 40
PROCESS_SCALE  = 0.667
MIN_BLOB_AREA  = 350
MORPH_K        = (5, 25)
H_MORPH_K      = (10, 3)
DILATE_K       = 1
HL_V_THRESH    = 160
HL_DILATION    = 80
CONFIRM_FRAMES = 2
EVICT_SEC      = 5.0
MAX_CAPACITY   = 10
WARMUP_FRAMES  = 1500
COUNT_ALPHA    = 0.3
# ─────────────────────────────────────────────────────────────────────────


def tier(count):
    if count <= 3: return "LOW"
    if count <= 6: return "MEDIUM"
    return "HIGH"


def main():
    # Load calibration
    if not os.path.exists(CAL_PATH):
        print(f"[ERROR] {CAL_PATH} not found — run calibrate_area.py first.")
        sys.exit(1)
    with open(CAL_PATH) as f:
        cal = json.load(f)
    px_per_person = float(cal["avg_pixels_per_person"])
    area_baseline = float(cal["area_at_zero"])

    # Load ground truth
    with open(GT_PATH) as f:
        gt_raw = json.load(f)
    gt_counts = {int(k): len(v) for k, v in gt_raw["frames"].items()}

    # Only evaluate annotated frames after warmup
    eval_frames = sorted(f for f in gt_counts if f > WARMUP_FRAMES)
    if not eval_frames:
        print("[ERROR] No ground-truth frames found after warmup. Check GT file.")
        sys.exit(1)
    eval_set = set(eval_frames)
    print(f"Ground-truth frames to evaluate: {len(eval_frames)}  "
          f"(frames {eval_frames[0]}–{eval_frames[-1]})")

    # Build kernels
    open_k      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fusion_k    = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_K)
    h_fusion_k  = cv2.getStructuringElement(cv2.MORPH_RECT, H_MORPH_K)
    d_kernel    = np.ones((DILATE_K, DILATE_K), np.uint8)
    close_k     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=MOG2_HISTORY, varThreshold=MOG2_THRESH, detectShadows=False
    )

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open {VIDEO_PATH}")
        sys.exit(1)

    ret, first = cap.read()
    if not ret:
        print("[ERROR] Could not read first frame.")
        sys.exit(1)

    proc_h = int(first.shape[0] * PROCESS_SCALE)
    proc_w = int(first.shape[1] * PROCESS_SCALE)

    raw_mask = cv2.imread(MASK_PATH, 0)
    if raw_mask is not None:
        roi_mask = cv2.resize(raw_mask, (proc_w, proc_h), interpolation=cv2.INTER_NEAREST)
    else:
        roi_mask = np.ones((proc_h, proc_w), np.uint8) * 255
        print("[WARN] mask_layer1.png not found — using full frame.")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    occupancy = OccupancyMap(
        (proc_h, proc_w),
        confirm_frames=CONFIRM_FRAMES,
        evict_frames=int(EVICT_SEC * 30),
        dark_v_thresh=40,
    )

    count_ema      = 0.0
    loop_idx       = 0
    results        = {}          # frame_idx -> {"gt": int, "pred": int}
    last_eval      = max(eval_frames)

    print("\nRunning pipeline...\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
        hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # MOG2
        fg = bg_sub.apply(frame)
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        thresh = cv2.bitwise_and(fg, fg, mask=roi_mask)

        # Headlight suppression
        thresh = suppress_headlights(thresh, hsv, HL_V_THRESH, HL_DILATION)

        # Morphology
        cleaned = cv2.morphologyEx(thresh,  cv2.MORPH_OPEN,  open_k)
        fused   = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, fusion_k)
        fused   = cv2.morphologyEx(fused,   cv2.MORPH_CLOSE, h_fusion_k)
        fused   = cv2.dilate(fused, d_kernel, iterations=1)

        # Warmup gate
        if loop_idx == WARMUP_FRAMES:
            occupancy.reset()

        # Pre-filter → occupancy map
        _conts, _ = cv2.findContours(fused, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        _fused_occ = np.zeros_like(fused)
        for _c in _conts:
            if cv2.contourArea(_c) >= MIN_BLOB_AREA:
                cv2.drawContours(_fused_occ, [_c], -1, 255, -1)
        persistent_fg = occupancy.update(_fused_occ, hsv)

        # Triple hybrid count
        _pfg_closed = cv2.morphologyEx(persistent_fg, cv2.MORPH_CLOSE, close_k)
        ws_occ, _   = count_people_watershed_scene(
            _pfg_closed, min_blob_area=MIN_BLOB_AREA, dt_thresh=0.50
        )
        ar_count = count_people_by_area(persistent_fg, px_per_person, area_baseline)
        total    = max(ws_occ, ar_count)

        if loop_idx <= WARMUP_FRAMES:
            total     = 0
            count_ema = 0.0

        count_ema = COUNT_ALPHA * total + (1 - COUNT_ALPHA) * count_ema
        pred      = round(count_ema)

        if loop_idx in eval_set:
            gt_n = gt_counts[loop_idx]
            results[loop_idx] = {"gt": gt_n, "pred": pred}
            match = "✓" if tier(pred) == tier(gt_n) else "✗"
            print(f"  f={loop_idx:>6} | GT={gt_n:>3}  PRED={pred:>3}  err={pred-gt_n:+d}"
                  f"  |  {tier(gt_n):6s} → {tier(pred):6s}  {match}")

        loop_idx += 1

        if loop_idx % 1000 == 0:
            print(f"  ... frame {loop_idx} processed")

        # Stop early once we're past the last annotated frame
        if loop_idx > last_eval + 60:
            break

    cap.release()

    if not results:
        print("\n[ERROR] No frames were evaluated.")
        return

    # ── Metrics ───────────────────────────────────────────────────────────
    errors     = [abs(r["pred"] - r["gt"]) for r in results.values()]
    sq_errors  = [(r["pred"] - r["gt"]) ** 2 for r in results.values()]
    signed     = [r["pred"] - r["gt"] for r in results.values()]

    mae  = sum(errors) / len(errors)
    rmse = (sum(sq_errors) / len(sq_errors)) ** 0.5
    bias = sum(signed) / len(signed)

    tier_correct = sum(
        1 for r in results.values() if tier(r["pred"]) == tier(r["gt"])
    )
    tier_acc = tier_correct / len(results) * 100

    print("\n" + "=" * 52)
    print(f"  EVALUATION SUMMARY  ({len(results)} annotated frames)")
    print("=" * 52)
    print(f"  MAE           : {mae:.2f} people")
    print(f"  RMSE          : {rmse:.2f} people")
    print(f"  Bias          : {bias:+.2f}  (+ = overcounting, – = undercounting)")
    print(f"  Tier Accuracy : {tier_acc:.1f}%  ({tier_correct}/{len(results)} correct)")
    print()

    for t in ["LOW", "MEDIUM", "HIGH"]:
        subset = [r for r in results.values() if tier(r["gt"]) == t]
        if not subset:
            continue
        t_mae  = sum(abs(r["pred"] - r["gt"]) for r in subset) / len(subset)
        t_ok   = sum(1 for r in subset if tier(r["pred"]) == t)
        print(f"  {t:6s}  n={len(subset):>3}  MAE={t_mae:.2f}  "
              f"tier_acc={t_ok/len(subset)*100:.0f}%")

    print()


if __name__ == "__main__":
    main()
