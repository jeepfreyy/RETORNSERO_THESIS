#!/usr/bin/env python3
"""
Offline pipeline evaluation against annotated ground truth.

Pipeline mirrors app.py / vision_engine.py EXACTLY:
  detectShadows=True, mog2_threshold=40, confirm_frames=3, evict_sec=10.0,
  headlight_v_thresh=160, headlight_dilation=80, COUNT_ALPHA=0.4,
  collaborative 3-stage fusion, 2-second census with shadow validation.

Reports:
  - Overall MAE, RMSE, signed bias
  - Tier classification accuracy  (LOW <= 3 / MEDIUM <= 6 / HIGH > 6)
  - Per-tier breakdown
  - Phase-separated breakdown  (ENTRY / PEAK / DEPART)

NOTE ON VIDEO / GT PATHS
------------------------
Set VIDEO_PATH and GT_PATH to match whichever pair you are evaluating.

Main demo video  -> VIDEO_PATH = "videos/main_video.mp4"
                   GT_PATH    = "ground truths/barangay_ground_truth.json"
                   WARMUP     = 1500

Calibration video (recommended -- same scene as calibration) ->
                   VIDEO_PATH = "videos/calibration.MOV"
                   GT_PATH    = "ground truths/barangay_ground_truth_calibration.json"
                   WARMUP     = 1000  (adjust to match empty frames at start)

KNOWN LIMITATION: calibration and evaluation run on the same video.
Any per-pixel px/person bias present in calibration will be present in
evaluation too, so reported MAE is optimistic.  Disclose this in the thesis.
"""
import json, cv2, numpy as np, os, sys

# ── YOLO census flag (temporary integration test) ─────────────────────────────
USE_YOLO      = True
YOLO_MODEL    = "yolov8n.pt"   # or "yolov8n.onnx" for faster CPU
YOLO_CONF     = 0.4
YOLO_IMGSZ    = 640            # internal resize; proc frame downscaled before inference
# ─────────────────────────────────────────────────────────────────────────────

from vision_engine import (
    OccupancyMap,
    suppress_headlights,
    count_people_watershed_scene,
    count_people_by_area,
)

# -- Configuration — edit here, nowhere else ----------------------------------
VIDEO_PATH    = "videos/calibration.MOV"
MASK_PATH     = "masks/mask_layer_calibration.png"
GT_PATH       = "ground truths/barangay_ground_truth_calibration.json"
CAL_PATH      = "ground truths/area_calibration.json"

# -- Pipeline params — must match app.py / vision_engine.py exactly -----------
MOG2_HISTORY    = 30000
MOG2_THRESH     = 40
DETECT_SHADOWS  = True      # app uses True; >200 threshold strips shadow labels
PROCESS_SCALE   = 0.667
MIN_BLOB_AREA   = 350
MORPH_K         = (5, 25)
H_MORPH_K       = (10, 3)
DILATE_K        = 1
HL_V_THRESH     = 160       # headlight V cutoff (lowered from 200 to kill halo)
HL_DILATION     = 80        # px halo kill radius
CONFIRM_FRAMES  = 3         # occupancy map: px must appear 3 consecutive frames
EVICT_SEC       = 10.0      # occupancy map: 10 s before evicting an absent pixel
WARMUP_FRAMES   = 1000      # suppress count while MOG2 stabilises (per-video)
COUNT_ALPHA     = 0.4       # EMA smoothing alpha (matches vision_engine.py)
RECOUNT_INTERVAL = 60       # frames between census runs (60 ~= 2 s at 30 fps)
FLOOR_DECAY      = 0.80     # per-frame decay multiplier applied to recount_floor
                             # only fires when occupancy map is empty (see below)
# -----------------------------------------------------------------------------


def tier(count):
    """Evaluation bucket; not the app's SAFE/WARNING/CRITICAL density tiers."""
    if count <= 3: return "LOW"
    if count <= 6: return "MEDIUM"
    return "HIGH"


def main():
    # -- YOLO model (optional) ------------------------------------------------
    yolo_model = None
    if USE_YOLO:
        from ultralytics import YOLO as _YOLO
        yolo_model = _YOLO(YOLO_MODEL)
        print(f"[YOLO] Loaded {YOLO_MODEL}  conf={YOLO_CONF}")

    # -- Calibration ----------------------------------------------------------
    px_per_person = None   # None = area counting disabled
    area_baseline = 0.0
    if not os.path.exists(CAL_PATH):
        print("[Cal] area_calibration.json not found — area counting disabled.")
    else:
        with open(CAL_PATH) as f:
            cal = json.load(f)
        _r2 = float(cal.get("r_squared", 0.0))
        if _r2 >= 0.70:
            px_per_person = float(cal["avg_pixels_per_person"])
            area_baseline = float(cal["area_at_zero"])
            print(f"[Cal] {px_per_person:.1f} px/person  baseline={area_baseline:.1f}"
                  f"  R2={_r2:.4f}")
        else:
            print(f"[Cal] Calibration rejected — R2={_r2:.4f} < 0.70 "
                  f"(area-count relationship is non-linear for this scene). "
                  f"Area counting disabled; using watershed + census only.")

    # -- Ground truth ---------------------------------------------------------
    if not os.path.exists(GT_PATH):
        print(f"[ERROR] {GT_PATH} not found.")
        sys.exit(1)
    with open(GT_PATH) as f:
        gt_raw = json.load(f)

    # Support both {"frames": {str_idx: [{"x":...,"y":...}, ...]}}
    # and          {"frames": {str_idx: int_count}} formats.
    gt_counts = {}
    for k, v in gt_raw["frames"].items():
        gt_counts[int(k)] = len(v) if isinstance(v, list) else int(v)

    eval_frames = sorted(f for f in gt_counts if f > WARMUP_FRAMES)
    if not eval_frames:
        print("[ERROR] No GT frames found after warmup. "
              "Check GT file or WARMUP_FRAMES setting.")
        sys.exit(1)
    eval_set = set(eval_frames)
    print(f"GT frames to evaluate: {len(eval_frames)}"
          f"  (f{eval_frames[0]}-f{eval_frames[-1]})")

    # -- Phase detection from GT trajectory -----------------------------------
    peak_gt    = max(gt_counts[f] for f in eval_frames)
    peak_frame = max(
        (f for f in eval_frames if gt_counts[f] >= peak_gt * 0.80),
        key=lambda f: f,
    )

    def phase(fidx):
        gt_n = gt_counts[fidx]
        if gt_n >= peak_gt * 0.80:
            return "PEAK"
        if fidx <= peak_frame:
            return "ENTRY"
        return "DEPART"

    # -- Build kernels --------------------------------------------------------
    open_k     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fusion_k   = cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_K)
    h_fusion_k = cv2.getStructuringElement(cv2.MORPH_RECT, H_MORPH_K)
    d_kernel   = np.ones((DILATE_K, DILATE_K), np.uint8)

    # -- MOG2 -----------------------------------------------------------------
    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=MOG2_HISTORY, varThreshold=MOG2_THRESH,
        detectShadows=DETECT_SHADOWS
    )

    # -- Video ----------------------------------------------------------------
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
        roi_mask = cv2.resize(raw_mask, (proc_w, proc_h),
                              interpolation=cv2.INTER_NEAREST)
    else:
        roi_mask = np.ones((proc_h, proc_w), np.uint8) * 255
        print("[WARN] Mask not found — using full frame.")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # -- Occupancy map --------------------------------------------------------
    occupancy = OccupancyMap(
        (proc_h, proc_w),
        confirm_frames=CONFIRM_FRAMES,
        evict_frames=int(EVICT_SEC * 30),
        dark_v_thresh=40,
    )

    # -- Loop state -----------------------------------------------------------
    count_ema     = 0.0
    recount_floor = 0.0     # float so decay math works
    loop_idx      = 0        # 0 = first frame read by cap.read()
    mog2_lr       = -1       # -1 = auto; set to 0 after warmup to freeze background
    bg_reference  = None     # captured at warmup end for static-diff detection
    results       = {}       # frame_idx -> {"gt": int, "pred": int, "phase": str}
    last_eval     = max(eval_frames)

    print("\nRunning pipeline...\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
        hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Phase 1: MOG2 background subtraction
        fg = bg_sub.apply(frame, learningRate=mog2_lr)
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        thresh = cv2.bitwise_and(fg, fg, mask=roi_mask)

        # Headlight suppression
        thresh = suppress_headlights(thresh, hsv, HL_V_THRESH, HL_DILATION)

        # Static background reference diff (fused with MOG2 output)
        if bg_reference is not None:
            _diff      = cv2.absdiff(frame, bg_reference)
            _diff_gray = cv2.cvtColor(_diff, cv2.COLOR_BGR2GRAY)
            _, _diff_mask = cv2.threshold(_diff_gray, 30, 255, cv2.THRESH_BINARY)
            _diff_mask = cv2.bitwise_and(_diff_mask, roi_mask)
            _diff_mask = suppress_headlights(_diff_mask, hsv, HL_V_THRESH, HL_DILATION)
            thresh = cv2.bitwise_or(thresh, _diff_mask)

        # Phase 2: Morphological sculpting
        cleaned = cv2.morphologyEx(thresh,  cv2.MORPH_OPEN,  open_k)
        fused   = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, fusion_k)
        fused   = cv2.morphologyEx(fused,   cv2.MORPH_CLOSE, h_fusion_k)
        fused   = cv2.dilate(fused, d_kernel, iterations=1)

        # Warmup gate — mirrors vision_engine.py exactly
        if loop_idx == WARMUP_FRAMES:
            occupancy.reset()
            recount_floor = 0.0
            mog2_lr      = 0
            bg_reference = bg_sub.getBackgroundImage()

        # Occupancy map — feed raw fused (not pre-filtered blobs)
        # This matches: persistent_fg = occupancy.update(fused, hsv_frame)
        persistent_fg = occupancy.update(fused, hsv)

        # Phase 4: Collaborative Incremental Fusion
        ws_fused_count, _ = count_people_watershed_scene(
            fused, min_blob_area=MIN_BLOB_AREA, dt_thresh=0.30
        )
        ws_occ_count, _ = count_people_watershed_scene(
            persistent_fg, min_blob_area=MIN_BLOB_AREA, dt_thresh=0.30
        )
        ar_count = (count_people_by_area(persistent_fg, px_per_person, area_baseline)
                    if px_per_person is not None else 0)

        # Stage 1: motion anchor
        total = ws_fused_count

        # Stage 2: absorption correction (capped at motion count)
        occ_surplus    = max(0, ws_occ_count - total)
        absorption_cap = max(total, 2)
        total          = total + min(occ_surplus, absorption_cap)

        # Stage 3: dense-crowd area correction (only when calibration is valid)
        if px_per_person is not None and ar_count > total and ws_occ_count >= total:
            total = min(ar_count, ws_occ_count)

        # 2-second census with shadow validation
        if loop_idx > WARMUP_FRAMES and loop_idx % RECOUNT_INTERVAL == 0:
            _pf_conts, _ = cv2.findContours(
                persistent_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            _validated = np.zeros_like(persistent_fg)
            for _bc in _pf_conts:
                if cv2.contourArea(_bc) < MIN_BLOB_AREA:
                    continue
                _bx, _by, _bw, _bh = cv2.boundingRect(_bc)
                if max(_bw, _bh) / max(1, min(_bw, _bh)) > 4.5:
                    continue    # shadow silhouette: too elongated
                _bm = np.zeros(persistent_fg.shape, dtype=np.uint8)
                cv2.drawContours(_bm, [_bc], -1, 255, -1)
                _v_mean = cv2.mean(hsv[:, :, 2], mask=_bm)[0]
                _s_mean = cv2.mean(hsv[:, :, 1], mask=_bm)[0]
                if _v_mean < 45 and _s_mean < 32:
                    continue    # shadow profile: dark + desaturated
                cv2.drawContours(_validated, [_bc], -1, 255, -1)

            _cws, _ = count_people_watershed_scene(
                _validated, min_blob_area=MIN_BLOB_AREA, dt_thresh=0.30
            )
            _car = (count_people_by_area(_validated, px_per_person, area_baseline)
                    if px_per_person is not None else 0)

            # YOLO census — appearance-based, catches stationary/absorbed people
            _yolo_n = 0
            if yolo_model is not None:
                _yolo_res = yolo_model(
                    frame, device="cpu", imgsz=YOLO_IMGSZ,
                    conf=YOLO_CONF, iou=0.35, classes=[0], verbose=False
                )[0]
                for _b in _yolo_res.boxes:
                    _bx1, _by1, _bx2, _by2 = _b.xyxy[0].tolist()
                    _bcx, _bcy = int((_bx1 + _bx2) / 2), int((_by1 + _by2) / 2)
                    _bcy = max(0, min(_bcy, roi_mask.shape[0] - 1))
                    _bcx = max(0, min(_bcx, roi_mask.shape[1] - 1))
                    if roi_mask[_bcy, _bcx] > 127:
                        _yolo_n += 1

            recount_floor = float(max(_cws, _car, _yolo_n))

        # Floor decay — only when the occupancy map itself is empty.
        # This distinguishes "stationary crowd absorbed by MOG2" (PEAK: occ has pixels)
        # from "scene truly empty" (late DEPART: occ evicted everyone).
        _occ_area = cv2.countNonZero(persistent_fg)
        if total < recount_floor and _occ_area < MIN_BLOB_AREA:
            recount_floor = max(float(total), recount_floor * FLOOR_DECAY)

        # Apply census floor
        total = max(total, int(round(recount_floor)))

        # Warmup suppression
        if loop_idx < WARMUP_FRAMES:
            total         = 0
            count_ema     = 0.0
            recount_floor = 0.0

        # EMA smoothing
        count_ema = COUNT_ALPHA * total + (1 - COUNT_ALPHA) * count_ema
        pred      = round(count_ema)

        # Record annotated frames
        if loop_idx in eval_set:
            gt_n = gt_counts[loop_idx]
            ph   = phase(loop_idx)
            results[loop_idx] = {"gt": gt_n, "pred": pred, "phase": ph}
            match = "x" if tier(pred) != tier(gt_n) else "o"
            print(f"  f={loop_idx:>6} | GT={gt_n:>3}  PRED={pred:>3}"
                  f"  err={pred - gt_n:+d}"
                  f"  |  {tier(gt_n):6s} -> {tier(pred):6s}  {match}"
                  f"  [{ph}]")

        loop_idx += 1

        if loop_idx % 1000 == 0:
            print(f"  ... frame {loop_idx}")

        if loop_idx > last_eval + 60:
            break

    cap.release()

    if not results:
        print("\n[ERROR] No frames were evaluated.")
        return

    # -- Overall metrics ------------------------------------------------------
    errors    = [abs(r["pred"] - r["gt"]) for r in results.values()]
    sq_errors = [(r["pred"] - r["gt"]) ** 2 for r in results.values()]
    signed    = [r["pred"] - r["gt"] for r in results.values()]

    mae  = sum(errors) / len(errors)
    rmse = (sum(sq_errors) / len(sq_errors)) ** 0.5
    bias = sum(signed) / len(signed)

    tier_correct = sum(
        1 for r in results.values() if tier(r["pred"]) == tier(r["gt"])
    )
    tier_acc = tier_correct / len(results) * 100

    print("\n" + "=" * 56)
    print(f"  EVALUATION SUMMARY  ({len(results)} annotated frames)")
    print("=" * 56)
    print(f"  MAE           : {mae:.2f} people")
    print(f"  RMSE          : {rmse:.2f} people")
    print(f"  Bias          : {bias:+.2f}  (+ = overcounting, - = undercounting)")
    print(f"  Tier Accuracy : {tier_acc:.1f}%  ({tier_correct}/{len(results)} correct)")
    print()

    # -- Per-tier breakdown ---------------------------------------------------
    print("  Tier breakdown:")
    for t in ["LOW", "MEDIUM", "HIGH"]:
        subset = [r for r in results.values() if tier(r["gt"]) == t]
        if not subset:
            continue
        t_mae = sum(abs(r["pred"] - r["gt"]) for r in subset) / len(subset)
        t_ok  = sum(1 for r in subset if tier(r["pred"]) == t)
        print(f"  {t:6s}  n={len(subset):>3}  MAE={t_mae:.2f}"
              f"  tier_acc={t_ok / len(subset) * 100:.0f}%")

    print()

    # -- Phase breakdown ------------------------------------------------------
    print(f"  Phase breakdown  (GT peak={peak_gt}, peak frame~={peak_frame}):")
    for ph in ["ENTRY", "PEAK", "DEPART"]:
        subset = [r for r in results.values() if r["phase"] == ph]
        if not subset:
            continue
        ph_mae  = sum(abs(r["pred"] - r["gt"]) for r in subset) / len(subset)
        ph_bias = sum(r["pred"] - r["gt"] for r in subset) / len(subset)
        print(f"  {ph:6s}  n={len(subset):>3}  MAE={ph_mae:.2f}"
              f"  bias={ph_bias:+.2f}")

    print()


if __name__ == "__main__":
    main()
