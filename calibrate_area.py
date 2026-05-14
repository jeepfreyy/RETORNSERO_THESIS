import cv2
import json
import numpy as np
import os
from vision_engine import suppress_headlights, OccupancyMap

PIPELINE_PARAMS = {
    'varThreshold':   40,
    'history':        30000,
    'morph_kernel':   (5, 25),
    'dilate_kernel':  1,
    'h_morph_kernel': (10, 3),
}
# These must match app.py so calibration and live pipeline are identical
HEADLIGHT_V_THRESH   = 160   # lowered — halo sits at V=160-190
HEADLIGHT_DILATION   = 80   # widened — kills full halo radius
OCCUPANCY_CONFIRM    = 3     # frames — must match app.py occupancy_confirm_frames
OCCUPANCY_EVICT_SEC  = 10.0  # must match app.py occupancy_evict_sec
OCCUPANCY_DARK_THRESH= 40
PROCESS_SCALE = 0.667
VIDEO_PATH    = 'videos/calibration.MOV'
MASK_PATH     = 'mask_layer_calibration.png'
GT_JSON       = 'barangay_ground_truth_calibration.json'
OUTPUT_JSON   = 'area_calibration.json'
WARMUP_FRAMES = 1000


def load_ground_truth():
    with open(GT_JSON, 'r') as f:
        db = json.load(f)
    roi_mask = cv2.imread(MASK_PATH, 0)
    frames_db = {}
    for k, v in db['frames'].items():
        count_in_roi = 0
        for pt in v:
            x, y = int(pt['x']), int(pt['y'])
            if roi_mask is not None and y < roi_mask.shape[0] and x < roi_mask.shape[1]:
                if roi_mask[y, x] > 127:
                    count_in_roi += 1
            else:
                count_in_roi += 1
        frames_db[int(k)] = count_in_roi
    return frames_db


def main():
    print("=" * 60)
    print(" BARANGAY SENTINEL: AREA CALIBRATION ")
    print("=" * 60)

    frames_db = load_ground_truth()
    annotated_indices = sorted(frames_db.keys())
    max_frame = annotated_indices[-1]
    print(f"Ground truth: {len(frames_db)} annotated frames, max frame {max_frame}")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {VIDEO_PATH}")
        return

    ret, first = cap.read()
    if not ret:
        print("ERROR: Cannot read first frame.")
        cap.release()
        return

    proc_h = int(first.shape[0] * PROCESS_SCALE)
    proc_w = int(first.shape[1] * PROCESS_SCALE)
    h, w = proc_h, proc_w

    raw_mask = cv2.imread(MASK_PATH, 0)
    if raw_mask is not None:
        roi_mask = cv2.resize(raw_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        roi_mask = np.ones((h, w), dtype=np.uint8) * 255

    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=PIPELINE_PARAMS['history'],
        varThreshold=PIPELINE_PARAMS['varThreshold'],
        detectShadows=False
    )
    open_k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fusion_k  = cv2.getStructuringElement(cv2.MORPH_RECT, PIPELINE_PARAMS['morph_kernel'])
    h_fuse_k  = cv2.getStructuringElement(cv2.MORPH_RECT, PIPELINE_PARAMS['h_morph_kernel'])
    d_kernel  = np.ones((PIPELINE_PARAMS['dilate_kernel'], PIPELINE_PARAMS['dilate_kernel']), np.uint8)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Occupancy map — same parameters as app.py so calibration matches live system
    occupancy = OccupancyMap(
        (h, w),
        confirm_frames=OCCUPANCY_CONFIRM,
        evict_frames=int(OCCUPANCY_EVICT_SEC * 30),
        dark_v_thresh=OCCUPANCY_DARK_THRESH,
    )

    # {person_count: [area, area, ...]}
    raw_data = {}
    frame_idx = 0
    scored = 0

    print(f"Processing frames 0–{max_frame}...")
    while frame_idx <= max_frame:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        fg_mask = bg_subtractor.apply(frame)
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        thresh = cv2.bitwise_and(thresh, thresh, mask=roi_mask)

        # Apply same headlight suppression as the live pipeline
        thresh = suppress_headlights(thresh, hsv,
                                     v_threshold=HEADLIGHT_V_THRESH,
                                     dilation_px=HEADLIGHT_DILATION)

        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, open_k)
        fused   = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, fusion_k)
        fused   = cv2.morphologyEx(fused, cv2.MORPH_CLOSE, h_fuse_k)
        fused   = cv2.dilate(fused, d_kernel, iterations=1)

        # Update occupancy map — measure the SAME thing the live system will count
        persistent_fg = occupancy.update(fused, hsv)

        if frame_idx in frames_db and frame_idx >= WARMUP_FRAMES:
            person_count = frames_db[frame_idx]
            # Measure occupancy map area, not raw fused area
            area = int(cv2.countNonZero(persistent_fg))
            raw_data.setdefault(person_count, []).append(area)
            scored += 1

        frame_idx += 1
        if frame_idx % 500 == 0:
            print(f"  frame {frame_idx}/{max_frame}  scored={scored}")

    cap.release()
    print(f"Done. {scored} frames scored across {len(raw_data)} unique crowd sizes.")

    if len(raw_data) < 2:
        print("ERROR: Need at least 2 distinct crowd sizes for regression. Add more annotations.")
        return

    # Median area per person count
    aggregated = {k: float(np.median(v)) for k, v in raw_data.items()}

    # Exclude count=0 from regression:
    # End-of-video "empty" frames have a full occupancy map from prior activity,
    # so their area is artifically inflated and poisons the linear fit.
    # The intercept (area_at_zero) is set to 0 — an empty scene = 0 confirmed pixels
    # once the system has had time to evict all previous occupants.
    regression_counts = sorted(k for k in aggregated if k > 0)
    if len(regression_counts) < 2:
        print("ERROR: Need at least 2 distinct non-zero crowd sizes for regression.")
        return
    regression_areas = [aggregated[k] for k in regression_counts]

    counts = sorted(aggregated.keys())
    areas  = [aggregated[k] for k in counts]

    slope, intercept = np.polyfit(regression_counts, regression_areas, 1)
    avg_pixels_per_person = float(slope)
    # Clamp intercept: a negative intercept is physically impossible (area ≥ 0)
    area_at_zero          = float(max(0.0, intercept))

    # R² computed only over the regression set (non-zero counts)
    predicted_reg = [slope * c + intercept for c in regression_counts]
    mean_a_reg    = sum(regression_areas) / len(regression_areas)
    ss_res        = sum((a - p) ** 2 for a, p in zip(regression_areas, predicted_reg))
    ss_tot        = sum((a - mean_a_reg) ** 2 for a in regression_areas)
    r_squared     = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Build predicted for ALL counts (including 0) for display
    predicted = [slope * c + intercept for c in counts]

    print()
    print(f"{'Count':>6}  {'Median Area':>12}  {'Predicted':>10}  {'Samples':>7}  {'In Fit':>6}")
    print("-" * 52)
    for c, a, p in zip(counts, areas, predicted):
        n = len(raw_data[c])
        in_fit = "YES" if c > 0 else "EXCL"
        print(f"{c:>6}  {a:>12.0f}  {p:>10.0f}  {n:>7}  {in_fit:>6}")
    print()
    print(f"avg_pixels_per_person : {avg_pixels_per_person:.2f}")
    print(f"area_at_zero          : {area_at_zero:.2f}")
    print(f"R²                    : {r_squared:.4f}")

    result = {
        "avg_pixels_per_person": avg_pixels_per_person,
        "area_at_zero":          area_at_zero,
        "r_squared":             r_squared,
        "data_points":           scored,
        "per_count_median_area": {str(k): int(aggregated[k]) for k in counts},
        "pipeline_params":       {
            **PIPELINE_PARAMS,
            "morph_kernel":   list(PIPELINE_PARAMS["morph_kernel"]),
            "h_morph_kernel": list(PIPELINE_PARAMS["h_morph_kernel"]),
        },
    }
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(result, f, indent=4)
    print(f"\nSaved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
