import cv2
import json
import itertools
import numpy as np
from vision_engine import is_human_blob, suppress_headlights, OccupancyMap
import os

# -------------------------------------------------------------
# Configuration Matrix — ROUND 4 (Counting Gate Fix)
# Round 3 confirmed:
#   - max_aspect fix only marginally improved MAE (2.98 → 2.95)
#   - min_blob_area=600 made things worse (EMPTY fix needs different approach)
#   - PRIMARY BOTTLENECK: solidity_threshold=0.75 in count_people_in_box
#     collapses 8-10 person compact group blobs to count=1
#   - SECONDARY: DT valley threshold=0.4 too high, merges peaks of nearby people
# Strategy:
#   - Lower solidity_threshold to allow sub-counting within dense groups
#   - Lower dt_thresh so watershed separates closely-seated person peaks
#   - Keep all other parameters locked to round 2/3 winners
# -------------------------------------------------------------
PARAM_GRID = {
    'varThreshold':       [40],               # Locked winner
    'history':            [10000, 30000],     # THE KEY: 5-16 minutes of absorption resistance
    'morph_kernel':       [(5, 25)],          # Locked heavy vertical fusion
    'dilate_kernel':      [1],                # Locked
    'min_blob_area':      [350],              # Locked winner
    'ghost_threshold':    [90],               # Locked
    'h_morph_kernel':     [(10, 3)],          # Locked heavy horizontal fusion
    'merge_thresh':       [20],               # Locked
    'dist_thresh':        [150],              # Locked
    'max_aspect':         [2.5, 3.5],         # Wide allowance for seated fused groups
    'base_width':         [90],               # Unused by area counter, kept for compat
    'solidity_threshold': [0.6],              # Unused by area counter
    'dt_thresh':          [0.3],              # Unused by area counter
}

def load_ground_truth(mask_path="mask_layer1.png"):
    with open('barangay_ground_truth.json', 'r') as f:
        db = json.load(f)
    roi_mask = cv2.imread(mask_path, 0)
    
    frames_db = {}
    for k, v in db['frames'].items():
        count_in_roi = 0
        for pt in v:
            x, y = int(pt["x"]), int(pt["y"])
            if roi_mask is not None and y < roi_mask.shape[0] and x < roi_mask.shape[1]:
                if roi_mask[y, x] > 127:
                    count_in_roi += 1
            else:
                count_in_roi += 1
        frames_db[int(k)] = count_in_roi
    return db['video_source'], frames_db

def test_configuration(video_path, frames_db, max_frame, params, mask_path="mask_layer1.png"):
    """
    Runs the full vision pipeline for one parameter combination up to the
    maximum annotated frame and returns Mean Absolute Error (MAE).

    Pipeline mirrors app.py exactly:
      MOG2 → headlight suppression → morphology → occupancy map → area count
    """
    process_scale = 0.667
    warmup_frames = min(500, params['history'] // 2)

    open_k     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fusion_k   = cv2.getStructuringElement(cv2.MORPH_RECT, params['morph_kernel'])
    h_fusion_k = cv2.getStructuringElement(cv2.MORPH_RECT, params['h_morph_kernel'])
    d_kernel   = np.ones((params['dilate_kernel'], params['dilate_kernel']), np.uint8)

    mog2 = cv2.createBackgroundSubtractorMOG2(
        history=params['history'],
        varThreshold=params['varThreshold'],
        detectShadows=False,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return float('inf')

    ret, first_frame = cap.read()
    if not ret:
        return float('inf')

    h = int(first_frame.shape[0] * process_scale)
    w = int(first_frame.shape[1] * process_scale)

    raw_mask = cv2.imread(mask_path, 0)
    if raw_mask is not None:
        roi_mask = cv2.resize(raw_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        roi_mask = np.ones((h, w), dtype=np.uint8) * 255

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    occupancy = OccupancyMap(
        (h, w),
        confirm_frames=10,
        evict_frames=int(5.0 * 30),
        dark_v_thresh=40,
    )

    frame_idx = 0
    errors = []

    while frame_idx <= max_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if process_scale != 1.0:
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # ── PHASE 1: MOG2 ─────────────────────────────────────────────
        fg_raw = mog2.apply(frame)
        _, fg_mask = cv2.threshold(fg_raw, 200, 255, cv2.THRESH_BINARY)
        thresh = cv2.bitwise_and(fg_mask, fg_mask, mask=roi_mask)

        # ── HEADLIGHT SUPPRESSION ─────────────────────────────────────
        thresh = suppress_headlights(thresh, hsv, v_threshold=200, dilation_px=40)

        # ── PHASE 2: Morphological Sculpting ─────────────────────────
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, open_k)
        fused   = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, fusion_k)
        fused   = cv2.morphologyEx(fused, cv2.MORPH_CLOSE, h_fusion_k)
        fused   = cv2.dilate(fused, d_kernel, iterations=1)

        # ── OCCUPANCY MAP ─────────────────────────────────────────────
        persistent_fg = occupancy.update(fused, hsv)

        # ── PHASE 3: Blob Detection (is_human_blob gate) ──────────────
        conts, _ = cv2.findContours(fused, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for c in conts:
            x, y, w_box, h_box = cv2.boundingRect(c)
            roi_check = fused[y : y + h_box, x : x + w_box]
            if is_human_blob(roi_check, x, y, w_box, h_box, h, params['min_blob_area'], params['max_aspect']):
                detections.append([x, y, w_box, h_box])

        # ── PHASE 4: Area-Based Scoring (from occupancy map) ──────────
        if frame_idx in frames_db and frame_idx >= warmup_frames:
            true_count = frames_db[frame_idx]
            area = int(cv2.countNonZero(persistent_fg))
            predicted_count = max(0, round(
                (area - params['area_baseline']) / max(1.0, params['area_px_per_person'])
            ))
            errors.append(abs(predicted_count - true_count))

        frame_idx += 1

    cap.release()

    if len(errors) == 0:
        return float('inf')
    return sum(errors) / len(errors)

def main():
    print("="*60)
    print(" BARANGAY SENTINEL: NIGHTTIME GRID SEARCH OPTIMIZATION ")
    print("="*60)

    if not os.path.exists('barangay_ground_truth.json'):
        print("ERROR: missing ground truth JSON. Run annotation tool first.")
        return

    if not os.path.exists('area_calibration.json'):
        print("ERROR: area_calibration.json not found. Run calibrate_area.py first.")
        return

    with open('area_calibration.json', 'r') as f:
        cal = json.load(f)
    area_px_per_person = float(cal['avg_pixels_per_person'])
    area_baseline      = float(cal['area_at_zero'])
    print(f"[Area Cal] {area_px_per_person:.1f} px/person  baseline={area_baseline:.1f}px  R²={cal.get('r_squared', '?')}")

    video_path, frames_db = load_ground_truth("mask_layer1.png")
    annotated_indices = sorted(list(frames_db.keys()))
    max_frame = annotated_indices[-1]

    print(f"Video Source: {video_path}")
    print(f"Loaded {len(frames_db)} ground truth frames, max frame {max_frame}")
    print()

    keys = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    # Inject calibration constants into every combination
    for combo in combinations:
        combo['area_px_per_person'] = area_px_per_person
        combo['area_baseline']      = area_baseline

    print(f"Beginning rigorous testing of {len(combinations)} algorithmic mutations...")
    print("-" * 60)

    best_mae = float('inf')
    best_params = None

    for i, params in enumerate(combinations):
        print(f"[{i+1}/{len(combinations)}] MAE Testing...", end='', flush=True)

        mae = test_configuration(video_path, frames_db, max_frame, params, "mask_layer1.png")

        print(f" Result = {mae:.2f} people")

        if mae < best_mae:
            best_mae = mae
            best_params = params

    print("\n" + "="*60)
    print(" GRID SEARCH COMPLETED - OPTIMAL PARAMETERS FOUND ")
    print("="*60)
    print(f"Lowest Mean Absolute Error: {best_mae:.2f} people discrepancy")
    print("Update vision_engine.py / app.py with these optimal nighttime values:")
    display_keys = [k for k in best_params if k not in ('area_px_per_person', 'area_baseline')]
    for k in display_keys:
        print(f" -> {k}: {best_params[k]}")

if __name__ == "__main__":
    main()
