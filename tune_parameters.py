import cv2
import json
import itertools
import numpy as np
from vision_engine import RobustSentinelTracker, is_human_blob, count_people_in_box
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
    'varThreshold':       [20, 25, 30],          # Now serves as the absdiff threshold
    'history':            [1000],                # Unused by Static BG, but kept for compat
    'morph_kernel':       [(3, 3), (5, 5), (3, 10)],  # Dramatically lighter vertical fusion
    'dilate_kernel':      [1],                   # Locked
    'min_blob_area':      [250, 350],            # Can afford smaller with static ref
    'ghost_threshold':    [90],                  # Locked
    'h_morph_kernel':     [(3, 3), (5, 3)],     # Dramatically lighter horizontal fusion
    'merge_thresh':       [20],                  # Locked
    'dist_thresh':        [150],                 # Locked
    'max_aspect':         [2.0, 2.5],            # Locked to reasonable overhead aspect
    'base_width':         [80, 90],              # Calibrated person width
    'solidity_threshold': [0.60, 0.70],          # Watershed trigger gate
    'dt_thresh':          [0.30, 0.40],          # Watershed valley floor
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
    Runs the static-reference vision pipeline for one parameter combination
    up to the maximum annotated frame.
    Returns: Mean Absolute Error (MAE)
    """
    process_scale = 0.667

    open_k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fusion_k  = cv2.getStructuringElement(cv2.MORPH_RECT, params['morph_kernel'])
    h_fusion_k = cv2.getStructuringElement(cv2.MORPH_RECT, params['h_morph_kernel'])
    d_kernel  = np.ones((params['dilate_kernel'], params['dilate_kernel']), np.uint8)
    clahe     = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    tracker = RobustSentinelTracker(
        max_ghost=params['ghost_threshold'],
        min_lifetime=10,
        dist_thresh=params['dist_thresh'],
        merge_thresh=params['merge_thresh'],
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return float('inf')

    # Read frame 0 — this is both the dims source and the static reference
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, bg_frame = cap.read()
    if not ret:
        return float('inf')

    if process_scale != 1.0:
        proc_h = int(bg_frame.shape[0] * process_scale)
        proc_w = int(bg_frame.shape[1] * process_scale)
        bg_frame = cv2.resize(bg_frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA)

    h, w = bg_frame.shape[:2]

    # Build static reference from frame 0
    ref_gray = cv2.cvtColor(bg_frame, cv2.COLOR_BGR2GRAY)
    ref_gray = clahe.apply(ref_gray)
    bg_reference = cv2.GaussianBlur(ref_gray, (5, 5), 0)

    raw_mask = cv2.imread(mask_path, 0)
    if raw_mask is not None:
        roi_mask = cv2.resize(raw_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        if roi_mask.shape[:2] != (h, w):
            roi_mask = np.ones((h, w), dtype=np.uint8) * 255
    else:
        roi_mask = np.ones((h, w), dtype=np.uint8) * 255

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_idx = 0
    errors = []

    while frame_idx <= max_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if process_scale != 1.0:
            proc_h = int(frame.shape[0] * process_scale)
            proc_w = int(frame.shape[1] * process_scale)
            frame = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA)

        # ── PHASE 1: Static Reference Differencing ────────────────────
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        curr_gray = clahe.apply(curr_gray)
        curr_gray = cv2.GaussianBlur(curr_gray, (5, 5), 0)
        diff = cv2.absdiff(curr_gray, bg_reference)
        _, fg_mask = cv2.threshold(diff, params['varThreshold'], 255, cv2.THRESH_BINARY)
        thresh = cv2.bitwise_and(fg_mask, fg_mask, mask=roi_mask)

        # ── PHASE 2: Morphological Sculpting ─────────────────────────
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, open_k)
        fused   = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, fusion_k)
        fused   = cv2.morphologyEx(fused, cv2.MORPH_CLOSE, h_fusion_k)
        fused   = cv2.dilate(fused, d_kernel, iterations=1)

        # ── PHASE 3: Blob Detection + Tracking ───────────────────────
        conts, _ = cv2.findContours(fused, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for c in conts:
            x, y, w_box, h_box = cv2.boundingRect(c)
            roi_check = thresh[y : y + h_box, x : x + w_box]
            if is_human_blob(roi_check, x, y, w_box, h_box, h, params['min_blob_area'], params['max_aspect']):
                detections.append([x, y, w_box, h_box])

        tracks = tracker.update(detections)

        # ── Scoring (skip frame 0 — it IS the reference) ─────────────
        if frame_idx in frames_db and frame_idx > 0:
            true_count = frames_db[frame_idx]
            predicted_count = 0
            for tid, data in tracks.items():
                if data['ghost'] == 0 and data['lifetime'] >= tracker.min_lifetime:
                    px, py, pw_box, ph_box = data['box']
                    roi = thresh[py : py + ph_box, px : px + pw_box]
                    predicted_count += count_people_in_box(
                        roi, pw_box, py + ph_box, h,
                        solidity_threshold=params['solidity_threshold'],
                        base_width=params['base_width'],
                        dt_thresh=params['dt_thresh'],
                    )
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

    # Use the new mask_layer1.png
    video_path, frames_db = load_ground_truth("mask_layer1.png")
    annotated_indices = sorted(list(frames_db.keys()))
    max_frame = annotated_indices[-1]
    
    print(f"Video Source: {video_path}")
    print(f"Loaded {len(frames_db)} mathematical ground truth frames.")
    print(f"Max required frame processing depth: {max_frame}")
    print()

    # Generate all combinations
    keys = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]

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
    for k, v in best_params.items():
        print(f" -> {k}: {v}")

if __name__ == "__main__":
    main()
