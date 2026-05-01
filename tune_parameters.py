import cv2
import json
import itertools
import numpy as np
from vision_engine import RobustSentinelTracker, is_human_blob, count_people_in_box
import os

# -------------------------------------------------------------
# Configuration Matrix (The "Grid") - NIGHTTIME / WELL-LIT TUNING
# -------------------------------------------------------------
PARAM_GRID = {
    'varThreshold':    [8],
    'history':         [1000],
    'morph_kernel':    [(7, 50)],
    'dilate_kernel':   [1],
    'min_blob_area':   [450],
    'ghost_threshold': [180],
    'h_morph_kernel':  [(20, 5), (35, 7), (50, 9)],
    'merge_thresh':    [20, 35, 50],
    'dist_thresh':     [100, 150, 200],
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
    Runs the vision engine logic using the specific parameter combo
    up to the maximum annotated frame.
    Returns: Mean Absolute Error (MAE)
    """
    process_scale = 0.667
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=params['history'], 
        varThreshold=params['varThreshold'], 
        detectShadows=False
    )
    # Build kernels matching new engine logic
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fusion_k = cv2.getStructuringElement(cv2.MORPH_RECT, params['morph_kernel'])
    h_fusion_k = cv2.getStructuringElement(cv2.MORPH_RECT, params['h_morph_kernel'])
    d_kernel = np.ones((params['dilate_kernel'], params['dilate_kernel']), np.uint8)
    
    # Enable Temporal Validation in the tuner
    tracker = RobustSentinelTracker(
        max_ghost=params['ghost_threshold'],
        min_lifetime=10,
        dist_thresh=params['dist_thresh'],
        merge_thresh=params['merge_thresh'],
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return float('inf')

    # Read first frame to get dims
    ret, first = cap.read()
    if not ret: return float('inf')
    
    if process_scale != 1.0:
        proc_h = int(first.shape[0] * process_scale)
        proc_w = int(first.shape[1] * process_scale)
        first = cv2.resize(first, (proc_w, proc_h), interpolation=cv2.INTER_AREA)

    h, w = first.shape[:2]
    
    raw_mask = cv2.imread(mask_path, 0)
    if raw_mask is not None:
        roi_mask = cv2.resize(raw_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        if roi_mask.shape[:2] != (h, w):
            roi_mask = np.ones((h, w), dtype=np.uint8) * 255
    else:
        roi_mask = np.ones((h, w), dtype=np.uint8) * 255

    # Reset video
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    frame_idx = 0
    errors = []

    while frame_idx <= max_frame:
        ret, frame = cap.read()
        if not ret: break

        if process_scale != 1.0:
            proc_h = int(frame.shape[0] * process_scale)
            proc_w = int(frame.shape[1] * process_scale)
            frame = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA)

        # ── Pipeline (MATCHING NEW ENGINE) ──────────
        # Nighttime: normalize local contrast before MOG2
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_ch = clahe.apply(l_ch)
        lab = cv2.merge((l_ch, a_ch, b_ch))
        frame_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        fg_mask = bg_subtractor.apply(frame_enhanced)
        
        # 1. Median Blur
        fg_mask = cv2.medianBlur(fg_mask, 3)
        
        _, thresh = cv2.threshold(fg_mask, 254, 255, cv2.THRESH_BINARY)
        thresh = cv2.bitwise_and(thresh, thresh, mask=roi_mask)

        # 2. Opening & Fusion
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, open_k)
        fused = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, fusion_k)
        fused = cv2.morphologyEx(fused, cv2.MORPH_CLOSE, h_fusion_k)
        fused = cv2.dilate(fused, d_kernel, iterations=1)

        conts, _ = cv2.findContours(fused, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for c in conts:
            x, y, w_box, h_box = cv2.boundingRect(c)
            roi_check = thresh[y : y + h_box, x : x + w_box]
            if is_human_blob(roi_check, x, y, w_box, h_box, h, params['min_blob_area']):
                detections.append([x, y, w_box, h_box])

        tracks = tracker.update(detections)

        # ── Mathematical Grading ──────────────────────
        if frame_idx in frames_db and frame_idx > 0:
            true_count = frames_db[frame_idx]
            
            # Count the engine's detections (WITH LIFETIME CHECK)
            predicted_count = 0
            for tid, data in tracks.items():
                if data['ghost'] == 0 and data['lifetime'] >= tracker.min_lifetime:
                    px, py, pw_box, ph_box = data['box']
                    roi = thresh[py : py + ph_box, px : px + pw_box]
                    predicted_count += count_people_in_box(roi, pw_box, py + ph_box, h)
                        
            # Calculate Absolute Error
            error = abs(predicted_count - true_count)
            errors.append(error)

        frame_idx += 1

    cap.release()
    
    if len(errors) == 0: 
        return float('inf')
    return sum(errors) / len(errors)  # True MAE

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
