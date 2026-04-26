import cv2
import json
import itertools
import numpy as np
from vision_engine import RobustSentinelTracker, is_human_blob, count_people_in_box
import os

# -------------------------------------------------------------
# Configuration Matrix (The "Grid") - NIGHTTIME TUNING
# -------------------------------------------------------------
PARAM_GRID = {
    'varThreshold': [8, 12, 16],        # Lower is more sensitive (good for night)
    'history': [200, 300, 500],         # Faster adaptation for fluctuating night light
    'morph_kernel': [(5, 35), (7, 50)], # Smaller kernels may capture thin night silhouettes
    'dilate_kernel': [1, 3],            # Minimal dilation to avoid merging noise
    'min_blob_area': [400, 600, 800]    # Smaller threshold for distant/dim people
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
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=params['history'], 
        varThreshold=params['varThreshold'], 
        detectShadows=True
    )
    fusion_k = cv2.getStructuringElement(cv2.MORPH_RECT, params['morph_kernel'])
    d_kernel = np.ones((params['dilate_kernel'], params['dilate_kernel']), np.uint8)
    tracker = RobustSentinelTracker()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return float('inf')

    # Read first frame to get dims
    ret, first = cap.read()
    if not ret: return float('inf')
    h, w = first.shape[:2]
    
    roi_mask = cv2.imread(mask_path, 0)
    if roi_mask is None or roi_mask.shape[:2] != (h, w):
        roi_mask = np.ones((h, w), dtype=np.uint8) * 255

    # Reset video
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    frame_idx = 0
    errors = []

    while frame_idx <= max_frame:
        ret, frame = cap.read()
        if not ret: break

        # ── Pipeline ──────────────────────────────────
        fg_mask = bg_subtractor.apply(frame)
        _, thresh = cv2.threshold(fg_mask, 254, 255, cv2.THRESH_BINARY)
        thresh = cv2.bitwise_and(thresh, thresh, mask=roi_mask)

        fused = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, fusion_k)
        fused = cv2.dilate(fused, d_kernel, iterations=1)

        conts, _ = cv2.findContours(fused, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for c in conts:
            x, y, w_box, h_box = cv2.boundingRect(c)
            roi_check = thresh[y : y + h_box, x : x + w_box]
            # Pass tunable min_blob_area
            if is_human_blob(roi_check, x, y, w_box, h_box, h, params['min_blob_area']):
                detections.append([x, y, w_box, h_box])

        tracks = tracker.update(detections)

        # ── Mathematical Grading ──────────────────────
        if frame_idx in frames_db:
            true_count = frames_db[frame_idx]
            
            # Count the engine's detections
            predicted_count = 0
            for tid, data in tracks.items():
                if data['ghost'] == 0:
                    px, py, pw_box, ph_box = data['box']
                    roi = thresh[py : py + ph_box, px : px + pw_box]
                    
                    # Full-frame counting
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
