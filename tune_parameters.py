import cv2
import json
import itertools
import numpy as np
from vision_engine import RobustSentinelTracker, is_human_blob, count_people_in_box
import os

# -------------------------------------------------------------
# Configuration Matrix (The "Grid")
# -------------------------------------------------------------
PARAM_GRID = {
    'varThreshold': [16, 50, 75],    # Controls shadow/noise sensitivity
    'history': [300, 500],           # Background memory 
    'morph_kernel': [(7, 50), (10, 85)], # Structural mega-blob fusion sizes
    'dilate_kernel': [3, 5]          # Pixel widening
}

def load_ground_truth():
    with open('barangay_ground_truth.json', 'r') as f:
        db = json.load(f)
    frames_db = {int(k): len(v) for k, v in db['frames'].items()}
    return db['video_source'], frames_db

def test_configuration(video_path, frames_db, max_frame, params, mask_path="mask_layer.png"):
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

    # ZONES definition removed for full-frame processing

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
            if is_human_blob(roi_check, x, y, w_box, h_box, h):
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
                    cx, cy = int(px + pw_box/2), int(py + ph_box)
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
    print(" BARANGAY SENTINEL: AUTOMATED GRID SEARCH OPTIMIZATION ")
    print("="*60)
    
    if not os.path.exists('barangay_ground_truth.json'):
        print("ERROR: missing ground truth JSON. Run annotation tool first.")
        return

    video_path, frames_db = load_ground_truth()
    annotated_indices = sorted(list(frames_db.keys()))
    max_frame = annotated_indices[-1]
    
    print(f"Loaded {len(frames_db)} mathematical ground truth frames.")
    print(f"Max required frame processing depth: {max_frame}")
    print()

    # Generate all combinations
    keys = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
    
    print(f"Beggining rigorous testing of {len(combinations)} algorithmic mutations...")
    print("-" * 60)

    best_mae = float('inf')
    best_params = None

    for i, params in enumerate(combinations):
        print(f"[{i+1}/{len(combinations)}] Testing -> Var:{params['varThreshold']} Hist:{params['history']} Morph:{params['morph_kernel']} Dilate:{params['dilate_kernel']}...", end='', flush=True)
        
        mae = test_configuration(video_path, frames_db, max_frame, params)
        
        print(f" MAE Error = {mae:.2f} people")
        
        if mae < best_mae:
            best_mae = mae
            best_params = params

    print("\n" + "="*60)
    print(" GRID SEARCH COMPLETED - OPTIMAL PARAMETERS FOUND ")
    print("="*60)
    print(f"Lowest Mean Absolute Error: {best_mae:.2f} people discrepancy")
    print("You should update vision_engine.py with these values:")
    for k, v in best_params.items():
        print(f" -> {k}: {v}")

if __name__ == "__main__":
    main()
