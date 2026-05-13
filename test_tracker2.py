import cv2
import json
from vision_engine import RobustSentinelTracker, is_human_blob, count_people_in_box
import numpy as np

def test():
    with open('barangay_ground_truth.json', 'r') as f:
        db = json.load(f)
    frames_db = {}
    for k, v in db['frames'].items():
        frames_db[int(k)] = len(v)

    cap = cv2.VideoCapture('videos/main_video.mp4')
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, bg_frame = cap.read()
    
    process_scale = 0.667
    proc_h = int(bg_frame.shape[0] * process_scale)
    proc_w = int(bg_frame.shape[1] * process_scale)
    bg_frame = cv2.resize(bg_frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
    h, w = bg_frame.shape[:2]
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ref_gray = cv2.cvtColor(bg_frame, cv2.COLOR_BGR2GRAY)
    ref_gray = clahe.apply(ref_gray)
    bg_reference = cv2.GaussianBlur(ref_gray, (5, 5), 0)

    roi_mask = cv2.imread('mask_layer1.png', 0)
    roi_mask = cv2.resize(roi_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    tracker = RobustSentinelTracker(max_ghost=30, dist_thresh=150, min_lifetime=10, merge_thresh=20)
    
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fusion_k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 10))
    h_fusion_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    d_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))

    frame_idx = 0
    errors = []
    
    while frame_idx <= 2500:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        curr_gray = clahe.apply(curr_gray)
        curr_gray = cv2.GaussianBlur(curr_gray, (5, 5), 0)
        diff = cv2.absdiff(curr_gray, bg_reference)
        _, fg_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        thresh = cv2.bitwise_and(fg_mask, fg_mask, mask=roi_mask)

        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, open_k)
        fused = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, fusion_k)
        fused = cv2.morphologyEx(fused, cv2.MORPH_CLOSE, h_fusion_k)
        
        conts, _ = cv2.findContours(fused, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for c in conts:
            x, y, w_box, h_box = cv2.boundingRect(c)
            roi_check = fused[y:y+h_box, x:x+w_box]
            if is_human_blob(roi_check, x, y, w_box, h_box, h, 350, 2.5):
                detections.append([x, y, w_box, h_box])
        
        tracks = tracker.update(detections)
        
        if frame_idx in frames_db and frame_idx > 0:
            true_count = frames_db[frame_idx]
            predicted_count = 0
            for tid, data in tracks.items():
                if data['ghost'] == 0 and data['lifetime'] >= 10:
                    px, py, pw_box, ph_box = data['box']
                    roi = fused[py : py + ph_box, px : px + pw_box]
                    predicted_count += count_people_in_box(roi, pw_box, py + ph_box, h, 0.60, 90, 0.3)
            
            error = abs(predicted_count - true_count)
            errors.append(error)
            print(f"Frame {frame_idx:4d} | True: {true_count:2d} | Pred: {predicted_count:2d} | Err: {error:2d}")
            
        frame_idx += 1

    if errors:
        print(f"MAE so far: {sum(errors)/len(errors):.2f}")

test()
