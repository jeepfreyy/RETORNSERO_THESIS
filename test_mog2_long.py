import cv2
import json
import numpy as np

def test():
    with open('barangay_ground_truth.json', 'r') as f:
        db = json.load(f)
    frames_db = {}
    for k, v in db['frames'].items():
        count = 0
        for pt in v:
            count += 1
        frames_db[int(k)] = count

    cap = cv2.VideoCapture('videos/main_video.mp4')
    ret, frame = cap.read()
    h = int(frame.shape[0] * 0.667)
    w = int(frame.shape[1] * 0.667)
    
    roi_mask = cv2.imread('mask_layer1.png', 0)
    roi_mask = cv2.resize(roi_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    mog2 = cv2.createBackgroundSubtractorMOG2(history=30000, varThreshold=40, detectShadows=False)
    
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fusion_k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 25))
    h_fusion_k = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 3))

    results = {}
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        fg_mask = mog2.apply(frame)
        
        if frame_idx in frames_db and frame_idx > 500:
            thresh = cv2.bitwise_and(fg_mask, fg_mask, mask=roi_mask)
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, open_k)
            fused = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, fusion_k)
            fused = cv2.morphologyEx(fused, cv2.MORPH_CLOSE, h_fusion_k)
            
            area = cv2.countNonZero(fused)
            gt = frames_db[frame_idx]
            if gt not in results:
                results[gt] = []
            results[gt].append(area)
            
        frame_idx += 1

    aggregated = {}
    for count in sorted(results.keys()):
        med = int(np.median(results[count]))
        aggregated[count] = med
        
    counts = sorted(aggregated.keys())
    areas = [aggregated[c] for c in counts]
    
    if len(counts) > 1:
        slope, intercept = np.polyfit(counts, areas, 1)
        predicted = [slope * c + intercept for c in counts]
        ss_res = sum((a - p)**2 for a, p in zip(areas, predicted))
        ss_tot = sum((a - sum(areas)/len(areas))**2 for a in areas)
        r2 = 1 - (ss_res / ss_tot)
        print(f"MOG2 History 30000 R2: {r2:.4f}")
        for c in counts:
            print(f"Count {c:2d}: {aggregated[c]} area")

test()
