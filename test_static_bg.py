import cv2
import json
import numpy as np

def test_static_correlation():
    # Load ground truth
    with open('barangay_ground_truth.json', 'r') as f:
        db = json.load(f)
        
    roi_mask = cv2.imread('mask_layer1.png', 0)
    
    frames_db = {}
    for k, v in db['frames'].items():
        count = 0
        for pt in v:
            x, y = int(pt["x"]), int(pt["y"])
            if y < roi_mask.shape[0] and x < roi_mask.shape[1] and roi_mask[y, x] > 127:
                count += 1
        frames_db[int(k)] = count

    cap = cv2.VideoCapture('videos/main_video.mp4')
    
    # ── 1. GRAB STATIC REFERENCE (Frame 0 is empty) ──
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, bg_frame = cap.read()
    
    h = int(bg_frame.shape[0] * 0.667)
    w = int(bg_frame.shape[1] * 0.667)
    
    bg_frame = cv2.resize(bg_frame, (w, h), interpolation=cv2.INTER_AREA)
    roi_mask = cv2.resize(roi_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    bg_gray = cv2.cvtColor(bg_frame, cv2.COLOR_BGR2GRAY)
    bg_gray = clahe.apply(bg_gray)
    bg_gray = cv2.GaussianBlur(bg_gray, (5, 5), 0)
    
    results = {}
    
    for frame_idx in sorted(frames_db.keys()):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret: continue
        
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        curr_gray = clahe.apply(curr_gray)
        curr_gray = cv2.GaussianBlur(curr_gray, (5, 5), 0)
        
        # ── 2. STATIC DIFFERENCING ──
        diff = cv2.absdiff(curr_gray, bg_gray)
        
        # Threshold the difference
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        
        # Apply ROI
        thresh = cv2.bitwise_and(thresh, thresh, mask=roi_mask)
        
        # Morphological clean up (remove tiny sensor noise)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Count the absolute foreground area
        area = cv2.countNonZero(thresh)
        
        gt_count = frames_db[frame_idx]
        if gt_count not in results:
            results[gt_count] = []
        results[gt_count].append(area)

    # Calculate medians and R^2
    print(f"{'Count':>5} | {'Static Diff Area px':>20} | {'Samples':>7}")
    print("-" * 40)
    
    aggregated = {}
    for count in sorted(results.keys()):
        med = int(np.median(results[count]))
        aggregated[count] = med
        print(f"{count:5d} | {med:20d} | {len(results[count]):7d}")
        
    counts = sorted(aggregated.keys())
    areas = [aggregated[c] for c in counts]
    
    if len(counts) > 1:
        slope, intercept = np.polyfit(counts, areas, 1)
        predicted = [slope * c + intercept for c in counts]
        ss_res = sum((a - p)**2 for a, p in zip(areas, predicted))
        ss_tot = sum((a - sum(areas)/len(areas))**2 for a in areas)
        r2 = 1 - (ss_res / ss_tot)
        print("\nMathematical Correlation (R²): {:.4f}".format(r2))

test_static_correlation()
