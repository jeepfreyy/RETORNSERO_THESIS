import cv2
import json
import numpy as np

def test_edge_correlation():
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
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    results = {}
    
    for frame_idx in sorted(frames_db.keys()):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret: continue
        
        # Resize to match pipeline
        h = int(frame.shape[0] * 0.667)
        w = int(frame.shape[1] * 0.667)
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(roi_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Classical Texture Extraction (Grayscale + CLAHE + Canny)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = clahe.apply(gray)
        
        # Slight blur to remove pure sensor noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Apply ROI mask
        roi_edges = cv2.bitwise_and(edges, edges, mask=mask)
        
        # Count edge pixels
        edge_count = cv2.countNonZero(roi_edges)
        
        gt_count = frames_db[frame_idx]
        if gt_count not in results:
            results[gt_count] = []
        results[gt_count].append(edge_count)

    # Calculate medians and R^2
    print(f"{'Count':>5} | {'Median Edge px':>15} | {'Samples':>7}")
    print("-" * 35)
    
    aggregated = {}
    for count in sorted(results.keys()):
        med = int(np.median(results[count]))
        aggregated[count] = med
        print(f"{count:5d} | {med:15d} | {len(results[count]):7d}")
        
    counts = sorted(aggregated.keys())
    edges = [aggregated[c] for c in counts]
    
    if len(counts) > 1:
        slope, intercept = np.polyfit(counts, edges, 1)
        predicted = [slope * c + intercept for c in counts]
        ss_res = sum((a - p)**2 for a, p in zip(edges, predicted))
        ss_tot = sum((a - sum(edges)/len(edges))**2 for a in edges)
        r2 = 1 - (ss_res / ss_tot)
        print("\nMathematical Correlation (R²): {:.4f}".format(r2))

test_edge_correlation()
