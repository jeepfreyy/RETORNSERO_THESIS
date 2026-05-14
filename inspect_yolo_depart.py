#!/usr/bin/env python3
"""
Visualise YOLO detections on late-DEPART census frames.

Saves annotated images to  yolo_inspect/  so you can open them and see
exactly what YOLO is picking up when GT is low but PRED stays high.

Usage:
    python3 inspect_yolo_depart.py
"""
import cv2, json, os, numpy as np
from ultralytics import YOLO

VIDEO_PATH    = "videos/calibration.MOV"
MASK_PATH     = "mask_layer_calibration.png"
GT_PATH       = "barangay_ground_truth_calibration.json"
OUT_DIR       = "yolo_inspect"
PROCESS_SCALE = 0.667
YOLO_CONF     = 0.40

# Frames to inspect — every census frame in the problematic late-DEPART window
# (every 60 frames from f=11700 onward so we see each YOLO census run)
INSPECT_FRAMES = list(range(11700, 12780, 60))

os.makedirs(OUT_DIR, exist_ok=True)

# Ground truth
with open(GT_PATH) as f:
    gt_raw = json.load(f)
gt_counts = {int(k): (len(v) if isinstance(v, list) else int(v))
             for k, v in gt_raw["frames"].items()}

# ROI mask
raw_mask = cv2.imread(MASK_PATH, 0)

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(VIDEO_PATH)
ret, first = cap.read()
proc_h = int(first.shape[0] * PROCESS_SCALE)
proc_w = int(first.shape[1] * PROCESS_SCALE)

roi_mask = cv2.resize(raw_mask, (proc_w, proc_h), interpolation=cv2.INTER_NEAREST) \
           if raw_mask is not None else None

# Semi-transparent ROI overlay colour
ROI_COLOUR = (0, 255, 255)   # cyan

print(f"Saving annotated frames to  ./{OUT_DIR}/\n")
print(f"{'Frame':>7}  {'GT':>4}  {'YOLO(in ROI)':>13}  {'YOLO(total)':>12}")
print("-" * 46)

for fidx in INSPECT_FRAMES:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
    vis   = frame.copy()

    # Draw ROI boundary
    if roi_mask is not None:
        contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, ROI_COLOUR, 2)

    # Run YOLO
    results = model(frame, device="cpu", conf=YOLO_CONF,
                    classes=[0], verbose=False)[0]

    in_roi = 0
    total_det = len(results.boxes)

    for box in results.boxes:
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        conf_score = float(box.conf[0])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Decide if inside ROI
        cy_c = max(0, min(cy, proc_h - 1))
        cx_c = max(0, min(cx, proc_w - 1))
        inside = roi_mask is None or roi_mask[cy_c, cx_c] > 127

        colour = (0, 255, 0) if inside else (128, 128, 128)  # green / grey
        cv2.rectangle(vis, (x1, y1), (x2, y2), colour, 2)
        label = f"{conf_score:.2f}"
        cv2.putText(vis, label, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv2.LINE_AA)
        # Mark centre
        cv2.circle(vis, (cx, cy), 4, colour, -1)

        if inside:
            in_roi += 1

    # Overlay frame info
    gt_n = gt_counts.get(fidx, "?")
    info = f"f={fidx}  GT={gt_n}  YOLO_in_ROI={in_roi}  YOLO_total={total_det}"
    cv2.putText(vis, info, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, info, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0),       1, cv2.LINE_AA)

    fname = os.path.join(OUT_DIR, f"f{fidx:06d}_gt{gt_n}_yolo{in_roi}.jpg")
    cv2.imwrite(fname, vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"{fidx:>7}  {str(gt_n):>4}  {in_roi:>13}  {total_det:>12}")

cap.release()
print(f"\nDone — {len(INSPECT_FRAMES)} frames saved to ./{OUT_DIR}/")
print("Open the folder and look at the green boxes (in-ROI detections).")
print("Grey boxes = detected outside ROI (not counted).")
