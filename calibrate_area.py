import cv2
import json
import numpy as np
import os

PIPELINE_PARAMS = {
    'varThreshold':   40,
    'history':        1000,
    'morph_kernel':   (5, 25),
    'dilate_kernel':  1,
    'h_morph_kernel': (10, 3),
}
PROCESS_SCALE = 0.667
VIDEO_PATH    = 'videos/main_video.mp4'
MASK_PATH     = 'mask_layer1.png'
GT_JSON       = 'barangay_ground_truth.json'
OUTPUT_JSON   = 'area_calibration.json'
WARMUP_FRAMES = 1000


def load_ground_truth():
    with open(GT_JSON, 'r') as f:
        db = json.load(f)
    roi_mask = cv2.imread(MASK_PATH, 0)
    frames_db = {}
    for k, v in db['frames'].items():
        count_in_roi = 0
        for pt in v:
            x, y = int(pt['x']), int(pt['y'])
            if roi_mask is not None and y < roi_mask.shape[0] and x < roi_mask.shape[1]:
                if roi_mask[y, x] > 127:
                    count_in_roi += 1
            else:
                count_in_roi += 1
        frames_db[int(k)] = count_in_roi
    return frames_db


def main():
    print("=" * 60)
    print(" BARANGAY SENTINEL: AREA CALIBRATION ")
    print("=" * 60)

    frames_db = load_ground_truth()
    annotated_indices = sorted(frames_db.keys())
    max_frame = annotated_indices[-1]
    print(f"Ground truth: {len(frames_db)} annotated frames, max frame {max_frame}")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {VIDEO_PATH}")
        return

    ret, first = cap.read()
    if not ret:
        print("ERROR: Cannot read first frame.")
        cap.release()
        return

    proc_h = int(first.shape[0] * PROCESS_SCALE)
    proc_w = int(first.shape[1] * PROCESS_SCALE)
    h, w = proc_h, proc_w

    raw_mask = cv2.imread(MASK_PATH, 0)
    if raw_mask is not None:
        roi_mask = cv2.resize(raw_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        roi_mask = np.ones((h, w), dtype=np.uint8) * 255

    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=PIPELINE_PARAMS['history'],
        varThreshold=PIPELINE_PARAMS['varThreshold'],
        detectShadows=False
    )
    open_k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fusion_k  = cv2.getStructuringElement(cv2.MORPH_RECT, PIPELINE_PARAMS['morph_kernel'])
    h_fuse_k  = cv2.getStructuringElement(cv2.MORPH_RECT, PIPELINE_PARAMS['h_morph_kernel'])
    d_kernel  = np.ones((PIPELINE_PARAMS['dilate_kernel'], PIPELINE_PARAMS['dilate_kernel']), np.uint8)
    clahe     = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # {person_count: [area, area, ...]}
    raw_data = {}
    frame_idx = 0
    scored = 0

    print(f"Processing frames 0–{max_frame}...")
    while frame_idx <= max_frame:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA)

        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        l_ch = clahe.apply(l_ch)
        lab = cv2.merge((l_ch, a_ch, b_ch))
        frame_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        fg_mask = bg_subtractor.apply(frame_enhanced)

        fg_mask = cv2.medianBlur(fg_mask, 3)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        shadow_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([179, 255, 45]))
        fg_mask = cv2.bitwise_and(fg_mask, fg_mask, mask=cv2.bitwise_not(shadow_mask))

        _, thresh = cv2.threshold(fg_mask, 254, 255, cv2.THRESH_BINARY)
        thresh = cv2.bitwise_and(thresh, thresh, mask=roi_mask)

        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, open_k)
        fused   = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, fusion_k)
        fused   = cv2.morphologyEx(fused, cv2.MORPH_CLOSE, h_fuse_k)
        fused   = cv2.dilate(fused, d_kernel, iterations=1)

        if frame_idx in frames_db and frame_idx >= WARMUP_FRAMES:
            person_count = frames_db[frame_idx]
            area = int(cv2.countNonZero(fused))
            raw_data.setdefault(person_count, []).append(area)
            scored += 1

        frame_idx += 1
        if frame_idx % 500 == 0:
            print(f"  frame {frame_idx}/{max_frame}  scored={scored}")

    cap.release()
    print(f"Done. {scored} frames scored across {len(raw_data)} unique crowd sizes.")

    if len(raw_data) < 2:
        print("ERROR: Need at least 2 distinct crowd sizes for regression. Add more annotations.")
        return

    # Median area per person count
    aggregated = {k: float(np.median(v)) for k, v in raw_data.items()}

    counts = sorted(aggregated.keys())
    areas  = [aggregated[k] for k in counts]

    slope, intercept = np.polyfit(counts, areas, 1)
    avg_pixels_per_person = float(slope)
    area_at_zero          = float(intercept)

    predicted = [slope * c + intercept for c in counts]
    mean_a    = sum(areas) / len(areas)
    ss_res    = sum((a - p) ** 2 for a, p in zip(areas, predicted))
    ss_tot    = sum((a - mean_a) ** 2 for a in areas)
    r_squared = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    print()
    print(f"{'Count':>6}  {'Median Area':>12}  {'Predicted':>10}  {'Samples':>7}")
    print("-" * 42)
    for c, a, p in zip(counts, areas, predicted):
        n = len(raw_data[c])
        print(f"{c:>6}  {a:>12.0f}  {p:>10.0f}  {n:>7}")
    print()
    print(f"avg_pixels_per_person : {avg_pixels_per_person:.2f}")
    print(f"area_at_zero          : {area_at_zero:.2f}")
    print(f"R²                    : {r_squared:.4f}")

    result = {
        "avg_pixels_per_person": avg_pixels_per_person,
        "area_at_zero":          area_at_zero,
        "r_squared":             r_squared,
        "data_points":           scored,
        "per_count_median_area": {str(k): int(aggregated[k]) for k in counts},
        "pipeline_params":       {
            **PIPELINE_PARAMS,
            "morph_kernel":   list(PIPELINE_PARAMS["morph_kernel"]),
            "h_morph_kernel": list(PIPELINE_PARAMS["h_morph_kernel"]),
        },
    }
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(result, f, indent=4)
    print(f"\nSaved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
