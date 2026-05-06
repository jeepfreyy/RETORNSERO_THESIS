"""
diagnose_errors.py — Per-Frame Error Diagnostic for Barangay Sentinel
======================================================================
Run this AFTER the tuner finishes to understand exactly which frames
are driving your MAE. It answers three questions:

  1. Is the system overcounting or undercounting (and by how much)?
  2. Which density tier (EMPTY / LOW / MEDIUM / HIGH) has the worst errors?
  3. Which specific frame numbers are the biggest contributors?

USAGE
-----
After the tuner finishes, paste the winning parameters into WINNING_PARAMS
below, then run:

    python3 diagnose_errors.py

OUTPUT
------
  - Terminal: sorted error report, tier breakdown, direction analysis
  - error_report.csv: all per-frame results, openable in Excel / Numbers
"""

import cv2
import json
import csv
import numpy as np
from vision_engine import RobustSentinelTracker, is_human_blob, count_people_in_box

# ── Paste the winning parameters from the tuner here ────────────────────────
WINNING_PARAMS = {
    'varThreshold':       40,
    'history':            1000,
    'morph_kernel':       (5, 25),
    'dilate_kernel':      1,
    'min_blob_area':      350,
    'ghost_threshold':    90,
    'h_morph_kernel':     (10, 3),
    'merge_thresh':       20,
    'dist_thresh':        150,
    'max_aspect':         2.5,
    'base_width':         90,
    'solidity_threshold': 0.6,
    'dt_thresh':          0.3,
}

MASK_PATH       = 'mask_layer1.png'
GT_JSON         = 'barangay_ground_truth.json'
PROCESS_SCALE   = 0.667
OUTPUT_CSV      = 'error_report.csv'

# Density tier labels (must match your dashboard thresholds)
def density_tier(count):
    if count == 0:            return 'EMPTY'
    if 1 <= count <= 5:       return 'LOW'
    if 6 <= count <= 9:       return 'MEDIUM'
    return 'HIGH'


def load_ground_truth():
    with open(GT_JSON) as f:
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
    return db['video_source'], frames_db


def run_diagnostic(params):
    video_path, frames_db = load_ground_truth()
    warmup_frames = params['history']
    annotated_indices = sorted(frames_db.keys())
    max_frame = annotated_indices[-1]

    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=params['history'],
        varThreshold=params['varThreshold'],
        detectShadows=False
    )
    open_k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fusion_k  = cv2.getStructuringElement(cv2.MORPH_RECT, params['morph_kernel'])
    h_fusion_k= cv2.getStructuringElement(cv2.MORPH_RECT, params['h_morph_kernel'])
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
        print(f"ERROR: Cannot open {video_path}")
        return

    ret, first = cap.read()
    if not ret:
        return

    if PROCESS_SCALE != 1.0:
        proc_h = int(first.shape[0] * PROCESS_SCALE)
        proc_w = int(first.shape[1] * PROCESS_SCALE)
        first  = cv2.resize(first, (proc_w, proc_h), interpolation=cv2.INTER_AREA)

    h, w = first.shape[:2]

    raw_mask = cv2.imread(MASK_PATH, 0)
    if raw_mask is not None:
        roi_mask = cv2.resize(raw_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        roi_mask = np.ones((h, w), dtype=np.uint8) * 255

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_idx = 0
    results   = []   # list of dicts: {frame, ground_truth, predicted, error, tier}
    frames_set = set(frames_db.keys())

    print(f"Processing {max_frame} frames (warm-up: {warmup_frames})...")

    while frame_idx <= max_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if PROCESS_SCALE != 1.0:
            proc_h = int(frame.shape[0] * PROCESS_SCALE)
            proc_w = int(frame.shape[1] * PROCESS_SCALE)
            frame  = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA)

        # ── Identical pipeline to vision_engine.py ───────────────────
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        l_ch = clahe.apply(l_ch)
        lab  = cv2.merge((l_ch, a_ch, b_ch))
        frame_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        fg_mask = bg_subtractor.apply(frame_enhanced)
        fg_mask = cv2.medianBlur(fg_mask, 3)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        shadow_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([179, 255, 45]))
        fg_mask = cv2.bitwise_and(fg_mask, fg_mask, mask=cv2.bitwise_not(shadow_mask))

        _, thresh = cv2.threshold(fg_mask, 254, 255, cv2.THRESH_BINARY)
        thresh    = cv2.bitwise_and(thresh, thresh, mask=roi_mask)

        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, open_k)
        fused   = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, fusion_k)
        fused   = cv2.morphologyEx(fused,   cv2.MORPH_CLOSE, h_fusion_k)
        fused   = cv2.dilate(fused, d_kernel, iterations=1)

        conts, _ = cv2.findContours(fused, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for c in conts:
            x, y, w_box, h_box = cv2.boundingRect(c)
            roi_check = thresh[y:y+h_box, x:x+w_box]
            if is_human_blob(roi_check, x, y, w_box, h_box, h,
                             params['min_blob_area'], params['max_aspect']):
                detections.append([x, y, w_box, h_box])

        tracks = tracker.update(detections)

        if frame_idx in frames_set and frame_idx >= warmup_frames:
            true_count = frames_db[frame_idx]
            predicted  = 0
            for tid, data in tracks.items():
                if data['ghost'] == 0 and data['lifetime'] >= tracker.min_lifetime:
                    px, py, pw_box, ph_box = data['box']
                    roi = thresh[py:py+ph_box, px:px+pw_box]
                    predicted += count_people_in_box(roi, pw_box, py+ph_box, h,
                                                     solidity_threshold=params['solidity_threshold'],
                                                     base_width=params['base_width'],
                                                     dt_thresh=params['dt_thresh'])

            signed_err = predicted - true_count
            results.append({
                'frame':        frame_idx,
                'ground_truth': true_count,
                'predicted':    predicted,
                'error':        signed_err,
                'abs_error':    abs(signed_err),
                'tier':         density_tier(true_count),
            })

        frame_idx += 1
        if frame_idx % 1000 == 0:
            print(f"  ...frame {frame_idx}/{max_frame}")

    cap.release()
    return results


def print_report(results):
    if not results:
        print("No scored frames found. Check warmup_frames and ground truth.")
        return

    mae = sum(r['abs_error'] for r in results) / len(results)
    overcounts  = [r for r in results if r['error'] > 0]
    undercounts = [r for r in results if r['error'] < 0]
    exact       = [r for r in results if r['error'] == 0]

    print()
    print("=" * 65)
    print("  PER-FRAME ERROR DIAGNOSTIC REPORT")
    print("=" * 65)
    print(f"  Scored frames : {len(results)}")
    print(f"  Overall MAE   : {mae:.3f} people")
    print()
    print(f"  Direction of error:")
    print(f"    Overcounting  (predicted > ground truth): {len(overcounts):3d} frames "
          f"({100*len(overcounts)/len(results):.1f}%)")
    print(f"    Undercounting (predicted < ground truth): {len(undercounts):3d} frames "
          f"({100*len(undercounts)/len(results):.1f}%)")
    print(f"    Exact match   (predicted = ground truth): {len(exact):3d} frames "
          f"({100*len(exact)/len(results):.1f}%)")
    if overcounts:
        avg_over = sum(r['error'] for r in overcounts) / len(overcounts)
        print(f"    Avg overcount magnitude : +{avg_over:.2f} people")
    if undercounts:
        avg_under = sum(r['error'] for r in undercounts) / len(undercounts)
        print(f"    Avg undercount magnitude: {avg_under:.2f} people")

    print()
    print("  MAE by density tier:")
    for tier in ['EMPTY', 'LOW', 'MEDIUM', 'HIGH']:
        tier_frames = [r for r in results if r['tier'] == tier]
        if tier_frames:
            tier_mae = sum(r['abs_error'] for r in tier_frames) / len(tier_frames)
            tier_over  = sum(1 for r in tier_frames if r['error'] > 0)
            tier_under = sum(1 for r in tier_frames if r['error'] < 0)
            print(f"    {tier:6s}: MAE={tier_mae:.2f}  "
                  f"frames={len(tier_frames):3d}  "
                  f"over={tier_over}  under={tier_under}")

    print()
    print("  Top 20 worst frames (sorted by absolute error):")
    print(f"  {'Frame':>7}  {'GT':>4}  {'Pred':>5}  {'Err':>5}  {'Tier'}")
    print("  " + "-" * 42)
    worst = sorted(results, key=lambda r: r['abs_error'], reverse=True)[:20]
    for r in worst:
        direction = f"+{r['error']}" if r['error'] > 0 else str(r['error'])
        print(f"  {r['frame']:>7}  {r['ground_truth']:>4}  "
              f"{r['predicted']:>5}  {direction:>5}  {r['tier']}")

    print()
    print(f"  Full report saved to: {OUTPUT_CSV}")
    print("=" * 65)


def save_csv(results):
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'frame', 'ground_truth', 'predicted', 'error', 'abs_error', 'tier'
        ])
        writer.writeheader()
        writer.writerows(results)


if __name__ == '__main__':
    print("=" * 65)
    print("  BARANGAY SENTINEL — PER-FRAME ERROR DIAGNOSTIC")
    print("=" * 65)
    print(f"  Parameters: varThreshold={WINNING_PARAMS['varThreshold']}, "
          f"history={WINNING_PARAMS['history']}, "
          f"ghost={WINNING_PARAMS['ghost_threshold']}")
    print()

    results = run_diagnostic(WINNING_PARAMS)
    if results:
        print_report(results)
        save_csv(results)
