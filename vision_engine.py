import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
import time

class RobustSentinelTracker:
    def __init__(self, max_ghost=30, dist_thresh=100):
        self.next_id = 1
        self.tracks = {} 
        self.max_ghost = max_ghost
        self.dist_thresh = dist_thresh

    def get_center(self, box):
        return (int(box[0] + box[2]/2), int(box[1] + box[3]/2))

    def update(self, detections):
        # Standard tracking logic...
        detections = self.merge_nearby_boxes(detections)
        track_ids = list(self.tracks.keys())
        track_centers = [self.get_center(self.tracks[tid]['box']) for tid in track_ids]
        det_centers = [self.get_center(box) for box in detections]

        if len(track_centers) > 0 and len(det_centers) > 0:
            cost = distance.cdist(track_centers, det_centers)
            row_idx, col_idx = linear_sum_assignment(cost)
            assigned_tracks, assigned_dets = set(), set()
            for r, c in zip(row_idx, col_idx):
                if cost[r, c] < self.dist_thresh:
                    tid = track_ids[r]
                    self.tracks[tid]['box'] = detections[c]
                    self.tracks[tid]['ghost'] = 0
                    assigned_tracks.add(tid)
                    assigned_dets.add(c)
            for i, tid in enumerate(track_ids):
                if tid not in assigned_tracks: self.tracks[tid]['ghost'] += 1
            for i in range(len(detections)):
                if i not in assigned_dets: self.register(detections[i])
        else:
            for tid in self.tracks: self.tracks[tid]['ghost'] += 1
            for det in detections: self.register(det)

        for tid in list(self.tracks.keys()):
            if self.tracks[tid]['ghost'] > self.max_ghost: del self.tracks[tid]
        return self.tracks

    def merge_nearby_boxes(self, boxes, thresh=15):
        if not boxes: return []
        merged = []
        while len(boxes) > 0:
            curr = boxes.pop(0)
            combined = False
            for i, other in enumerate(merged):
                if self.is_close(curr, other, thresh):
                    merged[i] = self.combine_boxes(curr, other)
                    combined = True
                    break
            if not combined: merged.append(curr)
        return merged

    def is_close(self, b1, b2, t):
        return not (b1[0]+b1[2]+t < b2[0] or b2[0]+b2[2]+t < b1[0] or 
                    b1[1]+b1[3]+t < b2[1] or b2[1]+b2[3]+t < b1[1])

    def combine_boxes(self, b1, b2):
        x = min(b1[0], b2[0])
        y = min(b1[1], b2[1])
        w = max(b1[0]+b1[2], b2[0]+b2[2]) - x
        h = max(b1[1]+b1[3], b2[1]+b2[3]) - y
        return [x, y, w, h]

    def register(self, box):
        self.tracks[self.next_id] = {'box': box, 'ghost': 0}
        self.next_id += 1

# --- NEW: IMPROVED SUB-COUNTING FUNCTION ---
def count_people_in_box(roi, box_width):
    # 1. Sanity Check: If box is too small, it's just 1 person
    # Assuming an average person is at least 40-50 pixels wide
    avg_person_width = 50 
    max_possible_people = max(1, int(box_width / avg_person_width))
    
    if roi is None or roi.size == 0:
        return 0
    
    # 2. Stronger Noise Removal (Erosion separates connected blobs)
    kernel = np.ones((3,3), np.uint8)
    eroded = cv2.erode(roi, kernel, iterations=1)
    
    # 3. Distance Transform
    dist = cv2.distanceTransform(eroded, cv2.DIST_L2, 5)
    
    # 4. Strict Thresholding (0.6 instead of 0.4)
    # Only counts the "thickest" centers (torsos), ignoring arms/legs
    _, sure_fg = cv2.threshold(dist, 0.6 * dist.max(), 255, 0)
    
    sure_fg = np.uint8(sure_fg)
    
    # 5. Filter by Area (Ignore tiny specks)
    contours, _ = cv2.findContours(sure_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_blobs = 0
    for c in contours:
        if cv2.contourArea(c) > 50: # Minimum area to be a "person core"
            valid_blobs += 1
            
    # Result: The count is the detected blobs, but clamped by physical width limits
    final_count = min(valid_blobs, max_possible_people)
    
    # Fallback: If watershed fails (0 detected), but tracker saw a box, return 1
    return max(1, final_count)

# --- SHARED STATE FOR FLASK STREAMING ---
# Latest JPEG frame for CAM-01 (used by /video_feed)
_cam1_latest_jpeg = None

# Latest stats for CAM-01 (used by /api/stats)
_cam1_latest_stats = {
    "count": 0,
    "density": 0,
    "status": "SAFE",
    "locations": []
}


def _cam1_update_loop(video_path='video1.mp4', mask_path='mask_layer.png'):
    """
    Long-running loop that reads frames from video_path, runs the robust
    sentinel pipeline, draws the green and yellow boxes, and updates:
      - _cam1_latest_jpeg (JPEG-encoded frame)
      - _cam1_latest_stats (count, density, status)

    This is intended to be run in a background thread from app.py.
    """
    global _cam1_latest_jpeg, _cam1_latest_stats

    # Zones (copied from original pipeline)
    z7_x1, z7_y1, z7_x2, z7_y2 = 240, 290, 325, 625
    z6_x1, z6_y1, z6_x2, z6_y2 = 340, 280, 855, 990

    # Fusion kernel
    fusion_k = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 85))

    tracker = RobustSentinelTracker()

    while True:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[vision_engine] Error: Cannot open video source '{video_path}'")
            time.sleep(1.0)
            continue

        # Use the first frame as the background reference
        ret, bg = cap.read()
        if not ret:
            print(f"[vision_engine] Error: Cannot read first frame from '{video_path}'")
            cap.release()
            time.sleep(1.0)
            continue

        frame_height, frame_width = bg.shape[:2]

        # Load ROI mask; if missing or mismatched, fall back to full-frame mask
        roi_mask = cv2.imread(mask_path, 0)
        if roi_mask is None or roi_mask.shape[:2] != (frame_height, frame_width):
            roi_mask = np.ones((frame_height, frame_width), dtype="uint8") * 255

        bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)

        # Rewind to the start so we process all frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while True:
            ret, frame = cap.read()
            if not ret:
                # End of video; break inner loop to restart from the beginning
                break

            # 1. Processing
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(bg_gray, frame_gray)
            _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            thresh = cv2.bitwise_and(thresh, thresh, mask=roi_mask)

            # 2. Fusion
            fused = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, fusion_k)
            fused = cv2.dilate(fused, np.ones((5, 5), np.uint8), iterations=1)

            # 3. Detection
            conts, _ = cv2.findContours(fused, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            detections = []
            for c in conts:
                if cv2.contourArea(c) > 1500:
                    detections.append(list(cv2.boundingRect(c)))

            tracks = tracker.update(detections)

            # --- RESET COUNTS ---
            count_z7 = 0
            count_z6 = 0

            # Draw Zones (green boxes)
            cv2.rectangle(frame, (z7_x1, z7_y1), (z7_x2, z7_y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (z6_x1, z6_y1), (z6_x2, z6_y2), (0, 255, 0), 2)

            # 4. ANALYZE TRACKS
            for tid, data in tracks.items():
                if data['ghost'] == 0:
                    x, y, w, h = data['box']
                    cx, cy = int(x + w / 2), int(y + h / 2)

                    # Extract ROI from RAW thresh
                    roi = thresh[y:y + h, x:x + w]

                    # --- APPLY NEW COUNTING LOGIC ---
                    people_in_this_box = count_people_in_box(roi, w)

                    # Draw Visuals (yellow boxes)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

                    # Label Logic: only show count if > 1 to keep it clean
                    label_text = f"ID {tid}"
                    if people_in_this_box > 1:
                        label_text += f" ({people_in_this_box})"

                    cv2.putText(
                        frame,
                        label_text,
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )

                    # Add to Zone Totals
                    if (z7_x1 < cx < z7_x2) and (z7_y1 < cy < z7_y2):
                        count_z7 += people_in_this_box
                    elif (z6_x1 < cx < z6_x2) and (z6_y1 < cy < z6_y2):
                        count_z6 += people_in_this_box

            # --- DISPLAY TOTALS ON FRAME ---
            total_count = count_z7 + count_z6
            text = f"Count of people: {total_count}"

            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            tx = frame_width - tw - 20
            ty = 50

            cv2.rectangle(frame, (tx - 10, ty - 30), (tx + tw + 10, ty + 10), (0, 0, 0), -1)
            cv2.putText(
                frame,
                text,
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

            # Display individual zone counts
            cv2.putText(
                frame,
                f"Z7: {count_z7}",
                (z7_x1, z7_y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Z6: {count_z6}",
                (z6_x1, z6_y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            # --- UPDATE SHARED STATS ---
            # Approximate density as people per normalized area unit of the two zones
            area_z7 = (z7_x2 - z7_x1) * (z7_y2 - z7_y1)
            area_z6 = (z6_x2 - z6_x1) * (z6_y2 - z6_y1)
            norm_area = max(1.0, (area_z7 + area_z6) / 10000.0)
            density = int(min(100, (total_count / norm_area)))

            if total_count == 0:
                status = "SAFE"
            elif total_count < 10:
                status = "WARNING"
            else:
                status = "CRITICAL"

            _cam1_latest_stats = {
                "count": int(total_count),
                "density": int(density),
                "status": status,
                "locations": [],
            }

            # --- ENCODE FRAME AS JPEG FOR MJPEG STREAM ---
            ok, buf = cv2.imencode(".jpg", frame)
            if ok:
                _cam1_latest_jpeg = buf.tobytes()

            # Target ~30 FPS
            time.sleep(0.033)

        cap.release()


if __name__ == "__main__":
    # Optional: run the loop directly for debugging (no Flask)
    _cam1_update_loop("video1.mp4")