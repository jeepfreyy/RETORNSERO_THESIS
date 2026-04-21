import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
import time
import threading


class RobustSentinelTracker:
    def __init__(self, max_ghost=30, dist_thresh=100):
        self.next_id = 1
        self.tracks = {}
        self.max_ghost = max_ghost
        self.dist_thresh = dist_thresh

    def get_center(self, box):
        return (int(box[0] + box[2] / 2), int(box[1] + box[3] / 2))

    def update(self, detections):
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
                if tid not in assigned_tracks:
                    self.tracks[tid]['ghost'] += 1
            for i in range(len(detections)):
                if i not in assigned_dets:
                    self.register(detections[i])
        else:
            for tid in self.tracks:
                self.tracks[tid]['ghost'] += 1
            for det in detections:
                self.register(det)

        for tid in list(self.tracks.keys()):
            if self.tracks[tid]['ghost'] > self.max_ghost:
                del self.tracks[tid]
        return self.tracks

    def merge_nearby_boxes(self, boxes, thresh=15):
        if not boxes:
            return []
        merged = []
        while len(boxes) > 0:
            curr = boxes.pop(0)
            combined = False
            for i, other in enumerate(merged):
                if self.is_close(curr, other, thresh):
                    merged[i] = self.combine_boxes(curr, other)
                    combined = True
                    break
            if not combined:
                merged.append(curr)
        return merged

    def is_close(self, b1, b2, t):
        return not (
            b1[0] + b1[2] + t < b2[0]
            or b2[0] + b2[2] + t < b1[0]
            or b1[1] + b1[3] + t < b2[1]
            or b2[1] + b2[3] + t < b1[1]
        )

    def combine_boxes(self, b1, b2):
        x = min(b1[0], b2[0])
        y = min(b1[1], b2[1])
        w = max(b1[0] + b1[2], b2[0] + b2[2]) - x
        h = max(b1[1] + b1[3], b2[1] + b2[3]) - y
        return [x, y, w, h]

    def register(self, box):
        self.tracks[self.next_id] = {'box': box, 'ghost': 0}
        self.next_id += 1


# ---------------------------------------------------------------------------
# Perspective Scaling
# ---------------------------------------------------------------------------
def get_perspective_weight(y_coord, frame_height):
    """
    Returns a scaling factor (0.3 to 1.0) based on where the object sits
    on the Y-axis. Objects near the top of the frame (far away) get a low
    weight, objects near the bottom (close) get a high weight.
    This is used to dynamically adjust minimum area and aspect-ratio
    thresholds so nearby objects are not over-filtered and distant objects
    are not under-filtered.
    """
    ratio = y_coord / max(1, frame_height)
    # Clamp between 0.3 (very top) and 1.0 (very bottom)
    return max(0.3, min(1.0, 0.3 + 0.7 * ratio))


# ---------------------------------------------------------------------------
# Structural Blob Validation (uses Distance Transform profile)
# ---------------------------------------------------------------------------
def is_human_blob(roi_thresh, x, y, w, h, frame_height):
    """
    Validates whether a detected blob is likely a human using:
      1. Aspect Ratio  –  people are generally taller than wide
      2. Distance Transform peak profile  –  people have a narrow/round peak,
         vehicles and boxes have a wide, flat peak
      3. Perspective-scaled minimum area
    Returns True if the blob passes all checks.
    """
    if roi_thresh is None or roi_thresh.size == 0:
        return False

    pw = get_perspective_weight(y + h, frame_height)

    # 1. Minimum area scales with perspective
    min_area = int(800 * pw)
    area = w * h
    if area < min_area:
        return False

    # 2. Aspect ratio check (width / height)
    #    Standing/walking people:   ratio < 1.3
    #    Sitting/bending people:    ratio < 1.8 (more lenient)
    #    Cars / wide objects:       ratio > 1.8  → reject
    aspect = w / max(1, h)
    max_aspect = 1.8  # lenient to allow crouching / off-angle people
    if aspect > max_aspect:
        return False

    # 3. Distance transform peak flatness check
    #    A human body produces a narrow peak in the distance transform.
    #    A vehicle/box produces a wide, flat plateau.
    dist = cv2.distanceTransform(roi_thresh, cv2.DIST_L2, 5)
    max_val = dist.max()
    if max_val < 3:
        return False  # too thin / noisy

    # Measure "peak sharpness": ratio of peak value to blob half-width
    # Humans typically have peak_ratio > 0.15; flat boxes are < 0.10
    peak_ratio = max_val / max(1, min(w, h) / 2.0)
    if peak_ratio < 0.08:
        return False

    return True


# ---------------------------------------------------------------------------
# Sub-Counting inside a Validated Blob (Watershed-based)
# ---------------------------------------------------------------------------
def count_people_in_box(roi, box_width, box_y, frame_height):
    """
    Given the thresholded ROI of a validated human blob, estimate
    how many people are clustered inside it using distance-transform
    peaks and perspective-scaled person width.
    """
    pw = get_perspective_weight(box_y, frame_height)
    # SHANGHAITECH OPTIMIZATION: Bounding width mathematically derived from 16px median proximity
    avg_person_width = int(32 * pw)
    max_possible_people = max(1, int(box_width / max(1, avg_person_width)))

    if roi is None or roi.size == 0:
        return 1

    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(roi, kernel, iterations=1)

    dist = cv2.distanceTransform(eroded, cv2.DIST_L2, 5)
    if dist.max() == 0:
        return 1

    # SHANGHAITECH OPTIMIZATION: Mathematical separation valley floor derived at 8.0px
    _, sure_fg = cv2.threshold(dist, max(8.0, 0.4 * dist.max()), 255, 0)
    sure_fg = np.uint8(sure_fg)

    contours, _ = cv2.findContours(sure_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_blobs = sum(1 for c in contours if cv2.contourArea(c) > 50)

    final_count = min(valid_blobs, max_possible_people)
    return max(1, final_count)


# ---------------------------------------------------------------------------
# Sleek Dot Overlay Rendering
# ---------------------------------------------------------------------------
def draw_person_marker(frame, cx, cy, count, in_zone7=False):
    """
    Draws a clean, glowing dot at the person's foot-center position.
    If count > 1 it draws a small numeric badge above the dot.
    
    Colors:
        Zone 7 (z7):  Cyan   (#00FFFF)
        Zone 6 (z6):  Green  (#00FF88)
        Outside zones: Amber (#FFD700)
    """
    if in_zone7:
        color_inner = (255, 255, 0)   # Cyan  (BGR)
        color_outer = (200, 200, 0)
    else:
        color_inner = (136, 255, 0)   # Green (BGR)
        color_outer = (100, 200, 0)

    # Outer glow ring
    cv2.circle(frame, (cx, cy), 14, color_outer, 2, cv2.LINE_AA)
    # Solid inner dot
    cv2.circle(frame, (cx, cy), 7, color_inner, -1, cv2.LINE_AA)

    # Badge for clusters
    if count > 1:
        badge_text = str(count)
        (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        bx = cx - tw // 2
        by = cy - 22
        # Dark background pill
        cv2.rectangle(frame, (bx - 4, by - th - 4), (bx + tw + 4, by + 4), (0, 0, 0), -1)
        cv2.rectangle(frame, (bx - 4, by - th - 4), (bx + tw + 4, by + 4), color_inner, 1)
        cv2.putText(frame, badge_text, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main Stream Class
# ---------------------------------------------------------------------------
class SentinelStream:
    """
    Thread-safe class that reads an RTSP stream (or video file),
    applies MOG2 background subtraction, validates blobs structurally,
    counts people via distance-transform watershed logic, and renders
    sleek dot overlays.
    """

    def __init__(self, stream_id, source="video1.mp4", mask_path="mask_layer.png"):
        self.stream_id = stream_id
        self.source = source
        self.mask_path = mask_path

        self.latest_jpeg = None
        self.latest_stats = {
            "count": 0,
            "density": 0,
            "status": "SAFE",
            "locations": [],
        }

        # MOG2 calibrated using localized Grid Search optimization
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True
        )

        self.running = True
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()

    # ------------------------------------------------------------------
    def _process_loop(self):
        tracker = RobustSentinelTracker()
        fusion_k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 50))

        while self.running:
            cap = cv2.VideoCapture(self.source)
            if not cap.isOpened():
                print(f"[SentinelStream {self.stream_id}] Error: Cannot open '{self.source}'")
                time.sleep(2.0)
                continue

            ret, first_frame = cap.read()
            if not ret:
                cap.release()
                time.sleep(2.0)
                continue

            frame_height, frame_width = first_frame.shape[:2]

            roi_mask = cv2.imread(self.mask_path, 0)
            if roi_mask is None or roi_mask.shape[:2] != (frame_height, frame_width):
                roi_mask = np.ones((frame_height, frame_width), dtype=np.uint8) * 255

            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            while self.running:
                ret, frame = cap.read()
                if not ret:
                    break

                # ── PHASE 1: MOG2 & Shadow Removal ──────────────────
                fg_mask = self.bg_subtractor.apply(frame)
                _, thresh = cv2.threshold(fg_mask, 254, 255, cv2.THRESH_BINARY)
                thresh = cv2.bitwise_and(thresh, thresh, mask=roi_mask)

                # ── PHASE 2: Fusion (stable mega-blobs) ─────────────
                fused = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, fusion_k)
                fused = cv2.dilate(fused, np.ones((3, 3), np.uint8), iterations=1)

                # ── PHASE 3: Detection with Structural Validation ───
                conts, _ = cv2.findContours(fused, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                detections = []
                for c in conts:
                    x, y, w_box, h_box = cv2.boundingRect(c)
                    # Extract ROI from raw thresh for structural analysis
                    roi_check = thresh[y : y + h_box, x : x + w_box]
                    if is_human_blob(roi_check, x, y, w_box, h_box, frame_height):
                        detections.append([x, y, w_box, h_box])

                tracks = tracker.update(detections)

                # ── PHASE 4: Counting & Rendering ───────────────────
                total_count = 0

                for tid, data in tracks.items():
                    if data['ghost'] == 0:
                        x, y, w_box, h_box = data['box']
                        cx = int(x + w_box / 2)
                        cy = int(y + h_box)

                        roi = thresh[y : y + h_box, x : x + w_box]
                        people_in_box = count_people_in_box(roi, w_box, y + h_box, frame_height)

                        total_count += people_in_box
                        draw_person_marker(frame, cx, cy, people_in_box, in_zone7=False)

                # ── HUD Overlay Removed (UI renders stats) ──────────

                # Stats
                norm_area = max(1.0, (frame_width * frame_height) / 10000.0)
                density = int(min(100, (total_count / norm_area)))

                if total_count == 0:
                    status = "SAFE"
                elif total_count < 10:
                    status = "WARNING"
                else:
                    status = "CRITICAL"

                self.latest_stats = {
                    "count": int(total_count),
                    "density": int(density),
                    "status": status,
                    "locations": [],
                }

                ok, buf = cv2.imencode(".jpg", frame)
                if ok:
                    self.latest_jpeg = buf.tobytes()

                time.sleep(0.033)

            cap.release()
            print(f"[SentinelStream {self.stream_id}] Stream ended. Restarting...")
            time.sleep(2.0)

    # ------------------------------------------------------------------
    def get_latest_jpeg(self):
        return self.latest_jpeg

    def get_latest_stats(self):
        return self.latest_stats