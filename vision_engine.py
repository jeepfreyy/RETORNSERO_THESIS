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
        return (int(box[0] + box[2]/2), int(box[1] + box[3]/2))

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

def count_people_in_box(roi, box_width):
    avg_person_width = 50 
    max_possible_people = max(1, int(box_width / avg_person_width))
    
    if roi is None or roi.size == 0:
        return 0
    
    kernel = np.ones((3,3), np.uint8)
    eroded = cv2.erode(roi, kernel, iterations=1)
    
    dist = cv2.distanceTransform(eroded, cv2.DIST_L2, 5)
    
    _, sure_fg = cv2.threshold(dist, 0.4 * dist.max(), 255, 0)
    
    sure_fg = np.uint8(sure_fg)
    
    contours, _ = cv2.findContours(sure_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_blobs = 0
    for c in contours:
        if cv2.contourArea(c) > 50: 
            valid_blobs += 1
            
    final_count = min(valid_blobs, max_possible_people)
    return max(1, final_count)

class SentinelStream:
    """
    Thread-safe class that reads an RTSP stream (or video file),
    applies MOG2, extracts coarse tracking blobs, and measures density.
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
            "locations": []
        }
        
        # Initialize MOG2
        # varThreshold=50 works better against minor changes, detectShadows=True handles dark marks
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
        
        # Set Zones (Will be moved to database/configuration later)
        self.z7_x1, self.z7_y1, self.z7_x2, self.z7_y2 = 240, 290, 325, 625
        self.z6_x1, self.z6_y1, self.z6_x2, self.z6_y2 = 340, 280, 855, 990
        
        self.running = True
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()
        
    def _process_loop(self):
        tracker = RobustSentinelTracker()
        fusion_k = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 85))

        while self.running:
            cap = cv2.VideoCapture(self.source)
            if not cap.isOpened():
                print(f"[SentinelStream {self.stream_id}] Error: Cannot open source '{self.source}'")
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
                    
                # 1. Processing (MOG2 replaces static absdiff)
                fg_mask = self.bg_subtractor.apply(frame)
                # Filter out shadows (value 127) to maintain sharp people blobs
                _, thresh = cv2.threshold(fg_mask, 254, 255, cv2.THRESH_BINARY)
                thresh = cv2.bitwise_and(thresh, thresh, mask=roi_mask)
                
                # 2. Fusion (Restore original solid mega-blobs)
                fused = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, fusion_k)
                fused = cv2.dilate(fused, np.ones((5,5), np.uint8), iterations=1)
                
                # 3. Detection
                conts, _ = cv2.findContours(fused, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                detections = []
                for c in conts:
                    if cv2.contourArea(c) > 1500:
                        detections.append(list(cv2.boundingRect(c)))
                        
                tracks = tracker.update(detections)
                
                # Reset counts
                count_z7 = 0
                count_z6 = 0
                
                # Draw Zones
                cv2.rectangle(frame, (self.z7_x1, self.z7_y1), (self.z7_x2, self.z7_y2), (0, 255, 0), 2)
                cv2.rectangle(frame, (self.z6_x1, self.z6_y1), (self.z6_x2, self.z6_y2), (0, 255, 0), 2)
                
                # 4. Analyze Tracks
                for tid, data in tracks.items():
                    if data['ghost'] == 0:
                        x, y, w_box, h_box = data['box']
                        cx, cy = int(x + w_box/2), int(y + h_box/2)
                        
                        # Extract ROI from RAW thresh containing exactly this object box
                        roi = thresh[y:y+h_box, x:x+w_box]
                        
                        # Apply local counting logic inside the stable tracker box
                        people_in_this_box = count_people_in_box(roi, w_box)
                        
                        # Draw Visuals
                        cv2.rectangle(frame, (x, y), (x+w_box, y+h_box), (0, 255, 255), 2)
                        
                        label_text = f"ID {tid}"
                        if people_in_this_box > 1:
                            label_text += f" ({people_in_this_box})"
                            
                        cv2.putText(frame, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        
                        # Add to Zone Totals
                        if (self.z7_x1 < cx < self.z7_x2) and (self.z7_y1 < cy < self.z7_y2):
                            count_z7 += people_in_this_box
                        elif (self.z6_x1 < cx < self.z6_x2) and (self.z6_y1 < cy < self.z6_y2):
                            count_z6 += people_in_this_box
                            
                # Calculate Status
                total_count = count_z7 + count_z6
                
                # Render Totals
                text = f"Count of people: {total_count}"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                tx = frame_width - tw - 20
                ty = 50
                
                cv2.rectangle(frame, (tx - 10, ty - 30), (tx + tw + 10, ty + 10), (0, 0, 0), -1)
                cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Display individual zone counts
                cv2.putText(frame, f"Z7: {count_z7}", (self.z7_x1, self.z7_y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Z6: {count_z6}", (self.z6_x1, self.z6_y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Shared Stats Update
                area_z7 = (self.z7_x2 - self.z7_x1) * (self.z7_y2 - self.z7_y1)
                area_z6 = (self.z6_x2 - self.z6_x1) * (self.z6_y2 - self.z6_y1)
                norm_area = max(1.0, (area_z7 + area_z6) / 10000.0)
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
                    "locations": []
                }
                
                # Encode Buffer
                ok, buf = cv2.imencode(".jpg", frame)
                if ok:
                    self.latest_jpeg = buf.tobytes()
                    
                time.sleep(0.033) 
                
            cap.release()
            print(f"[SentinelStream {self.stream_id}] Stream disconnected. Retrying...")
            time.sleep(2.0)
            
    def get_latest_jpeg(self):
        return self.latest_jpeg
        
    def get_latest_stats(self):
        return self.latest_stats