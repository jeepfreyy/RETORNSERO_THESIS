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


class SentinelStream:
    """
    Thread-safe class that reads an RTSP stream (or video file),
    applies MOG2 and Watershed segmentation, tracks objects natively, 
    and exposes the latest JPEG/stats frame.
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
        
        # 1. Initialize MOG2
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
        
        # Set Zones (Will be moved to database/configuration later)
        self.z7_x1, self.z7_y1, self.z7_x2, self.z7_y2 = 240, 290, 325, 625
        self.z6_x1, self.z6_y1, self.z6_x2, self.z6_y2 = 340, 280, 855, 990
        
        self.running = True
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()
        
    def _process_loop(self):
        tracker = RobustSentinelTracker()
        
        # Watershed logic kernels
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_bg = np.ones((7, 7), np.uint8)

        while self.running:
            cap = cv2.VideoCapture(self.source)
            if not cap.isOpened():
                print(f"[SentinelStream {self.stream_id}] Error: Cannot open source '{self.source}'")
                time.sleep(2.0)
                continue
                
            # Attempt to read mask
            ret, first_frame = cap.read()
            if not ret:
                cap.release()
                time.sleep(2.0)
                continue
            
            frame_height, frame_width = first_frame.shape[:2]
            
            roi_mask = cv2.imread(self.mask_path, 0)
            if roi_mask is None or roi_mask.shape[:2] != (frame_height, frame_width):
                # Fallback to pure white mask if not found
                roi_mask = np.ones((frame_height, frame_width), dtype=np.uint8) * 255
                
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    # Video ended or stream disconnected
                    break
                    
                # ### PHASE 1: MOG2 & Shadow Removal ###
                fg_mask = self.bg_subtractor.apply(frame)
                # Strict thresholding removes shadows (gray pixels)
                _, thresh = cv2.threshold(fg_mask, 254, 255, cv2.THRESH_BINARY)
                
                thresh = cv2.bitwise_and(thresh, thresh, mask=roi_mask)
                
                # ### PHASE 2: True Marker-Based Watershed Algorithm ###
                
                # 1. Noise Removal
                opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open, iterations=1)
                closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_open, iterations=2)
                
                # 2. Sure Background (dilate the blobs)
                sure_bg = cv2.dilate(closed, kernel_bg, iterations=3)
                
                # 3. Sure Foreground (Distance transform peaks)
                dist_transform = cv2.distanceTransform(closed, cv2.DIST_L2, 5)
                # Threshold at 40% of max - optimal classical separation point for crowds
                max_val = dist_transform.max()
                if max_val > 0:
                    _, sure_fg = cv2.threshold(dist_transform, 0.4 * max_val, 255, 0)
                else:
                    sure_fg = np.zeros_like(dist_transform)
                sure_fg = np.uint8(sure_fg)
                
                # 4. Unknown Region
                unknown = cv2.subtract(sure_bg, sure_fg)
                
                # 5. Marker Generation
                ret_markers, markers = cv2.connectedComponents(sure_fg)
                # Background should not be 0, we set it to 1
                markers = markers + 1
                # Mark the unknown region with 0
                markers[unknown == 255] = 0
                
                # 6. Apply Watershed
                cv2.watershed(frame, markers)
                
                # ### PHASE 3: Detection & Tracking ###
                detections = []
                # Loop through all detected unique markers (ignoring background=1)
                for label in range(2, ret_markers + 1):
                    # Isolate the specific object segmented by watershed
                    obj_mask = np.zeros_like(markers, dtype=np.uint8)
                    obj_mask[markers == label] = 255
                    
                    conts, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if conts:
                        c = max(conts, key=cv2.contourArea)
                        if cv2.contourArea(c) > 100:  # Minimum acceptable size for a person
                            x, y, w_box, h_box = cv2.boundingRect(c)
                            detections.append([x, y, w_box, h_box])
                            
                tracks = tracker.update(detections)
                
                # Reset counts
                count_z7 = 0
                count_z6 = 0
                
                # Draw Zones
                cv2.rectangle(frame, (self.z7_x1, self.z7_y1), (self.z7_x2, self.z7_y2), (0, 255, 0), 2)
                cv2.rectangle(frame, (self.z6_x1, self.z6_y1), (self.z6_x2, self.z6_y2), (0, 255, 0), 2)
                
                # Analyze tracks
                for tid, data in tracks.items():
                    if data['ghost'] == 0:
                        x, y, w_box, h_box = data['box']
                        cx, cy = int(x + w_box/2), int(y + h_box/2)
                        
                        # Draw bounding box derived from watershed
                        cv2.rectangle(frame, (x, y), (x+w_box, y+h_box), (0, 255, 255), 2)
                        cv2.putText(frame, f"ID {tid}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        
                        if (self.z7_x1 < cx < self.z7_x2) and (self.z7_y1 < cy < self.z7_y2):
                            count_z7 += 1
                        elif (self.z6_x1 < cx < self.z6_x2) and (self.z6_y1 < cy < self.z6_y2):
                            count_z6 += 1
                            
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
                    
                time.sleep(0.033) # Simulate ~30 FPS if reading from fast file
                
            # If we break, release and retry
            cap.release()
            print(f"[SentinelStream {self.stream_id}] Stream disconnected. Retrying...")
            time.sleep(2.0)
            
    def get_latest_jpeg(self):
        return self.latest_jpeg
        
    def get_latest_stats(self):
        return self.latest_stats