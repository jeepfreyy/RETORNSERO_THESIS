import cv2
import json
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
import collections
import time
import threading
import os
import psutil
from datetime import datetime


# ---------------------------------------------------------------------------
# Headlight Suppression
# ---------------------------------------------------------------------------
def suppress_headlights(fg_mask, hsv_frame, v_threshold=200, dilation_px=60):
    """
    Remove vehicle headlight blobs from a foreground mask.

    Motorcycle headlights are the only objects in a nighttime barangay street
    that exceed V=200 in HSV — human clothing and skin never reach this level
    under ambient street lighting.  The bright core is dilated by dilation_px
    to also erase the dim halo that survives a pixel-level brightness cut
    (which is why the earlier brightness-scrubber attempt failed — it cut
    the core but left the halo, which then fused via morphology into a
    phantom foreground blob).
    """
    v = hsv_frame[:, :, 2]
    bright_core = (v > v_threshold).astype(np.uint8) * 255
    if dilation_px > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_px, dilation_px))
        bright_zone = cv2.dilate(bright_core, k)
    else:
        bright_zone = bright_core
    return cv2.bitwise_and(fg_mask, cv2.bitwise_not(bright_zone))


# ---------------------------------------------------------------------------
# Occupancy Map  (decoupled presence tracker)
# ---------------------------------------------------------------------------
class OccupancyMap:
    """
    Persistent presence tracker that is completely decoupled from motion.

    WHY THIS EXISTS
    ---------------
    MOG2 is a motion detector.  It removes stationary pixels from the
    foreground once they stop changing.  This is the Stationary Absorption
    Paradox: a seated person disappears from the count within 1–2 minutes
    even though they are physically still there.

    HOW IT WORKS
    ------------
    Pixels ENTER the map only after they have been consistently foreground
    for `confirm_frames` consecutive frames.  This blocks transient headlight
    flashes (which last 2–5 frames) from entering.

    Pixels LEAVE the map only when they have been PHYSICALLY DARK
    (V-channel < dark_v_thresh) for `evict_frames` consecutive frames.
    "Physically dark" means the actual road surface is visible — no person,
    no reflected light, no clothing.  A person standing still is NOT dark,
    so they stay in the map indefinitely.  A person who walks away exposes
    the dark road beneath them, and after `evict_frames` that pixel is
    evicted.

    HEADLIGHT INTERACTION
    ---------------------
    When a headlight illuminates the road, the road pixels become very
    BRIGHT — the opposite of dark.  Their dark_count resets to zero.
    The moment the light passes, those pixels return to their natural
    dark state and begin accumulating dark_count again.  If suppress_headlights()
    is applied upstream, headlight pixels never enter the map at all.
    """

    def __init__(self, shape, confirm_frames=10, evict_frames=150):
        h, w = shape[:2]
        self._map         = np.zeros((h, w), dtype=np.uint8)
        self._fg_count    = np.zeros((h, w), dtype=np.int32)
        # Per-pixel V value stored at the moment of confirmation.
        # Eviction fires when the pixel has been absent from fg_mask long enough
        # AND its current V differs noticeably from the confirmation-time V —
        # meaning the spot now looks different (person has left, background is back).
        self._baseline_v  = np.zeros((h, w), dtype=np.uint8)
        self._absent_count = np.zeros((h, w), dtype=np.int32)
        self._confirm     = confirm_frames
        self._evict       = evict_frames
        # V_CHANGE_THRESH: how much brightness change at a location signals departure.
        # 25 catches dark-clothed (V≈40) person leaving concrete (V≈70): Δ=30 > 25.
        # Also catches light-clothed (V≈120) person leaving concrete (V≈80): Δ=40 > 25.
        # Medium clothing on similar-brightness concrete may be missed — acceptable.
        self._v_change    = 25

    def update(self, fg_additions, hsv_frame):
        v       = hsv_frame[:, :, 2]           # HSV value channel
        fg_bool = fg_additions > 127

        # ── Confirmation ─────────────────────────────────────────────────
        # Track consecutive foreground frames; enter the map after confirm_frames.
        self._fg_count[fg_bool]  += 1
        self._fg_count[~fg_bool]  = 0

        new_confirmed = (self._fg_count >= self._confirm) & (self._map == 0)
        if np.any(new_confirmed):
            self._map[new_confirmed]        = 255
            self._baseline_v[new_confirmed] = v[new_confirmed]   # snapshot V at entry
            self._absent_count[new_confirmed] = 0

        # ── Absence tracking ──────────────────────────────────────────────
        # Only count absence for pixels already in the map.
        confirmed_mask = self._map > 0
        absent = confirmed_mask & ~fg_bool
        present = confirmed_mask & fg_bool

        self._absent_count[absent]  += 1
        self._absent_count[present]  = 0    # reset when pixel reappears in fg

        # ── Eviction ──────────────────────────────────────────────────────
        # Evict a confirmed pixel when it has been absent for evict_frames AND
        # the current brightness at that spot differs from the stored baseline.
        # This distinguishes "person sitting still (absorbed by MOG2, V unchanged)"
        # from "person left (empty background, V changed back to original)".
        v_changed  = np.abs(v.astype(np.int32) - self._baseline_v.astype(np.int32)) > self._v_change
        to_evict   = confirmed_mask & (self._absent_count >= self._evict) & v_changed
        self._map[to_evict]          = 0
        self._absent_count[to_evict] = 0

        return self._map

    def reset(self):
        self._map[:]          = 0
        self._fg_count[:]     = 0
        self._baseline_v[:]   = 0
        self._absent_count[:] = 0



def apply_gamma_correction(frame, target_mean=100):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    current_mean = gray.mean()
    if current_mean < 1:
        return frame
    gamma = np.log(target_mean / 255.0) / np.log(current_mean / 255.0)
    gamma = float(np.clip(gamma, 0.4, 2.5))
    lut = np.array([min(255, int((i / 255.0) ** gamma * 255)) for i in range(256)], dtype=np.uint8)
    return cv2.LUT(frame, lut)


class RobustSentinelTracker:
    def __init__(self, max_ghost=30, dist_thresh=150, min_lifetime=10, merge_thresh=40):
        self.next_id = 1
        self.tracks = {}
        self.max_ghost = max_ghost
        self.dist_thresh = dist_thresh
        self.min_lifetime = min_lifetime
        self.merge_thresh = merge_thresh

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
                    self.tracks[tid]['lifetime'] += 1
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

    def merge_nearby_boxes(self, boxes, thresh=None):
        if thresh is None:
            thresh = self.merge_thresh
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
        self.tracks[self.next_id] = {'box': box, 'ghost': 0, 'lifetime': 1}
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
def is_human_blob(roi_thresh, x, y, w, h, frame_height, min_blob_area=800, max_aspect=1.8):
    """
    Validates whether a detected blob is likely a human using:
      1. Aspect Ratio  –  tunable via max_aspect (default 1.8 for eye-level;
         use 2.5–3.5 for elevated/overhead cameras with seated subjects)
      2. Distance Transform peak profile  –  people have a narrow/round peak,
         vehicles and boxes have a wide, flat peak
      3. Perspective-scaled minimum area (tunable for nighttime)
    Returns True if the blob passes all checks.
    """
    if roi_thresh is None or roi_thresh.size == 0:
        return False

    pw = get_perspective_weight(y + h, frame_height)

    # 1. Minimum area scales with perspective — base is tunable per camera
    min_area = int(min_blob_area * pw)
    area = w * h
    if area < min_area:
        return False

    # 2. Aspect ratio check (width / height)
    #    Standing/walking people (eye-level):  ratio < 1.3
    #    Seated people (45-60° overhead):      ratio up to 3.0+ (chair+body = wide blob)
    #    max_aspect is now a tunable parameter calibrated per camera angle.
    aspect = w / max(1, h)
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
    peak_ratio = max_val / max(1, min(w, h) / 2.0)
    if peak_ratio < 0.08:
        return False

    return True


# ---------------------------------------------------------------------------
# Area-Based Crowd Counter
# ---------------------------------------------------------------------------
def count_people_by_area(fused_mask, avg_pixels_per_person, area_at_zero):
    """
    Area-based crowd counter. Estimates crowd size from total foreground
    pixel area rather than blob topology.

    Rationale: morphological CLOSE operations merge nearby people into compact
    blobs, destroying the inter-person separation that watershed sub-counting
    needs. Total foreground area remains proportional to crowd size regardless
    of blob merging, bypassing this limitation entirely.

    Args:
        fused_mask:            Post-morphology binary mask after ROI masking.
        avg_pixels_per_person: Slope from linear calibration.
        area_at_zero:          Intercept — baseline noise at 0 people.
    Returns:
        Integer estimated person count, clamped to >= 0.
    """
    area = int(cv2.countNonZero(fused_mask))
    estimated = (area - area_at_zero) / max(1.0, avg_pixels_per_person)
    return max(0, round(estimated))


# ---------------------------------------------------------------------------
# Watershed Scene-Level Counter
# ---------------------------------------------------------------------------
def count_people_watershed_scene(fused_mask, min_blob_area=350, dt_thresh=0.30,
                                  min_peak_area=50):
    """
    Scene-level watershed crowd counter using distance-transform peak detection.

    Operates on the entire fused foreground mask (not per-blob): finds every
    independent blob, applies distance-transform peak separation, and counts
    the resulting watershed nuclei.  Works best on sparse/moderately-dense
    crowds where individuals are not completely merged.

    Args:
        fused_mask:     Post-morphology binary mask.
        min_blob_area:  Minimum contour area to be considered a crowd blob.
        dt_thresh:      Fraction of each blob's DT max to use as the valley floor.
                        Lower values (0.25–0.35) separate tightly-seated peaks.
        min_peak_area:  Minimum area (px²) of a DT peak to count as one person.

    Returns:
        (total_count, centroids)
            total_count: integer estimated person count (>= 0)
            centroids:   list of (cx, cy) tuples — one per watershed segment
    """
    contours, _ = cv2.findContours(fused_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total_count = 0
    centroids   = []

    erode_k = np.ones((3, 3), np.uint8)

    for c in contours:
        if cv2.contourArea(c) < min_blob_area:
            continue

        x, y, w, h = cv2.boundingRect(c)

        # Reject shadow silhouettes: cast shadows are very elongated (narrow
        # and long).  A real person blob from an overhead camera has aspect < 4.
        # Lamp-post shadows are typically 8:1 or more.
        if max(w, h) / max(1, min(w, h)) > 4.5:
            continue

        roi = fused_mask[y:y + h, x:x + w].copy()

        # Erode slightly to break thin bridges between bodies
        roi_e = cv2.erode(roi, erode_k, iterations=1)

        dist = cv2.distanceTransform(roi_e, cv2.DIST_L2, 5)
        if dist.max() < 4.0:
            # Blob too thin after erosion — count as 1 person
            M = cv2.moments(c)
            if M['m00'] > 0:
                centroids.append((int(M['m10'] / M['m00']),
                                  int(M['m01'] / M['m00'])))
            total_count += 1
            continue

        # Threshold the distance transform to isolate peak regions (sure foreground)
        _, sure_fg = cv2.threshold(
            dist, max(4.0, dt_thresh * dist.max()), 255, cv2.THRESH_BINARY
        )
        sure_fg = np.uint8(sure_fg)

        # Count connected peaks — each peak is one person nucleus
        peak_contours, _ = cv2.findContours(sure_fg, cv2.RETR_EXTERNAL,
                                             cv2.CHAIN_APPROX_SIMPLE)
        blob_people = 0
        for pc in peak_contours:
            if cv2.contourArea(pc) < min_peak_area:
                continue
            blob_people += 1
            M = cv2.moments(pc)
            if M['m00'] > 0:
                # Map centroid back to full-frame coordinates
                cx_roi = int(M['m10'] / M['m00'])
                cy_roi = int(M['m01'] / M['m00'])
                centroids.append((x + cx_roi, y + cy_roi))

        total_count += max(1, blob_people)

    return total_count, centroids





# ---------------------------------------------------------------------------
# Main Stream Class
# ---------------------------------------------------------------------------
class SentinelStream:
    """
    Thread-safe class that reads an RTSP stream (or video file), applies
    MOG2 background subtraction, validates blobs structurally, and counts
    people via a 3-stage fusion pipeline (watershed + area + YOLO census).
    """

    def __init__(
        self,
        stream_id,
        source="video1.mp4",
        mask_path="mask_layer.png",
        # --- Independent Hyperparameters per Camera ---
        mog2_history=500,
        mog2_threshold=16,
        min_blob_area=800,
        ghost_threshold=30,
        max_capacity=30,
        # --- Morphological Tuning (nighttime-critical) ---
        morph_kernel=(7, 50),      # MORPH_CLOSE fusion kernel (w, h)
        h_morph_kernel=(40, 7),   # Horizontal MORPH_CLOSE kernel (w, h) — fuses seated-person fragments
        dilate_kernel=3,           # Square dilation size in pixels
        merge_thresh=40,      # px gap tolerance for merging nearby body-part fragments
        dist_thresh=150,      # px max centroid shift the tracker tolerates before dropping a track
        detect_shadows=True,
        enable_gamma=False,
        process_scale=1.0,    # Downscale factor for processing
        max_aspect=1.8,       # Max blob w/h ratio for is_human_blob (use 2.5-3.5 for overhead cameras)
        area_px_per_person=None,   # If set, use area-based counting instead of watershed
        area_baseline=0.0,         # Intercept from area_calibration.json
        # --- Headlight Suppression ---
        headlight_v_thresh=200,    # HSV V-channel cutoff; pixels above this are headlights
        headlight_dilation_px=60,  # px radius to expand bright core (kills the halo)
        # --- Occupancy Map ---
        occupancy_confirm_frames=10,   # fg frames before a pixel enters the map (blocks transients)
        occupancy_evict_sec=5.0,       # seconds a pixel must be absent + changed before eviction
        # --- Warmup ---
        warmup_frames=1500,            # frames to suppress count while MOG2 stabilises
        # --- YOLO Census (appearance-based fallback for absorbed stationary people) ---
        yolo_model_path="yolov8n.pt",  # set to None to disable; path relative to cwd
        yolo_conf=0.40,                # detection confidence threshold
        yolo_iou=0.35,                 # NMS IoU threshold — lower = fewer double-detections
        # --- MOG2 Persistence ---
        bg_model_path="mog2_bg.yml",   # save/load pre-warmed MOG2 state; None to disable
        # --- Video Identity ---
        video_label="Main Video",      # human-readable label shown in notifications and clips
    ):
        self.stream_id = stream_id
        self.source = source
        self.mask_path = mask_path
        self.min_blob_area = min_blob_area
        self.max_capacity = max_capacity
        self._morph_kernel = morph_kernel
        self._h_morph_kernel = h_morph_kernel
        self._dilate_kernel = dilate_kernel
        self._merge_thresh = merge_thresh
        self._dist_thresh = dist_thresh
        self._enable_gamma = enable_gamma
        self._process_scale = process_scale
        self._max_aspect = max_aspect
        self._area_px_per_person = area_px_per_person
        self._area_baseline = area_baseline
        self._headlight_v_thresh = headlight_v_thresh
        self._headlight_dilation_px = headlight_dilation_px
        self._occupancy_confirm_frames = occupancy_confirm_frames
        self._occupancy_evict_sec = occupancy_evict_sec

        # Store MOG2 construction params so we can recreate the subtractor on
        # video switch (MOG2 has no reset() method — must be re-instantiated).
        self._mog2_history    = mog2_history
        self._mog2_threshold  = mog2_threshold
        self._detect_shadows  = detect_shadows

        # MOG2 — high history resists stationary-crowd absorption
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=mog2_history, varThreshold=mog2_threshold, detectShadows=detect_shadows
        )

        # YOLO census — loaded once at startup; None if disabled or model not found
        self._yolo_model = None
        self._yolo_conf  = yolo_conf
        self._yolo_iou   = yolo_iou
        if yolo_model_path is not None:
            if os.path.exists(yolo_model_path):
                try:
                    from ultralytics import YOLO as _YOLO
                    self._yolo_model = _YOLO(yolo_model_path)
                    print(f"[YOLO] Loaded {yolo_model_path}  conf={yolo_conf}  iou={yolo_iou}")
                except Exception as _e:
                    print(f"[YOLO] Could not load {yolo_model_path}: {_e} — census disabled.")
            else:
                print(f"[YOLO] {yolo_model_path} not found — census disabled.")

        # MOG2 persistence — path for save/load of pre-warmed background model
        self._bg_model_path = bg_model_path

        # Video identity — shown in notifications and clip metadata
        self._video_label = video_label

        # Per-video warmup / source-switch state
        self._warmup_frames        = warmup_frames
        self._switch_source        = None   # set by switch_source(); outer loop picks it up
        self._switch_warmup        = None
        self._switch_recreate_mog2 = False  # whether to throw away the MOG2 model on next switch
        self._switch_mask_path     = None   # None = keep current mask_path on next switch
        self._switch_video_label   = None   # None = keep current label on next switch
        self._skip_model_load      = False  # set True when recreate_mog2=True to block load
        self.is_warming_up         = True   # True until the first warmup window expires

        self.latest_jpeg = None
        self.latest_stats = {
            "count": 0,
            "density": 0,
            "status": "SAFE",
            "locations": [],
            "fps": 0,
            "latency_ms": 0,
            # --- Performance Profiling ---
            "cpu_percent": 0.0,
            "ram_mb": 0.0,
        }

        # Pass ghost_threshold into tracker
        self._ghost_threshold = ghost_threshold

        # --- Ring Buffer & Event Clip Recording ---
        self.pause_saving = os.environ.get('PAUSE_SAVING', 'False').lower() in ['true', '1', 't']
        self.frame_buffer = collections.deque(maxlen=150)   # ~5 sec @ 30fps
        self.clip_writer = None
        self.clip_recording = False
        self.clip_start_time = None
        self.clip_filename = None
        self.clip_min_duration = 5        # minimum seconds recorded before a clip can close
        self.clip_max_duration = 15       # hard cap: stop clip after this many seconds
        self.temp_clips_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Temp_Clips")
        self.temp_clips = []              # in-memory metadata list
        self._prev_clip_level = "LOW"
        self._cooldown_until = 0.0        # epoch when short inter-clip gap expires (5 s)
        self._last_clip_density = None    # density level that triggered the last auto clip
        self._last_clip_was_manual = False  # True when the last clip was operator-triggered
        # Per-density cooldown: MEDIUM and HIGH each get their own 2-minute timer.
        # After a MEDIUM clip ends, no new MEDIUM clip fires for 2 minutes — but
        # a HIGH clip can still fire immediately, and vice versa.
        self._density_cooldown_until = {"MEDIUM": 0.0, "HIGH": 0.0}
        self.DENSITY_COOLDOWN_SECS = 120.0   # 2-minute per-density cooldown
        self._high_since = 0.0            # epoch when current HIGH streak began (0 = not HIGH)
        self.manual_clip_active = False        # True while a user-triggered clip is recording
        self._manual_clip_requested = False    # one-shot flag set by API
        self._manual_clip_stop_requested = False  # one-shot flag set by API
        os.makedirs(self.temp_clips_dir, exist_ok=True)
        
        # Load persisted temp clips
        for f in os.listdir(self.temp_clips_dir):
            if f.endswith('.mp4') and f.startswith(self.stream_id):
                thumb = f.replace('.mp4', '.jpg')
                parts = f.replace('.mp4', '').split('_')
                if len(parts) >= 4:
                    timestamp = parts[1] + " " + parts[2].replace('-', ':')
                    density_tag = parts[3]
                    self.temp_clips.append({
                        "filename": f,
                        "thumbnail": thumb,
                        "timestamp": timestamp,
                        "density": density_tag,
                        "camera_id": self.stream_id,
                        "duration": 0
                    })
        self.temp_clips.sort(key=lambda x: x['timestamp'], reverse=True)
        self.temp_clips = self.temp_clips[:20]

        self.running = True
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()

    # ------------------------------------------------------------------
    def switch_source(self, new_source, new_warmup_frames=None,
                      recreate_mog2=False, new_mask_path=None, new_video_label=None):
        """
        Hot-swap to a different video file without stopping the thread.

        The inner frame-read loop checks self._switch_source on every iteration.
        When it is set, the loop breaks immediately, the outer loop applies the
        new source + warmup, optionally recreates MOG2 and swaps the ROI mask,
        then restarts.

        new_warmup_frames : frames to suppress count on the new video.
        recreate_mog2     : True  → throw away the current background model and
                                    let MOG2 re-learn from the new video's first
                                    frames.  Required when the new video has a
                                    different background from the current one
                                    (e.g. truck is parked throughout VIDEO3 but
                                    was absent in the main-video background model).
                                    Set warmup_frames to the number of empty-scene
                                    frames so MOG2 finishes learning before people
                                    arrive and the occupancy map resets.
                            False → keep the existing background model.  Correct
                                    when the new video shares the same empty-alley
                                    background (VIDEO2: people present from frame 0,
                                    no empty phase to re-learn from).
        new_mask_path     : path to the ROI mask PNG for the new video.
                            Pass None to keep the current mask unchanged.
        """
        self._switch_source        = new_source
        self._switch_warmup        = new_warmup_frames if new_warmup_frames is not None \
                                     else self._warmup_frames
        self._switch_recreate_mog2 = recreate_mog2
        self._switch_mask_path     = new_mask_path    # None = keep current mask
        self._switch_video_label   = new_video_label  # None = keep current label
        self.is_warming_up         = True

    # ------------------------------------------------------------------
    def _process_loop(self):
        # Build kernels from tunable params (nighttime-critical).
        # These never change between videos, so they live outside the outer loop.
        open_k     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fusion_k   = cv2.getStructuringElement(cv2.MORPH_RECT, self._morph_kernel)
        h_fusion_k = cv2.getStructuringElement(cv2.MORPH_RECT, self._h_morph_kernel)
        d_kernel   = np.ones((self._dilate_kernel, self._dilate_kernel), np.uint8)
        proc       = psutil.Process(os.getpid())
        ema_frame_time = 0.033

        # Tracker is recreated inside the outer loop so track state does not
        # carry over from a previous video loop or after a source switch.
        tracker = None

        while self.running:
            # ── Apply any pending source switch ──────────────────────────
            # switch_source() sets _switch_source; the inner loop breaks when
            # it sees it set.  We apply it here at the start of the outer loop
            # so the new source takes effect before we open the cap.
            if self._switch_source is not None:
                self.source          = self._switch_source
                self._warmup_frames  = self._switch_warmup
                do_recreate          = self._switch_recreate_mog2
                if self._switch_mask_path is not None:
                    self.mask_path   = self._switch_mask_path
                if self._switch_video_label is not None:
                    self._video_label = self._switch_video_label
                self._switch_source        = None
                self._switch_warmup        = None
                self._switch_recreate_mog2 = False
                self._switch_mask_path     = None
                self._switch_video_label   = None
                # Reset per-density cooldowns on source switch so the new
                # video scenario is not blocked by timers from the old one.
                self._density_cooldown_until = {"MEDIUM": 0.0, "HIGH": 0.0}
                self._cooldown_until = 0.0

                if do_recreate:
                    # Recreate MOG2 so it re-learns the new video's background.
                    # Required when the scenario video has objects (e.g. the
                    # parked truck in VIDEO1/VIDEO3) that were ABSENT in the
                    # current background model.  If we kept the old model those
                    # objects would appear as foreground indefinitely → haywire.
                    # Set warmup_frames = the number of empty-scene frames in the
                    # new video so MOG2 finishes learning BEFORE people arrive,
                    # at which point the occupancy map resets cleanly.
                    self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                        history=self._mog2_history,
                        varThreshold=self._mog2_threshold,
                        detectShadows=self._detect_shadows,
                    )
                    # Block the model-load step below — loading the saved model
                    # here would immediately undo the recreate, putting the old
                    # background back into the fresh subtractor.
                    self._skip_model_load = True
                # else: keep the existing background model.
                #   Correct for VIDEO2 (people present from frame 0 with no
                #   empty-background phase to re-learn from).  The main video's
                #   "empty alley" model lets the people appear as foreground
                #   immediately rather than being absorbed during bootstrapping.

            # Fresh tracker on every (re)start so stale tracks don't bleed across videos.
            tracker = RobustSentinelTracker(
                max_ghost=self._ghost_threshold,
                min_lifetime=10,
                dist_thresh=self._dist_thresh,
                merge_thresh=self._merge_thresh,
            )

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

            if self._process_scale != 1.0:
                proc_h = int(first_frame.shape[0] * self._process_scale)
                proc_w = int(first_frame.shape[1] * self._process_scale)
                first_frame = cv2.resize(first_frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA)

            frame_height, frame_width = first_frame.shape[:2]

            raw_mask = cv2.imread(self.mask_path, 0)
            if raw_mask is not None:
                roi_mask = cv2.resize(raw_mask, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)
                if roi_mask.shape[:2] != (frame_height, frame_width):
                    roi_mask = np.ones((frame_height, frame_width), dtype=np.uint8) * 255
            else:
                roi_mask = np.ones((frame_height, frame_width), dtype=np.uint8) * 255

            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # Occupancy map is reset on every video restart so stale
            # presence state from a previous loop does not corrupt the new one.
            occupancy = OccupancyMap(
                (frame_height, frame_width),
                confirm_frames=self._occupancy_confirm_frames,
                evict_frames=int(self._occupancy_evict_sec * 30),
            )

            # How many frames MOG2 needs before its background model is stable.
            # During warmup: count is forced to 0 so startup chair/background
            # noise never reaches the dashboard.
            # At the END of warmup: occupancy map is reset, purging any false
            # positives (chairs, walls, vehicles) accumulated while MOG2 was
            # still learning.
            # WARMUP_FRAMES is now per-video (set via switch_source or __init__
            # warmup_frames= param).  Default 1500 for the main demo video with
            # vehicles; 60–150 for no-vehicle scenario videos.
            warmup_frames = self._warmup_frames

            # Temporal smoothing: EMA over recent count values damps single-frame
            # spikes caused by mass simultaneous movement (everybody shifting at once
            # briefly inflates fused_px, which inflates the area estimate).
            # alpha=0.4 means each new reading contributes 40%; prior history 60%.
            # This gives ~3-frame settling time while staying responsive to real changes.
            count_ema        = 0.0
            COUNT_ALPHA      = 0.4
            loop_frame_idx   = 0
            mog2_lr          = -1     # -1 = auto learning; frozen to 0 after warmup
            bg_reference     = None   # set at warmup end; used for static diff detection
            recount_floor    = 0.0    # sticky lower-bound; updated every 2 s by census
            RECOUNT_INTERVAL = 60     # frames between census runs (~2 s at 30 fps)

            # ── MOG2 PERSISTENCE: load pre-warmed model to skip warmup ────────
            # On first run the file won't exist; MOG2 learns from scratch and the
            # model is saved at frame==warmup_frames.  On all subsequent runs the
            # model is loaded here, warmup is skipped (loop_frame_idx jumps past
            # the gate), and the background reference is restored from the
            # companion PNG so the static-diff detector is immediately accurate.
            #
            # _skip_model_load is set when recreate_mog2=True was requested on a
            # source switch — loading the saved model there would silently undo
            # the recreate by putting the old background back into the fresh subtractor.
            _do_load = (
                self._bg_model_path
                and os.path.exists(self._bg_model_path)
                and not self._skip_model_load
            )
            self._skip_model_load = False  # consume the flag regardless

            if _do_load:
                try:
                    fs = cv2.FileStorage(self._bg_model_path, cv2.FILE_STORAGE_READ)
                    self.bg_subtractor.read(fs.getNode("bg_model"))
                    fs.release()
                    companion_png = self._bg_model_path.replace(".yml", "_ref.png")
                    if os.path.exists(companion_png):
                        bg_reference = cv2.imread(companion_png)
                    loop_frame_idx = self._warmup_frames + 1  # skip warmup gate
                    mog2_lr        = 0                         # freeze learning
                    print(f"[MOG2] Loaded pre-warmed model from {self._bg_model_path} — warmup skipped.")
                except Exception as _e:
                    print(f"[MOG2] Failed to load {self._bg_model_path}: {_e} — running warmup normally.")
                    loop_frame_idx = 0
            # ─────────────────────────────────────────────────────────────────

            while self.running:
                # Break immediately when a source switch is requested so the
                # outer loop can apply the new source and restart cleanly.
                if self._switch_source is not None:
                    break

                start_t = time.time()
                ret, frame = cap.read()
                if not ret:
                    break

                if self._process_scale != 1.0:
                    proc_h = int(frame.shape[0] * self._process_scale)
                    proc_w = int(frame.shape[1] * self._process_scale)
                    frame = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA)

                if self._enable_gamma:
                    frame = apply_gamma_correction(frame)

                # Pre-compute HSV once per frame (used by headlight suppressor
                # and occupancy map — avoids redundant color conversions)
                hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                # ── PHASE 1: MOG2 Background Subtraction ──
                fg_mask = self.bg_subtractor.apply(frame, learningRate=mog2_lr)
                # Scrub MOG2 shadows (gray=127 when detectShadows=True)
                _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
                # Apply ROI mask
                thresh = cv2.bitwise_and(fg_mask, fg_mask, mask=roi_mask)

                # ── HEADLIGHT SUPPRESSION ─────────────────────────────
                # Must run AFTER ROI masking (so we don't fight the mask)
                # and BEFORE morphology (so the halo never gets fused into
                # a phantom blob).  dilation_px is already in process-space
                # coordinates (the frame is already downscaled).
                if self._headlight_v_thresh < 255:
                    thresh = suppress_headlights(
                        thresh, hsv_frame,
                        v_threshold=self._headlight_v_thresh,
                        dilation_px=self._headlight_dilation_px,
                    )


                # ── STATIC BACKGROUND REFERENCE DIFF ────────────────
                # Secondary detector: compares current frame directly against
                # the empty-scene reference captured at warmup end.
                # Catches people who are seated and static long enough that
                # MOG2 absorption / high variance makes them invisible to
                # motion detection.  Fused with MOG2 output so both moving
                # and stationary people reach the morphological pipeline.
                if bg_reference is not None:
                    _diff      = cv2.absdiff(frame, bg_reference)
                    _diff_gray = cv2.cvtColor(_diff, cv2.COLOR_BGR2GRAY)
                    _, _diff_mask = cv2.threshold(_diff_gray, 40, 255, cv2.THRESH_BINARY)
                    _diff_mask = cv2.bitwise_and(_diff_mask, roi_mask)
                    if self._headlight_v_thresh < 255:
                        _diff_mask = suppress_headlights(
                            _diff_mask, hsv_frame,
                            v_threshold=self._headlight_v_thresh,
                            dilation_px=self._headlight_dilation_px,
                        )
                    thresh = cv2.bitwise_or(thresh, _diff_mask)

                # ── PHASE 2: Fusion & Opening (Shape Cleanup) ──────
                # Rub out tiny artifacts before merging
                cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, open_k)
                fused = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, fusion_k)    # vertical: closes gaps within upright silhouettes
                fused = cv2.morphologyEx(fused, cv2.MORPH_CLOSE, h_fusion_k)    # horizontal: fuses seated/foreshortened fragments
                fused = cv2.dilate(fused, d_kernel, iterations=1)

                # ── WARMUP GATE ───────────────────────────────────────
                # At the exact frame MOG2 stabilises:
                #   1. Occupancy map is wiped — erases chairs/walls from startup.
                #   2. MOG2 learning rate frozen to 0 — background locked on the
                #      empty-scene state.
                #   3. Background reference frame captured from MOG2's learned
                #      background model.  Used by the static-diff detector to find
                #      seated people who fall within MOG2's variance envelope and
                #      are therefore missed by motion detection alone.
                if loop_frame_idx == warmup_frames:
                    occupancy.reset()
                    mog2_lr      = 0
                    bg_reference = self.bg_subtractor.getBackgroundImage()
                    # ── MOG2 PERSISTENCE: save after warmup ───────────────────
                    # This block only runs on a fresh warmup (no YAML existed, or
                    # recreate_mog2=True was used).  When a YAML is loaded above,
                    # loop_frame_idx is set to warmup_frames+1, so this gate is
                    # never reached — the existing file is never overwritten.
                    if self._bg_model_path:
                        try:
                            fs = cv2.FileStorage(self._bg_model_path, cv2.FILE_STORAGE_WRITE)
                            self.bg_subtractor.write(fs, "bg_model")
                            fs.release()
                            if bg_reference is not None:
                                companion_png = self._bg_model_path.replace(".yml", "_ref.png")
                                cv2.imwrite(companion_png, bg_reference)
                            print(f"[MOG2] Saved warmed model to {self._bg_model_path}.")
                        except Exception as _e:
                            print(f"[MOG2] Could not save model: {_e}")
                    # ─────────────────────────────────────────────────────────
                loop_frame_idx += 1

                # ── OCCUPANCY MAP UPDATE ──────────────────────────────
                # Feed the post-morphology fused mask into the occupancy map.
                # The map adds pixels that have been consistently foreground
                # (confirm_frames) and only removes them when physically dark
                # (evict_frames).  This preserves seated people after MOG2
                # absorbs them, while not accumulating transient headlight hits.
                persistent_fg = occupancy.update(fused, hsv_frame)

                # ── PHASE 3: Detection with Structural Validation ───
                conts, _ = cv2.findContours(fused, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                detections = []
                for c in conts:
                    x, y, w_box, h_box = cv2.boundingRect(c)
                    # Extract ROI from raw thresh for structural analysis
                    roi_check = fused[y : y + h_box, x : x + w_box]
                    if is_human_blob(roi_check, x, y, w_box, h_box, frame_height, self.min_blob_area, self._max_aspect):
                        detections.append([x, y, w_box, h_box])

                tracks = tracker.update(detections)

                # ── PHASE 4: Collaborative Incremental Fusion ────────────
                #
                # Three algorithms, three roles — each stage builds on the
                # previous rather than competing for the highest value.
                #
                #  Stage 1 │ ws_fused  │ Motion anchor.  Who MOG2 currently sees.
                #           │           │ The most trustworthy real-time snapshot.
                #           │           │
                #  Stage 2 │ ws_occ    │ Absorption corrector.  Adds confirmed-
                #           │           │ stationary people that MOG2 absorbed.
                #           │           │ Capped at the motion baseline so stale
                #           │           │ OccupancyMap pixels cannot inflate alone.
                #           │           │
                #  Stage 3 │ ar_count  │ Dense-crowd corrector.  Recovers the
                #           │           │ shortfall when tightly packed blobs merge
                #           │           │ and watershed undercounts peaks.  Bounded
                #           │           │ by occ confirmation — stale pixels that
                #           │           │ are not supported by occ are ignored.
                #
                # No single stale estimator can dominate the output because
                # each stage can only raise the count within defined limits.

                # Run all three estimators
                ws_fused_count, ws_fused_centroids = count_people_watershed_scene(
                    fused,
                    min_blob_area=self.min_blob_area,
                    dt_thresh=0.30,
                )
                ws_occ_count, ws_occ_centroids = count_people_watershed_scene(
                    persistent_fg,
                    min_blob_area=self.min_blob_area,
                    dt_thresh=0.30,
                )
                ws_count     = ws_fused_count
                ws_centroids = ws_fused_centroids

                if self._area_px_per_person is not None:
                    ar_count = count_people_by_area(
                        persistent_fg,
                        self._area_px_per_person,
                        self._area_baseline,
                    )
                else:
                    ar_count = 0

                # Stage 1 — motion anchor
                total_count = ws_fused_count

                # Stage 2 — absorption correction
                # OccupancyMap can add confirmed-stationary people that motion
                # missed, but only up to the motion count itself.  If motion
                # sees 3 people, occ cannot claim 8 absorbed — that would mean
                # 8 invisible people appeared, which is physically implausible.
                occ_surplus = max(0, ws_occ_count - total_count)
                absorption_cap = max(total_count, 2)   # floor: allow at least 2 absorbed
                total_count = total_count + min(occ_surplus, absorption_cap)

                # Stage 3 — dense-crowd area correction
                # When blobs merge in a dense crowd, watershed undercounts because
                # distance-transform peaks collapse.  ar_count (pixel-mass formula)
                # recovers that shortfall.  Only applied when:
                #   a) area says MORE than the current estimate (genuine shortfall)
                #   b) occ ALSO confirms the density (not just stale pixel noise)
                # Bounded by occ_count to prevent area from overshooting reality.
                if ar_count > total_count and ws_occ_count >= total_count:
                    total_count = min(ar_count, ws_occ_count)

                # ── 2-SECOND CENSUS WITH SHADOW VALIDATION ────────────
                # Every 60 frames rescan the OccupancyMap to recount all
                # confirmed people, including those absorbed by MOG2.
                if (loop_frame_idx > warmup_frames and
                        loop_frame_idx % RECOUNT_INTERVAL == 0):
                    _pf_conts, _ = cv2.findContours(
                        persistent_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    _validated = np.zeros_like(persistent_fg)
                    for _bc in _pf_conts:
                        if cv2.contourArea(_bc) < self.min_blob_area:
                            continue
                        _bx, _by, _bw, _bh = cv2.boundingRect(_bc)
                        if max(_bw, _bh) / max(1, min(_bw, _bh)) > 4.5:
                            continue
                        _bm = np.zeros(persistent_fg.shape, dtype=np.uint8)
                        cv2.drawContours(_bm, [_bc], -1, 255, -1)
                        _v_mean = cv2.mean(hsv_frame[:, :, 2], mask=_bm)[0]
                        _s_mean = cv2.mean(hsv_frame[:, :, 1], mask=_bm)[0]
                        if _v_mean < 45 and _s_mean < 32:
                            continue
                        cv2.drawContours(_validated, [_bc], -1, 255, -1)

                    _cws, _ = count_people_watershed_scene(
                        _validated,
                        min_blob_area=self.min_blob_area,
                        dt_thresh=0.30,
                    )
                    _car = (count_people_by_area(
                                _validated,
                                self._area_px_per_person,
                                self._area_baseline)
                            if self._area_px_per_person is not None else 0)
                    # YOLO census — appearance-based, catches people MOG2 absorbed
                    _yolo_n = 0
                    if self._yolo_model is not None:
                        _yolo_res = self._yolo_model(
                            frame, device="cpu", imgsz=640,
                            conf=self._yolo_conf, iou=self._yolo_iou,
                            classes=[0], verbose=False
                        )[0]
                        for _b in _yolo_res.boxes:
                            _bx1, _by1, _bx2, _by2 = _b.xyxy[0].tolist()
                            _bcx = int((_bx1 + _bx2) / 2)
                            _bcy = int((_by1 + _by2) / 2)
                            _bcy = max(0, min(_bcy, roi_mask.shape[0] - 1))
                            _bcx = max(0, min(_bcx, roi_mask.shape[1] - 1))
                            if roi_mask[_bcy, _bcx] > 127:
                                _yolo_n += 1

                    # YOLO is the unconditional floor anchor.
                    # It is stateless (appearance-based, no history) so it
                    # reflects the actual current headcount — not stale occupancy
                    # ghosts left by recently-departed people.  When YOLO returns
                    # 0 the floor is 0: Stage 2 still provides real-time
                    # absorption correction via ws_occ_count, and the next census
                    # will raise the floor again if people reappear.
                    # The old fallback to _cws when _yolo_n==0 caused departure
                    # spikes: occupancy retains confirmed positions for evict_sec
                    # after people leave, and without YOLO to override it the
                    # stale _cws locked a falsely-high floor until eviction fired.
                    recount_floor = float(_yolo_n)

                # Floor decay — only when occupancy map has no confirmed blobs,
                # meaning the scene is genuinely empty (not just absorbed by MOG2).
                _occ_area = cv2.countNonZero(persistent_fg)
                if total_count < recount_floor and _occ_area < self.min_blob_area:
                    recount_floor = max(float(total_count), recount_floor * 0.80)

                # Apply census floor — raises count when MOG2 misses stationary people.
                total_count = max(total_count, int(round(recount_floor)))

                # Suppress the count during MOG2 warmup so startup
                # chair/background noise never triggers alerts.
                if loop_frame_idx <= warmup_frames:
                    total_count    = 0
                    count_ema      = 0.0
                    recount_floor  = 0.0
                    self.is_warming_up = True
                else:
                    self.is_warming_up = False

                # Temporal EMA smoothing — damps single-frame spikes from
                # mass simultaneous movement without adding noticeable lag.
                count_ema   = COUNT_ALPHA * total_count + (1 - COUNT_ALPHA) * count_ema
                total_count = round(count_ema)

                # Phase 2.1 & 2.2: Density and Status
                density = int(min(100, (total_count / self.max_capacity) * 100))

                # density% is kept for the timeline chart (capacity gauge).
                # status uses raw count tiers so thresholds are independent
                # of max_capacity (which is venue-specific).
                if total_count <= 3:
                    status = "SAFE"
                elif total_count <= 6:
                    status = "WARNING"
                else:
                    status = "CRITICAL"

                # Calculate latency/fps
                proc_time = time.time() - start_t
                ema_frame_time = 0.9 * ema_frame_time + 0.1 * max(0.001, proc_time)

                # --- Performance Profiling ---
                cpu_pct = proc.cpu_percent(interval=None)
                ram_mb = round(proc.memory_info().rss / 1024 / 1024, 1)

                self.latest_stats = {
                    "count": int(total_count),
                    "density": int(density),
                    "status": status,
                    "locations": [],
                    "fps": int(1.0 / ema_frame_time),
                    "latency_ms": int(ema_frame_time * 1000),
                    "cpu_percent": cpu_pct,
                    "ram_mb": ram_mb,
                    # Hybrid pipeline breakdown (exposed for UI / thesis demo)
                    "watershed_count": int(ws_count) if self._area_px_per_person is not None else int(total_count),
                    "area_count":      int(ar_count) if self._area_px_per_person is not None else 0,
                    "manual_clip_active": self.manual_clip_active,
                    "video_label": self._video_label,
                }

                # Phase 2.3: EVENT CLIP RECORDING
                # Use raw count tiers (≤3=LOW, ≤6=MEDIUM, >6=HIGH) so that
                # 4–6 people triggers clips regardless of max_capacity setting.
                if total_count <= 3:
                    clip_level = "LOW"
                elif total_count <= 6:
                    clip_level = "MEDIUM"
                else:
                    clip_level = "HIGH"

                # Always buffer frames for pre-event pre-roll
                self.frame_buffer.append(frame.copy())

                now_t = time.time()

                # Track continuous HIGH streak
                if clip_level == "HIGH":
                    if self._high_since == 0.0:
                        self._high_since = now_t
                else:
                    self._high_since = 0.0

                in_global_cooldown = now_t < self._cooldown_until

                # Manual clip triggered by operator — bypasses all cooldowns
                if self._manual_clip_requested and not self.clip_recording:
                    self._start_event_clip(frame, clip_level, frame_width, frame_height)
                    self._manual_clip_requested = False
                    self.manual_clip_active = True
                    self._last_clip_was_manual = True
                    self._last_clip_density = None   # manual clips don't consume a density slot

                if not self.pause_saving and not in_global_cooldown:
                    # Check per-density 2-minute cooldown independently for MEDIUM and HIGH
                    in_density_cooldown = now_t < self._density_cooldown_until.get(clip_level, 0.0)
                    trigger_normal = (clip_level in ("MEDIUM", "HIGH")
                                      and not self.clip_recording
                                      and not in_density_cooldown)
                    if trigger_normal:
                        self._start_event_clip(frame, clip_level, frame_width, frame_height)
                        self._last_clip_density = clip_level
                        self._last_clip_was_manual = False

                if self.clip_recording:
                    self.clip_writer.write(frame)
                    elapsed = now_t - self.clip_start_time
                    stop_manual = self.manual_clip_active and self._manual_clip_stop_requested
                    stop_auto   = (not self.manual_clip_active and
                                   clip_level == "LOW" and elapsed >= self.clip_min_duration)
                    stop_cap    = elapsed > self.clip_max_duration
                    if stop_manual or stop_auto or stop_cap:
                        self._stop_event_clip()
                        was_manual = self.manual_clip_active
                        self.manual_clip_active = False
                        self._manual_clip_stop_requested = False
                        self._cooldown_until = now_t + 5.0   # 5 s short inter-clip gap

                        # Apply per-density 2-minute cooldown for auto-triggered clips only.
                        # Manual clips do not consume the cooldown so operators can record
                        # at will without delaying automatic detection.
                        if not was_manual and self._last_clip_density in self._density_cooldown_until:
                            self._density_cooldown_until[self._last_clip_density] = (
                                now_t + self.DENSITY_COOLDOWN_SECS
                            )
                            print(
                                f"[{self.stream_id}] {self._last_clip_density} cooldown started — "
                                f"next {self._last_clip_density} clip in "
                                f"{self.DENSITY_COOLDOWN_SECS:.0f}s"
                            )

                self._prev_clip_level = clip_level

                ok, buf = cv2.imencode(".jpg", frame)
                if ok:
                    self.latest_jpeg = buf.tobytes()
                    
                # Real-time synchronization: skip frames if processing took too long
                proc_time_total = time.time() - start_t
                frames_to_skip = int(proc_time_total / 0.033)
                if frames_to_skip > 0:
                    for _ in range(frames_to_skip):
                        cap.grab()

                # Small sleep to yield CPU if processing is too fast
                sleep_t = max(0, 0.033 - proc_time_total)
                time.sleep(sleep_t)

            cap.release()
            # Stop any in-progress clip on video restart
            if self.clip_recording:
                self._stop_event_clip()
            self._cooldown_until = time.time() + 3.0   # suppress clips during MOG2 re-learning
            # Reset per-density cooldowns so a fresh loop isn't blocked by a stale timer
            self._density_cooldown_until = {"MEDIUM": 0.0, "HIGH": 0.0}
            print(f"[SentinelStream {self.stream_id}] Stream ended. Restarting...")
            time.sleep(2.0)

    # ------------------------------------------------------------------
    def _start_event_clip(self, current_frame, level, width, height):
        """Begin recording an event clip, dumping the ring buffer as pre-event footage."""
        now = datetime.now()
        tag = now.strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{self.stream_id}_{tag}_{level}.mp4"
        filepath = os.path.join(self.temp_clips_dir, filename)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.clip_writer = cv2.VideoWriter(filepath, fourcc, 30.0, (width, height))

        # Dump ring buffer (pre-event history)
        for buffered_frame in self.frame_buffer:
            self.clip_writer.write(buffered_frame)

        # Save thumbnail JPEG for the frontend tray
        thumb_filename = filename.replace('.mp4', '.jpg')
        thumb_path = os.path.join(self.temp_clips_dir, thumb_filename)
        cv2.imwrite(thumb_path, current_frame)

        self.clip_recording = True
        self.clip_start_time = time.time()
        self.clip_filename = filename
        print(f"[{self.stream_id}] Event clip started: {filename}")

    def _stop_event_clip(self):
        """Finalize and save the current event clip."""
        if self.clip_writer:
            self.clip_writer.release()
            self.clip_writer = None

            duration = round(time.time() - self.clip_start_time, 1)
            thumb_filename = self.clip_filename.replace('.mp4', '.jpg')
            density_tag = self.clip_filename.rsplit('_', 1)[-1].replace('.mp4', '')

            self.temp_clips.append({
                "filename": self.clip_filename,
                "thumbnail": thumb_filename,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "density": density_tag,
                "camera_id": self.stream_id,
                "video_label": self._video_label,
                "duration": duration
            })

            # Keep only the last 20 clips in metadata
            if len(self.temp_clips) > 20:
                self.temp_clips = self.temp_clips[-20:]

            print(f"[{self.stream_id}] Event clip saved: {self.clip_filename} ({duration}s)")

        self.clip_recording = False
        self.clip_filename = None

    # ------------------------------------------------------------------
    def get_temp_clips(self):
        """Return a copy of the current temp clip metadata list."""
        return list(self.temp_clips)

    def get_latest_jpeg(self):
        return self.latest_jpeg

    def get_latest_stats(self):
        return self.latest_stats