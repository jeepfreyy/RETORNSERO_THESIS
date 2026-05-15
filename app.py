from dotenv import load_dotenv
load_dotenv()  # Load .env before anything reads os.environ

from flask import Flask, render_template, request, jsonify, session, Response, send_file
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
import secrets
import random
import shutil
import os
import time
import cv2
import bleach
import logging

from mailer import send_reset_code

logger = logging.getLogger(__name__)

app = Flask(__name__)

# Vision engine: Sentinel Streams
from vision_engine import SentinelStream

import json as _json_cal

# Load area-based counting calibration if available
_AREA_CAL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ground truths', 'area_calibration.json')
_area_px_per_person = None
_area_baseline = 0.0
if os.path.exists(_AREA_CAL_PATH):
    with open(_AREA_CAL_PATH) as _f:
        _cal = _json_cal.load(_f)
    _r2 = float(_cal.get('r_squared', 0.0))
    if _r2 >= 0.70:
        _area_px_per_person = float(_cal['avg_pixels_per_person'])
        _area_baseline      = float(_cal['area_at_zero'])
        print(f"[Area Cal] {_area_px_per_person:.1f} px/person  baseline={_area_baseline:.1f}px  R²={_r2:.4f}")
    else:
        print(f"[Area Cal] Calibration rejected — R²={_r2:.4f} < 0.70 (non-linear area-count relationship). Using watershed fallback.")
else:
    print("[Area Cal] area_calibration.json not found — using watershed fallback.")

# ---------------------------------------------------------------------------
# VIDEO CATALOG — one entry per switchable video source.
#
# recreate_mog2=False  : keep the main video's learned empty-scene background
#                        model on switch.  Correct for videos that start with
#                        people already present — they immediately appear as
#                        foreground against the carried-over empty background.
# warmup_frames=30     : 1-second suppression window only (not re-learning).
#                        YOLO census at frame 60 sets the correct floor.
# ---------------------------------------------------------------------------
VIDEO_CATALOG = {
    "main_video": {
        "label":          "Main Demonstration",
        "source":         "videos/calibration.MOV",
        "mask_path":      "masks/mask_layer_calibration.png",
        "warmup_frames":  800,
        "recreate_mog2":  False,
        "description":    "Full crowd entry → peak → departure",
    },
    "video1": {
        "label":          "Scenario 1",
        "source":         "videos/VIDEO1.MOV",
        "mask_path":      "masks/mask_layer_calibration.png",
        "warmup_frames":  30,
        "recreate_mog2":  False,
        "description":    "People present from frame 0. Carries over calibration background model.",
    },
    "video2": {
        "label":          "Scenario 2",
        "source":         "videos/VIDEO2.MOV",
        "mask_path":      "masks/mask_layer_calibration.png",
        "warmup_frames":  30,
        "recreate_mog2":  False,
        "description":    "People present from frame 0. Carries over calibration background model.",
    },
}

# --- CAM-01: Main Entrance ---
# IMPORTANT: mog2_history and morph_kernel must match the values in
# area_calibration.json["pipeline_params"] exactly.  Mismatching these
# causes the area measurement to differ from the calibration baseline,
# which shifts the entire count by a fixed offset.
#
# After changing headlight_v_thresh or occupancy parameters, re-run:
#   python calibrate_area.py
# to regenerate area_calibration.json with the corrected px/person slope.
cam1_stream = SentinelStream(
    stream_id="CAM-01",
    source="videos/calibration.MOV",
    mask_path="masks/mask_layer_calibration.png",
    mog2_history=30000,       # must match calibration (was 2000 — wrong)
    mog2_threshold=40,        # must match calibration (was 4 — wrong)
    min_blob_area=350,
    ghost_threshold=90,
    max_capacity=30,
    morph_kernel=(5, 25),     # must match calibration
    h_morph_kernel=(10, 3),   # must match calibration
    dilate_kernel=1,
    process_scale=0.667,
    detect_shadows=True,
    area_px_per_person=_area_px_per_person,
    area_baseline=_area_baseline,
    # Headlight suppression — eliminates motorcycle glare inflation
    # V > 200 in HSV = definitely a headlight in nighttime barangay footage.
    # dilation_px=40 at process_scale=0.667 covers ~60px at full resolution,
    # which kills the halo ring that survived the old pixel-level brightness cut.
    headlight_v_thresh=160,    # lowered from 200 — halo sits at V=160-190
    headlight_dilation_px=80,  # widened from 40 — kills full halo radius
    # Occupancy map — preserves seated people after MOG2 absorbs them.
    # confirm_frames=3: a pixel must appear in fg_mask for 3 consecutive frames
    #   before entering the map. MOG2 absorbs newly-seated people in ~5-8 frames,
    #   so 3 frames is fast enough to catch them. Still blocks 1-2 frame headlight
    #   transients (which are suppressed at V=160 before reaching here).
    # evict_sec=10.0: after 10 seconds of absence from fg_mask, the pixel is
    #   evicted IF its brightness also differs from the value stored at confirmation
    #   time (i.e. the spot now looks different — person has left, background back).
    #   Seated people absorbed by MOG2 keep their V unchanged → NOT evicted.
    #   People who physically leave expose different-brightness background → evicted.
    # dark_v_thresh: kept for API compat, eviction no longer uses raw darkness.
    occupancy_confirm_frames=3,
    occupancy_evict_sec=10.0,
    yolo_model_path="yolov8n.pt",
    yolo_conf=0.40,
    yolo_iou=0.35,
    bg_model_path="mog2/mog2_bg_cam1.yml",  # saved after first warmup; skipped on restart
    warmup_frames=VIDEO_CATALOG["main_video"]["warmup_frames"],
)

# CAM-02 is temporarily disabled while CAM-01 tuning targets MAE < 1.
# Re-enable by restoring the SentinelStream instantiation here.

# CONFIGURATION
# Secret key for session management (Keep this secret in production!)
app.secret_key = os.environ.get('SECRET_KEY', 'dev_fallback_secret_key_12345')
if not app.secret_key:
    raise RuntimeError("SECRET_KEY environment variable is required")

# Database configuration
# Use PostgreSQL if DATABASE_URL is set; otherwise fall back to SQLite for local dev.
# Example (Postgres): postgresql+psycopg://user:password@localhost:5432/sentinel_db
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get(
    'DATABASE_URL',
    'sqlite:///sentinel_users.db'
)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# --- DATABASE MODEL ---
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)  # Hash stored for security
    role = db.Column(db.String(20), default='Tanod')


class IncidentArchive(db.Model):
    """Permanent archive for incidents saved by Tanods with annotated reports."""
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    camera_id = db.Column(db.String(20), nullable=False)
    title = db.Column(db.String(200), nullable=True)
    report_html = db.Column(db.Text, nullable=True)
    status = db.Column(db.String(20), nullable=False)      # VALID_THREAT or FALSE_ALARM
    density_tag = db.Column(db.String(10), nullable=False)  # LOW, MEDIUM, HIGH, MANUAL
    threat_level = db.Column(db.Integer, nullable=True, default=5)  # 1–10 scale
    clip_filename = db.Column(db.String(200), nullable=False)
    clip_path = db.Column(db.String(500), nullable=False)
    duration = db.Column(db.Float, nullable=True)
    # Response-center fields
    incident_status = db.Column(db.String(20), default='OPEN')  # OPEN/RESPONDING/RESOLVED/CLOSED
    reporter_name = db.Column(db.String(100), nullable=True)
    resolution_note = db.Column(db.Text, nullable=True)
    people_count = db.Column(db.Integer, nullable=True, default=0)
    location = db.Column(db.String(200), nullable=True)
    responders = db.relationship('IncidentResponder', backref='incident',
                                 lazy=True, cascade='all, delete-orphan')
    status_logs = db.relationship('IncidentStatusLog', backref='incident',
                                  lazy=True, cascade='all, delete-orphan',
                                  order_by='IncidentStatusLog.changed_at')


class IncidentResponder(db.Model):
    """Tracks barangay officials assigned to respond to a specific incident."""
    __tablename__ = 'incident_responder'
    id = db.Column(db.Integer, primary_key=True)
    incident_id = db.Column(db.Integer, db.ForeignKey('incident_archive.id'), nullable=False)
    responder_name = db.Column(db.String(100), nullable=False)
    responder_role = db.Column(db.String(50), default='Barangay Tanod')
    assigned_at = db.Column(db.DateTime, default=datetime.utcnow)
    note = db.Column(db.String(300), nullable=True)


class IncidentStatusLog(db.Model):
    """Immutable audit trail of every status transition for an incident."""
    __tablename__ = 'incident_status_log'
    id = db.Column(db.Integer, primary_key=True)
    incident_id = db.Column(db.Integer, db.ForeignKey('incident_archive.id'), nullable=False)
    status     = db.Column(db.String(30), nullable=False)   # PUBLISHED / OPEN / RESPONDING / RESOLVED / CLOSED
    changed_at = db.Column(db.DateTime, default=datetime.utcnow)
    changed_by = db.Column(db.String(100), nullable=True)


class PasswordResetToken(db.Model):
    """Stores hashed OTP codes for the password-recovery flow."""
    __tablename__ = "password_reset_tokens"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    code_hash = db.Column(db.String(256), nullable=False)
    expires_at = db.Column(db.DateTime, nullable=False)
    attempts = db.Column(db.Integer, default=0)
    used_at = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    requester_ip = db.Column(db.String(64), nullable=True)


# --- PASSWORD RESET CONFIGURATION ---
PASSWORD_RESET_TTL = int(os.environ.get('PASSWORD_RESET_TTL_MINUTES', '10'))
PASSWORD_RESET_MAX_ATTEMPTS = int(os.environ.get('PASSWORD_RESET_MAX_ATTEMPTS', '5'))

# itsdangerous serializer for reset tickets
_ticket_serializer = URLSafeTimedSerializer(app.secret_key, salt='pw-reset-ticket')

# Simple in-memory rate-limiter: {email: [timestamps]}
_reset_request_log: dict[str, list[float]] = {}

# --- ROUTES ---

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'success': False, 'message': 'Unauthorized. Please log in.'}), 401
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def home():
    """Serves the frontend HTML."""
    return render_template('index.html')

@app.route('/api/register', methods=['POST'])
def register():
    """Handles user registration."""
    data = request.json
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')

    if not username or not email or not password:
        return jsonify({'success': False, 'message': 'All fields are required.'}), 400

    # Check if user already exists
    if User.query.filter_by(username=username).first():
        return jsonify({'success': False, 'message': 'Username already exists.'}), 409
    
    if User.query.filter_by(email=email).first():
        return jsonify({'success': False, 'message': 'Email already registered.'}), 409

    # Create new user WITH HASHED PASSWORD
    hashed_pw = generate_password_hash(password)
    new_user = User(username=username, email=email, password_hash=hashed_pw)
    
    try:
        db.session.add(new_user)
        db.session.commit()
        return jsonify({'success': True, 'message': 'Registration successful!'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/login', methods=['POST'])
def login():
    """Handles user login."""
    data = request.json
    email = data.get('email')
    password_attempt = data.get('password')

    user = User.query.filter_by(email=email).first()

    if user and check_password_hash(user.password_hash, password_attempt):
        session['user_id'] = user.id
        session['username'] = user.username
        return jsonify({'success': True, 'message': 'Login successful!', 'username': user.username})
    
    return jsonify({'success': False, 'message': 'Invalid email or password.'}), 401

@app.route('/api/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'success': True})



# ---------------------------------------------------------------------------
# PASSWORD RECOVERY ENDPOINTS
# ---------------------------------------------------------------------------

def _rate_limit_ok(email: str, max_requests: int = 3, window_seconds: int = 900) -> bool:
    """Return True if *email* hasn't exceeded the rate limit."""
    now = time.time()
    timestamps = _reset_request_log.get(email, [])
    timestamps = [t for t in timestamps if now - t < window_seconds]
    _reset_request_log[email] = timestamps
    return len(timestamps) < max_requests


@app.route('/api/forgot/request', methods=['POST'])
def forgot_request():
    """Step 1 — Accept an email; if the account exists, send a 6-digit OTP.

    Always returns 200 with a generic message to prevent email enumeration.
    """
    data = request.json or {}
    email = (data.get('email') or '').strip().lower()
    generic_msg = 'If an account exists for that email, a verification code has been sent.'

    if not email:
        return jsonify({'success': False, 'message': 'Email is required.'}), 400

    user = User.query.filter_by(email=email).first()

    if not user:
        # Flat timing — sleep a random interval so response time matches the happy path
        time.sleep(random.uniform(0.2, 0.5))
        return jsonify({'success': True, 'message': generic_msg})

    # Per-email rate limit: 3 requests per 15 minutes
    if not _rate_limit_ok(email):
        return jsonify({'success': True, 'message': generic_msg})  # silent

    _reset_request_log.setdefault(email, []).append(time.time())

    # Invalidate any previous active tokens for this user
    PasswordResetToken.query.filter_by(user_id=user.id, used_at=None).update(
        {'used_at': datetime.utcnow()}
    )

    # Generate a 6-digit code and hash it
    code = str(secrets.randbelow(1_000_000)).zfill(6)
    token = PasswordResetToken(
        user_id=user.id,
        code_hash=generate_password_hash(code),
        expires_at=datetime.utcnow() + timedelta(minutes=PASSWORD_RESET_TTL),
        requester_ip=request.remote_addr,
    )
    db.session.add(token)
    db.session.commit()

    # Send the email (errors are caught so the user always sees the generic message)
    try:
        send_reset_code(recipient=email, code=code)
    except Exception:
        logger.exception('Failed to send password-reset email')

    return jsonify({'success': True, 'message': generic_msg})


@app.route('/api/forgot/verify', methods=['POST'])
def forgot_verify():
    """Step 2 — Verify the 6-digit OTP and return a signed reset ticket."""
    data = request.json or {}
    email = (data.get('email') or '').strip().lower()
    code = (data.get('code') or '').strip()

    if not email or not code:
        return jsonify({'success': False, 'message': 'Email and code are required.'}), 400

    user = User.query.filter_by(email=email).first()
    if not user:
        return jsonify({'success': False, 'message': 'Invalid or expired verification code.'}), 400

    token = (
        PasswordResetToken.query
        .filter_by(user_id=user.id, used_at=None)
        .filter(PasswordResetToken.expires_at > datetime.utcnow())
        .filter(PasswordResetToken.attempts < PASSWORD_RESET_MAX_ATTEMPTS)
        .order_by(PasswordResetToken.created_at.desc())
        .first()
    )

    if not token:
        return jsonify({'success': False, 'message': 'Invalid or expired verification code.'}), 400

    if not check_password_hash(token.code_hash, code):
        token.attempts += 1
        db.session.commit()
        remaining = PASSWORD_RESET_MAX_ATTEMPTS - token.attempts
        if remaining <= 0:
            return jsonify({'success': False, 'message': 'Too many failed attempts. Please request a new code.'}), 400
        return jsonify({'success': False, 'message': f'Invalid code. {remaining} attempt(s) remaining.'}), 400

    # Code is correct — issue a signed ticket
    ticket = _ticket_serializer.dumps({'tid': token.id, 'uid': user.id})
    return jsonify({'success': True, 'reset_ticket': ticket})


@app.route('/api/forgot/reset', methods=['POST'])
def forgot_reset():
    """Step 3 — Accept the signed ticket + new password and update the hash."""
    data = request.json or {}
    ticket = data.get('reset_ticket', '')
    new_password = data.get('new_password', '')

    # --- validate password strength server-side ---
    if len(new_password) < 8:
        return jsonify({'success': False, 'message': 'Password must be at least 8 characters.'}), 400
    if not any(c.isupper() for c in new_password):
        return jsonify({'success': False, 'message': 'Password must contain at least one uppercase letter.'}), 400
    if not any(c.isdigit() or not c.isalnum() for c in new_password):
        return jsonify({'success': False, 'message': 'Password must contain at least one digit or symbol.'}), 400

    # --- unsign ticket (10-minute TTL) ---
    try:
        payload = _ticket_serializer.loads(ticket, max_age=PASSWORD_RESET_TTL * 60)
    except (BadSignature, SignatureExpired):
        return jsonify({'success': False, 'message': 'Reset session expired or invalid. Please start over.'}), 400

    tid = payload.get('tid')
    uid = payload.get('uid')

    token = PasswordResetToken.query.get(tid)
    if not token or token.user_id != uid or token.used_at is not None:
        return jsonify({'success': False, 'message': 'Reset session expired or invalid. Please start over.'}), 400

    if token.expires_at < datetime.utcnow():
        return jsonify({'success': False, 'message': 'Reset session expired. Please start over.'}), 400

    user = User.query.get(uid)
    if not user:
        return jsonify({'success': False, 'message': 'Account not found.'}), 400

    # Update password
    user.password_hash = generate_password_hash(new_password)

    # Mark this token (and any siblings) as used
    PasswordResetToken.query.filter_by(user_id=uid, used_at=None).update(
        {'used_at': datetime.utcnow()}
    )
    db.session.commit()

    # Clear any active session (don't auto-login)
    session.clear()

    return jsonify({'success': True, 'message': 'Your password has been reset successfully.'})


@app.route('/video_feed')
@login_required
def video_feed():
    """
    MJPEG stream for CAM-01.
    """
    def generate():
        while True:
            jpeg = cam1_stream.get_latest_jpeg()
            if jpeg is None:
                time.sleep(0.05)
                continue
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                + jpeg
                + b"\r\n"
            )
            time.sleep(0.033)

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store"},
    )


@app.route('/cam1_frame')
@login_required
def cam1_frame():
    """
    Single JPEG frame for CAM-01.
    """
    jpeg = cam1_stream.get_latest_jpeg()
    if jpeg is None:
        return Response(status=204)

    return Response(
        jpeg,
        mimetype="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )

@app.route('/api/stats')
@login_required
def api_stats():
    """
    JSON stats for CAM-01 (people count, density, status).
    Includes is_warming_up flag so the UI can show a warmup overlay.
    """
    stats = cam1_stream.get_latest_stats()
    stats['is_warming_up'] = cam1_stream.is_warming_up
    return jsonify(stats)


@app.route('/api/videos')
@login_required
def api_videos():
    """Return the video catalog for the frontend dropdown selector."""
    videos = []
    for key, v in VIDEO_CATALOG.items():
        abs_source = os.path.join(os.path.dirname(os.path.abspath(__file__)), v['source'])
        videos.append({
            'key':         key,
            'label':       v['label'],
            'description': v['description'],
            'active':      cam1_stream.source == v['source'],
            'available':   os.path.exists(abs_source),
        })
    return jsonify(videos)


@app.route('/api/switch_video', methods=['POST'])
@login_required
def api_switch_video():
    """Switch CAM-01 to a different video from the catalog."""
    data      = request.json or {}
    video_key = data.get('video_key', '').strip()

    if video_key not in VIDEO_CATALOG:
        return jsonify({'success': False, 'message': f'Unknown video: {video_key}'}), 400

    v             = VIDEO_CATALOG[video_key]
    source        = v['source']
    mask_path     = v.get('mask_path')
    warmup        = v['warmup_frames']
    recreate_mog2 = v.get('recreate_mog2', False)

    abs_source = os.path.join(os.path.dirname(os.path.abspath(__file__)), source)
    if not os.path.exists(abs_source):
        return jsonify({'success': False, 'message': f'Video file not found: {source}'}), 404

    if mask_path:
        abs_mask = os.path.join(os.path.dirname(os.path.abspath(__file__)), mask_path)
        if not os.path.exists(abs_mask):
            return jsonify({'success': False, 'message': f'Mask file not found: {mask_path}'}), 404

    cam1_stream.switch_source(
        source, warmup,
        recreate_mog2=recreate_mog2,
        new_mask_path=mask_path,
    )

    return jsonify({
        'success':        True,
        'message':        f'Switching to {v["label"]}',
        'warmup_frames':  warmup,
        'warmup_seconds': round(warmup / 30, 1),
        'label':          v['label'],
        'recreate_mog2':  recreate_mog2,
    })

# CAM-02 routes are disabled while the camera is offline for calibration.


@app.route('/api/system/health')
@login_required
def api_system_health():
    """Live CPU & RAM performance metrics for the whole process."""
    import psutil
    proc = psutil.Process(os.getpid())
    mem = proc.memory_info()
    return jsonify({
        'cpu_percent': psutil.cpu_percent(interval=0.1),
        'ram_used_mb': round(mem.rss / 1024 / 1024, 1),
        'ram_total_mb': round(psutil.virtual_memory().total / 1024 / 1024, 1),
        'ram_percent': psutil.virtual_memory().percent,
        'cam1': {
            'fps': cam1_stream.get_latest_stats().get('fps', 0),
            'latency_ms': cam1_stream.get_latest_stats().get('latency_ms', 0),
        },
        'cam2': {'fps': 0, 'latency_ms': 0},  # offline
    })


# ---------------------------------------------------------------------------
# TEMP CLIP & INCIDENT ARCHIVE ENDPOINTS
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_CLIPS_DIR = os.path.join(BASE_DIR, "Temp_Clips")
ARCHIVE_DIR = os.path.join(BASE_DIR, "Archive")
os.makedirs(TEMP_CLIPS_DIR, exist_ok=True)
os.makedirs(ARCHIVE_DIR, exist_ok=True)

# Track server start time for uptime calculation
_SERVER_START_TIME = time.time()


@app.route('/api/dashboard_stats')
@login_required
def api_dashboard_stats():
    """
    Aggregated stats for the four top-bar status cards:
      cameras_online / cameras_total, alerts_today,
      storage_percent, uptime_seconds.
    """
    import psutil
    from datetime import date

    # ── Cameras online ────────────────────────────────────────────────────
    cam1_fps   = cam1_stream.get_latest_stats().get('fps', 0)
    cam1_online = 1 if cam1_stream.running and cam1_fps > 0 else 0
    cam2_online = 0   # CAM-02 is a placeholder — offline

    # ── Alerts today ─────────────────────────────────────────────────────
    today_start = datetime.combine(date.today(), datetime.min.time())
    archived_today = IncidentArchive.query.filter(
        IncidentArchive.timestamp >= today_start
    ).count()
    temp_today = len(cam1_stream.temp_clips)   # unsubmitted clips in tray
    alerts_today = archived_today + temp_today

    # ── Storage used ─────────────────────────────────────────────────────
    try:
        du = psutil.disk_usage(BASE_DIR)
        storage_percent = round(du.percent, 1)
        storage_used_gb = round(du.used / (1024 ** 3), 1)
        storage_total_gb = round(du.total / (1024 ** 3), 1)
    except Exception:
        storage_percent = 0
        storage_used_gb = 0
        storage_total_gb = 0

    # ── Uptime ───────────────────────────────────────────────────────────
    uptime_seconds = int(time.time() - _SERVER_START_TIME)

    return jsonify({
        'cameras_online':  cam1_online + cam2_online,
        'cameras_total':   2,
        'alerts_today':    alerts_today,
        'storage_percent': storage_percent,
        'storage_used_gb': storage_used_gb,
        'storage_total_gb': storage_total_gb,
        'uptime_seconds':  uptime_seconds,
    })


@app.route('/api/manual_clip/start', methods=['POST'])
@login_required
def start_manual_clip():
    """Signal the CV thread to begin a user-triggered manual clip immediately."""
    if cam1_stream.manual_clip_active or cam1_stream.clip_recording:
        return jsonify({'ok': False, 'reason': 'already_recording'}), 409
    cam1_stream._manual_clip_requested = True
    return jsonify({'ok': True})


@app.route('/api/manual_clip/stop', methods=['POST'])
@login_required
def stop_manual_clip():
    """Signal the CV thread to finalize the current manual clip."""
    if not cam1_stream.manual_clip_active:
        return jsonify({'ok': False, 'reason': 'not_recording'}), 409
    cam1_stream._manual_clip_stop_requested = True
    return jsonify({'ok': True})


@app.route('/api/temp_clips')
@login_required
def api_temp_clips():
    """Returns metadata for all temporary event clips from CAM-01."""
    clips = cam1_stream.get_temp_clips()
    return jsonify(clips)


@app.route('/api/temp_clips/<filename>', methods=['DELETE'])
@login_required
def dismiss_clip(filename):
    """Dismiss (delete) a temp clip."""
    filename = secure_filename(filename)
    cam1_stream.temp_clips = [c for c in cam1_stream.temp_clips if c['filename'] != filename]
    clip_path = os.path.join(TEMP_CLIPS_DIR, filename)
    if os.path.exists(clip_path):
        os.remove(clip_path)
    thumb_path = clip_path.replace('.mp4', '.jpg')
    if os.path.exists(thumb_path):
        os.remove(thumb_path)
    return jsonify({'success': True})


@app.route('/clips/<filename>')
@login_required
def serve_clip(filename):
    """Serve a temp clip video or thumbnail from Temp_Clips/."""
    filename = secure_filename(filename)
    clip_path = os.path.join(TEMP_CLIPS_DIR, filename)
    if os.path.exists(clip_path):
        mimetype = 'video/mp4' if filename.endswith('.mp4') else 'image/jpeg'
        return send_file(clip_path, mimetype=mimetype)
    return Response(status=404)


def _log_status(incident_id, status, changed_by=None):
    """Append an immutable status-log entry for an incident."""
    db.session.add(IncidentStatusLog(
        incident_id=incident_id,
        status=status,
        changed_by=changed_by,
    ))


@app.route('/api/incidents', methods=['GET'])
@login_required
def list_incidents():
    """List all permanently archived incident reports."""
    incidents = IncidentArchive.query.order_by(IncidentArchive.timestamp.desc()).all()
    result = []
    for i in incidents:
        # Build thumbnail URL from clip_path
        try:
            rel = os.path.relpath(i.clip_path, ARCHIVE_DIR).replace('\\', '/')
            thumb_rel = rel.replace('.mp4', '.jpg')
            thumbnail_url = f'/archive_media/{thumb_rel}'
        except Exception:
            thumbnail_url = None

        result.append({
            'id': i.id,
            'timestamp': i.timestamp.isoformat(),
            'camera_id': i.camera_id,
            'title': i.title,
            'report_html': i.report_html,
            'status': i.status,
            'density_tag': i.density_tag,
            'threat_level': i.threat_level,
            'clip_filename': i.clip_filename,
            'duration': i.duration,
            'incident_status': i.incident_status or 'OPEN',
            'reporter_name': i.reporter_name,
            'resolution_note': i.resolution_note,
            'people_count': i.people_count or 0,
            'location': i.location,
            'thumbnail_url': thumbnail_url,
            'responders': [{
                'id': r.id,
                'responder_name': r.responder_name,
                'responder_role': r.responder_role,
                'assigned_at': r.assigned_at.isoformat(),
                'note': r.note,
            } for r in i.responders],
            'status_logs': [{
                'status': log.status,
                'changed_at': log.changed_at.isoformat(),
                'changed_by': log.changed_by,
            } for log in i.status_logs],
        })
    return jsonify(result)


@app.route('/api/incidents', methods=['POST'])
@login_required
def save_incident():
    """Save an incident: copies clip from Temp_Clips/ to Archive/ and writes DB record."""
    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'message': 'No data received.'}), 400

        clip_filename = data.get('clip_filename')
        if not clip_filename:
            return jsonify({'success': False, 'message': 'clip_filename is required.'}), 400

        clip_filename = secure_filename(clip_filename)

        ALLOWED_TAGS = ['b', 'i', 'u', 'strong', 'em', 'p', 'br', 'ul', 'ol', 'li', 'h1', 'h2', 'h3', 'a', 'span']
        ALLOWED_ATTRS = {'a': ['href', 'title', 'target'], 'span': ['class']}
        raw_html = data.get('report_html', '')
        report_html = bleach.clean(raw_html, tags=ALLOWED_TAGS, attributes=ALLOWED_ATTRS, strip=True)

        status       = data.get('status', 'FALSE_ALARM')
        density_tag  = data.get('density_tag', 'LOW')
        camera_id    = data.get('camera_id', 'CAM-01')
        duration     = data.get('duration', 0)
        title        = (data.get('title') or '').strip() or None
        threat_level = int(data.get('threat_level') or 5)
        people_count = int(data.get('people_count') or 0)
        location     = (data.get('location') or '').strip() or None
        reporter_name = session.get('username') or data.get('reporter_name') or 'Unknown'

        # Create date-stamped archive subdirectory
        today = datetime.now().strftime("%Y-%m-%d")
        archive_subdir = os.path.join(ARCHIVE_DIR, f"Archive_{today}")
        os.makedirs(archive_subdir, exist_ok=True)

        # Copy clip from Temp → Archive (copy2 avoids WinError 32 file-lock issue)
        temp_path    = os.path.join(TEMP_CLIPS_DIR, clip_filename)
        archive_path = os.path.join(archive_subdir, clip_filename)

        if os.path.exists(temp_path):
            shutil.copy2(temp_path, archive_path)
            # Retry delete up to 3× in case cv2 still holds a brief write-lock
            for _ in range(3):
                try:
                    os.remove(temp_path)
                    break
                except OSError:
                    time.sleep(0.3)

            # Copy thumbnail too
            thumb_name = clip_filename.replace('.mp4', '.jpg')
            temp_thumb = os.path.join(TEMP_CLIPS_DIR, thumb_name)
            if os.path.exists(temp_thumb):
                shutil.copy2(temp_thumb, os.path.join(archive_subdir, thumb_name))
                try:
                    os.remove(temp_thumb)
                except OSError:
                    pass
        else:
            archive_path = os.path.join(archive_subdir, clip_filename)  # best-effort

        # Remove from in-memory temp clips list
        cam1_stream.temp_clips = [c for c in cam1_stream.temp_clips if c['filename'] != clip_filename]

        # Persist to database
        incident = IncidentArchive(
            camera_id     = camera_id,
            title         = title,
            report_html   = report_html,
            status        = status,
            density_tag   = density_tag,
            threat_level  = threat_level,
            clip_filename = clip_filename,
            clip_path     = archive_path,
            duration      = duration,
            incident_status = 'OPEN',
            reporter_name = reporter_name,
            people_count  = people_count,
            location      = location,
        )
        db.session.add(incident)
        db.session.flush()   # get incident.id before committing

        # Seed status log: PUBLISHED + OPEN (same moment, same actor)
        _log_status(incident.id, 'PUBLISHED', reporter_name)
        _log_status(incident.id, 'OPEN',      reporter_name)
        db.session.commit()

        return jsonify({'success': True, 'message': 'Incident archived successfully', 'id': incident.id})

    except Exception as e:
        db.session.rollback()
        logger.exception('save_incident failed')
        return jsonify({'success': False, 'message': f'Server error: {str(e)}'}), 500


@app.route('/archive_media/<path:filepath>')
@login_required
def serve_archive_media(filepath):
    """Serve a permanently archived clip or thumbnail."""
    if '..' in filepath or filepath.startswith('/'):
        return Response(status=403)
    full_path = os.path.join(ARCHIVE_DIR, filepath)
    if os.path.exists(full_path):
        mimetype = 'video/mp4' if filepath.endswith('.mp4') else 'image/jpeg'
        return send_file(full_path, mimetype=mimetype)
    return Response(status=404)


# ── Responder management ─────────────────────────────────────────────────────

@app.route('/api/incidents/<int:incident_id>/respond', methods=['POST'])
@login_required
def add_responder(incident_id):
    """Assign a responder to an incident."""
    incident = IncidentArchive.query.get_or_404(incident_id)
    data = request.json or {}
    name = (data.get('responder_name') or '').strip()
    if not name:
        return jsonify({'error': 'responder_name is required'}), 400
    role = (data.get('responder_role') or 'Barangay Tanod').strip()
    note = (data.get('note') or '').strip() or None

    responder = IncidentResponder(
        incident_id    = incident_id,
        responder_name = name,
        responder_role = role,
        note           = note,
    )
    db.session.add(responder)

    # Auto-advance OPEN → RESPONDING on first assignment
    if incident.incident_status == 'OPEN':
        incident.incident_status = 'RESPONDING'
        _log_status(incident_id, 'RESPONDING', session.get('username'))

    db.session.commit()
    return jsonify({
        'success': True,
        'responder': {
            'id': responder.id,
            'responder_name': responder.responder_name,
            'responder_role': responder.responder_role,
            'assigned_at': responder.assigned_at.isoformat(),
            'note': responder.note,
        }
    })


@app.route('/api/incidents/<int:incident_id>/responders/<int:rid>', methods=['DELETE'])
@login_required
def remove_responder(incident_id, rid):
    """Remove a responder from an incident."""
    responder = IncidentResponder.query.filter_by(id=rid, incident_id=incident_id).first_or_404()
    db.session.delete(responder)

    # Revert to OPEN if no responders remain
    incident = IncidentArchive.query.get_or_404(incident_id)
    remaining = IncidentResponder.query.filter_by(incident_id=incident_id).count()
    if remaining == 0 and incident.incident_status == 'RESPONDING':
        incident.incident_status = 'OPEN'

    db.session.commit()
    return jsonify({'success': True})


@app.route('/api/incidents/<int:incident_id>/status', methods=['PATCH'])
@login_required
def update_incident_status(incident_id):
    """Update incident_status and optional resolution_note."""
    incident = IncidentArchive.query.get_or_404(incident_id)

    data = request.json or {}
    new_status = (data.get('status') or '').strip().upper()
    valid_statuses = {'OPEN', 'RESPONDING', 'RESOLVED', 'CLOSED'}
    if new_status not in valid_statuses:
        return jsonify({'error': f'Invalid status. Must be one of: {", ".join(valid_statuses)}'}), 400

    old_status = incident.incident_status
    incident.incident_status = new_status
    note = (data.get('resolution_note') or '').strip()
    if note:
        incident.resolution_note = note

    # Only log if status actually changed
    if old_status != new_status:
        _log_status(incident_id, new_status, session.get('username'))

    db.session.commit()
    return jsonify({'success': True, 'incident_status': incident.incident_status})


# ── Permanent deletion ───────────────────────────────────────────────────────

@app.route('/api/incidents/<int:incident_id>', methods=['DELETE'])
@login_required
def delete_incident(incident_id):
    """Permanently delete an incident report, its clip, and its thumbnail."""
    try:
        incident = IncidentArchive.query.get_or_404(incident_id)
        clip_path = incident.clip_path

        # Delete DB record (cascade removes responders too)
        db.session.delete(incident)
        db.session.commit()

        # Delete clip file from disk
        if clip_path and os.path.exists(clip_path):
            try:
                os.remove(clip_path)
            except OSError:
                pass
            # Delete thumbnail too
            thumb_path = clip_path.replace('.mp4', '.jpg')
            if os.path.exists(thumb_path):
                try:
                    os.remove(thumb_path)
                except OSError:
                    pass

        return jsonify({'success': True, 'message': 'Incident permanently deleted.'})

    except Exception as e:
        db.session.rollback()
        logger.exception('delete_incident failed')
        return jsonify({'success': False, 'message': f'Server error: {str(e)}'}), 500


@app.route('/api/users/status')
@login_required
def api_users_status():
    """Return all registered users with their current availability status.

    Availability:
      - "Responding" = user's username appears as a responder_name in any
        incident whose incident_status is OPEN or RESPONDING.
      - "Available"  = everyone else.
    """
    # Collect names that are actively responding
    active_responder_names = set()
    active_incidents = IncidentArchive.query.filter(
        IncidentArchive.incident_status.in_(['OPEN', 'RESPONDING'])
    ).all()
    for inc in active_incidents:
        for r in inc.responders:
            active_responder_names.add(r.responder_name.strip().lower())

    users = User.query.order_by(User.username).all()
    result = []
    for u in users:
        availability = 'Responding' if u.username.strip().lower() in active_responder_names else 'Available'
        result.append({
            'id':           u.id,
            'username':     u.username,
            'role':         u.role or 'Tanod',
            'availability': availability,
        })
    return jsonify(result)


def _mjpeg_from_file(video_path):
    """Generator: reads a video file frame-by-frame and yields MJPEG frames."""
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # Loop the clip
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break
        ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        time.sleep(0.033)
    cap.release()


@app.route('/clip_stream/<filename>')
@login_required
def clip_stream(filename):
    """Stream a temp clip as MJPEG so browsers can display it."""
    filename = secure_filename(filename)
    filepath = os.path.join(TEMP_CLIPS_DIR, filename)
    if not os.path.exists(filepath):
        return Response(status=404)
    return Response(_mjpeg_from_file(filepath),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/archive_stream/<path:filepath>')
@login_required
def archive_stream(filepath):
    """Stream an archived clip as MJPEG so browsers can display it."""
    if '..' in filepath or filepath.startswith('/'):
        return Response(status=403)
    full_path = os.path.join(ARCHIVE_DIR, filepath)
    if not os.path.exists(full_path):
        return Response(status=404)
    return Response(_mjpeg_from_file(full_path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ── DB initialisation (runs whether started with `python app.py` or `flask run`) ──
def _migrate_schema():
    """Safely add any columns that may not exist yet (idempotent)."""
    from sqlalchemy import text
    cols = {
        'incident_archive': [
            ("incident_status", "VARCHAR(20) DEFAULT 'OPEN'"),
            ("reporter_name",   "VARCHAR(100)"),
            ("resolution_note", "TEXT"),
            ("people_count",    "INTEGER DEFAULT 0"),
            ("location",        "VARCHAR(200)"),
            ("title",           "VARCHAR(200)"),
            ("threat_level",    "INTEGER DEFAULT 5"),
        ]
    }
    with db.engine.connect() as conn:
        for table, columns in cols.items():
            for col_name, col_def in columns:
                try:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_def}"))
                    conn.commit()
                except Exception:
                    conn.rollback()  # Column already exists — skip silently


with app.app_context():
    db.create_all()
    _migrate_schema()


if __name__ == '__main__':
    is_debug = os.environ.get('FLASK_DEBUG', 'False').lower() in ['true', '1', 't']
    app.run(debug=is_debug, port=5001)