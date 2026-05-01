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

# --- CAM-01: Main Entrance (NIGHTTIME OPTIMIZED - MAE: 2.02) ---
cam1_stream = SentinelStream(
    stream_id="CAM-01",
    source="videos/vid1-angle1.MOV",
    mask_path="mask_layer1.png",
    mog2_history=2000,
    mog2_threshold=4,
    min_blob_area=450,
    ghost_threshold=30,
    max_capacity=30,
    morph_kernel=(5, 35),
    dilate_kernel=1,
    process_scale=0.667,
    detect_shadows=False
)

# --- CAM-02: Secondary Entrance (NIGHTTIME OPTIMIZED - MAE: 2.02) ---
cam2_stream = SentinelStream(
    stream_id="CAM-02",
    source="videos/vid2-angle2.MOV",
    mask_path="mask_layer1.png",
    mog2_history=2000,
    mog2_threshold=4,
    min_blob_area=450,
    ghost_threshold=30,
    max_capacity=30,
    morph_kernel=(5, 35),
    dilate_kernel=1,
    process_scale=0.667,
    detect_shadows=False
)

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
    report_html = db.Column(db.Text, nullable=True)
    status = db.Column(db.String(20), nullable=False)      # VALID_THREAT or FALSE_ALARM
    density_tag = db.Column(db.String(10), nullable=False)  # LOW, MEDIUM, HIGH
    clip_filename = db.Column(db.String(200), nullable=False)
    clip_path = db.Column(db.String(500), nullable=False)
    duration = db.Column(db.Float, nullable=True)


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
    """
    return jsonify(cam1_stream.get_latest_stats())

# --- CAM-02 ENDPOINTS ---
@app.route('/cam2_frame')
@login_required
def cam2_frame():
    """Single JPEG frame for CAM-02."""
    jpeg = cam2_stream.get_latest_jpeg()
    if jpeg is None:
        return Response(status=204)
    return Response(jpeg, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})

@app.route('/api/stats/cam2')
@login_required
def api_stats_cam2():
    """JSON stats for CAM-02."""
    return jsonify(cam2_stream.get_latest_stats())


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
        'cam2': {
            'fps': cam2_stream.get_latest_stats().get('fps', 0),
            'latency_ms': cam2_stream.get_latest_stats().get('latency_ms', 0),
        },
    })


# ---------------------------------------------------------------------------
# TEMP CLIP & INCIDENT ARCHIVE ENDPOINTS
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_CLIPS_DIR = os.path.join(BASE_DIR, "Temp_Clips")
ARCHIVE_DIR = os.path.join(BASE_DIR, "Archive")
os.makedirs(TEMP_CLIPS_DIR, exist_ok=True)
os.makedirs(ARCHIVE_DIR, exist_ok=True)


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


@app.route('/api/incidents', methods=['GET'])
@login_required
def list_incidents():
    """List all permanently archived incident reports."""
    incidents = IncidentArchive.query.order_by(IncidentArchive.timestamp.desc()).all()
    return jsonify([{
        'id': i.id,
        'timestamp': i.timestamp.isoformat(),
        'camera_id': i.camera_id,
        'report_html': i.report_html,
        'status': i.status,
        'density_tag': i.density_tag,
        'clip_filename': i.clip_filename,
        'duration': i.duration
    } for i in incidents])


@app.route('/api/incidents', methods=['POST'])
@login_required
def save_incident():
    """Save an incident: moves clip from Temp_Clips/ to Archive/ and writes DB record."""
    data = request.json
    clip_filename = data.get('clip_filename')
    raw_html = data.get('report_html', '')
    
    ALLOWED_TAGS = ['b', 'i', 'u', 'strong', 'em', 'p', 'br', 'ul', 'ol', 'li', 'h1', 'h2', 'h3', 'a', 'span']
    ALLOWED_ATTRS = {'a': ['href', 'title', 'target'], 'span': ['style', 'class']}
    report_html = bleach.clean(raw_html, tags=ALLOWED_TAGS, attributes=ALLOWED_ATTRS, strip=True)
    status = data.get('status', 'FALSE_ALARM')
    density_tag = data.get('density_tag', 'LOW')
    camera_id = data.get('camera_id', 'CAM-01')
    duration = data.get('duration', 0)

    # Create date-stamped archive subdirectory
    today = datetime.now().strftime("%Y-%m-%d")
    archive_subdir = os.path.join(ARCHIVE_DIR, f"Archive_{today}")
    os.makedirs(archive_subdir, exist_ok=True)

    # Move clip video from Temp → Archive
    temp_path = os.path.join(TEMP_CLIPS_DIR, clip_filename)
    archive_path = os.path.join(archive_subdir, clip_filename)
    if os.path.exists(temp_path):
        shutil.move(temp_path, archive_path)
        thumb_name = clip_filename.replace('.mp4', '.jpg')
        temp_thumb = os.path.join(TEMP_CLIPS_DIR, thumb_name)
        if os.path.exists(temp_thumb):
            shutil.move(temp_thumb, os.path.join(archive_subdir, thumb_name))

    # Remove from in-memory temp clips list
    cam1_stream.temp_clips = [c for c in cam1_stream.temp_clips if c['filename'] != clip_filename]

    # Persist to database
    incident = IncidentArchive(
        camera_id=camera_id,
        report_html=report_html,
        status=status,
        density_tag=density_tag,
        clip_filename=clip_filename,
        clip_path=archive_path,
        duration=duration
    )
    db.session.add(incident)
    db.session.commit()

    return jsonify({'success': True, 'message': 'Incident archived successfully', 'id': incident.id})


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

if __name__ == '__main__':
    # Create database tables if they don't exist
    with app.app_context():
        db.create_all()
    is_debug = os.environ.get('FLASK_DEBUG', 'False').lower() in ['true', '1', 't']
    app.run(debug=is_debug, port=5001)