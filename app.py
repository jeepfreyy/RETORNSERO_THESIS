from flask import Flask, render_template, request, jsonify, session, Response, send_file
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from datetime import datetime
import bleach
import shutil
import os
import time
import cv2
import atexit
import signal

app = Flask(__name__)

# Vision engine: Sentinel Streams
from vision_engine import SentinelStream

# Initialize cameras immediately when app starts
# Using same local video for both for now until true RTSP URLs are provided
cam1_stream = SentinelStream(stream_id="CAM-01", source="video1.mp4", mask_path="mask_layer.png")
cam2_stream = SentinelStream(stream_id="CAM-02", source="video1.mp4", mask_path="mask_layer.png")  # Placeholder

# --- Auto-cleanup temp clips on shutdown ---
def cleanup_all_streams():
    """Stop all streams and delete their temporary clips."""
    print("[App] Shutting down — cleaning up temporary clips...")
    cam1_stream.stop()
    cam2_stream.stop()
    print("[App] All temporary clips deleted.")

atexit.register(cleanup_all_streams)

# Also clean up any leftover temp clips from previous sessions on startup
cam1_stream.cleanup_temp_clips()
cam2_stream.cleanup_temp_clips()

# ---------------------------------------------------------------------------
# CONFIGURATION — Phase 1.3: Require explicit SECRET_KEY
# ---------------------------------------------------------------------------
_secret = os.environ.get('SECRET_KEY')
if not _secret:
    raise RuntimeError(
        "SECRET_KEY environment variable is required.\n"
        "Set it before starting the app, e.g.:\n"
        "  $env:SECRET_KEY='change-me-to-a-random-string'   (PowerShell)\n"
        "  export SECRET_KEY='change-me-to-a-random-string'  (bash)\n"
        "See .env.example for reference."
    )
app.secret_key = _secret

# Database configuration
# Use PostgreSQL if DATABASE_URL is set; otherwise fall back to SQLite for local dev.
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get(
    'DATABASE_URL',
    'sqlite:///sentinel_users.db'
)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)


# ---------------------------------------------------------------------------
# DATABASE MODELS — Phase 1.1: password_hash replaces plaintext password
# ---------------------------------------------------------------------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(20), default='Tanod')

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


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


# ---------------------------------------------------------------------------
# Phase 1.4: Authentication decorators
# ---------------------------------------------------------------------------
def login_required(f):
    """Decorator: returns 401 if no authenticated session."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('user_id'):
            return jsonify({'error': 'Authentication required'}), 401
        return f(*args, **kwargs)
    return decorated


# ---------------------------------------------------------------------------
# Phase 1.5: HTML sanitization allow-lists (conservative)
# ---------------------------------------------------------------------------
ALLOWED_TAGS = [
    'p', 'br', 'strong', 'em', 'u', 's', 'ul', 'ol', 'li',
    'h1', 'h2', 'h3', 'blockquote', 'pre', 'code', 'span',
]
ALLOWED_ATTRS = {
    '*': ['class'],
}


# ---------------------------------------------------------------------------
# PUBLIC ROUTES (no auth required)
# ---------------------------------------------------------------------------
@app.route('/')
def home():
    """Serves the frontend HTML."""
    return render_template('index.html')


@app.route('/api/register', methods=['POST'])
def register():
    """Handles user registration — Phase 1.1: hashes password with Werkzeug."""
    data = request.json
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')

    if not username or not email or not password:
        return jsonify({'success': False, 'message': 'All fields are required.'}), 400

    if User.query.filter_by(username=username).first():
        return jsonify({'success': False, 'message': 'Username already exists.'}), 409

    if User.query.filter_by(email=email).first():
        return jsonify({'success': False, 'message': 'Email already registered.'}), 409

    new_user = User(username=username, email=email)
    new_user.set_password(password)

    try:
        db.session.add(new_user)
        db.session.commit()
        return jsonify({'success': True, 'message': 'Registration successful!'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/login', methods=['POST'])
def login():
    """Handles user login — Phase 1.1: verifies against hashed password."""
    data = request.json
    email = data.get('email')
    password_attempt = data.get('password')

    user = User.query.filter_by(email=email).first()

    if user and user.check_password(password_attempt):
        session['user_id'] = user.id
        session['username'] = user.username
        session['role'] = user.role
        return jsonify({'success': True, 'message': 'Login successful!', 'username': user.username})

    return jsonify({'success': False, 'message': 'Invalid email or password.'}), 401


@app.route('/api/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'success': True})


# ---------------------------------------------------------------------------
# AUTHENTICATED ROUTES — Phase 1.4: all gated behind @login_required
# ---------------------------------------------------------------------------

@app.route('/video_feed')
@login_required
def video_feed():
    """MJPEG stream for CAM-01."""
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
    """Single JPEG frame for CAM-01."""
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
    """JSON stats for CAM-01 (people count, density, status)."""
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
    result = []
    for i in incidents:
        # Derive the relative archive path for the frontend
        archive_rel = ''
        if i.clip_path and os.path.exists(i.clip_path):
            archive_rel = os.path.relpath(i.clip_path, ARCHIVE_DIR).replace('\\', '/')
        else:
            # Fallback: try to locate the file by scanning Archive subdirs
            for subdir in os.listdir(ARCHIVE_DIR):
                candidate = os.path.join(ARCHIVE_DIR, subdir, i.clip_filename)
                if os.path.isfile(candidate):
                    archive_rel = f"{subdir}/{i.clip_filename}"
                    break

        result.append({
            'id': i.id,
            'timestamp': i.timestamp.isoformat(),
            'camera_id': i.camera_id,
            'report_html': i.report_html,
            'status': i.status,
            'density_tag': i.density_tag,
            'clip_filename': i.clip_filename,
            'archive_path': archive_rel,
            'duration': i.duration
        })
    return jsonify(result)


@app.route('/api/incidents', methods=['POST'])
@login_required
def save_incident():
    """Save an incident: moves clip from Temp_Clips/ to Archive/ and writes DB record."""
    try:
        data = request.json
        clip_filename = data.get('clip_filename')
        report_html = data.get('report_html', '')
        status = data.get('status', 'FALSE_ALARM')
        density_tag = data.get('density_tag', 'LOW')
        camera_id = data.get('camera_id', 'CAM-01')
        duration = data.get('duration', 0)

        if not clip_filename:
            return jsonify({'success': False, 'message': 'Missing clip filename.'}), 400

        # Phase 1.5: Sanitize report HTML (strip scripts, iframes, on* attrs)
        report_html = bleach.clean(
            report_html,
            tags=ALLOWED_TAGS,
            attributes=ALLOWED_ATTRS,
            strip=True,
        )

        # Create date-stamped archive subdirectory
        today = datetime.now().strftime("%Y-%m-%d")
        archive_subdir = os.path.join(ARCHIVE_DIR, f"Archive_{today}")
        os.makedirs(archive_subdir, exist_ok=True)

        # Move clip video from Temp -> Archive
        temp_path = os.path.join(TEMP_CLIPS_DIR, clip_filename)
        archive_path = os.path.join(archive_subdir, clip_filename)
        clip_moved = False
        if os.path.exists(temp_path):
            shutil.move(temp_path, archive_path)
            clip_moved = True
            thumb_name = clip_filename.replace('.mp4', '.jpg')
            temp_thumb = os.path.join(TEMP_CLIPS_DIR, thumb_name)
            if os.path.exists(temp_thumb):
                shutil.move(temp_thumb, os.path.join(archive_subdir, thumb_name))
        else:
            print(f"[save_incident] Warning: temp clip '{clip_filename}' not found at '{temp_path}'")
            archive_path = ''

        # Remove from in-memory temp clips list (both cams)
        cam1_stream.temp_clips = [c for c in cam1_stream.temp_clips if c['filename'] != clip_filename]
        cam2_stream.temp_clips = [c for c in cam2_stream.temp_clips if c['filename'] != clip_filename]

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

        msg = 'Incident archived successfully'
        if not clip_moved:
            msg += ' (warning: clip file was not found in temp folder)'

        return jsonify({'success': True, 'message': msg, 'id': incident.id})

    except Exception as e:
        db.session.rollback()
        print(f"[save_incident] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Server error: {str(e)}'}), 500


@app.route('/api/incidents/<int:incident_id>', methods=['DELETE'])
@login_required
def delete_incident(incident_id):
    """Delete an archived incident (admin-level action)."""
    incident = IncidentArchive.query.get(incident_id)
    if not incident:
        return jsonify({'success': False, 'message': 'Incident not found.'}), 404

    # Remove the archived file if it exists
    if incident.clip_path and os.path.exists(incident.clip_path):
        os.remove(incident.clip_path)
        thumb_path = incident.clip_path.replace('.mp4', '.jpg')
        if os.path.exists(thumb_path):
            os.remove(thumb_path)

    db.session.delete(incident)
    db.session.commit()
    return jsonify({'success': True, 'message': f'Incident #{incident_id} deleted.'})


@app.route('/archive_media/<path:filepath>')
@login_required
def serve_archive_media(filepath):
    """Serve a permanently archived clip or thumbnail."""
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
    filepath = os.path.join(TEMP_CLIPS_DIR, filename)
    if not os.path.exists(filepath):
        return Response(status=404)
    return Response(_mjpeg_from_file(filepath),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/archive_stream/<path:filepath>')
@login_required
def archive_stream(filepath):
    """Stream an archived clip as MJPEG so browsers can display it."""
    full_path = os.path.join(ARCHIVE_DIR, filepath)
    if not os.path.exists(full_path):
        return Response(status=404)
    return Response(_mjpeg_from_file(full_path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ---------------------------------------------------------------------------
# APP ENTRY POINT — Phase 1.2: debug off by default
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    # Create database tables if they don't exist
    with app.app_context():
        db.create_all()

    # Handle Ctrl+C gracefully
    def _signal_handler(sig, frame):
        print("\n[App] Received shutdown signal...")
        cleanup_all_streams()
        os._exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    debug_mode = os.environ.get('FLASK_DEBUG', 'false').lower() in ('1', 'true', 'yes')

    try:
        app.run(debug=debug_mode, port=5000)
    finally:
        cleanup_all_streams()