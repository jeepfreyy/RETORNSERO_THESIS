from flask import Flask, render_template, request, jsonify, session, Response
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import os
import threading
import time

app = Flask(__name__)

# Vision engine: CAM-01 stream (video_feed + api/stats)
_vision_started = False


def _ensure_vision_started():
    """
    Lazily start the CAM-01 vision engine loop in a background thread.
    """
    global _vision_started
    if not _vision_started:
        _vision_started = True
        try:
            from vision_engine import _cam1_update_loop

            t = threading.Thread(
                target=_cam1_update_loop,
                args=("video1.mp4",),
                daemon=True,
            )
            t.start()
        except Exception:
            # If something goes wrong, allow retry on next request
            _vision_started = False

#push postgress
# CONFIGURATION
# Secret key for session management (Keep this secret in production!)
app.secret_key = os.environ.get('SECRET_KEY', 'barangay_sentinel_secure_key')

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
    password_hash = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), default='Tanod') # Future-proofing for Admin/Tanod roles

# --- ROUTES ---

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

    # Create new user with Hashed Password (Security requirement)
    hashed_pw = generate_password_hash(password, method='pbkdf2:sha256')
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
    username = data.get('username')
    password = data.get('password')

    user = User.query.filter_by(username=username).first()

    # Verify user exists and password matches hash
    if user and check_password_hash(user.password_hash, password):
        session['user_id'] = user.id
        session['username'] = user.username
        return jsonify({'success': True, 'message': 'Login successful!', 'username': user.username})
    
    return jsonify({'success': False, 'message': 'Invalid username or password.'}), 401

@app.route('/api/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'success': True})


@app.route('/video_feed')
def video_feed():
    """
    MJPEG stream for CAM-01, backed by vision_engine._cam1_latest_jpeg.
    """
    _ensure_vision_started()

    def generate():
        from vision_engine import _cam1_latest_jpeg

        while True:
            if _cam1_latest_jpeg is None:
                time.sleep(0.05)
                continue
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                + _cam1_latest_jpeg
                + b"\r\n"
            )
            time.sleep(0.033)

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store"},
    )


@app.route('/cam1_frame')
def cam1_frame():
    """
    Single JPEG frame for CAM-01. This is useful in environments where
    multipart/x-mixed-replace streams are not handled correctly; the
    frontend can poll this endpoint to simulate video.
    """
    _ensure_vision_started()
    try:
        from vision_engine import _cam1_latest_jpeg
    except Exception:
        return Response(status=503)

    if _cam1_latest_jpeg is None:
        return Response(status=204)

    return Response(
        _cam1_latest_jpeg,
        mimetype="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )

@app.route('/api/stats')
def api_stats():
    """
    JSON stats for CAM-01 (people count, density, status).
    """
    _ensure_vision_started()
    try:
        from vision_engine import _cam1_latest_stats

        return jsonify(_cam1_latest_stats)
    except Exception:
        return jsonify({"count": 0, "density": 0, "status": "SAFE", "locations": []})

if __name__ == '__main__':
    # Create database tables if they don't exist
    with app.app_context():
        db.create_all()
    app.run(debug=True, port=5000)