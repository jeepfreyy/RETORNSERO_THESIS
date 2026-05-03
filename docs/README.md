# Barangay Sentinel CCTV — Crowd Monitoring System

Real-time crowd density monitoring and anomaly detection for barangay CCTV feeds, using background subtraction (MOG2), morphological processing, and a custom multi-person tracking engine.

## Quick Start

### 1. Create a virtual environment (outside the repo)

```powershell
# PowerShell (Windows)
python -m venv $HOME\.venvs\retornsero
& $HOME\.venvs\retornsero\Scripts\Activate.ps1
```

```bash
# Bash (Linux / macOS)
python3 -m venv ~/.venvs/retornsero
source ~/.venvs/retornsero/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set environment variables

Copy `.env.example` to `.env` and fill in the values, or set them directly:

```powershell
# PowerShell
$env:SECRET_KEY = (python -c "import secrets; print(secrets.token_hex(32))")
```

```bash
# Bash
export SECRET_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")
```

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SECRET_KEY` | **Yes** | — | Flask session signing key. App will not start without it. |
| `FLASK_DEBUG` | No | `false` | Set to `true` for hot-reload during development. |
| `DATABASE_URL` | No | `sqlite:///sentinel_users.db` | PostgreSQL connection string for production. |

### 4. Run the application

```bash
python app.py
```

Open `http://127.0.0.1:5000` in your browser. Register a new account to log in.

## Project Structure

```
├── app.py                  # Flask web server & API routes
├── vision_engine.py        # SentinelStream: MOG2 pipeline, tracking, clip recording
├── templates/index.html    # Single-page dashboard (Tailwind + Quill.js)
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable reference
├── barangay_ground_truth.json  # 15-frame annotated ground truth for MAE evaluation
├── tune_parameters.py      # Automated grid search optimizer
├── mask_layer.png          # ROI mask for the video feed
├── docs/baseline/          # Phase 0 baseline artifacts (MAE, screenshots)
├── tests/                  # Verification test suite
├── legacy/                 # Archived pre-refactor prototype (main.py)
└── instance/               # SQLite DB (auto-created, gitignored)
```

## Security Notes

- Passwords are hashed with `scrypt` via `werkzeug.security`.
- All API routes require session authentication (`@login_required`).
- Report HTML is sanitized server-side with `bleach` (no `<script>`, `<iframe>`, or `on*` attributes).
- `SECRET_KEY` must be set explicitly — no hardcoded fallback.
- Debug mode is off by default.

## Running Tests

```powershell
$env:SECRET_KEY = 'test-secret'; python -m tests.test_phase1
```
