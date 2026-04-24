# RETORNSERO — Baseline System Report

Prepared as a contextual baseline before further development. Written as a partner, not a cheerleader: the good parts are labeled, the weak parts are labeled, and the things that would get you cornered in a thesis panel are called out explicitly.

---

## 1. What this system appears to be

Judging only from the code and file layout, **RETORNSERO (Barangay Sentinel CCTV)** is a *classical-computer-vision* crowd-monitoring and incident-logging web application intended for a barangay-level surveillance use case. Nothing in the repo uses deep learning. The pipeline is 100% OpenCV + SciPy.

At a functional level it:

1. Ingests a video source (currently the local file `video1.mp4`; architected to accept RTSP) per camera.
2. Runs MOG2 background subtraction, filters the foreground mask through an ROI mask that hides the road/cars, morphologically "fuses" noisy blobs into stable mega-blobs, and structurally validates each blob as "human-shaped" using aspect ratio + distance-transform peak sharpness.
3. Tracks surviving blobs across frames with a Hungarian-assignment centroid tracker that tolerates short ghost frames.
4. For each tracked blob, estimates the *number of people inside the blob* using a distance-transform / watershed-style sub-counter — this is how the system handles merged groups.
5. Exposes the annotated live frame as JPEG/MJPEG, plus a JSON stats endpoint (`count`, `density`, `status`, `locations`), via Flask.
6. Maintains a ring buffer of the last ~5 seconds of frames. When the engine classifies a frame as MEDIUM or HIGH density, it dumps the ring buffer + ongoing footage into an event clip (`Temp_Clips/`) with a thumbnail.
7. Presents a heavy single-page Tailwind frontend (`templates/index.html`, 2,119 lines) with an auth screen (Operator / Admin modes), a Live dashboard showing two camera tiles, a maximize-camera modal with activity logs and a density timeline, an event-clip "tray" where a Tanod reviews alerts and writes a Quill-rich-text report, and an Archive view of saved incidents.
8. Persists confirmed incidents (`VALID_THREAT` / `FALSE_ALARM`) into a SQLite or Postgres DB via SQLAlchemy, moving the clip from `Temp_Clips/` to `Archive/Archive_YYYY-MM-DD/`.
9. Ships two offline tooling scripts used to *justify the vision-engine parameters mathematically*: `generate_ground_truth.py` (a click-to-annotate tool that produced `barangay_ground_truth.json`, 15 frames), `tune_parameters.py` (a grid search over MOG2 params minimizing MAE against that ground truth), and `shanghaitech_optimizer.py` (reads ShanghaiTech .mat ground-truth files, computes median nearest-neighbor head distance, and emits "recommended hyperparameters").
10. Keeps `create_mask.py`, a simple click-polygon tool that produced `mask_layer.png`, the road/car exclusion mask.

## 2. Repository inventory

| File / Folder | Role | State |
| --- | --- | --- |
| `app.py` | Flask backend: routes, DB models, stream plumbing, incident archive | Active, production-shaped |
| `vision_engine.py` | `SentinelStream` class + detection/tracking/counting pipeline | Active, this is the core |
| `main.py` | Older standalone OpenCV viewer with zone rectangles Z6/Z7 | **Dead code** — references `FOOTAGE/video2.mp4` which does not exist |
| `tune_parameters.py` | Grid search over MOG2 hyperparameters vs. `barangay_ground_truth.json` | Active |
| `shanghaitech_optimizer.py` | Reads ShanghaiTech .mat files to derive "mathematically justified" kernel/threshold values | Active, with caveats (see §4) |
| `generate_ground_truth.py` | Click-to-annotate tool for making `barangay_ground_truth.json` | Active |
| `create_mask.py` | Click-polygon tool to generate `mask_layer.png` | Active |
| `barangay_ground_truth.json` | 15 annotated frames of `video1.mp4` (every 60 frames, 3–10 people, avg ≈ 7.9) | Active, but small (see §4) |
| `mask_layer.png` | ROI exclusion mask for road/cars | Active |
| `templates/index.html` | 2,119-line Tailwind SPA; auth, live, maximize, archive, clip review | Active |
| `video1.mp4` | ~21 MB input footage | Active (only video source) |
| `background.jpg` | Static background reference (used by `main.py`, not by `vision_engine.py`) | Orphaned by the active pipeline |
| `requirements.txt` | Flask 3, Flask-SQLAlchemy 3, psycopg3, OpenCV ≥4.10, NumPy, SciPy | Active |
| `instance/` | SQLite DB directory | Active |
| `Temp_Clips/`, `Archive/` | Event-clip storage | Active |
| `venv/`, `__pycache__/` | Should not be in the repo tree; `.gitignore` excludes venv but it's still sitting in the folder | Noise |
| `.git/` | Git history | — |

## 3. End-to-end data flow

```
 video1.mp4 ──► VideoCapture ──► MOG2 bg subtraction ──► threshold(254)
                                          │
                                          ▼
                         ROI mask (mask_layer.png) AND
                                          │
                                          ▼
                         morph close (7×50) + dilate 3×3  ◄── "fusion" step
                                          │
                                          ▼
                         findContours ──► per blob:
                                            • perspective-weighted min area
                                            • aspect ratio ≤ 1.8
                                            • distance-transform peak sharpness
                                          │
                                          ▼
                         RobustSentinelTracker (Hungarian on centroids,
                                                ghost tolerance 30 frames)
                                          │
                                          ▼
                         count_people_in_box (watershed-style sub-count)
                                          │
                  ┌───────────────────────┼────────────────────────┐
                  ▼                       ▼                        ▼
     annotated JPEG                latest_stats               ring buffer (150 frames)
     (get_latest_jpeg)             (count/density/status)        │
                  │                       │                    MEDIUM/HIGH
                  ▼                       ▼                    triggers
          /cam1_frame, /video_feed    /api/stats          event clip → Temp_Clips/
                  │                                              │
                  ▼                                              ▼
          <img> tag in SPA                         /api/temp_clips ──► clip tray
                                                                  │
                                                                  ▼
                                                       Tanod reviews + writes Quill
                                                                  │
                                                                  ▼
                                                       POST /api/incidents
                                                                  │
                                                                  ▼
                                          SQLAlchemy IncidentArchive row
                                          + file moved to Archive/Archive_YYYY-MM-DD/
```

The FE polls `/cam1_frame?t=<ts>` with a fresh timestamp per tick to defeat the browser cache. A streaming `/video_feed` endpoint exists but the live dashboard does not actually use it.

## 4. Honest assessment — what is right, what is wrong, what is fragile

### 4.1 Strengths

- **Separation of concerns is clean.** `vision_engine.py` is self-contained and testable; `app.py` only wires HTTP around it; the SPA is HTML-only. A grader can read each layer independently.
- **The pipeline is defensible as "classical CV, end-to-end."** Every step has a textbook justification: MOG2 → threshold → morphology → contours → Hungarian tracker → watershed-ish sub-count. Nothing hand-wavy in the *structure*.
- **Perspective-aware thresholds** (`get_perspective_weight`) are a nice touch — it shows awareness that far-field and near-field blobs cannot share a single min-area.
- **Structural blob validation** via distance-transform peak-to-half-width ratio is genuinely clever. It is the right answer to "how do you tell a car apart from a person with no neural net?".
- **Ring-buffer pre-event recording** (150 frames dumped before the trigger) is the right design for surveillance clips. Most student projects forget the "before" part and only record after.
- **Ground-truth tooling and automated tuning exist.** Having `generate_ground_truth.py` + `tune_parameters.py` + `shanghaitech_optimizer.py` gives you something to point at when a panelist asks "how did you pick your numbers?".
- **The incident-archive model is sensible**: temp clips are disposable, confirmed incidents are persistent on disk *and* in the DB, with a separate date-stamped folder per day.
- **Two storage backends (SQLite default, Postgres via `DATABASE_URL`)** — good forward thinking for deployment.

### 4.2 Problems that will hurt you at defense

These are the ones a sharp panelist will zero in on. Fix these before the defense or prepare a clean answer:

1. **Passwords are stored in plaintext, and the code literally says `# PLAIN TEXT FOR TESTING`.** `app.py` lines 33 and 90–94. Even if the panel is lenient on security, this one line makes the whole "production-ready surveillance system" framing collapse. Swap in `werkzeug.security.generate_password_hash` / `check_password_hash` (already a Flask transitive dep).
2. **Flask `debug=True` in `app.run`.** That's remote-code-execution via the Werkzeug debugger in any shared-network demo. At minimum, gate it on an env var.
3. **Hardcoded `app.secret_key` fallback.** `'barangay_sentinel_secure_key'` is the default if `SECRET_KEY` isn't set. Session forgery is trivial.
4. **Admin login is purely client-side.** `handleAdminLogin` in the SPA never calls the server, as far as the code I read. The "Admin" role column in `User` is declared but the backend never checks `role`. If this is meant to be a role-based system, that's a hole, not a feature.
5. **No auth on sensitive API endpoints.** `/api/temp_clips`, `/api/incidents` (both GET and POST), `/clips/...`, `/archive_media/...` are unauthenticated. Anyone who reaches the server can read incidents and delete clips.
6. **`report_html` is stored raw and re-rendered raw.** Quill produces HTML that goes into the DB and back out via `/api/incidents`. No sanitization. If anyone other than a trusted Tanod can write reports, that's stored XSS. Even in a single-user demo, panelists notice.
7. **The second camera is theater.** `cam1_stream` and `cam2_stream` both point at the same `video1.mp4`, with the same mask. In the UI, CAM-02's `<img src="/cam1_frame">` confirms it — it's literally the first camera's feed. For a barangay multi-camera pitch, this is the first question you'll get and the answer "placeholder" is weak. Either (a) swap in a second distinct test video and render from `/cam2_frame`, or (b) scope the thesis down honestly to a single camera and call the second one "future work".
8. **Your ground truth is far too small to defend the grid search.** 15 frames sampled every 60 frames from a single video. 15 MAE samples is not a tuning set — it is an anecdote. The "optimal" params from `tune_parameters.py` are almost guaranteed to be overfit to those 15 frames, and there's no held-out validation. Either expand to hundreds of frames across multiple scenes, or reframe the tuning as "illustrative, not conclusive." The latter is honest and survives scrutiny; the former is better if you have time.
9. **The MAE metric double-counts and mis-aligns.** In `tune_parameters.py`, `true_count = len(points)` over the *whole frame*, but the predicted count only comes from detections *inside the ROI mask*. If any annotated point sits outside the mask, the MAE is systematically biased toward over-predicting (or, more likely, toward false error). The ground-truth points should be filtered by the same ROI mask before comparison, or the mask should be disabled during grading.
10. **The ShanghaiTech "mathematical derivation" is thinner than it reads.**
    - `shanghaitech_optimizer.py` uses a **Windows absolute path** (`C:\Users\histo\...`). It cannot run on the machine the thesis runs on. Document that it was run once, elsewhere, and snapshot the output.
    - More importantly: ShanghaiTech gives you *head-point* nearest-neighbor distances from datasets whose camera geometry, resolution, crowd density, and pedestrian scale are *not* your barangay scene. Asserting that a 16 px median NN distance in ShanghaiTech justifies a 32 px "person width" in your video is a leap. Be ready to say: "I used ShanghaiTech as a *prior* to seed hyperparameters; final tuning came from my local ground truth." Don't oversell it as "mathematically derived".
    - In `count_people_in_box`, you do `max(8.0, 0.4 * dist.max())`. In most real blobs, `0.4 * dist.max()` is already > 8.0, so the ShanghaiTech-derived floor is ignored in practice. The claim "mathematical separation valley floor derived at 8.0px" in the comment is true in *derivation* but not in *effect* on most frames. A panelist who reads carefully will catch this.
11. **MOG2 cannot see stationary people.** With `history=500`, anyone who stops moving for ~15–17 seconds at 30 FPS fades into the background and disappears from the count. For a crowd-monitoring system this is a *fundamental* limitation worth naming out loud in your thesis. Don't let a panelist surprise you with it.
12. **The "density" metric is broken.** `density = min(100, total_count / ((W*H)/10000))`. For a 1920×1080 frame, the denominator is ~207. A count of 10 → density ≈ 0.048 → `int(0)`. You are emitting 0 every time. The UI's LOW/MEDIUM/HIGH buckets are effectively driven by `count` (via the `clip_level` thresholds at 4/7), not by `density`. Either redefine density as `people per unit ROI area`, or delete the field. Keeping it as-is is a landmine at defense.
13. **Status thresholds are arbitrary.** `SAFE` at 0, `WARNING` at <10, `CRITICAL` at ≥10. No justification. Similarly, MEDIUM at >4 and HIGH at >7. Ground these in something — density literature, barangay capacity, square-meter estimates — or concede they are user-configurable heuristics.
14. **The tracker has no re-ID.** Hungarian on centroid distance only. Any occlusion longer than 30 frames (one second) creates a new track ID. Counts are robust because you re-count every frame from fresh detections, but per-person trajectories are not. If your thesis claims "tracking", make sure the claim matches what the tracker actually does.
15. **In-memory `temp_clips` metadata is not persisted.** If the server restarts, clips on disk in `Temp_Clips/` exist, but the list `cam1_stream.temp_clips` is empty. The frontend will show nothing. Rescan the folder on startup.
16. **`pause_saving = True` is the default.** Event-clip recording is off until someone toggles it. That's fine for dev but will trip you during a live demo if you forget.
17. **`main.py` is confusing dead code.** It predates `vision_engine.py`, uses `absdiff` against a static `background.jpg` (not MOG2), has hard-coded zones Z6/Z7 that no longer exist in the new engine, and points at a video file that isn't in the repo. If the panel opens it, you'll spend ten minutes explaining what's actually being used. Delete it, move it to `legacy/`, or put a header comment at the top.
18. **No FPS / latency measurements anywhere.** For a real-time surveillance system, "it works" is not enough — you need to report processing FPS on your test hardware. Add a simple EMA frame-time counter in `_process_loop`.
19. **No evaluation beyond MAE.** You have no precision/recall for detection, no confusion matrix for SAFE/WARNING/CRITICAL, no comparison to even a trivial baseline (e.g., static threshold on contour count). A thesis on detection should report at least one of these.
20. **The `venv/` folder is inside the project directory** (it's .gitignored, so not in git, but it bloats the workspace). Put it outside the repo or under a different name; makes the repo noisy when anyone opens it.

### 4.3 Smaller, cheaper-to-fix issues

- Event-clip cooldown of 90 frames after video restart is a sensible band-aid but should be a named constant, not a magic number.
- `_mjpeg_from_file` re-decodes every request — fine for one viewer, fragile at scale, but acceptable at this scope.
- The FE polling `/cam1_frame?t=<ts>` at ~10 FPS is wasteful vs. just consuming `/video_feed`. Not broken, but you have the streaming endpoint already; use it.
- `drawCameraSVG` in the SPA is called with `camId` but always sets `src="/cam1_frame"` regardless — again, the second camera shortcut.
- `setReportStatus` / Quill report body: sanitize on the server before storing.
- `generate_ground_truth.py` does not dedupe points or warn on doubly-clicked people.
- `create_mask.py` doesn't close the polygon visibly, which makes its output easy to get wrong.
- `IncidentArchive` has `status` as a free-form string ("VALID_THREAT" / "FALSE_ALARM"). Constrain it with an Enum.
- Flask app never calls `db.create_all()` outside `if __name__ == '__main__'`. If run under a WSGI server, the tables are not created. Move it to an `init_db()` function called at import time or via a CLI.

## 5. What the thesis actually demonstrates (as of today)

If a panelist asks "what did you build?", the *honest* one-paragraph answer based on the current state is:

> A single-camera, classical-CV crowd counting and incident-logging web app. It uses MOG2 background subtraction, morphological fusion, distance-transform-based structural blob validation, and a Hungarian-assignment centroid tracker to estimate people counts in frame. It emits density-tagged event clips that a human Tanod reviews, annotates, and either dismisses or archives. Hyperparameters were initialized from statistics computed over the ShanghaiTech ground truth and then refined via grid search against a locally-annotated 15-frame validation set (MAE). A Flask/SQLAlchemy backend serves the live MJPEG, stats JSON, clip metadata, and incident archive, and a Tailwind SPA provides the operator UI.

The second camera, the Admin role, the activity-log sidebar, the density timeline graph, and the password-reset flow are currently **UI-complete but not functionally backed**. That's fine for a pitch but you need to know which is which when defending.

## 6. Recommended next-step priorities

Ordered by "biggest return per hour of work":

1. Replace plaintext passwords with hashes, disable `debug=True`, require `SECRET_KEY`, add a login-required decorator to the sensitive API routes. ~1–2 hours. Removes the embarrassing findings.
2. Either plug in a second real video source or cleanly label CAM-02 as "reserved / future work" in the UI. ~30 min.
3. Fix the `density` calculation (or drop it). Re-derive the thresholds against whatever metric you keep. ~1 hour.
4. Expand `barangay_ground_truth.json` to at least ~100 frames, ideally across multiple clips/times-of-day. Split into tune/test. Re-run `tune_parameters.py`. Report MAE on the held-out set, not the tuning set. Half a day, but it is the single most defensible thing you can do.
5. Filter ground-truth points by `mask_layer.png` inside `tune_parameters.py` before computing MAE. ~15 min, removes a systematic bias.
6. Add FPS and per-stage latency measurement to `_process_loop`. Report in the thesis. ~30 min.
7. Persist `temp_clips` by scanning `Temp_Clips/` on startup, and sanitize `report_html` server-side. ~1 hour.
8. Delete or clearly archive `main.py`. ~1 min.
9. Add a precision/recall evaluation on per-frame detections (needs point-to-box association with a distance threshold — reuse the Hungarian solver). Half a day, but it's the second most defensible thing you can do.
10. Document the ShanghaiTech derivation carefully — include the output from running `shanghaitech_optimizer.py` as a plaintext artifact in the repo, since the script's hardcoded path cannot be re-executed on this machine.

## 7. One-line verdict

There is a real, working, classical-CV surveillance system here, and the architecture is honest. The software-engineering hygiene (auth, secrets, debug mode, dead code) and the evaluation story (15-frame MAE, "mathematically derived" hyperparameters that mostly aren't, broken density metric) are the two surfaces where this thesis is currently soft. Both are fixable in well under a week of focused work, and fixing them is worth more at defense than any new feature.
