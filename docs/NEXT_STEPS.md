# RETORNSERO — What To Do Next

A working roadmap to bring the thesis from *"it runs"* to *"it defends cleanly."* Organized into phases by priority, with concrete file-level actions, effort estimates, and acceptance criteria per item. Read this as a checklist you can execute; each item is small enough to finish and verify before moving on.

## How to use this document

Work **top to bottom**. Do not skip ahead. The ordering is chosen so that:

1. The quickest, highest-impact fixes come first so that even if you run out of time, the thesis is still defensible.
2. Evaluation rigor is prioritized over new features, because evaluation is what you will be graded on.
3. Every phase has a "**Done when**" line. Do not mark an item done until that line is true.

Total estimated effort, end to end: **4 to 6 focused working days**, not counting new footage collection or writing the thesis chapters themselves.

---

## Phase 0 — Snapshot the current state before you break anything (30 min)

Before touching a single line, lock in what you have today. When a panelist asks "compared to what?", you want an honest before-vs-after.

- **0.1 Tag a baseline git commit.**
  - `git add -A && git commit -m "baseline: pre-hardening snapshot"`
  - `git tag v0.1-baseline`
  - Done when `git log --oneline -1` shows the tagged commit.
- **0.2 Run the current pipeline once, end-to-end, and record its behavior.**
  - Launch the app, open the dashboard, let it run for 60 seconds, take a screen-recording or screenshots of: the live feed, `/api/stats`, a triggered event clip, the archive view.
  - Save artifacts into `docs/baseline/` inside the repo.
  - Done when `docs/baseline/` contains at least: `live_feed.png`, `api_stats.json`, `event_clip.mp4`, `archive_view.png`.
- **0.3 Capture baseline numbers on the 15-frame ground truth.**
  - Run `tune_parameters.py` with the current hyperparameters only (not the whole grid). Log the MAE.
  - Paste the output into `docs/baseline/mae_before.txt`.
  - Done when that file exists and shows a single MAE number.

> Why: when you report "MAE dropped from X to Y" in the thesis, you need X on paper. Without this step the whole improvement narrative becomes anecdote.

---

## Phase 1 — Remove the defense-ending findings (half a day, max one day)

These are the items a panel will fixate on. None of them is research; all of them are hygiene. Ship this phase before anything else.

- **1.1 Hash passwords with Werkzeug.**
  - Files: `app.py` lines 33, 90–94, 101–106.
  - Replace the plaintext `password` column with a `password_hash` column; use `werkzeug.security.generate_password_hash` on register and `check_password_hash` on login.
  - Migrate the existing DB: easiest is to drop `instance/sentinel_users.db`, re-run, and re-create test accounts. Note in the thesis that the old DB was wiped intentionally during the hardening pass.
  - Done when: grepping the repo for `# PLAIN TEXT` returns nothing, and a new account can register + log in.
- **1.2 Disable Flask debug by default.**
  - File: `app.py` bottom block.
  - Change `app.run(debug=True, port=5000)` to read `FLASK_DEBUG` from env, defaulting to `False`.
  - Done when: starting the app with no env vars shows "Debug mode: off" in the console.
- **1.3 Require an explicit SECRET_KEY.**
  - File: `app.py` line ~22.
  - If `SECRET_KEY` is not set, raise `RuntimeError("SECRET_KEY is required")` instead of falling back to a hardcoded default.
  - For local dev, add a `.env.example` documenting the variable.
  - Done when: starting the app without `SECRET_KEY` crashes with a clear message.
- **1.4 Put authentication in front of the sensitive API routes.**
  - File: `app.py`.
  - Write a `@login_required` decorator that checks `session.get('user_id')` and returns 401 otherwise. Apply it to: `/api/temp_clips`, `/api/temp_clips/<filename>` DELETE, `/api/incidents` GET and POST, `/clips/<filename>`, `/archive_media/...`, `/clip_stream/...`, `/archive_stream/...`, `/cam1_frame`, `/cam2_frame`, `/video_feed`, `/api/stats`, `/api/stats/cam2`.
  - Done when: curling any of those endpoints without a logged-in session returns 401.
- **1.5 Sanitize Quill HTML server-side.**
  - Files: `app.py` `save_incident` handler, `requirements.txt`.
  - Add `bleach>=6` to requirements. On POST, pass `report_html` through `bleach.clean(..., tags=ALLOWED_TAGS, attributes=ALLOWED_ATTRS, strip=True)` with a conservative allow-list (no `script`, no `iframe`, no `on*` attrs, no `style`). Store the cleaned result.
  - Done when: submitting a report containing `<script>alert(1)</script>` results in a DB row whose `report_html` no longer contains the script tag.
- **1.6 Either wire up the Admin role or remove its UI.**
  - Files: `app.py` (decide on role semantics), `templates/index.html` (`handleAdminLogin`).
  - Minimum viable path: make Admin log in through the same `/api/login` endpoint, and gate an `@admin_required` decorator on an endpoint you actually care about (e.g., deleting incidents, which doesn't exist yet — add it and gate it).
  - If you are not going to implement it, cleanly delete the Admin mode toggle from the SPA and say so in the thesis scope section.
  - Done when: the Admin button either (a) performs a real server-authenticated login and unlocks at least one admin-only action, or (b) is gone.
- **1.7 Move `instance/` and `venv/` out of the way.**
  - `venv/` should live *outside* the repo, e.g. in `~/.venvs/retornsero`. Put instructions in `README.md`.
  - Keep `.gitignore` as-is.
  - Done when: `ls RETORNSERO_THESIS/` shows no `venv/`.
- **1.8 Delete or clearly archive `main.py`.**
  - Option A (recommended): move it to `legacy/main_single_file_prototype.py` with a header comment explaining it is the pre-refactor prototype.
  - Option B: delete it and reference it in the thesis via git history only.
  - Done when: nothing in the active code path imports from `main.py` and no panelist will mistake it for production code.

> Why this phase matters: one plaintext password, one hardcoded secret, or one public `/api/incidents` is enough to derail a defense. These are cheap wins and they remove the cheap attacks from the panel.

---

## Phase 2 — Fix the metric and logic bugs that contradict your own claims (half a day)

These are not security issues. They are items where the code does not do what the UI or the thesis says it does. Each one is an easy "gotcha" for a careful panelist.

- **2.1 Replace the broken `density` metric.**
  - File: `vision_engine.py`, `_process_loop`.
  - Current formula effectively always yields 0 on a 1080p frame. Two options, pick one and stick with it:
    - Option A: density = `total_count / roi_area_in_m²` — requires one real-world length calibration (document it in the thesis as "monocular planar homography from two known landmarks").
    - Option B: density = `total_count / count_of_valid_roi_pixels * 10000` — unitless, but has *some* signal; document the 10000 normalization.
  - Update the SPA to display the new metric with correct units.
  - Done when: the `density` field in `/api/stats` varies meaningfully as people enter/leave the scene on the test video.
- **2.2 Ground the SAFE / WARNING / CRITICAL thresholds.**
  - Files: `vision_engine.py`, `templates/index.html`.
  - Pick thresholds based on something you can defend: barangay assembly guidelines, floor-area-to-person ratios (NBCP: ~0.5 m² per person standing), or an empirical histogram of counts on your footage. Document the reasoning in the thesis.
  - Done when: the `status` buckets are derived in code from named constants that each trace back to a cited source or a recorded empirical choice.
- **2.3 Ground the `clip_level` MEDIUM/HIGH thresholds (4 / 7) the same way.**
  - File: `vision_engine.py`.
  - Same treatment as 2.2.
  - Done when: both sets of thresholds live in a single `config.py` and are referenced, not repeated.
- **2.4 Fix the MAE evaluation bias.**
  - File: `tune_parameters.py`.
  - Before computing `len(points)` for `true_count`, filter the annotated points through `mask_layer.png` — discard any point falling outside the ROI. Predict on the ROI-masked pipeline, compare against the ROI-masked ground truth.
  - Done when: running the script emits both `MAE (raw)` and `MAE (ROI-aligned)`; the ROI-aligned number is what you report.
- **2.5 Persist `temp_clips` across restarts.**
  - File: `vision_engine.py` `__init__`.
  - On construction, scan `Temp_Clips/` for `*.mp4`, rebuild the metadata list from filenames (they already encode `stream_id`, timestamp, and density level), and set `self.temp_clips`.
  - Done when: killing and restarting the server with 3 clips on disk shows 3 clips in the tray without any manual step.
- **2.6 Flip the default of `pause_saving` to `False`.**
  - File: `vision_engine.py`.
  - Add an env var (`CLIP_SAVING_ENABLED`, defaults to `True`) so it remains controllable for development.
  - Done when: running the system on the test video produces event clips without a code edit.
- **2.7 Make `_mjpeg_from_file` sanitize the filename.**
  - File: `app.py`.
  - Guard against path traversal via `werkzeug.utils.secure_filename`.
  - Done when: `curl /clips/../app.py` returns 404.
- **2.8 Add an FPS / per-stage latency meter.**
  - File: `vision_engine.py`.
  - At the top of `_process_loop` maintain an EMA of frame time. Expose `fps` and `ms_per_frame` in `/api/stats`.
  - Done when: `/api/stats` shows a non-zero `fps` value and it is stable within ±10% over a minute of playback.

---

## Phase 3 — Build an evaluation story that can actually be defended (two to three days)

This is the single biggest credibility gain available to you. A thesis lives and dies on its numbers. Fifteen ground-truth frames on one video with no held-out set is not a number; it is an anecdote.

- **3.1 Expand the ground-truth set.**
  - Target: **at least 150 annotated frames**, spanning at least **three distinct clips** (varying time-of-day, crowd level, and ideally weather if you can capture it). Distribute roughly: 60% "normal" scenes, 25% borderline MEDIUM, 15% HIGH density.
  - Tool: keep using `generate_ground_truth.py` but first make the improvements in 3.2 so annotation is less painful.
  - Time: this is the labor-intensive item. Budget 4–6 hours of clicking. Plan breaks; fatigue kills annotation quality.
  - Done when: `barangay_ground_truth.json` contains ≥150 frames across ≥3 videos, and a new field `video_source` is a per-frame value, not a single top-level key.
- **3.2 Harden the annotation tool.**
  - File: `generate_ground_truth.py`.
  - Add: (a) multi-video support via command-line argument, (b) a dedup warning when two points are within ~5 px, (c) a visible ROI overlay showing which points will be excluded, (d) keyboard shortcut to jump to any frame index, (e) autosave every 30 seconds.
  - Done when: you can annotate a second video without editing the script.
- **3.3 Split the ground truth into tune / validation / test.**
  - Use a **stratified, clip-aware split**: for each clip, allocate 70% of its annotated frames to tune, 15% to validation, 15% to test. Store the split as `splits.json` so it is reproducible.
  - Critical: the test set is touched **once**, at the end, for the final reported number. Do not peek.
  - Done when: `splits.json` exists, and `tune_parameters.py` only reads from tune; a new `evaluate.py` reads validation during development and test only for the final report.
- **3.4 Write `evaluate.py` with multiple metrics, not just MAE.**
  - Metrics to compute, per frame:
    - MAE: mean absolute error on total count.
    - RMSE: root mean squared error on total count (penalizes large misses).
    - GAME(L) for L in {1, 2, 3}: Grid Average Mean absolute Error — divides the frame into 4^L cells and averages per-cell MAE. This catches "right total, wrong location" errors and is standard in crowd-counting literature.
    - Precision / Recall / F1 on per-person detections, by associating predicted box-center to ground-truth points with a Hungarian matcher and a distance threshold (e.g., 50 px).
  - Output: a single JSON file `docs/eval/results_vX.json` plus a pretty Markdown table.
  - Done when: the script runs end-to-end and emits all five numbers.
- **3.5 Report at least one baseline comparison.**
  - Cheapest defensible baseline: "total count = number of contours after thresholding, no validation, no tracking." It will be terrible, and that's the point — it shows your pipeline earns its complexity.
  - Stretch baseline: a single off-the-shelf pretrained detector (e.g., YOLOv8n on "person") run on the same frames. Record its MAE and FPS. Yes, it will beat you on accuracy; you will beat it on FPS and on the classical-CV framing of your thesis. Name the trade-off out loud.
  - Done when: `docs/eval/results_vX.json` contains a `baselines` block with at least the trivial baseline.
- **3.6 Report FPS on target hardware.**
  - Run the pipeline on the same laptop you will demo with, on each of the three clips, and record average FPS + 95th-percentile frame time.
  - Done when: `docs/eval/runtime.json` exists.
- **3.7 Sensitivity analysis on the chosen hyperparameters.**
  - After grid-searching on tune + validating on validation, vary each chosen hyperparameter by ±20% one at a time and record the MAE change on the validation set. This is a one-page "robustness" figure that dramatically raises thesis quality.
  - Done when: `docs/eval/sensitivity.csv` exists with one row per (parameter, perturbation).
- **3.8 Re-run `shanghaitech_optimizer.py` on a machine where the dataset actually lives.**
  - Problem today: the script's path `C:\Users\histo\...` cannot be executed on the current machine. Either fix the path to read from an env var (`SHANGHAITECH_ROOT`), or snapshot the output from a one-time run into `docs/shanghaitech_run_YYYYMMDD.txt` and cite that.
  - Also: honestly reframe the role of ShanghaiTech in the thesis. It is a **prior to seed hyperparameter starting points**, not a ground-truth-aligned derivation. Do not claim it "proves" anything about your video.
  - Done when: running the script on a machine with `SHANGHAITECH_ROOT` set produces the same table it did originally, and the thesis text reflects the weaker-but-honest "seeding" framing.

> Why this phase matters: if every other phase is perfect but this one is weak, the thesis is weak. If this phase is strong, a surprising amount of the rest can be forgiven.

---

## Phase 4 — Address the fundamental detector limitations (one day)

Not every weakness can be closed; some must be acknowledged and bounded. This phase is about both.

- **4.1 Mitigate the "stationary person disappears" problem.**
  - MOG2 with `history=500` will absorb a person who stops for ~17 seconds at 30 FPS. Options:
    - **Short term (what you can ship)**: when a confirmed track goes "ghost" before leaving the frame edge, keep counting it as present for up to N frames even though MOG2 no longer sees it. This preserves the count for a pedestrian who stops briefly.
    - **Medium term**: lower `history` in zones where people are expected to linger; or add a motion-independent supplementary cue (e.g., sparse optical flow "keypoint presence" check) that keeps a track alive.
  - Even if you don't fix it, **name the limitation explicitly in the thesis** with a quantified example: "A stationary person is undercounted after approximately 500 frames (≈17 s at 30 FPS). See Figure X."
  - Done when: the thesis contains a labeled figure showing the failure mode, and (if you shipped the mitigation) a second figure showing the improved behavior.
- **4.2 Deal with the second camera honestly.**
  - Three acceptable paths:
    - **A — Cut it.** Remove CAM-02 from the SPA, scope the thesis as single-camera. Easiest and most honest.
    - **B — Second video.** Record or find a second barangay-like video, point `cam2_stream` at it, render `<img src="/cam2_frame">` in the SPA. Medium effort.
    - **C — Multi-camera story.** Add a real second RTSP/video source and claim multi-camera monitoring as a contribution. Highest effort; only do this if you have time budget and a second real scene.
  - Done when: the UI's second tile shows either (a) nothing, or (b) a genuinely different feed. Never the same feed relabeled.
- **4.3 Tracker re-ID expectation-setting.**
  - The current Hungarian-on-centroid tracker cannot re-identify a person after long occlusion. This is fine, but the thesis must not claim "trajectory tracking"; it must say "per-frame count with short-term centroid association."
  - Optional upgrade (do only if time allows): add a 32-bin color histogram per track and use it as a secondary cost in the Hungarian matrix. This dramatically improves re-ID after ≥1 s occlusion at a modest compute cost.
  - Done when: the thesis language accurately reflects tracker capabilities. If you do the upgrade, evaluate it in Phase 3 terms (MAE before vs after).

---

## Phase 5 — Engineering hygiene and deployment polish (half a day)

- **5.1 `config.py`.** Extract every magic number from `vision_engine.py`, `app.py`, `tune_parameters.py` into a single module. Thresholds, kernel sizes, frame counts, density cutoffs. Each named with a comment explaining the origin (ShanghaiTech prior / grid-search tuned / empirically chosen / cited source).
- **5.2 `README.md`.** Cover: what the project is in one paragraph, how to install, how to run, how to generate ground truth, how to run evaluation, environment variables, known limitations. Two pages max.
- **5.3 Logging.** Replace every `print(f"[SentinelStream ...]")` with `logging.getLogger(__name__).info(...)` so that the app can be run quietly in demo mode and verbosely in debug.
- **5.4 Graceful shutdown.** `SentinelStream` does not stop cleanly. Add a `stop()` that sets `self.running = False` and joins the thread. Call it from a Flask teardown hook.
- **5.5 Error handling in `_process_loop`.** Wrap the frame-processing body in a try/except that logs and continues, so a single corrupt frame doesn't kill the stream.
- **5.6 Pin dependencies.** `requirements.txt` uses soft bounds. Generate `requirements.lock.txt` via `pip freeze` for reproducibility. Include Python version in the README.
- **5.7 Delete cruft.** `background.jpg` is unused by the active pipeline; remove it or move to `legacy/`. Same for anything else unreferenced (run a quick `grep -R` for each asset name).
- **5.8 Switch the FE from polling `/cam1_frame` to consuming `/video_feed`.** The streaming endpoint already exists; the polling wastes bandwidth and causes jitter.
- **5.9 Add a one-page architecture diagram to the repo.** PNG or SVG, not ASCII. Use it in both the README and the thesis.

---

## Phase 6 — Thesis document work (one to two days)

This is where the code work converts into a defense. You want every weakness you cannot fix to be acknowledged *in the document* on *your* terms, not a panelist's.

- **6.1 Scope section.** Explicit, narrow, one paragraph: "This thesis presents a classical-computer-vision single-camera crowd-monitoring and incident-logging pipeline for barangay-level surveillance. It does *not* use deep learning. It does *not* perform long-term person re-identification. It reports metrics on a locally-annotated dataset of N frames across M scenes."
- **6.2 Contributions section.** Three to five bullet points, each defensible: (i) classical-CV-only crowd counting pipeline with structural validation, (ii) ShanghaiTech-seeded hyperparameters refined via local grid search, (iii) ring-buffered event-clip workflow integrated with a rich-text incident archive, (iv) empirical evaluation on N frames across M scenes with {MAE, RMSE, GAME(L), precision/recall, FPS}, (v) sensitivity analysis of chosen hyperparameters.
- **6.3 Methodology chapter.** Walk the pipeline stage by stage. Each stage has its own subsection, its own motivation, its own parameter table, and a reference to the source line(s). Do not write this as a narrative; write it as a reader-friendly specification.
- **6.4 Evaluation chapter.** Present the Phase 3 numbers. Include: the metrics, the baselines, the sensitivity table, the FPS table, and at least two qualitative figures (one success, one failure — the failure is where you gain credibility).
- **6.5 Limitations section.** Mandatory. State every known weakness, with quantification where possible. Stationary-person fade-out. Single-camera scope. Small annotated corpus. No re-ID. No low-light validation. Writing this yourself costs you nothing and removes it from the panel's arsenal.
- **6.6 Future work section.** Turn every "I didn't have time for this" into a labeled future direction. Deep-learning head detector. Multi-camera fusion. Temporal re-ID with appearance features. Density via metric homography. Each one is a sentence; together they show self-awareness.
- **6.7 Appendix: running the system.** Step-by-step from a clean clone to a running server. If a panelist tries to reproduce, they will reach you; if they succeed, you win. If they fail because of instructions, you lose.

---

## Phase 7 — Demo rehearsal (half a day)

You only get one shot on defense day. Rehearse the demo under real constraints.

- **7.1 Cold-start rehearsal.** From a reboot: launch the server, open the browser, log in, show a live event trigger, review a clip, write a report, archive it, reopen from the archive. Time it. Twice in a row without errors.
- **7.2 Failure rehearsal.** Rehearse the three most likely live failures: no network, no camera feed, DB locked. Have a one-sentence response for each.
- **7.3 FAQ sheet.** Write down the five questions you are most afraid of and draft honest answers. Examples:
  - "Why not deep learning?" — scope + FPS + reproducibility; see Limitations.
  - "Why 15 frames?" — we now have ≥150 across ≥3 scenes; see Evaluation.
  - "Why does CAM-02 show the same image as CAM-01?" — it doesn't anymore (or: it's out of scope, see Scope).
  - "How are your hyperparameters justified?" — seeded from ShanghaiTech statistics, refined via grid search on the tune split, validated on a held-out validation split, reported on a never-touched test split.
  - "What happens if a person stops moving?" — they fade after ~17 s at history=500. Mitigation and limitation both documented.
- **7.4 Kill the unknowns.** Go through the SPA and click every button. For each one that does nothing or half-works, either fix it or hide it. A broken button in the UI is worth five broken paragraphs in the thesis.

---

## Effort summary

| Phase | Effort | Risk if skipped |
| --- | --- | --- |
| 0. Snapshot | 0.5 h | Lose the before/after story |
| 1. Defense-ending hygiene | 4–6 h | Plaintext passwords / debug mode = instant credibility loss |
| 2. Metric & logic bugs | 3–4 h | Panelist catches a contradiction between code and claim |
| 3. Evaluation story | 2–3 days | This **is** the thesis |
| 4. Fundamental limitations | 1 day | Panelist brings up MOG2 stationary-fade; you have no answer |
| 5. Engineering hygiene | 4 h | Demo failures; unreproducible thesis |
| 6. Thesis document | 1–2 days | Good system, weak write-up |
| 7. Demo rehearsal | 4 h | Live demo goes wrong on the one day that matters |

Total floor: **4 focused days.** Total ceiling (with stretch items): **7 days.**

## A minimum-viable "if the defense is in 48 hours" subset

If you are under real time pressure, do **only these** and skip the rest:

- 1.1, 1.2, 1.3, 1.4 (security hygiene).
- 2.1 (fix density), 2.4 (ROI-aligned MAE).
- 3.1 at a lower target (≥60 frames across ≥2 clips), 3.3 (splits), 3.4 (MAE + RMSE + F1).
- 4.1 at the *acknowledgement only* level — write the Limitation paragraph.
- 6.1, 6.5 (Scope + Limitations sections).
- 7.3 (FAQ sheet).

This gets you to "defensible" in roughly one long day plus a second for evaluation. Everything else is quality-of-life.

## A "golden path" suggested order for a full pass

Day 1 AM: Phase 0 + Phase 1.
Day 1 PM: Phase 2.
Day 2 AM: Phase 3.2 (annotation tooling) + start Phase 3.1 (annotate clip #1).
Day 2 PM: Phase 3.1 (annotate clips #2 and #3).
Day 3 AM: Phase 3.3 (splits) + Phase 3.4 (`evaluate.py`).
Day 3 PM: Phase 3.5 (baselines) + 3.6 (FPS) + 3.7 (sensitivity) + 3.8 (ShanghaiTech).
Day 4 AM: Phase 4 + Phase 5.
Day 4 PM: Phase 6.
Day 5: Phase 7 + buffer for anything that slipped.

---

## What I (your partner) commit to

- When you come back and say "let's start Phase 1.1," I will touch only `app.py`, stick to the scope of that item, run the pipeline, and show you the diff + a verification step before we move on.
- I will not silently expand scope. If I think an adjacent item should come along for the ride, I will flag it and ask.
- I will push back on any item on this list if I change my mind about its priority once we're in the code. The list is a hypothesis, not a contract.
- When we finish a phase, I will update `BASELINE_REPORT.md` and this file to reflect what's true, so the documents don't drift out of sync with reality.

Your move. Pick a phase — or tell me to start at the top — and I'll begin.
