# Barangay Sentinel — User Manual

**System:** Barangay Sentinel Advanced Surveillance System  
**Version:** As of May 14, 2026  
**Audience:** Barangay Tanods, Barangay Officials, Operators

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Getting Started — Login & Registration](#2-getting-started--login--registration)
3. [Dashboard](#3-dashboard)
4. [Live Camera Feed](#4-live-camera-feed)
5. [Activity Logs](#5-activity-logs)
6. [Incident Tray](#6-incident-tray)
7. [Filing an Incident Report](#7-filing-an-incident-report)
8. [Incident Archive](#8-incident-archive)
9. [Incident Response Center](#9-incident-response-center)
10. [Appearance — Dark / Light Mode](#10-appearance--dark--light-mode)
11. [Language Switcher](#11-language-switcher)
12. [Password Recovery](#12-password-recovery)
13. [User Roles & Availability](#13-user-roles--availability)
14. [Frequently Asked Questions](#14-frequently-asked-questions)
15. [Changelog](#15-changelog)

---

## 1. System Overview

Barangay Sentinel is a real-time crowd-monitoring and incident-management platform designed for barangay-level public safety operations. The system uses computer vision to detect and count people in camera feeds, automatically flags crowd density events, and provides tools for operators to document, track, and resolve security incidents.

**Core capabilities:**

| Capability | Description |
|---|---|
| Live Monitoring | Real-time MJPEG stream from surveillance cameras with AI-powered people counting |
| Density Detection | Automatic classification of crowd levels (Normal / Crowding / Alert) |
| Incident Logging | Clip-based incident reports with threat level, notes, and location |
| Response Center | Assign barangay officials to incidents and track resolution in real time |
| Incident Archive | Permanent searchable record of all filed incidents with full status history |
| Activity Logs | Chronological feed of all detected events across all cameras |

---

## 2. Getting Started — Login & Registration

### Registering an Account

1. Open the system in your browser (default: `http://localhost:5001`).
2. On the login page, click **REGISTER**.
3. Fill in:
   - **Username** — your display name used throughout the system
   - **Email** — used for login and password recovery
   - **Password** — minimum 8 characters, must include at least one uppercase letter and one digit or symbol
4. Click **CREATE ACCOUNT**.
5. You will be redirected to the login page once registration succeeds.

> **Note:** All new accounts are assigned the role **Tanod** by default. Role changes must be made directly in the database by a system administrator.

### Logging In

1. Enter your registered **email** and **password**.
2. Click **LOGIN** (or **MAG-LOGIN** in Filipino mode).
3. On success you are taken directly to the **Dashboard**.

### Language Selection

Before logging in, you can switch the interface language using the flag buttons at the top-right of the login form:

- 🇺🇸 **EN** — English
- 🇵🇭 **TL** — Filipino (Tagalog)

Your language preference is saved and restored on your next visit.

---

## 3. Dashboard

The Dashboard is the main hub after login. It contains four sections arranged across the screen:

### Status Cards (top bar)

| Card | What it shows |
|---|---|
| **CAMERAS ONLINE** | How many cameras are currently streaming (e.g. `1 / 2`) |
| **ALERTS TODAY** | Total incidents filed + unsubmitted clips queued today |
| **STORAGE USED** | Disk usage percentage of the server drive |
| **UPTIME** | How long the server has been running since last restart |

The cards refresh automatically every 30 seconds.  
- Storage turns **amber** above 70 % and **red** above 90 %.  
- Cameras turn **amber** if only some are online and **red** if all are offline.

### Navigation Tabs

Use the tabs at the top or the sidebar to switch between:

- **MONITOR** — Live camera feed + Activity Logs
- **INCIDENT TRAY** — Unsubmitted clip queue
- **INCIDENT ARCHIVE** — All filed incident reports
- **RESPONSE CENTER** — Open incidents awaiting responders

---

## 4. Live Camera Feed

The **MONITOR** tab shows the live feed from CAM-01 (DHT Junkshop, Front Gate – MRC street).

### Camera Controls

| Control | Action |
|---|---|
| **Maximize button** (⛶) | Opens the feed in a fullscreen overlay |
| **Video selector dropdown** | Switches between available demonstration video scenarios |
| **MANUAL CLIP** button | Starts / stops a manually triggered recording |

### Crowd Density Indicators

The system automatically classifies the current scene:

| Level | Meaning |
|---|---|
| **NORMAL** | Crowd count is within safe limits |
| **CROWDING** | Moderate crowd — operators should monitor |
| **ALERT** | High density — immediate attention recommended |

### People Count & Warmup

When the camera first starts (or a video is switched), a **WARMING UP** overlay appears. During this period the background model is being calibrated and the people count may not be accurate. The overlay disappears once calibration is complete.

---

## 5. Activity Logs

The Activity Logs panel slides in from the right side of the MONITOR tab. Click **📊 ACTIVITY LOGS** to show or hide it.

Each log card shows:
- **Camera ID** — which camera detected the event
- **Location** — mapped location name
- **Description** — what was detected (e.g. "Crowd count exceeded threshold")
- **Timestamp** — when the event occurred

### Log Types & Colors

| Color | Type | Meaning |
|---|---|---|
| 🟢 Green | **NORMAL** | Low density — everything is fine |
| ➡️ Green | **ENTRY** | People entering the monitored area |
| ⬅️ Blue | **EXIT** | People leaving the monitored area |
| ⚠️ Amber | **CROWDING** | Crowd is building — moderate density |
| 🚨 Red | **ALERT** | High density or threat detected |

Logs are listed newest-first. The list is live and updates automatically as new events come in.

---

## 6. Incident Tray

The **INCIDENT TRAY** tab holds short video clips that were automatically captured when the system detected a crowd event, as well as any manual clips you recorded. These clips are **temporary** — they must either be filed as an incident report or dismissed.

### Tray Card Layout

Each clip card shows:
- Thumbnail preview of the clip
- Camera source, density tag, people count detected
- Duration and timestamp

### Actions

| Button | Action |
|---|---|
| **▶ REVIEW** | Opens a preview of the clip so you can watch it before deciding |
| **FILE REPORT** | Opens the incident report form for this clip |
| **✕ DISMISS** | Deletes the clip permanently without filing a report |

---

## 7. Filing an Incident Report

Clicking **FILE REPORT** on any clip opens the **Incident Report** form.

### Required Fields

| Field | Description |
|---|---|
| **Incident Title** | Short name for the event (leave blank for "Untitled Incident") |
| **Classification** | `VALID THREAT` or `FALSE ALARM` |
| **Density Tag** | `LOW`, `MEDIUM`, `HIGH`, or `MANUAL` |
| **Threat Level** | Slider from 1 (low) to 10 (critical) |

### Optional Fields

| Field | Description |
|---|---|
| **Operator Notes** | Rich-text field for observations, actions taken, context |
| **People Count** | Estimated number of people in the clip |
| **Location** | Pre-filled from camera metadata; editable if needed |

### Submitting

Click **SUBMIT REPORT** to file the incident. The system will:
1. Copy the clip to permanent storage (Archive folder)
2. Create a database record with status **OPEN**
3. Log a **PUBLISHED** and **OPEN** status entry with your username and the current timestamp
4. Remove the clip from the Incident Tray
5. Make the incident visible in the Incident Archive and Response Center

---

## 8. Incident Archive

The **INCIDENT ARCHIVE** tab shows every filed incident report, newest first. Each card is expandable.

### Viewing an Incident

Click any incident card to expand it. The expanded view shows:

- **Clip preview** — thumbnail of the recorded clip
- **Incident details** — title, camera, density, threat level, people count, location, timestamp
- **Operator notes** — the report written at time of filing
- **Reported by** — the username of the operator who filed the report

### Status Timeline

Every incident tracks five status milestones. The timeline panel shows each step with:
- A filled checkmark ✅ when that status has been reached
- The **exact date and time** the status was set
- The **username** of the operator who triggered the change

| Status | Meaning |
|---|---|
| **PUBLISHED** | Incident report was created and saved |
| **OPEN** | Incident is waiting for a response team |
| **RESPONDING** | One or more responders have been assigned |
| **RESOLVED** | The incident has been handled in the field |
| **CLOSED** | Report is complete and archived |

### Assigned Responders

The archive card shows all officials who were assigned to the incident at any point, including their role and the time they were assigned.

---

## 9. Incident Response Center

The **RESPONSE CENTER** tab (`INCIDENT RESPONSE CENTER`) is where operators coordinate the active response to open incidents.

### Filter Tabs

| Tab | Shows |
|---|---|
| **ALL** | Every filed incident regardless of status |
| **OPEN** | Incidents with no responders yet |
| **RESPONDING** | Incidents with at least one assigned responder |
| **RESOLVED** | Incidents marked as resolved |

Click **REFRESH** to reload the list manually.

### Incident Cards

Each card in the Response Center shows:
- Status badge (OPEN / RESPONDING / RESOLVED / CLOSED)
- Density and Threat Level tags
- Title, location, timestamp, reporter
- Number of assigned responders

Click **MANAGE** on any card to expand the full management panel.

### Assigning a Responder

Inside the expanded management panel, under **ASSIGN RESPONDER**:

1. Click the **Select a registered user…** dropdown.
2. A list of all system users appears, showing:
   - **Avatar** — first letter of the username
   - **Username** — the account name
   - **Role** — their registered role (e.g. Barangay Tanod, Barangay Captain)
   - **Availability** — 🟢 **Available** or 🟡 **Responding** (currently assigned to another active incident)
3. Click a user to select them. Their name and role fill the form automatically.
4. Optionally type a **note** (e.g. "En route", "On standby at checkpoint").
5. Click **ASSIGN OFFICIAL** to confirm.

**OR** click **TAKE THIS INCIDENT** to immediately assign yourself (the currently logged-in user) to the incident.

> When the first responder is assigned to an OPEN incident, the status automatically advances to **RESPONDING**.

### Removing a Responder

In the responder list inside the management panel, click **REMOVE** beside a responder's name.  
If all responders are removed from a RESPONDING incident, it reverts back to **OPEN**.

### Updating Incident Status

Under **UPDATE INCIDENT STATUS**:

1. Select the new status from the dropdown (`OPEN`, `RESPONDING`, `RESOLVED`, `CLOSED`).
2. Optionally add a **Resolution Note** describing what action was taken.
3. Click **APPLY**.

The status change is logged immediately with your username and the current timestamp.

### Deleting an Incident

Under **DANGER ZONE** at the bottom of the management panel:

1. Click **DELETE INCIDENT**.
2. A confirmation dialog appears: *"Are you sure you want to permanently delete this incident? This cannot be undone."*
3. Click **CONFIRM DELETE** to proceed.

> **Warning:** Deletion is permanent. The database record, the archived video clip, and its thumbnail are all removed. This action cannot be reversed.

---

## 10. Appearance — Dark / Light Mode

A theme toggle button is located in the **top navigation bar**, to the right of the **BARANGAY SENTINEL** brand title.

- Click the **moon icon** (🌙) to switch to **Light Mode**.
- Click the **sun icon** (☀️) to switch back to **Dark Mode**.

Your preference is saved automatically and restored the next time you open the system.

---

## 11. Language Switcher

On the **Login / Register** page, two flag buttons appear above the form:

| Button | Language |
|---|---|
| 🇺🇸 EN | English (default) |
| 🇵🇭 TL | Filipino / Tagalog |

Switching language changes all labels, buttons, and placeholders on the login page. The selected language is saved and persists across visits.

---

## 12. Password Recovery

If you forget your password:

1. On the login page, click **Forgot password?**
2. Enter your registered **email address** and click **Send Code**.
3. Check your email inbox for a **6-digit verification code** (valid for 10 minutes).
4. Enter the code in the verification field and click **Verify**.
5. Set a new password (same rules as registration: 8+ characters, uppercase, digit or symbol).
6. Click **Reset Password**. You will be returned to the login screen.

> You may request a reset code up to **3 times per 15 minutes**. After 5 incorrect code attempts the code is invalidated and you must request a new one.

---

## 13. User Roles & Availability

### Roles

When a user account is created, it is assigned the **Tanod** role by default. The role is displayed in the Assign Responder dropdown and on responder cards.

| Role | Typical User |
|---|---|
| Tanod | Front-line barangay patrol officer |
| Barangay Official | Administrative official |
| Barangay Captain | Head of the barangay |
| Investigator | Designated case investigator |
| Responder | General emergency responder |

> Role assignment is currently managed at the database level by a system administrator.

### Availability Status

The Assign Responder dropdown shows each user's live availability:

| Status | Meaning |
|---|---|
| 🟢 **Available** | User is not assigned to any currently active (OPEN or RESPONDING) incident |
| 🟡 **Responding** | User's name appears as an assigned responder on at least one active incident |

Availability refreshes every time the Response Center is loaded or manually refreshed.

---

## 14. Frequently Asked Questions

**Q: The camera feed shows a "WARMING UP" overlay — is it broken?**  
A: No. The AI background model needs a short calibration period (up to ~30 seconds) at startup or after switching video sources. The overlay disappears automatically once calibration is complete and counts become reliable.

**Q: I filed a report but it doesn't appear in the Response Center.**  
A: Click the **REFRESH** button in the Response Center, or switch away from and back to the tab to trigger a reload.

**Q: Why does the Incident Response Center show "Failed to load incidents"?**  
A: This typically means the server returned an error. Check that the Flask server (`app.py`) is running. If the problem persists, restart the server and refresh the browser.

**Q: I assigned a responder but the incident still shows as OPEN.**  
A: An incident moves to RESPONDING automatically when the **first** responder is successfully assigned. Confirm the assignment completed without an error alert.

**Q: Can I edit a filed incident report?**  
A: Status and resolution notes can be updated at any time via the Response Center. The original clip, title, and report notes cannot be edited after filing.

**Q: What happens to dismissed clips?**  
A: Dismissed clips are permanently deleted from both the tray and the server's Temp_Clips folder. They cannot be recovered.

**Q: Who can see other users' incident reports?**  
A: All logged-in users can see all incidents. There is no per-user access restriction in the current version.

---

## 15. Changelog

> This section is updated automatically whenever the system receives changes.

### May 14, 2026

- **Incident Response Center — Assign Responder redesigned**  
  Free-text name input replaced with a dropdown listing all registered system users. Each entry shows the user's name, role, and live availability (🟢 Available / 🟡 Responding). Availability is determined by whether the user is already assigned to an active incident.

- **Activity Logs — Filter dropdowns removed**  
  FILTER BY CAMERA and FILTER BY TYPE dropdowns removed from the Activity Logs sidebar. All log types are now shown together in a single live stream.

- **Activity Logs — Low density logs now green**  
  Log cards of type NORMAL (low crowd density) now display with a green background (matching ENTRY events) instead of the previous dark-slate color, making it easy to distinguish safe-state logs from alerts.

- **Incident Response Center — RESPONDING tab crash fixed**  
  Fixed a bug where opening the RESPONDING tab showed "Failed to load incidents." The cause was a field name mismatch (`r.name` vs `r.responder_name`) in the card renderer that threw a silent JavaScript exception.

- **Dark / Light mode toggle added**  
  Theme toggle button (moon/sun icon) added to the main navigation bar beside the BARANGAY SENTINEL brand title. Preference saved to localStorage.

- **Language switcher added**  
  English / Filipino (Tagalog) language toggle added to the login/register page. All labels, buttons, and placeholders switch language. Preference saved to localStorage.

- **Dashboard stat cards made functional**  
  CAMERAS ONLINE, ALERTS TODAY, STORAGE USED, and UPTIME cards now show live data pulled from the server every 30 seconds. Storage and camera cards color-code based on thresholds.

- **Incident Archive — Status timeline and responder history**  
  Archive cards now show a 5-step status timeline (PUBLISHED → OPEN → RESPONDING → RESOLVED → CLOSED) with timestamps and actor names for each step. Assigned responders are also listed on closed incidents.

- **Delete Incident feature**  
  REQUEST DELETION flow replaced with an immediate DELETE INCIDENT button under a DANGER ZONE section. Requires confirmation before permanently removing the report, clip, and thumbnail.

- **Responder assignment and status management**  
  Added ability to assign barangay officials to incidents, remove them, and update incident status (OPEN / RESPONDING / RESOLVED / CLOSED) with resolution notes. Status changes are logged with timestamp and username.
