# Barangay Sentinel — Session History

**Date:** 2026-06-05  
**Project:** RETORNSERO_THESIS-main (Barangay Sentinel CCTV + Mobile Incident Response)  
**Stack:** Flask + SQLAlchemy (backend) · Tailwind CSS (web) · React Native + Expo Router (mobile)

---

## 1. Web — Incident Clip Video Player in Response Center

### Problem
The Incident Response Center showed only a static `<img>` thumbnail per incident. The operator could not play the recorded video clip.

### Files Changed
- `templates/index.html`

### What Was Done
- Replaced the static thumbnail section in `_rcCard()` with a **clickable thumbnail overlay** showing a green play button and "PLAY CLIP" label.
- Added a **hidden full-width `<video controls>` element** (`rc-clip-player-{id}`) that appears below the info row when the thumbnail is clicked.
- Thumbnail hides itself when the player is visible; a "✕ Close" button stops playback, unloads the file handle, and restores the thumbnail.
- Added `toggleRcClip(incId)` JavaScript function — reads the clip URL from the thumbnail wrapper's `data-clip-url` attribute, lazy-loads it, and auto-plays.
- Incidents with no clip retain the "No clip available" placeholder unchanged.

---

## 2. Mobile — Incident Clip URL Fix

### Problem
The mobile app constructed the clip URL as `BASE_URL + /clips/ + clip_filename`, but `/clips/` only serves **temp clips** from `Temp_Clips/`. Archived incident clips live under `Archive/` and must be fetched via `/archive_media/...`. Videos never loaded on mobile.

### Files Changed
- `mobile/lib/api.ts`
- `mobile/app/incident/[id].tsx`

### What Was Done
- Added `clip_url: string | null` to the `Incident` interface (the server already returns it).
- Changed `clip_filename` field to `string | null` (nullable, reflecting reality).
- Mobile now uses `BASE_URL + incident.clip_url` (correct `/archive_media/` path) instead of the broken `/clips/` construction.
- `hasClip` simplified to `!!clipUrl` (no longer checks `clip_filename`).
- `Authorization: Bearer <token>` header already passed through `expo-av`'s `source.headers`, so auth works on `/archive_media/` too.

---

## 3. Web — Declined Requests Tab in Response Center

### Problem
Operators had no visibility into which mobile users had declined assignment requests or what reasons they gave. All declined entries were invisible from the web dashboard.

### Files Changed
- `app.py`
- `templates/index.html`

### What Was Done

**Backend (`app.py`):**
- Added `GET /api/assignments/declined` — returns all `AssignmentNotification` records with `status='declined'`, ordered newest-first, with incident summary fields (title, status, density, threat level, location) and `decline_reason`.
- Added `GET /api/assignments` — returns all assignment notifications regardless of status (for future use / full history).

**Frontend (`templates/index.html`):**
- Converted the Response Center header into a **sub-tab layout**: `INCIDENTS` and `DECLINED REQUESTS`.
- Added `switchRcTab(tab)` function to toggle between the two panels.
- Added `rcRefresh()` function — context-aware, refreshes incidents or declined requests depending on the active tab.
- Added `loadDeclinedAssignments()` and `renderDeclinedAssignments(rows)` functions.
- Added `_declineCard(r)` template function — renders each declined assignment as a card showing: responder avatar + name + `DECLINED` badge, who assigned them + when sent, incident summary strip (status/density/TL/title/location/ID), and the **decline reason** in a red-tinted quote block.
- Added `_updateDeclineBadge(count)` — shows a red counter on the DECLINED REQUESTS tab button.
- On opening the Response Center, pre-fetches the decline count so the badge appears immediately.
- Added `info` (amber) variant to `_showToast()` alongside the existing `success` (green) and `error` (red).

---

## 4. Pending Assignment Flow Redesign

### Problem
When the operator assigned a mobile user, that user **immediately appeared as a confirmed responder** and the incident status changed to RESPONDING — before the user had even seen the request. When the user declined, the backend correctly removed them from `IncidentResponder`, but the web view was stale and still showed them as assigned.

### Files Changed
- `app.py`
- `templates/index.html`
- `mobile/lib/api.ts`
- `mobile/app/incident/[id].tsx`

### What Was Done

**Backend (`app.py`):**

`add_responder()` — complete redesign with two paths:
- **Registered mobile user assigned by operator:** only creates `AssignmentNotification(pending)`. No `IncidentResponder` created yet. Incident status stays `OPEN`. Returns `{ pending: true }`.
- **Self-assignment or unregistered user:** direct `IncidentResponder` creation, incident advances `OPEN → RESPONDING`. Returns `{ pending: false }`.
- Guards against duplicate pending notifications (returns `409` if one already exists).

`mobile_respond_notification()` — **accept path extended:**
- On `accept`: now creates `IncidentResponder` (using the user's role from the `User` table) and advances `OPEN → RESPONDING` if no other confirmed responders exist.
- On `decline`: defensive cleanup retained; removes any legacy `IncidentResponder` if present.

`list_incidents()` — new field:
- Each incident now includes `pending_assignments: [{ id, assigned_to_name, assigned_by, created_at }]`.

New endpoint `DELETE /api/incidents/{id}/assignments/{notif_id}`:
- Operator can cancel a `pending` assignment notification before the user responds.

`remove_responder()` — extended:
- Now logs an `OPEN` status change when reverting from `RESPONDING`.

**Frontend (`templates/index.html`):**

Responder section in `_rcCard()`:
- **Confirmed responders** (from `IncidentResponder`) → green `CONFIRMED` badge.
- **Pending assignments** (from `pending_assignments`) → amber blinking `AWAITING RESPONSE` badge + `Cancel` button calling `cancelPendingAssignment()`.
- Collapsed header shows `2 confirmed · 1 pending` instead of a raw count.
- "Assign Selected User" help text updated to explain the pending-approval flow.

`assignResponder()`:
- Handles `{ pending: true }` → amber "Request Sent" info toast.
- Handles `{ pending: false }` → green "Assigned" toast.
- Shows error toast on `409` (duplicate pending) instead of `alert()`.

`cancelPendingAssignment(incId, notifId)` — new function:
- Calls `DELETE /api/incidents/{id}/assignments/{notifId}`, reloads the Response Center.

**Mobile (`mobile/lib/api.ts`):**
- Added `PendingAssignment` interface `{ id, assigned_to_name, assigned_by, created_at }`.
- Added `pending_assignments: PendingAssignment[]` to `Incident` interface.

**Mobile (`mobile/app/incident/[id].tsx`):**
- Confirmed responders display with a green `CONFIRMED` badge.
- Pending assignments render as a separate row with an amber `PENDING` badge.
- `hasPendingForMe` — if the current user has a pending assignment for this incident, the "I'll Take This" button is hidden and replaced with a `⏳ You have a pending assignment request` info box.
- New styles: `confirmedBadge`, `confirmedBadgeText`, `pendingRow`, `pendingAvatar`, `pendingInitial`, `pendingBadge`, `pendingBadgeText`, `pendingForMeBox`, `pendingForMeText`.

---

## 5. Mobile — Assignment Modal Submit Button Fix

### Problem
The red "Confirm Decline" button was invisible — it used `btnDeclineText` style which has `color: '#ef4444'` (red text on a red background). The label was also unclear ("Confirm Decline").

### Files Changed
- `mobile/components/AssignmentModal.tsx`
- `mobile/i18n/index.ts`

### What Was Done
- Added `btnSubmitText: { color: '#fff', fontSize: 15, fontWeight: '800' }` to the style factory.
- Changed the Confirm Decline button to use `s.btnSubmitText` instead of `s.btnDeclineText`.
- Updated translation key `modal.confirmDec`:
  - English: `'Confirm Decline'` → `'Submit'`
  - Tagalog: `'Kumpirmahin ang Pagtanggi'` → `'Isumite'`

---

## 6. Mobile — Status Bar / Safe Area Overlap Fix

### Problem
The greeting "Hello, [Name] 👋" on the Home screen overlapped the iPhone status bar (notification bar). All tab screens used a hardcoded `paddingTop: 56` as a guess for the status bar height — incorrect across device families. The Home screen had zero top padding.

`react-native-safe-area-context` was installed (`~5.6.0`) but **never imported anywhere** in the actual app code.

### Files Changed
- `mobile/app/(tabs)/index.tsx`
- `mobile/app/(tabs)/incidents.tsx`
- `mobile/app/(tabs)/tasks.tsx`
- `mobile/app/(tabs)/live.tsx`
- `mobile/app/(tabs)/profile.tsx`
- `mobile/app/incident/[id].tsx`

### What Was Done
Added `useSafeAreaInsets` from `react-native-safe-area-context` to all 6 screens. Replaced every hardcoded `paddingTop: 56` in `makeStyles()` with a dynamic inline style override using `insets.top + N`:

| Screen | Previous | Fixed |
|---|---|---|
| `(tabs)/index.tsx` | `padding: 20` (no top) | `paddingTop: insets.top + 16` on ScrollView `contentContainerStyle` |
| `(tabs)/incidents.tsx` | `paddingTop: 56` | `paddingTop: insets.top + 16` on header View |
| `(tabs)/tasks.tsx` | `paddingTop: 56` | `paddingTop: insets.top + 16` on header View |
| `(tabs)/live.tsx` | `paddingTop: 56` | `paddingTop: insets.top + 12` on header View |
| `(tabs)/profile.tsx` | `paddingTop: 56` | `paddingTop: insets.top + 16` on header View |
| `incident/[id].tsx` | `paddingTop: 56` | `paddingTop: insets.top + 8` on navbar View |

`insets.top` returns the exact pixel height of the status bar for the current device at runtime — handles all iPhone form factors (SE, standard, notch, Dynamic Island) correctly.

---

## Summary of All Files Modified

### Backend
| File | Changes |
|---|---|
| `app.py` | `add_responder()` redesigned (pending flow); `mobile_respond_notification()` creates `IncidentResponder` on accept; `list_incidents()` adds `pending_assignments`; `remove_responder()` logs status; new `cancel_pending_assignment()` endpoint; new `get_declined_assignments()` and `get_all_assignments()` endpoints |

### Web Frontend
| File | Changes |
|---|---|
| `templates/index.html` | Response Center: inline video player, sub-tabs (INCIDENTS / DECLINED REQUESTS), decline badge, `_declineCard()`, `toggleRcClip()`, `cancelPendingAssignment()`, `assignResponder()` updated, `_showToast()` `info` variant, confirmed/pending responder rows |

### Mobile
| File | Changes |
|---|---|
| `mobile/lib/api.ts` | `Incident.clip_url` added, `clip_filename` nullable, `PendingAssignment` interface, `pending_assignments` field |
| `mobile/app/(tabs)/index.tsx` | `useSafeAreaInsets`, dynamic `paddingTop` |
| `mobile/app/(tabs)/incidents.tsx` | `useSafeAreaInsets`, dynamic `paddingTop` |
| `mobile/app/(tabs)/tasks.tsx` | `useSafeAreaInsets`, dynamic `paddingTop` |
| `mobile/app/(tabs)/live.tsx` | `useSafeAreaInsets`, dynamic `paddingTop` |
| `mobile/app/(tabs)/profile.tsx` | `useSafeAreaInsets`, dynamic `paddingTop` |
| `mobile/app/incident/[id].tsx` | `useSafeAreaInsets`, dynamic `paddingTop`; `clip_url` usage; `pending_assignments` display; `hasPendingForMe`; `CONFIRMED`/`PENDING` badges; new styles |
| `mobile/components/AssignmentModal.tsx` | `btnSubmitText` style (white text); Confirm Decline button uses new style |
| `mobile/i18n/index.ts` | `modal.confirmDec` → `'Submit'` (EN) / `'Isumite'` (TL) |
