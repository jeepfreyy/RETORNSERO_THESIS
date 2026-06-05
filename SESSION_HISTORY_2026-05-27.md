# Session History — 2026-05-27

**Project:** Barangay Sentinel Advanced Surveillance System  
**Branch:** `master` (local) → `origin/main` (GitHub)  
**Session date:** May 27, 2026  
**Continued from:** SESSION_HISTORY_2026-05-26.md (previous session)

---

## Table of Contents

1. [Incident Response Center Simplification](#1-incident-response-center-simplification)
2. [Mobile Application — Architecture Plan](#2-mobile-application--architecture-plan)
3. [Mobile Application — Full Implementation](#3-mobile-application--full-implementation)
4. [Mobile App — PowerShell Execution Policy Fix](#4-mobile-app--powershell-execution-policy-fix)
5. [Mobile App — SDK Version Mismatch (52 → 54)](#5-mobile-app--sdk-version-mismatch-52--54)
6. [Mobile App — react-native-worklets Missing](#6-mobile-app--react-native-worklets-missing)
7. [Mobile App — Connection Error (Wrong IP & Port)](#7-mobile-app--connection-error-wrong-ip--port)
8. [Mobile App — Live Feed Attempt 1 (WebView MJPEG → JPEG polling)](#8-mobile-app--live-feed-attempt-1-webview-mjpeg--jpeg-polling)
9. [Mobile App — Live Feed Attempt 2 (Native Image component — final)](#9-mobile-app--live-feed-attempt-2-native-image-component--final)
10. [Files Changed This Session](#10-files-changed-this-session)
11. [How to Run the System](#11-how-to-run-the-system)

---

## 1. Incident Response Center Simplification

### Problem
User showed a screenshot of the expanded Incident Response Center (IRC) card saying it was "too complex for a beginner to use." The previous design had:
- 5-column left/right split layout
- ALL CAPS labels everywhere (bureaucratic feel)
- 7 separate metadata stat boxes (Camera, Density, People, TL, Timestamp, Location, Reporter)
- Two confusing buttons: "TAKE THIS INCIDENT" + "ASSIGN OFFICIAL"
- A "DANGER ZONE" red banner for the delete button

### Changes Made

**File:** `templates/index.html` — `_rcCard()` function completely rewritten

| Before | After |
|---|---|
| 5-column grid left/right split | Single vertical column flow |
| ALL CAPS section labels | Plain English labels with emoji (👮 🔄) |
| 7 separate metadata boxes | One compact info line: 📍 location · 🕒 time · 👥 people |
| "DANGER ZONE" red banner | Quiet "Delete Incident" link at the bottom |
| "ASSIGN OFFICIAL" / "TAKE THIS INCIDENT" buttons | "✋ I'll Take This" + "Assign Selected User" with one-line explanation |
| "APPLY" status button | "Save" — plain and obvious |
| Complex grid layout | Flex row thumbnail + info, then stacked sections below |

### New card structure
```
[Collapsed header — status badge, density, TL, title, time]
  ↓ click to expand
[Expanded]
  Row 1: thumbnail (left) + title, info line, operator notes (right)
  ──────────────────────────────────────
  👮 Assign a Responder
    [responder list]
    [user picker dropdown]
    [optional note field]
    [✋ I'll Take This]   [Assign Selected User]
    explanation text beneath
  ──────────────────────────────────────
  🔄 Update Incident Status
    [status select]  [Save]
    [resolution note textarea]
  ──────────────────────────────────────
  [Delete Incident]  ← quiet link, not alarming
```

All JavaScript functions (`assignResponder`, `takeIncident`, `removeResponder`, `updateIncidentStatus`, `openDeleteConfirm`, user picker dropdown) remained unchanged — only the HTML template changed.

---

## 2. Mobile Application — Architecture Plan

### Architecture overview

Full architecture produced for a React Native + Expo mobile companion app:

- **Tech stack:** React Native + Expo SDK 54, TypeScript, Expo Router, Axios, Zustand, expo-secure-store, expo-av, react-native-webview
- **Auth strategy:** New `POST /api/mobile/login` endpoint returns Bearer token stored in SecureStore; auto-injected via Axios interceptor on every request
- **Screen map:** Login → (tabs) Home, Incidents, Live Feed, Profile + Incident Detail (dynamic route)
- **Backend reuse:** 100% of core data endpoints already existed; only 2 new endpoints needed (`/api/mobile/login`, `/api/mobile/logout`)
- **Push notifications:** Planned via Firebase FCM + `MobileFCMToken` DB table (planned, not yet implemented)

### Development phases planned

| Phase | What | Estimate |
|---|---|---|
| 0 | Expo setup, Axios pointing to Flask | 1 day |
| 1 | Login + session persistence, Home stats polling | 2 days |
| 2 | Incidents list + Incident Detail | 2–3 days |
| 3 | Live Feed screen | 1 day |
| 4 | FCM push notifications | 2 days |
| 5 | Polish — icons, error states | 1–2 days |

---

## 3. Mobile Application — Full Implementation

### Backend changes (`app.py`)

| Change | Detail |
|---|---|
| `from flask import ... g` | Added Flask `g` for request-scoped user context |
| `_mobile_tokens: dict` | In-memory Bearer token store `{token_hex: {user_id, username, role}}` |
| `_current_username()` helper | Returns username from session OR `g` (set by mobile auth) |
| Modified `login_required` | Now accepts `Authorization: Bearer <token>` alongside existing session cookie |
| Updated 3 `session.get('username')` calls | Changed to `_current_username()` so mobile actions are attributed correctly in logs |
| `POST /api/mobile/login` | Email + password → returns `{success, token, username, role}` |
| `POST /api/mobile/logout` | Invalidates the token from `_mobile_tokens` dict |
| `host='0.0.0.0'` on `app.run()` | Makes Flask reachable from LAN devices (phone) not just localhost |

### Mobile app — 20 files created (`mobile/`)

```
mobile/
├── package.json              Expo SDK 54 dependency manifest
├── app.json                  Expo config (sdkVersion, scheme, plugins)
├── tsconfig.json             TypeScript config (strict mode)
├── babel.config.js           babel-preset-expo
├── .npmrc                    legacy-peer-deps=true (avoids install flags)
├── .gitignore
├── lib/
│   ├── constants.ts          SERVER_IP, SERVER_PORT, BASE_URL, density colors
│   └── api.ts                Axios client + typed helpers (apiLogin, apiIncidents, etc.)
├── store/
│   └── useAppStore.ts        Zustand store — token, username, role; persist via SecureStore
├── hooks/
│   ├── useStats.ts           Polls /api/stats every 5 seconds
│   └── useIncidents.ts       Fetches incident list on demand
├── components/
│   ├── DensityBadge.tsx      Color-coded LOW/MEDIUM/HIGH chip component
│   └── IncidentCard.tsx      Reusable incident list card (tappable → detail screen)
├── app/
│   ├── _layout.tsx           Root layout + auth guard (redirects to login if no token)
│   ├── (auth)/
│   │   └── index.tsx         Login screen — email + password → POST /api/mobile/login
│   ├── (tabs)/
│   │   ├── _layout.tsx       Bottom tab bar (Home, Incidents, Live Feed, Profile)
│   │   ├── index.tsx         Home: density card + active incidents list (top 3)
│   │   ├── incidents.tsx     Filterable incident list: OPEN/RESPONDING/RESOLVED/ALL
│   │   ├── live.tsx          Live feed (see sections 8 and 9 for evolution)
│   │   └── profile.tsx       Username, role, server info, sign out
│   └── incident/
│       └── [id].tsx          Incident detail: thumbnail, responders, "I'll Take This",
│                             update status modal (slide-up sheet)
└── assets/
    └── README.txt            Icon/splash file requirements for production build
```

### Key design decisions
- **Same credentials as web** — no separate mobile account; email + password identical
- **Token in SecureStore** — survives app restarts; only cleared on explicit logout
- **All core API calls reuse existing Flask endpoints** — no data duplication
- **Incident status update** — slide-up modal sheet with colored option list + resolution note field
- **Auth guard in root layout** — unauthenticated users auto-redirect to login; authenticated users auto-redirect to tabs

---

## 4. Mobile App — PowerShell Execution Policy Fix

### Error
```
npx : File C:\Program Files\nodejs\npx.ps1 cannot be loaded because running scripts is
disabled on this system. PSSecurityException: UnauthorizedAccess
```

### Root cause
Windows PowerShell blocks `.ps1` scripts (which `npx` is) by default on machines with Group Policy enforcement.

### Fix options
**Option A — Use CMD.exe (always works):**
```cmd
cd E:\RETORNSERO_THESIS-main\mobile
npx expo start
```

**Option B — Fix for current PowerShell session:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force
npx expo start
```

### npm dependency issues also encountered and fixed

| Issue | Fix |
|---|---|
| `react@18.3.2` doesn't exist on npm | Changed to `react@18.3.1` |
| `react-native@0.76.9` peer dep conflict | Added `--legacy-peer-deps` flag |
| Future `npm install` needing the flag every time | Created `.npmrc` with `legacy-peer-deps=true` |

---

## 5. Mobile App — SDK Version Mismatch (52 → 54)

### Error on phone (Expo Go)
```
ERROR  Project is incompatible with this version of Expo Go
• The installed version of Expo Go is for SDK 54.0.0.
• The project you opened uses SDK 52.
```

### Root cause
Project was initially scaffolded with Expo SDK 52 versions. The phone's Expo Go app is SDK 54.

### Fix
Ran `node node_modules/expo/bin/cli install --fix` — Expo auto-updated `package.json` to correct SDK 54 versions, then `npm install --legacy-peer-deps` to finish.

**Final SDK 54.0.34 versions:**

| Package | Version |
|---|---|
| expo | ~54.0.34 |
| react | 19.1.0 |
| react-native | 0.81.5 |
| expo-router | ~6.0.23 |
| expo-av | ~16.0.0 |
| expo-notifications | ~0.32.17 |
| expo-secure-store | ~15.0.8 |
| expo-status-bar | ~3.0.9 |
| expo-system-ui | ~6.0.9 |
| react-native-reanimated | ~4.1.1 |
| react-native-safe-area-context | ~5.6.0 |
| react-native-screens | ~4.16.0 |
| react-native-webview | 13.15.0 |
| @expo/vector-icons | ^15.0.3 |

**`app.json` also updated:**
- Added `"sdkVersion": "54.0.0"` 
- Added `"newArchEnabled": true`
- Removed icon/splash/notification-icon asset references (caused crash when image files were missing)

---

## 6. Mobile App — react-native-worklets Missing

### Error on phone (Expo Go)
```
node_modules\expo-router\entry.js: [BABEL]
Cannot find module 'react-native-worklets/plugin'
Require stack: react-native-reanimated\plugin\index.js
```

### Root cause
`react-native-reanimated@4.x` was split into two separate packages. The animation engine is in `react-native-reanimated`, but its Babel plugin now depends on a separate `react-native-worklets` package (`peerDependency: >=0.5.0`). This package was not listed as a direct dependency and so npm never installed it.

### Fix
```bash
npm install react-native-worklets@0.8.3 --legacy-peer-deps
```

Added to `package.json`:
```json
"react-native-worklets": "^0.8.3"
```

Then restarted Expo with `--clear` to force Metro to re-run Babel compilation:
```cmd
npx expo start --clear
```

---

## 7. Mobile App — Connection Error (Wrong IP & Port)

### Error on phone
```
Connection Error
timeout of 12000ms exceeded
Make sure the server is running and SERVER_IP is correct in lib/constants.ts.
```

### Root causes (3 separate problems)

| Problem | Detail |
|---|---|
| Wrong `SERVER_IP` | `constants.ts` still had placeholder `192.168.1.100` |
| Wrong `SERVER_PORT` | `constants.ts` had `5000` but Flask actually runs on `5001` |
| Flask localhost-only | `app.run(port=5001)` defaults to `127.0.0.1` — mobile devices on the LAN cannot reach it |

### PC network info (discovered via `ipconfig` / PowerShell)
- PC Ethernet adapter IP: `192.168.254.110`
- No Wi-Fi adapter active (PC is wired)
- Phone on Wi-Fi connects to same router → same `192.168.254.x` subnet

### Fixes applied

**`app.py` — bind Flask to all interfaces:**
```python
# Before:
app.run(debug=is_debug, port=5001)

# After:
app.run(debug=is_debug, host='0.0.0.0', port=5001)
```

**`mobile/lib/constants.ts` — correct IP and port:**
```typescript
export const SERVER_IP   = '192.168.254.110';  // PC's actual Ethernet IP
export const SERVER_PORT = '5001';             // Flask's actual port
```

**Windows Firewall — opened port 5001 inbound:**
```powershell
New-NetFirewallRule -DisplayName "Barangay Sentinel Flask (5001)" `
  -Direction Inbound -Protocol TCP -LocalPort 5001 -Action Allow
```

---

## 8. Mobile App — Live Feed Attempt 1 (WebView MJPEG → JPEG polling)

> ⚠️ This approach was superseded by Attempt 2. Documented here for completeness.

### Problem
Live Feed screen showed "Live feed unavailable" even though:
- Web dashboard showed video correctly
- Stats header showed "5 people" (proving network + auth were working)

### Root cause identified
**iOS WebKit does not support MJPEG streams** (`multipart/x-mixed-replace` content type). The original implementation used:
```html
<img src="http://192.168.254.110:5001/video_feed" />
```
This works on Android but silently fires `onerror` immediately on iOS.

### First fix applied

**Strategy:** Keep WebView but poll `/cam1_frame` (single JPEG) every 150ms via JavaScript.  
**Auth problem:** WebView `<img>` tags can't send `Authorization` headers, so the token was passed as `?token=` query param.

**`app.py` — extended `login_required` to accept `?token=` query param:**
```python
query_token = request.args.get('token', '').strip()
if query_token:
    info = _mobile_tokens.get(query_token)
    if info:
        g.user_id  = info['user_id']
        g.username = info['username']
        return f(*args, **kwargs)
```

**`live.tsx`** — rewrote to inject polling JavaScript into WebView inline HTML.

### Why it still failed
After applying this fix, the live feed still did not appear. Likely causes:
1. Flask was not restarted so the `?token=` code was not active
2. iOS WebView blocks HTTP (`http://`) img requests loaded from inline HTML due to App Transport Security (ATS)
3. WebView `about:blank` origin may restrict cross-origin requests differently than a native HTTP call

---

## 9. Mobile App — Live Feed Attempt 2 (Native Image component — final)

### Problem
After the WebView JPEG polling fix (Attempt 1), the live feed still showed "Live feed unavailable." The user reloaded "countless times" with no success.

### Root cause (final diagnosis)
iOS WebView has stricter HTTP request handling when the WebView is loaded from inline HTML (null/about:blank origin). iOS App Transport Security blocks or degrades `http://` img requests in that context. Additionally, the `?token=` Flask change required Flask to be restarted — a step that may not have occurred.

### Final fix — remove WebView entirely, use native RN `Image`

**Key insight:** React Native's built-in `Image` component supports a `headers` object in the `source` prop. It makes native HTTP requests (not browser requests), which:
- Are unaffected by iOS ATS/WebView restrictions
- Send proper `Authorization: Bearer TOKEN` headers (no `?token=` hack needed)
- Work identically to how `/api/stats` and `/api/incidents` already work

**`live.tsx` — completely rewritten, no WebView at all:**

```typescript
// Poll /cam1_frame every 200ms using native RN Image
const tick = useCallback(() => {
  setFrameUri(`${BASE_URL}/cam1_frame?t=${Date.now()}`);
}, []);

// Image renders with Authorization header — no Flask changes needed
<Image
  source={{
    uri: frameUri,
    headers: { Authorization: `Bearer ${token}` },
    cache: 'reload',  // forces network fetch every time
  }}
  style={styles.frameImage}
  resizeMode="contain"
  onLoad={onFrameLoad}      // → sets hasFrame=true, resets errorStreak
  onError={onFrameError}    // → increments errorStreak
  fadeDuration={0}          // Android: no fade between frames
/>
```

**Polling behavior:**
- Normal: every 200ms (~5fps) — smooth for surveillance monitoring
- After 5 consecutive errors: slow to 2s retry  
- Auto-recovers to fast mode when a frame succeeds after error

**UI states:**
- `!hasFrame && !showError` → loading spinner + contextual text ("warming up" vs "connecting")
- `showError` → error message + "Retry Now" button  
- `hasFrame` → frame fills the screen; header shows density badge + people count

### Comparison: Attempt 1 vs Attempt 2

| | Attempt 1 (WebView) | Attempt 2 (Native Image) |
|---|---|---|
| Approach | WebView + inline HTML + JS polling | Native RN Image polling |
| iOS HTTP | Blocked by ATS from inline HTML | Works natively |
| Auth | `?token=` query param (needed Flask restart) | `Authorization: Bearer` header (already worked) |
| Cache bust | `?t=Date.now()` in JS | `?t=Date.now()` + `cache:'reload'` |
| Flask changes needed | Yes — `?token=` in `login_required` | No — existing auth unchanged |
| Result | Still failing | ✅ Should work |

---

## 10. Files Changed This Session

| File | Change |
|---|---|
| `templates/index.html` | `_rcCard()` fully rewritten — simpler, plain English, single-column layout |
| `app.py` | Added `g` import; `_mobile_tokens`; `_current_username()`; modified `login_required` to accept Bearer token + `?token=` query param; added `/api/mobile/login`, `/api/mobile/logout`; changed `host='0.0.0.0'` on `app.run()` |
| `mobile/package.json` | SDK 52 → SDK 54.0.34; all package versions corrected; added `react-native-worklets` |
| `mobile/app.json` | Added `sdkVersion:"54.0.0"`, `newArchEnabled:true`; removed missing asset refs |
| `mobile/lib/constants.ts` | `SERVER_IP` → `192.168.254.110`; `SERVER_PORT` → `5001` |
| `mobile/app/(tabs)/live.tsx` | Rewritten twice — WebView MJPEG → WebView JPEG polling → Native Image polling (final) |
| `mobile/.npmrc` | Created with `legacy-peer-deps=true` |
| *(new)* `mobile/` directory | Entire mobile app: 20 files, ~1,100 lines of TypeScript |
| `SESSION_HISTORY_2026-05-27.md` | This file |

---

## 11. How to Run the System

### Flask backend
```cmd
cd E:\RETORNSERO_THESIS-main
python app.py
```
Confirm output: `* Running on http://0.0.0.0:5001`

### Mobile app (use CMD, not PowerShell)
```cmd
cd E:\RETORNSERO_THESIS-main\mobile
npx expo start --clear
```
Scan the QR code with **Expo Go** on your phone.  
Log in with the **same email and password** as the web dashboard.

### If live feed doesn't appear
1. Stop and restart Flask (`python app.py`) — confirms `host='0.0.0.0'` is active
2. Restart Expo with `--clear` flag
3. Log out and log back in on the phone (refreshes the Zustand token)
4. Tap `↻` in the top-right of the Live Feed tab to force reload
5. Confirm phone Wi-Fi IP is `192.168.254.x` (same subnet as PC)
6. Test in phone browser: `http://192.168.254.110:5001/` — web dashboard should load

### Troubleshooting reference

| Symptom | Likely cause | Fix |
|---|---|---|
| `npx` blocked in PowerShell | Execution policy | Use CMD instead, or run `Set-ExecutionPolicy Bypass -Scope Process` |
| SDK mismatch error in Expo Go | `package.json` versions wrong | Run `node node_modules/expo/bin/cli install --fix` then `npm install --legacy-peer-deps` |
| `react-native-worklets/plugin` error | Missing peer dependency | `npm install react-native-worklets@0.8.3 --legacy-peer-deps` |
| Connection timeout on login | Wrong IP/port or Flask localhost-only | Check `SERVER_IP`/`SERVER_PORT` in `constants.ts`; restart Flask with `host='0.0.0.0'` |
| Live feed shows "unavailable" | Native Image auth or network | Restart Flask, restart Expo `--clear`, log out and back in |
