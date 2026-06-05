# Barangay Sentinel — Mobile App Setup Guide

This is the companion mobile app for the Barangay Sentinel system.
It lets Barangay Tanods view live density alerts, manage incidents,
and update their status from the field — directly on their phone.

---

## Prerequisites

Install these on your **development PC** before you begin:

| Tool | Download |
|------|---------- |
| Node.js 18+ | https://nodejs.org (LTS version) |
| npm (included with Node) | — |
| Expo Go app (on your phone) | Search "Expo Go" in the App Store or Google Play |

You do **not** need Android Studio or Xcode for development.
Expo Go runs the app directly on your physical phone.

---

## Step 1 — Find your Flask server's IP address

Your phone and PC must be on the **same Wi-Fi network**.

**On Windows:**
1. Open Command Prompt (`Win + R` → type `cmd` → Enter)
2. Type: `ipconfig`
3. Look for **"IPv4 Address"** under your Wi-Fi adapter
   - Example: `192.168.1.105`

---

## Step 2 — Configure the server address

Open this file in a text editor:

```
mobile/lib/constants.ts
```

Find this line near the top:

```typescript
export const SERVER_IP = '192.168.1.100';   // ← CHANGE THIS
```

Replace `192.168.1.100` with your PC's actual IPv4 address.
Example:

```typescript
export const SERVER_IP = '192.168.1.105';
```

Save the file.

---

## Step 3 — Install dependencies

Open a terminal, navigate to the `mobile/` folder, and run:

```bash
cd mobile
npm install
```

This installs all packages (Expo, React Native, Axios, etc.).
It may take 1–3 minutes on first run.

---

## Step 4 — Start the Flask server

In a **separate terminal**, start the backend:

```bash
cd ..           # back to project root (RETORNSERO_THESIS-main)
python app.py
```

Make sure it says something like:
```
 * Running on http://0.0.0.0:5000
```

> ⚠️ The server must be running before you open the app.

---

## Step 5 — Start the Expo development server

Back in the `mobile/` terminal, run:

```bash
npx expo start
```

You will see a **QR code** in the terminal.

---

## Step 6 — Open on your phone

1. On **Android**: open the **Expo Go** app → tap **"Scan QR Code"**
2. On **iPhone**: open the built-in **Camera app** → point at the QR code → tap the notification

The app will load on your phone in about 30–60 seconds.

---

## Logging In

Use the **same email and password** you use on the web dashboard.

- The login uses a separate mobile endpoint (`/api/mobile/login`) that
  returns a Bearer token stored securely on your device.
- Your session persists across app restarts — you only log in once.

---

## Screens

| Screen | What it does |
|--------|-------------|
| **Home** | Shows current crowd density and active incidents. Pull down to refresh. |
| **Incidents** | Full list of all incidents, filterable by status (OPEN / RESPONDING / RESOLVED). Tap any card to open the detail view. |
| **Live Feed** | Embeds the `/video_feed` MJPEG stream from the server. Phone must be on the same Wi-Fi. |
| **Profile** | Shows your username, role, and server info. Sign out from here. |
| **Incident Detail** | View clip thumbnail, responders, assign yourself, and update the status. |

---

## Troubleshooting

### "Cannot reach server" on login
- Make sure `SERVER_IP` in `lib/constants.ts` matches your PC's IPv4 address (not `127.0.0.1` or `localhost`)
- Make sure the Flask server is running (`python app.py`)
- Make sure your phone and PC are on the same Wi-Fi network
- Temporarily disable Windows Firewall or allow port 5000

### Live feed shows "Live feed unavailable"
- Same as above — same network, server running, correct IP
- The MJPEG stream at `/video_feed` requires the vision engine to be active

### QR code scan fails / "Network response timed out"
- Make sure your phone's Wi-Fi is connected (not mobile data)
- Try running `npx expo start --tunnel` instead (uses a cloud tunnel — slower but works across networks)

### `npm install` fails
- Make sure you are inside the `mobile/` folder, not the root
- Try deleting `node_modules/` and running `npm install` again

---

## Running on Android Emulator (optional)

If you prefer a phone emulator instead of a physical device:
1. Install Android Studio
2. Create an AVD (Android Virtual Device) in AVD Manager
3. Start the emulator
4. In the Expo terminal, press `a` to open in the Android emulator

---

## Building a standalone APK (for distribution)

This requires an Expo account (free):

```bash
npm install -g eas-cli
eas login
eas build --platform android --profile preview
```

This produces an APK you can install on any Android phone without Expo Go.

---

## File Structure

```
mobile/
├── app/
│   ├── _layout.tsx              Root layout + auth guard
│   ├── (auth)/
│   │   └── index.tsx            Login screen
│   ├── (tabs)/
│   │   ├── _layout.tsx          Bottom tab bar
│   │   ├── index.tsx            Home / Dashboard
│   │   ├── incidents.tsx        Incident list
│   │   ├── live.tsx             Live feed WebView
│   │   └── profile.tsx          Profile & logout
│   └── incident/
│       └── [id].tsx             Incident detail screen
├── components/
│   ├── DensityBadge.tsx         Colour-coded LOW/MEDIUM/HIGH chip
│   └── IncidentCard.tsx         Reusable incident list card
├── hooks/
│   ├── useStats.ts              Polls /api/stats every 5 s
│   └── useIncidents.ts          Fetches incident list
├── lib/
│   ├── api.ts                   Axios client + typed API helpers
│   └── constants.ts             SERVER_IP, colours, poll intervals
├── store/
│   └── useAppStore.ts           Zustand: auth token + user info
├── assets/                      App icons and splash (see assets/README.txt)
├── app.json                     Expo config
├── package.json                 Dependencies
└── MOBILE_SETUP.md              This file
```
