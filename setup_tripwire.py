#!/usr/bin/env python3
"""
Barangay Sentinel — Tripwire Setup Tool

Draw the entry/exit line across your gate or monitored entrance.
The system counts people crossing this line to maintain an accurate
running total, independent of MOG2 absorption.

Usage:
    python setup_tripwire.py
    python setup_tripwire.py videos/other_video.mp4 mask_layer2.png

Controls:
    Left-click      — place P1 (first click) then P2 (second click)
    R               — reset and re-draw
    ENTER           — save tripwire.json and exit
    ESC             — cancel without saving

Direction rule:
    People crossing from the RIGHT of the P1→P2 direction = ENTERING (+1)
    People crossing from the LEFT                          = EXITING  (-1)
    If entries and exits are swapped when you test, re-run and click
    P1 and P2 in reverse order.
"""
import cv2
import json
import sys
import os

VIDEO_PATH  = "videos/main_video.mp4"
MASK_PATH   = "mask_layer1.png"
OUT_PATH    = "tripwire.json"
SEEK_FRAME  = 1600   # seek past warmup so the scene is clear

_pts = []


def _mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(_pts) < 2:
        _pts.append((x, y))
        print(f"  Placed P{len(_pts)}: ({x}, {y})")


def main():
    video     = sys.argv[1] if len(sys.argv) > 1 else VIDEO_PATH
    mask_path = sys.argv[2] if len(sys.argv) > 2 else MASK_PATH

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video}")
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    seek  = min(SEEK_FRAME, max(0, total - 10))
    cap.set(cv2.CAP_PROP_POS_FRAMES, seek)
    ret, base = cap.read()
    cap.release()

    if not ret:
        print("[ERROR] Could not read a frame from the video.")
        return

    # Dim the masked-out area so the ROI boundary is visible
    mask = cv2.imread(mask_path, 0)
    if mask is not None:
        m   = cv2.resize(mask, (base.shape[1], base.shape[0]))
        dim = (base * 0.3).astype("uint8")
        base[m == 0] = dim[m == 0]

    WIN = "Tripwire Setup  |  Click P1 then P2  |  ENTER=save  R=reset  ESC=cancel"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN, _mouse)

    print("\n=== Barangay Sentinel — Tripwire Setup ===")
    print(f"Video  : {video}")
    print(f"Output : {OUT_PATH}")
    print()
    print("Click exactly TWO points to draw the entry/exit line.")
    print("Place the line across the gate opening so every person")
    print("must walk through it when entering or leaving the area.")
    print()
    print("  ENTER  — save and quit")
    print("  R      — redo (clear both points)")
    print("  ESC    — quit without saving")
    print()

    while True:
        display = base.copy()

        # Draw placed points
        for i, p in enumerate(_pts):
            cv2.circle(display, p, 8, (0, 255, 0), -1)
            cv2.circle(display, p, 8, (0, 0, 0), 1)
            cv2.putText(display, f"P{i+1}", (p[0] + 12, p[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        if len(_pts) == 2:
            # Draw the tripwire line
            cv2.line(display, _pts[0], _pts[1], (0, 255, 255), 3)

            # Direction arrow at midpoint
            mx = (_pts[0][0] + _pts[1][0]) // 2
            my = (_pts[0][1] + _pts[1][1]) // 2

            # Normal vector (perpendicular, pointing "inside")
            dx = _pts[1][0] - _pts[0][0]
            dy = _pts[1][1] - _pts[0][1]
            length = max(1, (dx**2 + dy**2) ** 0.5)
            nx = int(-dy / length * 30)
            ny = int( dx / length * 30)
            cv2.arrowedLine(display, (mx, my), (mx + nx, my + ny),
                            (0, 200, 255), 2, tipLength=0.4)
            cv2.putText(display, "IN", (mx + nx + 5, my + ny + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

            cv2.putText(display, "Press ENTER to save, R to redo",
                        (10, display.shape[0] - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            remaining = 2 - len(_pts)
            cv2.putText(display, f"Click {remaining} more point(s)",
                        (10, display.shape[0] - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        cv2.imshow(WIN, display)
        key = cv2.waitKey(30) & 0xFF

        if key == 27:   # ESC — cancel
            print("Cancelled. tripwire.json was NOT updated.")
            break

        elif key == ord('r') or key == ord('R'):
            _pts.clear()
            print("  Reset. Click two new points.")

        elif key == 13 and len(_pts) == 2:   # ENTER — save
            data = {
                "x1": _pts[0][0],
                "y1": _pts[0][1],
                "x2": _pts[1][0],
                "y2": _pts[1][1],
                "note": (
                    "Full-resolution pixel coordinates. "
                    "SentinelStream scales these by process_scale automatically. "
                    "Do NOT edit manually."
                )
            }
            with open(OUT_PATH, "w") as f:
                json.dump(data, f, indent=2)

            print(f"\n[SAVED] {OUT_PATH}")
            print(f"  P1 = ({_pts[0][0]}, {_pts[0][1]})")
            print(f"  P2 = ({_pts[1][0]}, {_pts[1][1]})")
            print()
            print("Restart the Flask app to activate the tripwire.")
            print("If IN and OUT are swapped during testing, re-run")
            print("this tool and click P1 and P2 in the opposite order.\n")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
