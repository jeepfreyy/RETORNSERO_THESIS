import cv2
import json
import os

VIDEO_PATH = 'videos/vid1-angle1.MOV'
OUTPUT_JSON = 'barangay_ground_truth.json'
MASK_PATH = 'mask_layer1.png'
SKIP_FRAMES = 60  # Default step (2 seconds at 30fps)

current_points = []
ground_truth_db = {}
frame_idx = 0


def get_distribution(db):
    counts = {}
    for pts in db.values():
        n = len(pts)
        counts[n] = counts.get(n, 0) + 1
    return counts


def draw_ui(img, fidx, total_frames, mask_overlay=None, db=None):
    if mask_overlay is not None:
        overlay = img.copy()
        overlay[mask_overlay < 127] = [0, 0, 80]
        cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)

    for pt in current_points:
        cv2.circle(img, (pt["x"], pt["y"]), 8, (0, 255, 0), -1)
        cv2.circle(img, (pt["x"], pt["y"]), 10, (255, 255, 255), 2)

    # Top bar
    cv2.rectangle(img, (0, 0), (img.shape[1], 42), (0, 0, 0), -1)
    pct = int(fidx / max(1, total_frames - 1) * 100)
    cv2.putText(img, f"Frame {fidx}/{total_frames-1} ({pct}%)  |  People: {len(current_points)}",
                (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Controls bar
    cv2.rectangle(img, (0, img.shape[0] - 44), (img.shape[1], img.shape[0]), (0, 0, 0), -1)
    controls = "LClick:Mark  RClick:Undo  SPACE/ENTER:Next  B:Back  G:GoTo  ESC:Save&Quit"
    cv2.putText(img, controls, (10, img.shape[0] - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)

    # Distribution panel (bottom-right)
    if db is not None:
        dist = get_distribution(db)
        annotated = len(db)
        panel_lines = [f"Annotated: {annotated}"] + [
            f"  {k} person{'s' if k != 1 else ''}: {v}" for k, v in sorted(dist.items())
        ]
        px, py = img.shape[1] - 200, img.shape[0] - 44 - len(panel_lines) * 22 - 8
        cv2.rectangle(img, (px - 4, py - 4),
                      (img.shape[1] - 4, img.shape[0] - 48), (20, 20, 20), -1)
        for i, line in enumerate(panel_lines):
            color = (180, 255, 180) if i == 0 else (200, 200, 200)
            cv2.putText(img, line, (px, py + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return img


def mouse_callback(event, x, y, flags, param):
    global current_points
    if event == cv2.EVENT_LBUTTONDOWN:
        current_points.append({"x": x, "y": y})
    elif event == cv2.EVENT_RBUTTONDOWN:
        if current_points:
            current_points.pop()


def ask_frame_number(total_frames):
    """Prompt the user for a frame number via a simple OpenCV input loop."""
    buf = ""
    prompt_img = 255 * __import__("numpy").ones((120, 420, 3), dtype="uint8")
    cv2.putText(prompt_img, "Go to frame (Enter to confirm):", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(prompt_img, f"Range: 0 - {total_frames - 1}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)
    while True:
        disp = prompt_img.copy()
        cv2.putText(disp, buf + "_", (10, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 200), 2)
        cv2.imshow("Go To Frame", disp)
        k = cv2.waitKey(30) & 0xFF
        if k == 13 or k == 10:  # Enter
            break
        elif k == 27:  # ESC — cancel
            buf = ""
            break
        elif k == 8 and buf:  # Backspace
            buf = buf[:-1]
        elif chr(k).isdigit():
            buf += chr(k)
    cv2.destroyWindow("Go To Frame")
    return int(buf) if buf.isdigit() else None


def main():
    global frame_idx, current_points, ground_truth_db

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {VIDEO_PATH}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON, 'r') as f:
            try:
                data = json.load(f)
                ground_truth_db = data.get("frames", {})
                print(f"Loaded {len(ground_truth_db)} existing annotations.")
            except Exception:
                ground_truth_db = {}

    mask_img = cv2.imread(MASK_PATH, 0)
    if mask_img is None:
        print(f"Warning: {MASK_PATH} not found. Proceeding without overlay.")

    cv2.namedWindow('Ground Truth Annotator', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Ground Truth Annotator', mouse_callback)

    print("Controls: SPACE/ENTER = next frame | B = back | G = go to frame | ESC = save & quit")
    print("Goal: annotate frames with 0, 1, 2-3, 4-5, and 8+ people for a balanced dataset.")

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print("Reached end of video.")
            break

        if str(frame_idx) in ground_truth_db:
            current_points = ground_truth_db[str(frame_idx)].copy()
        else:
            current_points = []

        while True:
            disp = frame.copy()
            disp = draw_ui(disp, frame_idx, total_frames, mask_img, ground_truth_db)
            cv2.imshow('Ground Truth Annotator', disp)

            key = cv2.waitKey(15) & 0xFF

            if key in (32, 13):  # SPACE or ENTER — advance
                ground_truth_db[str(frame_idx)] = current_points
                frame_idx = min(frame_idx + SKIP_FRAMES, total_frames - 1)
                break

            elif key == ord('b') or key == ord('B'):  # Back
                ground_truth_db[str(frame_idx)] = current_points
                frame_idx = max(0, frame_idx - SKIP_FRAMES)
                break

            elif key == ord('g') or key == ord('G'):  # Go to frame
                ground_truth_db[str(frame_idx)] = current_points
                target = ask_frame_number(total_frames)
                if target is not None:
                    frame_idx = max(0, min(target, total_frames - 1))
                break

            elif key == 27:  # ESC — save & quit
                ground_truth_db[str(frame_idx)] = current_points
                print(f"Saving {len(ground_truth_db)} frames to {OUTPUT_JSON}...")
                with open(OUTPUT_JSON, 'w') as f:
                    json.dump({"video_source": VIDEO_PATH, "frames": ground_truth_db}, f, indent=4)
                dist = get_distribution(ground_truth_db)
                print("Count distribution:", {k: v for k, v in sorted(dist.items())})
                cap.release()
                cv2.destroyAllWindows()
                return

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Launching Ground Truth Annotation Tool...")
    main()
