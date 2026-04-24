import cv2
import json
import os

VIDEO_PATH = 'video1.mp4'
OUTPUT_JSON = 'barangay_ground_truth.json'
SKIP_FRAMES = 60  # Skip 2 seconds ahead each time to get unique crowd positions

# Global state for the UI
current_points = []
ground_truth_db = {}
frame_idx = 0

def draw_instructions(img, current_count):
    # Black background bar
    cv2.rectangle(img, (0, 0), (img.shape[1], 40), (0, 0, 0), -1)
    
    cv2.putText(img, f"Frame: {frame_idx} | People Counted: {current_count}", (10, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    cv2.putText(img, "Left Click: Mark Person | Right Click: Undo | SPACE: Next Frame | ESC: Save & Quit", 
                (350, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return img

def mouse_callback(event, x, y, flags, param):
    global current_points
    if event == cv2.EVENT_LBUTTONDOWN:
        current_points.append({"x": x, "y": y})
    elif event == cv2.EVENT_RBUTTONDOWN:
        if current_points:
            current_points.pop()

def main():
    global frame_idx, current_points, ground_truth_db

    # Load existing DB if they partially finished it before
    if os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON, 'r') as f:
            try:
                ground_truth_db = json.load(f)["frames"]
            except:
                ground_truth_db = {}
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {VIDEO_PATH}")
        return

    cv2.namedWindow('Ground Truth Annotator', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Ground Truth Annotator', mouse_callback)

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            print("Reached end of video.")
            break

        # Load existing points for this frame if any
        if str(frame_idx) in ground_truth_db:
            current_points = ground_truth_db[str(frame_idx)].copy()
        else:
            current_points = []

        while True:
            display_frame = frame.copy()
            
            # Draw markers
            for pt in current_points:
                cv2.circle(display_frame, (pt["x"], pt["y"]), 8, (0, 255, 0), -1)
                cv2.circle(display_frame, (pt["x"], pt["y"]), 10, (255, 255, 255), 2)

            display_frame = draw_instructions(display_frame, len(current_points))
            cv2.imshow('Ground Truth Annotator', display_frame)

            key = cv2.waitKey(15) & 0xFF

            if key == 32 or key == 13: # Space or Enter (NEXT FRAME)
                ground_truth_db[str(frame_idx)] = current_points
                frame_idx += SKIP_FRAMES
                break
                
            elif key == 27: # ESC (SAVE AND QUIT)
                ground_truth_db[str(frame_idx)] = current_points
                print(f"Saving {len(ground_truth_db)} frames of ground truth to {OUTPUT_JSON}...")
                with open(OUTPUT_JSON, 'w') as f:
                    json.dump({"video_source": VIDEO_PATH, "frames": ground_truth_db}, f, indent=4)
                cap.release()
                cv2.destroyAllWindows()
                return

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Launching Local Ground Truth Annotation Tool...")
    main()
