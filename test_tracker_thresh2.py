import cv2

cap = cv2.VideoCapture('videos/main_video.mp4')
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
ret, bg_frame = cap.read()

bg_gray = cv2.cvtColor(bg_frame, cv2.COLOR_BGR2GRAY)
bg_reference = cv2.GaussianBlur(bg_gray, (5, 5), 0)

cap.set(cv2.CAP_PROP_POS_FRAMES, 1500)
ret, frame = cap.read()
curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
curr_gray = cv2.GaussianBlur(curr_gray, (5, 5), 0)

diff = cv2.absdiff(curr_gray, bg_reference)
for thresh_val in [25, 30, 40, 50, 60]:
    _, fg_mask = cv2.threshold(diff, thresh_val, 255, cv2.THRESH_BINARY)
    print(f"Without CLAHE Threshold {thresh_val}: {cv2.countNonZero(fg_mask)} noise pixels")
