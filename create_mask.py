import cv2
import numpy as np

# CONFIGURATION
VIDEO_SOURCE = "video1.mp4"

# GLOBAL VARIABLES
points = []

def click_event(event, x, y, flags, params):
   if event == cv2.EVENT_LBUTTONDOWN:
      points.append((x, y))
      cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
      if len(points) >= 2:
         cv2.line(img, points[-2], points[-1], (0, 0, 255), 2)
      cv2.imshow('Draw Polygon (Press s to save, r to reset, q to quit)', img)

# LOAD VIDEO
cap = cv2.VideoCapture(VIDEO_SOURCE)
ret, frame = cap.read()
if not ret:
   print("Error: Cannot read video.")
   exit()

img = frame.copy()
cv2.imshow('Draw Polygon (Press s to save, r to reset, q to quit)', img)
cv2.setMouseCallback('Draw Polygon (Press s to save, r to reset, q to quit)', click_event)

print("INSTRUCTIONS:")
print("1. Click points to outline the ROAD/CARS (the area to IGNORE).")
print("2. Press 's' to save the mask.")
print("3. Press 'r' to reset.")
print("4. Press 'q' to quit without saving.")

while True:
   key = cv2.waitKey(1) & 0xFF
   if key == ord('s'):
      # Create black blank image
      mask = np.zeros(frame.shape[:2], dtype="uint8")
      # Fill the polygon with WHITE (255) - this is the area we want to REMOVE
      if len(points) > 0:
         pts = np.array(points, np.int32)
         pts = pts.reshape((-1, 1, 2))
         cv2.fillPoly(mask, [pts], 255) 
      
      # Invert mask: Black = Ignore, White = Keep
      # Actually, let's save the "Road Mask" directly. 
      # Any pixel that is WHITE in 'mask.png' will be ignored by the main engine.
      cv2.imwrite("mask_layer.png", mask)
      print("Success! 'mask_layer.png' saved.")
      break
   elif key == ord('r'):
      img = frame.copy()
      points = []
      cv2.imshow('Draw Polygon (Press s to save, r to reset, q to quit)', img)
   elif key == ord('q'):
      break

cap.release()
cv2.destroyAllWindows()