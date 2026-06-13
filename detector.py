from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("yolo11n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

# Set webcam resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Check webcam
if not cap.isOpened():
    print("Cannot open webcam")
    exit()

# Fullscreen window setup
window_name = "SafeVision AI"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Cannot read frame")
        break

    # Run YOLO detection
    results = model(frame, verbose=False)

    # Draw bounding boxes
    annotated_frame = results[0].plot()

    # Show fullscreen output
    cv2.imshow(window_name, annotated_frame)

    # Press q or ESC to quit
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()