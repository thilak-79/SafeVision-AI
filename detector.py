from ultralytics import YOLO
import cv2
import os
import time

# Load YOLO model
model = YOLO("yolo11n.pt")

# Create folder for alert screenshots
os.makedirs("alerts/screenshots", exist_ok=True)

# -------------------------------
# VIDEO SOURCE
# -------------------------------

# Use webcam
cap = cv2.VideoCapture(0)

# For CCTV MP4 later, use this instead:
# cap = cv2.VideoCapture("cctv.mp4")

# Set webcam resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Cannot open webcam/video")
    exit()

# -------------------------------
# RESTRICTED AREA
# -------------------------------

# Format: x1, y1, x2, y2
# x1,y1 = top-left corner
# x2,y2 = bottom-right corner
zone_x1, zone_y1 = 600, 100
zone_x2, zone_y2 = 1000, 500

# Alert cooldown
last_alert_time = 0
cooldown_seconds = 5

# -------------------------------
# FULLSCREEN WINDOW
# -------------------------------

window_name = "SafeVision AI"

cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Cannot read frame / video ended")
        break

    # Optional: print frame size once for checking
    # print(frame.shape)  # height, width, channels

    # Run YOLO detection
    results = model(frame, verbose=False)

    # Draw restricted zone
    cv2.rectangle(
        frame,
        (zone_x1, zone_y1),
        (zone_x2, zone_y2),
        (0, 0, 255),
        2
    )

    cv2.putText(
        frame,
        "RESTRICTED AREA",
        (zone_x1, zone_y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    restricted_alert = False

    # Process YOLO results
    for result in results:
        boxes = result.boxes

        for box in boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])

            # COCO class 0 = person
            if cls_id == 0 and confidence > 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                inside_zone = (
                    zone_x1 < center_x < zone_x2 and
                    zone_y1 < center_y < zone_y2
                )

                box_color = (0, 255, 0)

                if inside_zone:
                    box_color = (0, 0, 255)
                    restricted_alert = True

                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                label = f"Person {confidence:.2f}"
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    box_color,
                    2
                )

                cv2.circle(frame, (center_x, center_y), 5, box_color, -1)

    # Alert section
    if restricted_alert:
        cv2.putText(
            frame,
            "ALERT: Person in Restricted Area!",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

        current_time = time.time()

        if current_time - last_alert_time > cooldown_seconds:
            filename = f"alerts/screenshots/restricted_alert_{int(current_time)}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Restricted Area Alert! Screenshot saved: {filename}")
            last_alert_time = current_time

    # Show fullscreen output
    cv2.imshow(window_name, frame)

    # Press q or ESC to quit
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == 27:
        break

cap.release()
cv2.destroyAllWindows()