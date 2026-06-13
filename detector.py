from ultralytics import YOLO
import cv2
import os
import time
import csv
import requests
from datetime import datetime
from dotenv import load_dotenv

# =====================================================
# LOAD ENVIRONMENT VARIABLES
# =====================================================

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# =====================================================
# SAFEVISION AI SETTINGS
# =====================================================

# Use 0 for webcam
SOURCE = 0

# Use this for CCTV MP4 footage
# SOURCE = "cctv.mp4"

MODEL_PATH = "yolo11n.pt"

# Restricted area coordinates
# x1, y1 = top-left corner
# x2, y2 = bottom-right corner
zone_x1, zone_y1 = 600, 100
zone_x2, zone_y2 = 1000, 500

# YOLO confidence limit
CONFIDENCE_LIMIT = 0.65

# Crowd detection threshold
# For your room demo, use 2 or 3
# For real CCTV, use 5 or more
CROWD_LIMIT = 3

# Alert cooldown in seconds
cooldown_seconds = 5

# =====================================================
# FOLDERS AND LOG FILE
# =====================================================

os.makedirs("alerts/screenshots", exist_ok=True)

CSV_LOG_PATH = "alerts/alert_log.csv"

# Create CSV file with headings if it does not exist
if not os.path.exists(CSV_LOG_PATH):
    with open(CSV_LOG_PATH, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "alert_type", "confidence_or_count", "screenshot_path"])


# =====================================================
# TELEGRAM ALERT FUNCTION
# =====================================================

def send_telegram_alert(alert_type, confidence, screenshot_path):
    """
    Send Telegram alert message with screenshot.
    """

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram token or chat ID not found. Skipping Telegram alert.")
        return

    message = (
        f"🚨 SafeVision AI Alert\n\n"
        f"Type: {alert_type}\n"
        f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Confidence / Count: {confidence}\n"
    )

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"

    try:
        with open(screenshot_path, "rb") as image:
            files = {
                "photo": image
            }

            data = {
                "chat_id": TELEGRAM_CHAT_ID,
                "caption": message
            }

            response = requests.post(url, data=data, files=files, timeout=10)

        if response.status_code == 200:
            print("Telegram alert sent successfully.")
        else:
            print("Failed to send Telegram alert:", response.text)

    except Exception as e:
        print("Telegram alert error:", e)


# =====================================================
# HELPER FUNCTION: SAVE ALERT
# =====================================================

def save_alert(frame, alert_type, confidence):
    """
    Save screenshot, write alert details to CSV file,
    and send Telegram alert.
    """

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    safe_alert_name = alert_type.lower().replace(" ", "_")
    screenshot_path = f"alerts/screenshots/{safe_alert_name}_{timestamp}.jpg"

    # Save screenshot
    cv2.imwrite(screenshot_path, frame)

    # Save alert details to CSV
    with open(CSV_LOG_PATH, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            alert_type,
            round(confidence, 2),
            screenshot_path
        ])

    print(f"{alert_type} alert saved: {screenshot_path}")

    # Send Telegram alert with screenshot
    send_telegram_alert(alert_type, round(confidence, 2), screenshot_path)


# =====================================================
# LOAD YOLO MODEL
# =====================================================

model = YOLO(MODEL_PATH)

# =====================================================
# OPEN WEBCAM OR CCTV MP4
# =====================================================

cap = cv2.VideoCapture(SOURCE)

# If using webcam, set resolution
if SOURCE == 0:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Cannot open webcam/video source")
    exit()

# =====================================================
# FULLSCREEN WINDOW
# =====================================================

window_name = "SafeVision AI - MVP"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Separate cooldown timers for each alert type
last_alert_times = {
    "Restricted Area": 0,
    "Crowd": 0
}

# =====================================================
# MAIN LOOP
# =====================================================

while True:
    ret, frame = cap.read()

    if not ret:
        print("Video ended or cannot read frame")
        break

    # Detect only persons
    # COCO class 0 = person
    results = model(
        frame,
        classes=[0],
        conf=CONFIDENCE_LIMIT,
        verbose=False
    )

    # Draw restricted area
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

    person_count = 0
    restricted_alert = False
    max_restricted_confidence = 0

    # Process detections
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])

            if cls_id == 0:
                person_count += 1

                # Person bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Foot-point detection
                # Better for CCTV because it checks where the person is standing
                foot_x = (x1 + x2) // 2
                foot_y = y2

                # Check if foot point is inside restricted area
                inside_zone = (
                    zone_x1 < foot_x < zone_x2 and
                    zone_y1 < foot_y < zone_y2
                )

                box_color = (0, 255, 0)

                if inside_zone:
                    box_color = (0, 0, 255)
                    restricted_alert = True
                    max_restricted_confidence = max(max_restricted_confidence, confidence)

                # Draw person box
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                # Draw label
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

                # Draw foot point
                cv2.circle(frame, (foot_x, foot_y), 6, box_color, -1)

    # =====================================================
    # CROWD DETECTION
    # =====================================================

    crowd_alert = person_count >= CROWD_LIMIT

    # Show person count
    cv2.putText(
        frame,
        f"Persons: {person_count}",
        (50, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 0),
        2
    )

    # =====================================================
    # ALERT HANDLING
    # =====================================================

    current_time = time.time()

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

        if current_time - last_alert_times["Restricted Area"] > cooldown_seconds:
            save_alert(frame, "Restricted Area", max_restricted_confidence)
            last_alert_times["Restricted Area"] = current_time

    if crowd_alert:
        cv2.putText(
            frame,
            "ALERT: Crowd Detected!",
            (50, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 165, 255),
            3
        )

        if current_time - last_alert_times["Crowd"] > cooldown_seconds:
            save_alert(frame, "Crowd", person_count)
            last_alert_times["Crowd"] = current_time

    # =====================================================
    # SHOW OUTPUT
    # =====================================================

    cv2.imshow(window_name, frame)

    # Press q or ESC to quit
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == 27:
        break

cap.release()
cv2.destroyAllWindows()