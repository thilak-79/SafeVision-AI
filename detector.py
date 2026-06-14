from ultralytics import YOLO
import cv2
import os
import time
import csv
import requests
import pygame
import threading
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

SOURCE = 0
# SOURCE = "cctv.mp4"

MODEL_PATH = "yolo11n.pt"

# Restricted area - right bottom area
zone_x1, zone_y1 = 930, 400
zone_x2, zone_y2 = 1270, 710

CONFIDENCE_LIMIT = 0.65
CROWD_LIMIT = 5

# Alert cooldown for screenshot + CSV + Telegram
cooldown_seconds = 3

# Loitering settings
LOITERING_TIME_LIMIT = 10  # seconds

# =====================================================
# SOUND ALERT SETTINGS
# =====================================================

ENABLE_SOUND_ALERTS = True
SOUND_COOLDOWN_SECONDS = 15
sound_muted = False

SOUND_FILES = {
    "Restricted Area": "audio/high.wav",
    "Crowd": "audio/medium.wav",
    "Loitering": "audio/critical.wav"
}

last_sound_times = {
    "Restricted Area": 0,
    "Crowd": 0,
    "Loitering": 0
}

# =====================================================
# FOLDERS AND LOG FILE
# =====================================================

os.makedirs("alerts/screenshots", exist_ok=True)
os.makedirs("audio", exist_ok=True)

CSV_LOG_PATH = "alerts/alert_log.csv"

if not os.path.exists(CSV_LOG_PATH):
    with open(CSV_LOG_PATH, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "alert_type", "confidence_or_count", "screenshot_path"])

# =====================================================
# SOUND ALERT SETUP
# =====================================================

try:
    pygame.mixer.init()
    print("Sound system initialized.")
except Exception as e:
    print("Sound system error:", e)
    ENABLE_SOUND_ALERTS = False


# =====================================================
# SOUND ALERT FUNCTION
# =====================================================

def play_alert_sound(alert_type):
    global sound_muted

    if not ENABLE_SOUND_ALERTS or sound_muted:
        return

    current_time = time.time()

    if current_time - last_sound_times.get(alert_type, 0) < SOUND_COOLDOWN_SECONDS:
        return

    sound_path = SOUND_FILES.get(alert_type)

    if not sound_path or not os.path.exists(sound_path):
        print(f"Sound file not found for {alert_type}: {sound_path}")
        return

    def play_sound():
        try:
            sound = pygame.mixer.Sound(sound_path)
            sound.set_volume(0.8)
            sound.play()
        except Exception as e:
            print("Sound play error:", e)

    threading.Thread(target=play_sound, daemon=True).start()

    last_sound_times[alert_type] = current_time


# =====================================================
# TELEGRAM ALERT
# =====================================================

def send_telegram_alert(alert_type, confidence, screenshot_path):
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
            files = {"photo": image}
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
# SAVE ALERT
# =====================================================

def save_alert(frame, alert_type, confidence):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    safe_alert_name = alert_type.lower().replace(" ", "_")
    screenshot_path = f"alerts/screenshots/{safe_alert_name}_{timestamp}.jpg"

    cv2.imwrite(screenshot_path, frame)

    with open(CSV_LOG_PATH, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            alert_type,
            round(confidence, 2),
            screenshot_path
        ])

    print(f"{alert_type} alert saved: {screenshot_path}")

    # Play sound alert
    play_alert_sound(alert_type)

    # Send Telegram alert
    send_telegram_alert(alert_type, round(confidence, 2), screenshot_path)


# =====================================================
# LOAD YOLO MODEL
# =====================================================

model = YOLO(MODEL_PATH)

# =====================================================
# OPEN WEBCAM / VIDEO
# =====================================================

cap = cv2.VideoCapture(SOURCE)

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

# Cooldown timers
last_alert_times = {
    "Restricted Area": 0,
    "Crowd": 0,
    "Loitering": 0
}

# Loitering timer
restricted_start_time = None

# =====================================================
# MAIN LOOP
# =====================================================

while True:
    ret, frame = cap.read()

    if not ret:
        print("Video ended or cannot read frame")
        break

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

    # =====================================================
    # PROCESS PERSON DETECTIONS
    # =====================================================

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])

            if cls_id == 0:
                person_count += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Foot point
                foot_x = (x1 + x2) // 2
                foot_y = y2

                inside_zone = (
                    zone_x1 < foot_x < zone_x2 and
                    zone_y1 < foot_y < zone_y2
                )

                box_color = (0, 255, 0)

                if inside_zone:
                    box_color = (0, 0, 255)
                    restricted_alert = True
                    max_restricted_confidence = max(max_restricted_confidence, confidence)

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

                cv2.circle(frame, (foot_x, foot_y), 6, box_color, -1)

    # =====================================================
    # CROWD DETECTION
    # =====================================================

    crowd_alert = person_count >= CROWD_LIMIT

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
    # LOITERING DETECTION
    # =====================================================

    current_time = time.time()

    loitering_alert = False
    loitering_duration = 0

    if restricted_alert:
        if restricted_start_time is None:
            restricted_start_time = current_time

        loitering_duration = current_time - restricted_start_time

        cv2.putText(
            frame,
            f"Zone Time: {int(loitering_duration)}s",
            (50, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2
        )

        if loitering_duration >= LOITERING_TIME_LIMIT:
            loitering_alert = True

    else:
        restricted_start_time = None

    # =====================================================
    # ALERT HANDLING
    # =====================================================

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

    if loitering_alert:
        cv2.putText(
            frame,
            "ALERT: Loitering Detected!",
            (50, 210),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            3
        )

        if current_time - last_alert_times["Loitering"] > cooldown_seconds:
            save_alert(frame, "Loitering", loitering_duration)
            last_alert_times["Loitering"] = current_time

    # =====================================================
    # SOUND STATUS DISPLAY
    # =====================================================

    sound_status = "MUTED" if sound_muted else "SOUND ON"
    sound_color = (0, 0, 255) if sound_muted else (0, 255, 0)

    cv2.putText(
        frame,
        sound_status,
        (1050, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        sound_color,
        2
    )

    cv2.putText(
        frame,
        "Press M to mute/unmute",
        (980, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    # =====================================================
    # SHOW OUTPUT
    # =====================================================

    cv2.imshow(window_name, frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("m"):
        sound_muted = not sound_muted
        print("Sound muted" if sound_muted else "Sound enabled")

    if key == ord("q") or key == 27:
        break

cap.release()
cv2.destroyAllWindows()