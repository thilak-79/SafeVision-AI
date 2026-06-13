# SafeVision AI

SafeVision AI is a real-time AI-based CCTV safety monitoring system built using **Python**, **OpenCV**, and **YOLO11n**.  
The system detects people from webcam or CCTV footage, identifies restricted-area entry, detects crowd situations, displays alerts, and saves screenshots as evidence.

## Project Status

Current MVP is working successfully.

### Completed Features

- Real-time webcam detection
- CCTV MP4 video support
- Person detection using YOLO11n
- Restricted area detection
- Crowd detection
- Fullscreen display
- Alert message on screen
- Alert screenshot saving
- Alert log generation using CSV
- Foot-point based zone checking for better CCTV accuracy

## Demo Flow

```text
Webcam / CCTV video
        ↓
YOLO11n person detection
        ↓
Restricted area checking
        ↓
Crowd detection
        ↓
Alert display
        ↓
Screenshot + CSV alert log
```

## Tech Stack

```text
Python
OpenCV
Ultralytics YOLO11n
CSV logging
VS Code
GitHub
```

## Project Structure

```text
safevision-ai/
│
├── detector.py
├── yolo11n.pt
├── requirements.txt
├── README.md
├── .gitignore
│
└── alerts/
    ├── alert_log.csv
    └── screenshots/
```

## How It Works

The system reads frames from a webcam or CCTV video file using OpenCV.  
Each frame is passed into the YOLO11n model to detect people.

For each detected person:

1. A bounding box is drawn.
2. The foot-point of the person is calculated.
3. The foot-point is checked against the restricted zone.
4. If the person enters the zone, a restricted-area alert is triggered.
5. If the number of detected people is greater than the crowd threshold, a crowd alert is triggered.
6. Alert screenshots are saved inside `alerts/screenshots/`.
7. Alert details are saved inside `alerts/alert_log.csv`.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/thilak-79/SafeVision-AI.git
cd SafeVision-AI
```

### 2. Create virtual environment

```bash
python -m venv venv
```

### 3. Activate virtual environment

For Windows PowerShell:

```bash
venv\Scripts\activate
```

For Linux / macOS:

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available, install manually:

```bash
pip install ultralytics opencv-python
```

## Running the Project

```bash
python detector.py
```

Press:

```text
q   → quit
ESC → quit
```

## Using Webcam or CCTV Footage

Inside `detector.py`, use:

```python
SOURCE = 0
```

for webcam.

Use:

```python
SOURCE = "cctv.mp4"
```

for CCTV MP4 footage.

Place the video file in the same folder as `detector.py`.

Example:

```text
safevision-ai/
├── detector.py
├── cctv.mp4
└── yolo11n.pt
```

## Changing the Restricted Area

Inside `detector.py`, change these values:

```python
zone_x1, zone_y1 = 600, 100
zone_x2, zone_y2 = 1000, 500
```

Meaning:

```text
zone_x1, zone_y1 → top-left corner
zone_x2, zone_y2 → bottom-right corner
```

Example for left-side restricted zone:

```python
zone_x1, zone_y1 = 100, 100
zone_x2, zone_y2 = 500, 500
```

Example for right-side restricted zone:

```python
zone_x1, zone_y1 = 850, 100
zone_x2, zone_y2 = 1250, 550
```

## Crowd Detection

Crowd detection is controlled by:

```python
CROWD_LIMIT = 3
```

Meaning:

```text
If 3 or more people are detected, a crowd alert is triggered.
```

For small room testing:

```python
CROWD_LIMIT = 2
```

For real CCTV footage:

```python
CROWD_LIMIT = 5
```

## Alert Outputs

When an alert happens, the system creates:

```text
alerts/screenshots/
alerts/alert_log.csv
```

The CSV log stores:

```text
timestamp
alert_type
confidence
screenshot_path
```

Example:

```text
2026-06-13 13:17:40, Crowd, 3, alerts/screenshots/crowd_2026-06-13_13-17-40.jpg
```

## Why Foot-Point Detection Is Used

Instead of checking the center of the person’s body, this system checks the bottom-center point of the bounding box.

This is better for CCTV because the bottom point represents where the person is standing.

```text
Body center point → less accurate for zones
Foot point        → better for CCTV zone entry detection
```

## Current MVP Features

| Feature | Status |
|---|---|
| Person Detection | Done |
| Restricted Area Detection | Done |
| Crowd Detection | Done |
| Screenshot Saving | Done |
| CSV Alert Log | Done |
| Fullscreen Mode | Done |
| MP4 CCTV Support | Done |
| Dashboard | Planned |
| Fire/Smoke Detection | Planned |
| Fall Detection | Planned |
| Telegram/WhatsApp Alerts | Planned |

## Future Improvements

- Flask dashboard for live monitoring
- Alert history table with screenshots
- Telegram alert notification
- WhatsApp/SMS alert notification
- Fire and smoke detection using a custom YOLO model
- Fall detection using YOLO11n-pose
- Loitering detection
- Direction-of-travel analysis
- Multi-camera support
- Edge deployment using Raspberry Pi or Jetson device

## Project Goal

The goal of SafeVision AI is to provide a low-cost, real-time, AI-powered CCTV monitoring system for places such as:

```text
Universities
Hostels
Hospitals
Shops
Warehouses
Banks
Factories
Public buildings
```

The system is designed to help reduce manual CCTV monitoring effort and automatically detect important safety events.

## Author

**Ravichandran Thilakshan**  
GitHub: [thilak-79](https://github.com/thilak-79)

## License

This project is for academic and learning purposes.  
You can modify and improve it for future research, competitions, or real-world deployment.
