from flask import Flask, render_template, send_from_directory
import csv
import os

app = Flask(__name__)

CSV_LOG_PATH = "alerts/alert_log.csv"


def get_camera_source():
    camera_source = "Unknown"

    try:
        with open("detector.py", "r", encoding="utf-8") as f:
            for line in f:
                clean_line = line.strip()

                if clean_line.startswith("SOURCE"):
                    val = clean_line.split("=", 1)[1].strip()
                    clean_val = val.strip("\"'")

                    if clean_val == "0":
                        camera_source = "Webcam"
                    else:
                        camera_source = f"CCTV ({clean_val})"

                    break

    except Exception:
        camera_source = "Unknown"

    return camera_source


def get_alert_data():
    alerts = []
    total_alerts = 0
    restricted_count = 0
    crowd_count = 0
    loitering_count = 0
    latest_screenshot = None

    if os.path.exists(CSV_LOG_PATH):
        with open(CSV_LOG_PATH, mode="r", encoding="utf-8") as file:
            reader = csv.DictReader(file)

            for row in reader:
                alerts.append(row)
                total_alerts += 1

                alert_type = row.get("alert_type", "")

                if alert_type == "Restricted Area":
                    restricted_count += 1
                elif alert_type == "Crowd":
                    crowd_count += 1
                elif alert_type == "Loitering":
                    loitering_count += 1

                latest_screenshot = row.get("screenshot_path")

    alerts.reverse()
    alerts = alerts[:5]

    return {
        "total_alerts": total_alerts,
        "restricted_count": restricted_count,
        "crowd_count": crowd_count,
        "loitering_count": loitering_count,
        "latest_screenshot": latest_screenshot,
        "alerts": alerts,
        "camera_source": get_camera_source(),
    }


@app.route("/")
def dashboard():
    data = get_alert_data()
    return render_template("dashboard.html", **data)


@app.route("/alerts/<path:filename>")
def serve_alerts(filename):
    return send_from_directory("alerts", filename)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)