from flask import Flask, render_template, send_from_directory
import csv
import os

app = Flask(__name__)

CSV_LOG_PATH = "alerts/alert_log.csv"

def get_alert_data():
    alerts = []
    total_alerts = 0
    restricted_count = 0
    crowd_count = 0
    latest_screenshot = None

    if os.path.exists(CSV_LOG_PATH):
        with open(CSV_LOG_PATH, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                alerts.append(row)
                total_alerts += 1
                if row.get('alert_type') == 'Restricted Area':
                    restricted_count += 1
                elif row.get('alert_type') == 'Crowd':
                    crowd_count += 1
                
                # The latest in the file is typically the last row read
                latest_screenshot = row.get('screenshot_path')

    # Reverse alerts to show newest first in the table
    alerts.reverse()
    
    return {
        'total_alerts': total_alerts,
        'restricted_count': restricted_count,
        'crowd_count': crowd_count,
        'latest_screenshot': latest_screenshot,
        'alerts': alerts
    }

@app.route('/')
def dashboard():
    data = get_alert_data()
    return render_template('dashboard.html', **data)

@app.route('/alerts/<path:filename>')
def serve_alerts(filename):
    return send_from_directory('alerts', filename)

if __name__ == '__main__':
    # Run the Flask app on port 5000
    app.run(debug=True, host='0.0.0.0', port=5000)
