import pyttsx3
import os

os.makedirs("audio", exist_ok=True)

engine = pyttsx3.init()

# Voice settings
engine.setProperty("rate", 145)      # speaking speed
engine.setProperty("volume", 1.0)    # max volume

alerts = {
    "audio/high.wav": "Warning. Person entered restricted area.",
    "audio/medium.wav": "Alert. Crowd detected.",
    "audio/critical.wav": "Critical alert. Loitering detected in restricted area."
}

for file_path, text in alerts.items():
    engine.save_to_file(text, file_path)

engine.runAndWait()

print("Voice alert files created successfully.")