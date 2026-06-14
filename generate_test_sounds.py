import numpy as np
import wave
import struct
import os

os.makedirs("audio", exist_ok=True)

def write_wav(path, freq, duration, volume=0.6, sr=44100):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    wave_data = (np.sin(2 * np.pi * freq * t) * 32767 * volume).astype(np.int16)

    with wave.open(path, "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(struct.pack(f"<{len(wave_data)}h", *wave_data))

    print(f"Created {path}")

write_wav("audio/critical.wav", freq=1200, duration=0.4)
write_wav("audio/high.wav", freq=880, duration=0.3)
write_wav("audio/medium.wav", freq=520, duration=0.35)