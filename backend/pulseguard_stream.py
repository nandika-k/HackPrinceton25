# ============================================================
# PulseGuard Real-Time Backend
# Streams Arduino ECG + accelerometer data → ML features → LGBM → Grok → SOS
# ============================================================

import eventlet
eventlet.monkey_patch()

import time
import math
import random
from collections import deque

import numpy as np
import serial
from flask import Flask, jsonify
from flask_socketio import SocketIO

from pulseguard_infer import load_model, predict_from_data

# -----------------------------
# CONFIG
# -----------------------------
SERIAL_PORT = "/dev/tty.usbmodem1101"   # 🔧 Update with `ls /dev/tty.*` if it changes
BAUD_RATE = 9600
MODEL_PATH = "sos_gbt_model.joblib"

# Stream cadence and feature windowing
STREAM_INTERVAL = 0.5        # seconds between emissions to frontend
WINDOW_SECONDS = 5           # rolling window used to derive features
FS_ECG_GUESS = 100           # only used for simple “per-sec” proxies
FS_ACCEL_GUESS = 50

# Behavior
USE_SIMULATION_IF_FAIL = True    # fallback to simulated data if serial port not available
PRINT_COMPACT_LOG = True

# Scaling (rough normalization so features are stable)
ECG_SCALE = 200.0    # divide raw ECG by this (tune for your board)
ACCEL_SCALE = 9.8    # divide raw accel by this to get g-units

# -----------------------------
# INIT APP
# -----------------------------
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# -----------------------------
# LOAD MODEL
# -----------------------------
print("✅ Loading ML model...")
model, feature_names = load_model(MODEL_PATH)
print(f"✅ Model loaded with {len(feature_names)} features from {MODEL_PATH}")
# Expected: ['hr_mean','hr_std','hr_range','hr_slope','hrv_proxy',
#            'accel_mean','accel_std','accel_max','peaks_per_sec',
#            'stillness_duration','post_impact_still_flag']

# -----------------------------
# GROK & SOS LOGIC
# -----------------------------
def analyze_with_grok(data):
    """
    Simple rule-based placeholder. You can later swap this with an LLM call.
    """
    ecg = float(data.get("ecg", 0.0))
    pred = float(data.get("prediction", 0.0))
    ax, ay, az = float(data.get("ax", 0.0)), float(data.get("ay", 0.0)), float(data.get("az", 0.0))
    accel_mag = math.sqrt(ax * ax + ay * ay + az * az)

    if pred >= 0.6:
        return "⚠️ AI detected possible abnormal pattern."
    if abs(ecg) > 0.9:
        return "⚠️ ECG signal instability detected."
    if accel_mag > 2.0:
        return "⚠️ Sudden movement detected."
    return "✅ Vitals appear stable."

def detect_sos(data):
    """
    Decide if SOS alert should trigger (tune thresholds as needed).
    """
    ecg = float(data.get("ecg", 0.0))
    ax, ay, az = float(data.get("ax", 0.0)), float(data.get("ay", 0.0)), float(data.get("az", 0.0))
    pred = float(data.get("prediction", 0.0))
    accel_mag = math.sqrt(ax * ax + ay * ay + az * az)

    return (pred >= 0.7) or (abs(ecg) > 1.1) or (accel_mag > 2.5)

# -----------------------------
# SERIAL CONNECTION
# -----------------------------
def try_open_serial():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)  # allow board to reset
        ser.reset_input_buffer()
        print(f"🔌 Connected to {SERIAL_PORT} @ {BAUD_RATE} baud")
        return ser
    except Exception as e:
        print(f"⚠️ Serial connection failed: {e}")
        return None

ser = try_open_serial()

def parse_serial_line(raw: bytes):
    """
    Robustly parse one sensor line.
    Expected format: "ecg,ax,ay,az" (floats).
    Returns dict or None.
    """
    if not raw:
        return None

    # Be tolerant to weird bytes; ignore undecodable chars
    line = raw.decode("utf-8", errors="ignore").strip()
    if not line:
        return None

    # Accept commas/semicolons/spaces as separators
    for sep in [",", ";", " "]:
        if sep in line:
            parts = [p for p in line.split(sep) if p]
            break
    else:
        return None

    if len(parts) < 4:
        return None

    try:
        ecg, ax, ay, az = map(float, parts[:4])
        return {"ecg": ecg, "ax": ax, "ay": ay, "az": az}
    except Exception:
        return None

def read_sensor_data():
    """
    Read one sensor sample from serial; fall back to simulation if needed.
    """
    global ser
    # If serial isn't up, simulate
    if ser is None:
        if not USE_SIMULATION_IF_FAIL:
            return None
        t = time.time()
        ecg = 0.9 * math.sin(2 * math.pi * 1.2 * t) + 0.1 * random.uniform(-1, 1)
        ax = 0.25 * math.sin(2 * math.pi * 0.75 * t) + 0.05 * random.uniform(-1, 1)
        ay = 0.2 * math.sin(2 * math.pi * 0.55 * t + 1.0) + 0.05 * random.uniform(-1, 1)
        az = 0.15 * math.sin(2 * math.pi * 0.45 * t + 2.0) + 0.05 * random.uniform(-1, 1)
        return {"ecg": ecg, "ax": ax, "ay": ay, "az": az}

    try:
        raw = ser.readline()
        data = parse_serial_line(raw)
        if data is None:
            return None
        return data
    except Exception as e:
        print("⚠️ Serial read error:", e)
        ser = try_open_serial()  # try to heal
        return None

# -----------------------------
# ROLLING BUFFERS FOR FEATURES
# -----------------------------
# ~5 seconds of data; if your actual sample rate differs, adjust maxlen accordingly.
ECG_BUFFER = deque(maxlen=max(int(WINDOW_SECONDS * FS_ECG_GUESS), 20))
AX_BUFFER  = deque(maxlen=max(int(WINDOW_SECONDS * FS_ACCEL_GUESS), 20))
AY_BUFFER  = deque(maxlen=max(int(WINDOW_SECONDS * FS_ACCEL_GUESS), 20))
AZ_BUFFER  = deque(maxlen=max(int(WINDOW_SECONDS * FS_ACCEL_GUESS), 20))

def safe_std(x: np.ndarray) -> float:
    return float(np.std(x)) if x.size else 0.0

def extract_features(ecg_buf, ax_buf, ay_buf, az_buf):
    """
    Compute the 11 features expected by the model from recent buffers.
    Mirrors training-time approximations used in LGBMAlgo.py.
    """
    if len(ecg_buf) < 10 or len(ax_buf) < 10:
        return None

    ecg = np.asarray(ecg_buf, dtype=float)
    ax = np.asarray(ax_buf, dtype=float)
    ay = np.asarray(ay_buf, dtype=float)
    az = np.asarray(az_buf, dtype=float)

    accel_mag = np.sqrt(ax * ax + ay * ay + az * az)

    # --- ECG-derived ---
    hr_mean  = float(np.mean(ecg))
    hr_std   = float(np.std(ecg))
    hr_range = float(np.ptp(ecg))
    # slope as average gradient
    hr_slope = float(np.mean(np.gradient(ecg))) if ecg.size > 1 else 0.0
    hrv_proxy = float(np.std(np.diff(ecg))) if ecg.size > 2 else 0.0

    # --- Accel-derived ---
    accel_mean = float(np.mean(accel_mag))
    accel_std  = float(np.std(accel_mag))
    accel_max  = float(np.max(accel_mag))

    # Approximate peaks per second (very simple proxy)
    # Count local maxima in ecg; divide by window duration
    if ecg.size > 2:
        peaks = ((ecg[1:-1] > ecg[:-2]) & (ecg[1:-1] > ecg[2:])).sum()
    else:
        peaks = 0
    duration_s = max(len(ecg) / float(FS_ECG_GUESS), 1e-6)
    peaks_per_sec = float(peaks / duration_s)

    # Stillness features (very rough heuristic)
    low_thresh = float(np.percentile(accel_mag, 20)) if accel_mag.size else 0.0
    hi_thresh  = float(np.percentile(accel_mag, 90)) if accel_mag.size else 0.0
    # if an “impact” occurs, look for quiet period afterwards
    post_impact_still_flag = 1.0 if (accel_max > hi_thresh and accel_std < 0.1) else 0.0
    # “duration” proxy: count samples below low_thresh (not true seconds, but monotonic)
    stillness_duration = float((accel_mag < low_thresh).sum()) / float(FS_ACCEL_GUESS)

    return {
        "hr_mean": hr_mean,
        "hr_std": hr_std,
        "hr_range": hr_range,
        "hr_slope": hr_slope,
        "hrv_proxy": hrv_proxy,
        "accel_mean": accel_mean,
        "accel_std": accel_std,
        "accel_max": accel_max,
        "peaks_per_sec": peaks_per_sec,
        "stillness_duration": stillness_duration,
        "post_impact_still_flag": post_impact_still_flag,
    }

# -----------------------------
# STREAM LOOP
# -----------------------------
def stream_loop():
    """
    Continuously:
      read raw → scale → append to buffers → (if enough) extract 11 features →
      predict → Grok insight → SOS → emit to frontend
    """
    while True:
        try:
            raw = read_sensor_data()
            if not raw:
                eventlet.sleep(STREAM_INTERVAL)
                continue

            # Scale raw inputs toward training-like ranges
            ecg = float(raw["ecg"]) * .75
            ax  = float(raw["ax"])  / ACCEL_SCALE
            ay  = float(raw["ay"])  / ACCEL_SCALE
            az  = float(raw["az"])  / ACCEL_SCALE

            # Append to buffers
            ECG_BUFFER.append(ecg)
            AX_BUFFER.append(ax)
            AY_BUFFER.append(ay)
            AZ_BUFFER.append(az)

            # Only predict once buffers have enough samples
            features = extract_features(ECG_BUFFER, AX_BUFFER, AY_BUFFER, AZ_BUFFER)
            if features is None:
                # Emit raw for charts even before predictions
                payload = {
                    "ecg": ecg, "ax": ax, "ay": ay, "az": az,
                    "prediction": 0.0, "sos": False,
                    "grok_insight": "⏳ Warming up buffer...",
                    "timestamp": time.time(),
                }
                socketio.emit("sensor_data", payload)
                eventlet.sleep(STREAM_INTERVAL)
                continue

            # Align feature dict to model feature order
            live_for_model = {f: float(features.get(f, 0.0)) for f in feature_names}

            # Predict with the model (probability if available)
            pred = predict_from_data(model, feature_names, live_for_model)

            payload = {
                # raw (scaled) for charts
                "ecg": ecg, "ax": ax, "ay": ay, "az": az,
                # derived features
                **features,
                # model
                "prediction": float(pred),
                "grok_insight": analyze_with_grok({"ecg": ecg, "ax": ax, "ay": ay, "az": az, "prediction": float(pred)}),
                "sos": detect_sos({"ecg": ecg, "ax": ax, "ay": ay, "az": az, "prediction": float(pred)}),
                "timestamp": time.time(),
            }

            socketio.emit("sensor_data", payload)

            if PRINT_COMPACT_LOG:
                print(
                    f"ECG={ecg:+.3f} | ax={ax:+.3f} ay={ay:+.3f} az={az:+.3f} | "
                    f"pred={payload['prediction']:.2f} | SOS={'⚠️' if payload['sos'] else 'OK'}"
                )

        except Exception as e:
            print("⚠️ Stream error:", e)
        eventlet.sleep(STREAM_INTERVAL)

# -----------------------------
# ROUTES & SOCKET EVENTS
# -----------------------------
@app.route("/")
def home():
    return jsonify({"message": "PulseGuard real-time backend running."})

@socketio.on("connect")
def on_connect():
    print("✅ Client connected via Socket.IO")
    socketio.emit("message", {"status": "connected"})

# -----------------------------
# MAIN ENTRY
# -----------------------------
if __name__ == "__main__":
    socketio.start_background_task(stream_loop)
    socketio.run(app, host="0.0.0.0", port=5100, debug=True)
