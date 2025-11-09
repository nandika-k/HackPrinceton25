# --- MUST BE FIRST (before any other imports) ---
import eventlet
eventlet.monkey_patch()

import os
import sys
import time
from typing import Dict, Any

# Ensure local packages (Grok/, LightGradientBoostedTree/, etc.) are importable
sys.path.append(os.path.dirname(__file__))

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit

# Your modules
from pulseguard_stream import get_sensor_data, SensorInitError
from LightGradientBoostedTree.LGBMAlgo import load_model, predict_from_data
from Grok.grok_agent import analyze_with_grok
from Grok.sos_agent import detect_sos

# ------------------------
# Config
# ------------------------
PORT = int(os.getenv("PULSEGUARD_PORT", "5100"))
MODEL_PATH = os.getenv("PULSEGUARD_MODEL", os.path.join(os.path.dirname(__file__), "sos_gbt_model.joblib"))
STREAM_INTERVAL_SEC = float(os.getenv("PULSEGUARD_INTERVAL", "0.5"))  # seconds

# ------------------------
# App / Socket
# ------------------------
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# ------------------------
# Model load (once)
# ------------------------
print("✅ Initializing PulseGuard backend...")
model = load_model(MODEL_PATH)
print("✅ Model loaded from:", MODEL_PATH)

# ------------------------
# HTTP routes
# ------------------------
@app.route("/")
def home():
    return jsonify({"message": "PulseGuard backend is running!"})

@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": bool(model)})

@app.route("/api/vitals")
def api_vitals():
    """Return one reading (useful for quick tests without sockets)."""
    try:
        data = get_sensor_data()
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Expect JSON: {"features": [ecg, ax, ay, az, ...]}
    Returns: {"prediction": <value>}
    """
    body = request.get_json(silent=True) or {}
    features = body.get("features")
    if not isinstance(features, list):
        return jsonify({"error": "Missing or invalid 'features' list."}), 400
    pred = predict_from_data(model, features)
    val = pred[0] if hasattr(pred, "__len__") else float(pred)
    return jsonify({"prediction": float(val)})

# ------------------------
# Socket.IO
# ------------------------
@socketio.on("connect")
def on_connect():
    print("✅ Frontend connected via Socket.IO")
    emit("message", {"status": "connected"})

def stream_loop():
    """Continuously read sensor → run ML → Grok → SOS → emit."""
    while True:
        try:
            # 1) Sensor
            data: Dict[str, Any] = get_sensor_data()  # {"ecg": float, "ax": float, "ay": float, "az": float, ...}

            # 2) Predict
            # NOTE: adjust this feature order to match how your model was trained
            features = [data.get("ecg", 0.0), data.get("ax", 0.0), data.get("ay", 0.0), data.get("az", 0.0)]
            pred = predict_from_data(model, features)
            data["prediction"] = float(pred[0] if hasattr(pred, "__len__") else pred)

            # 3) Reasoning (Grok)
            data["grok_insight"] = analyze_with_grok(data)

            # 4) SOS logic
            data["sos"] = bool(detect_sos(data))

            # 5) Emit to frontend
            socketio.emit("sensor_data", data)

        except SensorInitError as se:
            # If serial isn’t available, message once and continue (sim mode will still produce data)
            print(f"⚠️ Sensor init issue: {se}")
            time.sleep(2.0)
        except Exception as e:
            print(f"⚠️ Stream error: {e}")

        eventlet.sleep(STREAM_INTERVAL_SEC)

# ------------------------
# Main
# ------------------------
if __name__ == "__main__":
    socketio.start_background_task(stream_loop)
    # 0.0.0.0 so your frontend can reach it; port must match your React socket URL
    socketio.run(app, host="0.0.0.0", port=PORT, debug=True)
