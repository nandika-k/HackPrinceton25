import logging
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import serial
import pandas as pd
from flask import Flask
from flask_socketio import SocketIO

# Ensure project root is on the Python path for local imports
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from LightGradientBoostedTree.LGBMAlgo import extract_accel, extract_ecg  # noqa: E402

try:
    import joblib
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise RuntimeError(
        "joblib is required to run the streaming predictor. "
        "Install dependencies via `pip install -r requirements.txt`."
    ) from exc

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
)


SERIAL_PORT = os.getenv("SERIAL_PORT", "/dev/tty.usbmodem21101")
BAUD_RATE = int(os.getenv("BAUD_RATE", "9600"))
MODEL_PATH = Path(
    os.getenv("MODEL_PATH", PROJECT_ROOT / "sos_gbt_model.joblib")
).resolve()
WINDOW_SEC = float(os.getenv("MODEL_WINDOW_SEC", "5"))
STEP_SEC = float(os.getenv("MODEL_STEP_SEC", "1"))
FS_ECG = int(os.getenv("ECG_FS", "125"))
FS_ACCEL = int(os.getenv("ACCEL_FS", str(FS_ECG)))
ALERT_THRESHOLD = float(os.getenv("MODEL_ALERT_THRESHOLD", "0.6"))
SIMULATE_STREAM = os.getenv("SIMULATE_SENSOR_STREAM", "0") == "1"


class StreamingPredictor:
    """Accumulates streaming samples and runs the LightGBM model on sliding windows."""

    def __init__(
        self,
        model_path: Path,
        fs_ecg: int,
        fs_accel: int,
        window_sec: float,
        step_sec: float,
        alert_threshold: float,
    ) -> None:
        self.enabled = False
        self._model = None
        self._feature_order = []
        self._alert_threshold = alert_threshold

        self.fs_ecg = fs_ecg
        self.fs_accel = fs_accel
        self.window_sec = window_sec
        self.step_sec = step_sec

        self.window_samples_ecg = max(int(window_sec * fs_ecg), 1)
        self.window_samples_acc = max(int(window_sec * fs_accel), 1)
        self.step_samples = max(int(step_sec * fs_ecg), 1)

        self.ecg_buffer: deque[float] = deque(maxlen=self.window_samples_ecg)
        self.acc_buffer: deque[tuple[float, float, float]] = deque(
            maxlen=self.window_samples_acc
        )
        self.samples_since_emit = 0

        try:
            bundle = joblib.load(str(model_path))
            self._model = bundle["model"]
            self._feature_order = list(bundle["features"])
            self.enabled = True
            logger.info("Loaded LightGBM model from %s", model_path)
        except FileNotFoundError:
            logger.warning("Model file not found at %s. Predictor disabled.", model_path)
        except Exception as exc:  # pragma: no cover - runtime safeguard
            logger.exception("Failed to load model: %s", exc)
            self.enabled = False

    def add_sample(
        self, ecg: float, ax: float, ay: float, az: float, timestamp: float
    ) -> Optional[Dict[str, float]]:
        if not self.enabled:
            return None

        self.ecg_buffer.append(ecg)
        self.acc_buffer.append((ax, ay, az))
        self.samples_since_emit += 1

        if (
            len(self.ecg_buffer) < self.window_samples_ecg
            or len(self.acc_buffer) < self.window_samples_acc
            or self.samples_since_emit < self.step_samples
        ):
            return None

        feature_map = self._compute_features()
        if feature_map is None:
            return None

        feature_vector = pd.DataFrame(
            [{name: feature_map.get(name, 0.0) for name in self._feature_order}],
            columns=self._feature_order,
        )

        try:
            proba = float(self._model.predict_proba(feature_vector)[0, 1])
        except Exception as exc:  # pragma: no cover - runtime safeguard
            logger.exception("Prediction failed: %s", exc)
            return None

        self.samples_since_emit = 0
        risk_level = self._risk_level(proba)

        return {
            "probability": proba,
            "risk_level": risk_level,
            "threshold": self._alert_threshold,
            "timestamp": timestamp,
            "features": feature_map,
        }

    def _compute_features(self) -> Optional[Dict[str, float]]:
        if len(self.ecg_buffer) == 0 or len(self.acc_buffer) == 0:
            return None

        ecg_arr = np.asarray(self.ecg_buffer, dtype=float)
        acc_arr = np.asarray(self.acc_buffer, dtype=float)

        feat_ecg = extract_ecg(ecg_arr, self.fs_ecg)
        feat_acc = extract_accel(acc_arr, self.fs_accel)

        return {**feat_ecg, **feat_acc}

    def _risk_level(self, probability: float) -> str:
        if probability >= self._alert_threshold:
            return "high"
        if probability >= 0.5 * self._alert_threshold:
            return "medium"
        return "low"


predictor = StreamingPredictor(
    model_path=MODEL_PATH,
    fs_ecg=FS_ECG,
    fs_accel=FS_ACCEL,
    window_sec=WINDOW_SEC,
    step_sec=STEP_SEC,
    alert_threshold=ALERT_THRESHOLD,
)


def _emit_sensor_event(ecg: float, ax: float, ay: float, az: float, ts: float) -> None:
    socketio.emit(
        "sensor_data",
        {
            "ecg": ecg,
            "ax": ax,
            "ay": ay,
            "az": az,
            "timestamp": ts,
        },
    )


def _process_prediction(ecg: float, ax: float, ay: float, az: float, ts: float) -> None:
    result = predictor.add_sample(ecg, ax, ay, az, ts)
    if result is None:
        return

    socketio.emit("prediction", result)


def read_serial() -> None:
    logger.info("Starting serial reader on %s @ %s baud", SERIAL_PORT, BAUD_RATE)

    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        ser.reset_input_buffer()
    except Exception as exc:
        logger.error("Serial connection error: %s", exc)
        if SIMULATE_STREAM:
            logger.info("Falling back to simulated sensor stream.")
            _simulate_sensor_stream()
        return

    with ser:
        while True:
            try:
                line = ser.readline().decode("utf-8").strip()
            except Exception as exc:
                logger.debug("Read error: %s", exc)
                continue

            if not line or "," not in line:
                continue

            try:
                ecg, ax, ay, az = map(float, line.split(","))
            except Exception as exc:
                logger.debug("Parse error for line '%s': %s", line, exc)
                continue

            ts = time.time()
            _emit_sensor_event(ecg, ax, ay, az, ts)
            _process_prediction(ecg, ax, ay, az, ts)


def _simulate_sensor_stream() -> None:
    """Generate synthetic ECG/accel values for local development without hardware."""
    logger.info("Starting simulated sensor stream (development mode).")
    t = 0
    dt = 1.0 / max(FS_ECG, 1)

    rng = np.random.default_rng()

    while True:
        ts = time.time()
        ecg = 100 * np.sin(2 * np.pi * 1.2 * t) + rng.normal(0, 5)
        ax = 0.1 * np.sin(2 * np.pi * 0.8 * t) + rng.normal(0, 0.05)
        ay = 0.1 * np.sin(2 * np.pi * 1.1 * t + np.pi / 6) + rng.normal(0, 0.05)
        az = 1.0 + 0.1 * np.sin(2 * np.pi * 0.6 * t + np.pi / 3) + rng.normal(0, 0.05)

        # Occasional simulated fall impulse
        if int(t) % 30 == 0 and 0.0 <= (t % 30.0) < dt:
            ax += rng.normal(2.5, 0.5)
            ay += rng.normal(2.5, 0.5)
            az += rng.normal(2.5, 0.5)

        _emit_sensor_event(ecg, ax, ay, az, ts)
        _process_prediction(ecg, ax, ay, az, ts)

        t += dt
        time.sleep(dt)


@socketio.on("connect")
def handle_connect() -> None:
    logger.info("Client connected to Socket.IO stream.")


if __name__ == "__main__":
    threading.Thread(target=read_serial, daemon=True).start()
    socketio.run(app, host="0.0.0.0", port=5100)
