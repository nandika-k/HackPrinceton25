from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np
from flask import Flask, jsonify
from flask_socketio import SocketIO

from pathlib import Path
import sys

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from simple_model import FeatureExtractor, FS_HZ, ensure_model

try:
    import serial  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    serial = None

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
logger = logging.getLogger("pulseguard")

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

SERIAL_PORT = "/dev/tty.usbmodem21101"
BAUD_RATE = 9600
ALERT_THRESHOLD = 0.65
SYNTHETIC_FALL_PROBABILITY = 0.35


@dataclass
class SensorSample:
    ecg: float
    ax: float
    ay: float
    az: float
    label: Optional[int] = None  # synthetic generator populates this for debugging


class SyntheticDataSource:
    """
    Generates realistic-looking ECG + accelerometer samples for demo/testing
    when no live serial hardware is connected.
    """

    def __init__(self, fs_hz: int = FS_HZ, fall_probability: float = SYNTHETIC_FALL_PROBABILITY) -> None:
        self.fs_hz = fs_hz
        self.window_size = int(fs_hz * 2)  # matches FeatureExtractor default
        self.fall_probability = fall_probability
        self._rng = np.random.default_rng(2025)

        self._ecg_window = np.zeros(self.window_size, dtype=float)
        self._accel_window = np.zeros((self.window_size, 3), dtype=float)
        self._label = 0
        self._index = 0
        self._generate_new_window()

    def __iter__(self) -> "SyntheticDataSource":
        return self

    def __next__(self) -> SensorSample:
        if self._index >= self.window_size:
            self._generate_new_window()
        ecg = float(self._ecg_window[self._index])
        ax, ay, az = map(float, self._accel_window[self._index])
        sample = SensorSample(ecg=ecg, ax=ax, ay=ay, az=az, label=self._label)
        self._index += 1
        return sample

    def _generate_new_window(self) -> None:
        self._label = int(self._rng.random() < self.fall_probability)
        t = np.linspace(0.0, 2.0, self.window_size, endpoint=False)

        ecg_base = 80.0 + 15.0 * np.sin(2 * np.pi * 1.2 * t)
        ecg_noise = self._rng.normal(scale=8.0, size=self.window_size)
        ecg = ecg_base + ecg_noise
        if self._label:
            drop = np.linspace(0, -45, self.window_size)
            ecg = ecg + drop
        self._ecg_window = ecg.astype(float)

        accel_mag = 1.0 + self._rng.normal(scale=0.08, size=self.window_size)
        if self._label:
            center = self._rng.integers(low=self.window_size // 4, high=3 * self.window_size // 4)
            width = self._rng.integers(low=3, high=10)
            height = self._rng.uniform(2.5, 4.5)
            start = max(0, center - width // 2)
            end = min(self.window_size, center + width // 2)
            accel_mag[start:end] += height
            if end < self.window_size:
                accel_mag[end:] *= self._rng.uniform(0.1, 0.3)

        orientation = self._rng.normal(size=(self.window_size, 3))
        orientation /= np.linalg.norm(orientation, axis=1, keepdims=True) + 1e-8
        self._accel_window = (orientation * accel_mag[:, None]).astype(float)
        self._index = 0


class SerialDataSource:
    """
    Wraps a serial port that emits comma-separated ecg, ax, ay, az readings.
    Falls back to synthetic data if the port is unavailable or fails mid-stream.
    """

    def __init__(self, port: str, baud_rate: int) -> None:
        if serial is None:
            raise RuntimeError("pyserial not installed; cannot use SerialDataSource.")

        self._port = port
        self._baud_rate = baud_rate
        self._ser = serial.Serial(port, baud_rate, timeout=1)
        time.sleep(2.0)
        self._ser.reset_input_buffer()
        logger.info("Connected to serial device on %s", port)

    def __iter__(self) -> "SerialDataSource":
        return self

    def __next__(self) -> SensorSample:
        while True:
            line = self._ser.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 4:
                logger.debug("Skipping malformed line: %s", line)
                continue
            try:
                ecg, ax, ay, az = map(float, parts)
            except ValueError:
                logger.debug("Non-numeric sensor line: %s", line)
                continue
            return SensorSample(ecg=ecg, ax=ax, ay=ay, az=az)

    def close(self) -> None:
        if self._ser.is_open:
            self._ser.close()
            logger.info("Serial connection closed")


def create_data_source() -> Iterator[SensorSample]:
    try:
        if serial is not None:
            return SerialDataSource(SERIAL_PORT, BAUD_RATE)
    except Exception as exc:
        logger.warning("Serial connection failed (%s). Falling back to synthetic data.", exc)
    logger.info("Using synthetic data source.")
    return SyntheticDataSource()


def stream_sensor_data() -> None:
    logger.info("Loading lightweight model bundle...")
    model, feature_names = ensure_model()
    extractor = FeatureExtractor()

    data_source = create_data_source()

    for sample in data_source:
        start_time = time.time()
        features = extractor.add_sample(sample.ecg, sample.ax, sample.ay, sample.az)

        payload = {
            "ecg": sample.ecg,
            "ax": sample.ax,
            "ay": sample.ay,
            "az": sample.az,
            "timestamp": start_time,
        }

        if features:
            feature_row = np.array([[features[name] for name in feature_names]], dtype=float)
            risk_score = float(model.predict_proba(feature_row)[0, 1])
            status = "ALERT" if risk_score >= ALERT_THRESHOLD else "NORMAL"
            payload.update(
                {
                    "riskScore": risk_score,
                    "status": status,
                    "featureWindowSeconds": extractor.window_size / extractor.fs_hz,
                }
            )
            # Provide ground truth label when using synthetic source for easier debugging.
            if sample.label is not None:
                payload["syntheticLabel"] = bool(sample.label)

        socketio.emit("sensor_data", payload)

        elapsed = time.time() - start_time
        sleep_time = max(0.0, (1.0 / FS_HZ) - elapsed)
        time.sleep(sleep_time)


@socketio.on("connect")
def handle_connect() -> None:  # pragma: no cover - network side effect
    logger.info("Socket client connected")


@app.route("/health", methods=["GET"])
def healthcheck():
    return jsonify({"status": "ok", "model": "simple-logistic", "fs_hz": FS_HZ})


def main() -> None:
    thread = threading.Thread(target=stream_sensor_data, daemon=True)
    thread.start()
    socketio.run(app, host="0.0.0.0", port=5100, allow_unsafe_werkzeug=True)


if __name__ == "__main__":
    main()
