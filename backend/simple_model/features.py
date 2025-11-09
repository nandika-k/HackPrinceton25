from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional

import numpy as np

# Sampling configuration shared between training and inference
FS_HZ: int = 125  # sampling frequency in Hz
WINDOW_SEC: float = 2.0  # size of feature window in seconds
WINDOW_SIZE: int = int(FS_HZ * WINDOW_SEC)

# Features produced for the classifier (order matters!)
FEATURE_NAMES = [
    "ecg_mean",
    "ecg_std",
    "ecg_range",
    "accel_mag_mean",
    "accel_mag_std",
    "accel_mag_max",
]


def compute_features(ecg_window: np.ndarray, accel_window: np.ndarray) -> Dict[str, float]:
    """
    Compute lightweight summary features for one window of ECG + accelerometer samples.

    Args:
        ecg_window: 1D numpy array of ECG samples (length WINDOW_SIZE)
        accel_window: 2D numpy array of accelerometer samples shaped (WINDOW_SIZE, 3)

    Returns:
        Dictionary mapping feature name -> float value.
    """
    ecg = np.asarray(ecg_window, dtype=float)
    accel = np.asarray(accel_window, dtype=float)

    if ecg.ndim != 1:
        raise ValueError(f"Expected ecg_window to be 1D, got shape {ecg.shape}")
    if accel.ndim != 2 or accel.shape[1] != 3:
        raise ValueError(f"Expected accel_window shape (N, 3), got {accel.shape}")
    if len(ecg) != len(accel):
        raise ValueError("ECG and accelerometer windows must have the same length")

    ecg_mean = float(np.mean(ecg))
    ecg_std = float(np.std(ecg))
    ecg_range = float(np.max(ecg) - np.min(ecg))

    accel_mag = np.linalg.norm(accel, axis=1)
    accel_mag_mean = float(np.mean(accel_mag))
    accel_mag_std = float(np.std(accel_mag))
    accel_mag_max = float(np.max(accel_mag))

    return {
        "ecg_mean": ecg_mean,
        "ecg_std": ecg_std,
        "ecg_range": ecg_range,
        "accel_mag_mean": accel_mag_mean,
        "accel_mag_std": accel_mag_std,
        "accel_mag_max": accel_mag_max,
    }


@dataclass
class FeatureExtractor:
    """
    Maintains rolling buffers of ECG and accelerometer data and exposes windowed features.
    """

    fs_hz: int = FS_HZ
    window_size: int = WINDOW_SIZE

    def __post_init__(self) -> None:
        self._ecg_buffer: Deque[float] = deque(maxlen=self.window_size)
        self._accel_buffer: Deque[np.ndarray] = deque(maxlen=self.window_size)

    def add_sample(self, ecg: float, ax: float, ay: float, az: float) -> Optional[Dict[str, float]]:
        """
        Add a single sensor sample to the buffers. When enough samples are accumulated
        to fill one window, return the computed feature dictionary; otherwise return None.
        """
        self._ecg_buffer.append(float(ecg))
        self._accel_buffer.append(np.array([ax, ay, az], dtype=float))

        if len(self._ecg_buffer) < self.window_size:
            return None

        ecg_window = np.array(self._ecg_buffer, dtype=float)
        accel_window = np.vstack(self._accel_buffer).astype(float)
        return compute_features(ecg_window, accel_window)


__all__ = [
    "FS_HZ",
    "WINDOW_SEC",
    "WINDOW_SIZE",
    "FEATURE_NAMES",
    "FeatureExtractor",
    "compute_features",
]
