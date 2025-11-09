from __future__ import annotations

from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from .features import FEATURE_NAMES, FS_HZ, WINDOW_SEC, WINDOW_SIZE, compute_features


def _simulate_window(is_fall: bool, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic window of ECG and accelerometer samples.
    The statistics loosely mimic calm vs fall scenarios.
    """
    t = np.linspace(0.0, WINDOW_SEC, WINDOW_SIZE, endpoint=False)

    # ECG baseline around 80 bpm (arbitrary units scaled for our visualizations)
    ecg_base = 80.0 + 15.0 * np.sin(2 * np.pi * 1.2 * t)
    ecg_noise = rng.normal(scale=8.0, size=WINDOW_SIZE)
    ecg = ecg_base + ecg_noise

    if is_fall:
        # ECG suppression towards end of window to mimic distress
        drop = np.linspace(0, -45, WINDOW_SIZE)
        ecg = ecg + drop

    # Accelerometer magnitude baseline around 1 g
    accel_mag = 1.0 + rng.normal(scale=0.08, size=WINDOW_SIZE)

    if is_fall:
        # Impact spike
        spike_center = rng.integers(low=WINDOW_SIZE // 4, high=3 * WINDOW_SIZE // 4)
        spike_width = rng.integers(low=3, high=10)
        spike_height = rng.uniform(2.5, 4.5)
        start = max(0, spike_center - spike_width // 2)
        end = min(WINDOW_SIZE, spike_center + spike_width // 2)
        accel_mag[start:end] += spike_height

        # Post-impact stillness (reduced magnitude)
        post = slice(end, WINDOW_SIZE)
        if post.start < WINDOW_SIZE:
            accel_mag[post] = accel_mag[post] * rng.uniform(0.1, 0.3)

    # Sample random orientation per timestep
    orientation = rng.normal(size=(WINDOW_SIZE, 3))
    orientation_norm = np.linalg.norm(orientation, axis=1, keepdims=True) + 1e-8
    unit_orientation = orientation / orientation_norm
    accel = unit_orientation * accel_mag[:, None]

    return ecg.astype(float), accel.astype(float)


def _generate_dataset(
    n_samples: int = 2400,
    fall_probability: float = 0.35,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_seed)

    feature_rows = []
    labels = []

    for _ in range(n_samples):
        is_fall = rng.random() < fall_probability
        ecg_window, accel_window = _simulate_window(is_fall, rng)
        feature_dict = compute_features(ecg_window, accel_window)
        feature_rows.append([feature_dict[name] for name in FEATURE_NAMES])
        labels.append(int(is_fall))

    X = np.array(feature_rows, dtype=float)
    y = np.array(labels, dtype=int)
    return X, y


def train_and_save(model_path: Path) -> None:
    """
    Train the lightweight logistic classifier and persist it to disk.
    """
    X, y = _generate_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=2024
    )

    clf = LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        solver="lbfgs",
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred)
    print("Simple logistic model evaluation:\n", report)

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": clf,
            "feature_names": FEATURE_NAMES,
            "fs_hz": FS_HZ,
            "window_sec": WINDOW_SEC,
        },
        model_path,
    )
    print(f"Saved simple model to {model_path}")


__all__ = ["train_and_save"]
