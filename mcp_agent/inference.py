"""
Inference module for fall detection using trained LightGBM model.
Processes sensor data and makes predictions.
"""
import os
import numpy as np
import pandas as pd
import joblib
from typing import Dict, Tuple, Optional
import sys

# Add parent directory to path to import feature extraction functions
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from LightGradientBoostedTree.LGBMAlgo import extract_ecg, extract_accel
except ImportError:
    # Fallback: define functions here if import fails
    import numpy as np
    
    def extract_ecg(ecg_signal, fs):
        """Extract ECG features (fallback implementation)."""
        x = np.asarray(ecg_signal, dtype=float)
        if x.size == 0:
            return {"hr_mean": 0.0, "hr_std": 0.0, "hr_range": 0.0, "hr_slope": 0.0, "hrv_proxy": 0.0}
        hr_mean = float(np.mean(x))
        hr_std = float(np.std(x))
        hr_range = float(np.max(x) - np.min(x))
        t = np.arange(len(x)) / fs
        slope = float(np.polyfit(t, x, 1)[0]) if len(t) > 1 else 0.0
        hrv_proxy = float(np.std(np.diff(x))) if len(x) > 1 else 0.0
        return {"hr_mean": hr_mean, "hr_std": hr_std, "hr_range": hr_range, "hr_slope": slope, "hrv_proxy": hrv_proxy}
    
    def extract_accel(accel_xyz, fs):
        """Extract accelerometer features (fallback implementation)."""
        a = np.asarray(accel_xyz, dtype=float)
        if a.ndim == 1:
            if len(a) % 3 == 0:
                a = a.reshape(-1, 3)
            else:
                return {"accel_mean": 0.0, "accel_std": 0.0, "accel_max": 0.0, "peaks_per_sec": 0.0, "stillness_duration": 0.0, "post_impact_still_flag": 0.0}
        if a.ndim != 2 or a.shape[0] == 0 or a.shape[1] != 3:
            return {"accel_mean": 0.0, "accel_std": 0.0, "accel_max": 0.0, "peaks_per_sec": 0.0, "stillness_duration": 0.0, "post_impact_still_flag": 0.0}
        mag = np.linalg.norm(a, axis=1)
        mag_mean = float(np.mean(mag))
        mag_std = float(np.std(mag))
        mag_max = float(np.max(mag))
        impact_thresh = float(np.percentile(mag, 90))
        peaks = mag > impact_thresh
        peaks_per_sec = float(peaks.sum() / (len(mag) / fs + 1e-6))
        low_thresh = float(np.percentile(mag, 20))
        last_peak_idx = np.where(peaks)[0]
        if len(last_peak_idx) > 0:
            last_peak_idx = last_peak_idx[-1]
            post_segment = mag[last_peak_idx:]
            still_mask = post_segment < low_thresh
            stillness_duration = float(still_mask.sum() / fs)
            post_impact_still_flag = float(stillness_duration > 3.0)
        else:
            stillness_duration = 0.0
            post_impact_still_flag = 0.0
        return {"accel_mean": mag_mean, "accel_std": mag_std, "accel_max": mag_max, "peaks_per_sec": peaks_per_sec, "stillness_duration": stillness_duration, "post_impact_still_flag": post_impact_still_flag}


class FallDetectionModel:
    """Wrapper for the trained fall detection model."""
    
    def __init__(self, model_path: str = "sos_gbt_model.joblib"):
        """
        Initialize the fall detection model.
        
        Args:
            model_path: Path to the saved model file
        """
        self.model_path = model_path
        self.model = None
        self.feature_cols = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model and feature list."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}. "
                "Please train the model first using LGBMAlgo.py"
            )
        
        model_data = joblib.load(self.model_path)
        self.model = model_data["model"]
        self.feature_cols = model_data["features"]
        print(f"Model loaded successfully. Features: {len(self.feature_cols)}")
    
    def extract_features(self, ecg_signal: np.ndarray, accel_xyz: np.ndarray, 
                        fs_ecg: float = 125.0, fs_accel: float = 125.0) -> Dict:
        """
        Extract features from sensor data.
        
        Args:
            ecg_signal: 1D array of ECG/HR data
            accel_xyz: (N, 3) array of accelerometer data [ax, ay, az]
            fs_ecg: Sampling frequency for ECG (Hz)
            fs_accel: Sampling frequency for accelerometer (Hz)
        
        Returns:
            Dictionary of extracted features
        """
        f_ecg = extract_ecg(ecg_signal, fs_ecg)
        f_acc = extract_accel(accel_xyz, fs_accel)
        
        # Combine features
        features = {**f_ecg, **f_acc}
        return features
    
    def predict(self, features: Dict) -> Tuple[int, float, Dict]:
        """
        Make a prediction on extracted features.
        
        Args:
            features: Dictionary of extracted features
        
        Returns:
            Tuple of (prediction, probability, feature_dict)
            - prediction: 0 (no fall) or 1 (fall detected)
            - probability: Confidence score [0, 1]
            - feature_dict: Ordered feature values for model input
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Convert features to array in correct order
        feature_array = np.array([features.get(col, 0.0) for col in self.feature_cols])
        feature_array = feature_array.reshape(1, -1)
        
        # Handle NaN/Inf
        feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Make prediction
        prediction = self.model.predict(feature_array)[0]
        probability = self.model.predict_proba(feature_array)[0, 1]  # Probability of fall
        
        feature_dict = {col: float(val) for col, val in zip(self.feature_cols, feature_array[0])}
        
        return int(prediction), float(probability), feature_dict
    
    def predict_from_sensors(self, ecg_signal: np.ndarray, accel_xyz: np.ndarray,
                            fs_ecg: float = 125.0, fs_accel: float = 125.0) -> Tuple[int, float, Dict]:
        """
        Complete pipeline: extract features and make prediction.
        
        Args:
            ecg_signal: 1D array of ECG/HR data
            accel_xyz: (N, 3) array of accelerometer data
            fs_ecg: Sampling frequency for ECG (Hz)
            fs_accel: Sampling frequency for accelerometer (Hz)
        
        Returns:
            Tuple of (prediction, probability, features)
        """
        features = self.extract_features(ecg_signal, accel_xyz, fs_ecg, fs_accel)
        return self.predict(features)

