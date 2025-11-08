"""
Fall Detection Service - Wraps the trained model for easy integration.
"""
import os
import numpy as np
import joblib
import logging
from typing import Dict, Optional, Tuple
from datetime import datetime

# Configure logging first
logger = logging.getLogger(__name__)

# Import feature extraction
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from LightGradientBoostedTree.LGBMAlgo import extract_ecg, extract_accel
except ImportError:
    # Fallback if import fails
    logger.warning("Could not import from LGBMAlgo. Using fallback implementations.")
    
    def extract_ecg(ecg_signal, fs):
        x = np.asarray(ecg_signal, dtype=float)
        if x.size == 0:
            return {"hr_mean": 0.0, "hr_std": 0.0, "hr_range": 0.0, "hr_slope": 0.0, "hrv_proxy": 0.0}
        return {
            "hr_mean": float(np.mean(x)),
            "hr_std": float(np.std(x)),
            "hr_range": float(np.max(x) - np.min(x)),
            "hr_slope": float(np.polyfit(np.arange(len(x)) / fs, x, 1)[0]) if len(x) > 1 else 0.0,
            "hrv_proxy": float(np.std(np.diff(x))) if len(x) > 1 else 0.0
        }
    
    def extract_accel(accel_xyz, fs):
        a = np.asarray(accel_xyz, dtype=float)
        if a.ndim != 2 or a.shape[0] == 0 or a.shape[1] != 3:
            return {"accel_mean": 0.0, "accel_std": 0.0, "accel_max": 0.0, "peaks_per_sec": 0.0, "stillness_duration": 0.0, "post_impact_still_flag": 0.0}
        mag = np.linalg.norm(a, axis=1)
        detection_threshold = float(np.percentile(mag, 90))
        peaks = mag > detection_threshold
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
        return {
            "accel_mean": float(np.mean(mag)),
            "accel_std": float(np.std(mag)),
            "accel_max": float(np.max(mag)),
            "peaks_per_sec": peaks_per_sec,
            "stillness_duration": stillness_duration,
            "post_impact_still_flag": post_impact_still_flag
        }


class FallDetectionService:
    """Service for fall detection using trained model."""
    
    def __init__(self, model_path: str = "sos_gbt_model.joblib", fs_ecg: float = 125.0, fs_accel: float = 125.0):
        """
        Initialize fall detection service.
        
        Args:
            model_path: Path to trained model file
            fs_ecg: ECG sampling frequency (Hz)
            fs_accel: Accelerometer sampling frequency (Hz)
        """
        self.model_path = model_path
        self.fs_ecg = fs_ecg
        self.fs_accel = fs_accel
        
        # Load model
        self.model = None
        self.feature_cols = None
        self.load_model()
    
    def load_model(self):
        """Load the trained fall detection model."""
        if not os.path.exists(self.model_path):
            logger.warning(f"Model file not found: {self.model_path}. Using placeholder.")
            return
        
        try:
            model_data = joblib.load(self.model_path)
            self.model = model_data["model"]
            self.feature_cols = model_data["features"]
            logger.info(f"Model loaded successfully. Features: {len(self.feature_cols)}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
    
    def detect_fall(
        self,
        ecg_data: np.ndarray,
        accel_data: np.ndarray,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Detect fall from sensor data.
        
        Args:
            ecg_data: ECG/HR signal (1D array)
            accel_data: Accelerometer data (N x 3 array)
            metadata: Additional metadata
        
        Returns:
            Dictionary with detection results
        """
        try:
            # Extract features
            f_ecg = extract_ecg(ecg_data, self.fs_ecg)
            f_acc = extract_accel(accel_data, self.fs_accel)
            
            # Combine features
            features = {**f_ecg, **f_acc}
            
            # Make prediction if model is loaded
            if self.model is not None and self.feature_cols is not None:
                # Convert to array in correct order
                feature_array = np.array([features.get(col, 0.0) for col in self.feature_cols])
                feature_array = feature_array.reshape(1, -1)
                
                # Handle NaN/Inf
                feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=1e6, neginf=-1e6)
                
                # Predict
                prediction = self.model.predict(feature_array)[0]
                probability = self.model.predict_proba(feature_array)[0, 1]
            else:
                # Fallback: use heuristic based on features
                prediction, probability = self._heuristic_detection(features)
            
            # Determine emergency level
            emergency_level = self._determine_emergency_level(probability, features)
            
            return {
                "fall_detected": bool(prediction == 1),
                "probability": float(probability),
                "emergency_level": emergency_level,
                "features": features,
                "timestamp": datetime.now().isoformat(),
                "model_used": self.model is not None
            }
        
        except Exception as e:
            logger.error(f"Error in fall detection: {e}")
            return {
                "fall_detected": False,
                "probability": 0.0,
                "emergency_level": "unknown",
                "features": {},
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    def _heuristic_detection(self, features: Dict) -> Tuple[int, float]:
        """
        Heuristic fall detection when model is not available.
        Based on key indicators: high accel max, stillness after impact.
        """
        accel_max = features.get("accel_max", 0)
        stillness_flag = features.get("post_impact_still_flag", 0)
        hr_range = features.get("hr_range", 0)
        
        # Simple heuristic
        if accel_max > 3.0 and stillness_flag > 0:
            return 1, 0.8
        elif accel_max > 2.0:
            return 1, 0.6
        else:
            return 0, 0.3
    
    def _determine_emergency_level(self, probability: float, features: Dict) -> str:
        """Determine emergency level based on probability and features."""
        if probability >= 0.9 and features.get("post_impact_still_flag", 0) > 0:
            return "critical"
        elif probability >= 0.8:
            return "high"
        elif probability >= 0.6:
            return "medium"
        elif probability >= 0.4:
            return "low"
        else:
            return "none"

