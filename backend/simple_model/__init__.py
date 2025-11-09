"""
Lightweight fall-risk model utilities shared by training and inference code.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib

from .features import (
    FEATURE_NAMES,
    FS_HZ,
    WINDOW_SEC,
    WINDOW_SIZE,
    FeatureExtractor,
    compute_features,
)

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "pulseguard_simple_model.joblib"


def ensure_model(model_path: Optional[Path] = None) -> Tuple[object, Tuple[str, ...]]:
    """
    Load the trained model bundle from disk. If the artifact does not exist yet,
    train a new classifier and persist it first.

    Returns:
        (sklearn_model, feature_names)
    """
    path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    if not path.exists():
        from .training import train_and_save

        train_and_save(path)

    bundle: Dict[str, object] = joblib.load(path)
    model = bundle["model"]
    features = tuple(bundle["feature_names"])
    return model, features


__all__ = [
    "DEFAULT_MODEL_PATH",
    "FEATURE_NAMES",
    "FS_HZ",
    "FeatureExtractor",
    "WINDOW_SEC",
    "WINDOW_SIZE",
    "compute_features",
    "ensure_model",
]
