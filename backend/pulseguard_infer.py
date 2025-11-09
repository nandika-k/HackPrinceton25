# ================================================
# LGBMAlgo_infer.py
# For PulseGuard real-time inference
# ================================================
import joblib
import pandas as pd
import numpy as np

def load_model(path="sos_gbt_model.joblib"):
    """Load model and its feature names from joblib bundle."""
    bundle = joblib.load(path)
    model = bundle["model"]
    feature_names = bundle["features"]
    print(f"✅ Model loaded with {len(feature_names)} features from {path}")
    return model, feature_names


def predict_from_data(model, feature_names, live_data):
    """
    live_data: dict like {"ecg": ..., "ax": ..., "ay": ..., "az": ...}
    Returns predicted SOS probability (float)
    """
    # Create DataFrame to preserve feature names & order
    X = pd.DataFrame([[live_data.get(f, 0.0) for f in feature_names]],
                     columns=feature_names)

    try:
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X)[:, 1]  # probability of SOS
            return float(y_score[0])
        else:
            y_pred = model.predict(X)
            return float(y_pred[0])
    except Exception as e:
        print(f"⚠️ Prediction error: {e}")
        # Fallback using NumPy in case of schema mismatch
        X_np = np.array([[live_data.get(f, 0.0) for f in feature_names]])
        return float(model.predict(X_np)[0])

if __name__ == "__main__":
    model, features = load_model("sos_gbt_model.joblib")
    print("Model expects features:", features)
