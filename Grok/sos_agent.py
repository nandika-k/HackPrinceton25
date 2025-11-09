# sos_training.py
import os
import glob
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    average_precision_score,
    precision_recall_curve,
)
import lightgbm as lgb
import joblib

# -----------------------------
# PATHS
# -----------------------------
ECG_ROOT = "data/ecg_mitbih"
WEDA_ROOT = "WEDA-FALL-data-source/dataset/125Hz"

# -----------------------------
# SAMPLING RATES
# -----------------------------
HR_FS = 125    # 125 Hz HR (SpO2 / HR proxy)
ACC_FS = 125   # 125 Hz accelerometer

# Aliases for old naming, in case you re-enable HIFD/WEDA helpers later
FS_HR_DEFAULT = HR_FS
FS_ACCEL_DEFAULT = ACC_FS

# -----------------------------
# GEO FUSION PARAMS (used at inference)
# -----------------------------
ALPHA_DEFAULT = 0.5   # weight of hazard index in risk fusion


# -----------------------------
# FEATURE EXTRACTION
# -----------------------------
def extract_ecg(ecg_signal, fs):
    """For MVP: treat ecg_signal as filtered RR proxy, converting visual to numerical.
    1D numpy array means len ~ window_sec * fs.
    """
    x = np.asarray(ecg_signal, dtype=float)
    if x.size == 0:
        return {
            "hr_mean": 0.0,
            "hr_std": 0.0,
            "hr_range": 0.0,
            "hr_slope": 0.0,
            "hrv_proxy": 0.0,
        }

    hr_mean = float(np.mean(x))
    hr_std = float(np.std(x))
    hr_range = float(np.max(x) - np.min(x))

    # Linear trend: fit HR slope over time
    t = np.arange(len(x)) / fs
    if len(t) > 1:
        slope = float(np.polyfit(t, x, 1)[0])
    else:
        slope = 0.0

    # HRV proxy: std of first differences (heart rate variability indicator)
    if len(x) > 1:
        hrv_proxy = float(np.std(np.diff(x)))
    else:
        hrv_proxy = 0.0

    return {
        "hr_mean": hr_mean,
        "hr_std": hr_std,
        "hr_range": hr_range,
        "hr_slope": slope,
        "hrv_proxy": hrv_proxy,
    }


def _extract_acc_core(acc_xyz, fs):
    """
    acc_xyz: (N, 3) array of ax, ay, az
    fs: sampling frequency in Hz
    """
    a = np.asarray(acc_xyz, dtype=float)

    # Handle different input shapes
    if a.ndim == 1:
        # If 1D, try to reshape to (N, 3) if length is divisible by 3
        if len(a) % 3 == 0:
            a = a.reshape(-1, 3)
        else:
            # invalid accel vector -> return neutral features
            return {
                "acc_mean": 0.0,
                "acc_std": 0.0,
                "acc_max": 0.0,
                "peak_per_sec": 0.0,
                "stillness_duration": 0.0,
                "after_impact_still_flag": 0.0,
            }

    if a.ndim != 2 or a.shape[0] == 0 or a.shape[1] != 3:
        # invalid accel matrix -> neutral
        return {
            "acc_mean": 0.0,
            "acc_std": 0.0,
            "acc_max": 0.0,
            "peak_per_sec": 0.0,
            "stillness_duration": 0.0,
            "after_impact_still_flag": 0.0,
        }

    value = np.linalg.norm(a, axis=1)
    value_mean = float(np.mean(value))
    value_std = float(np.std(value))
    value_max = float(np.max(value))

    # SOS situation detection: peaks > threshold
    detection_threshold = float(np.percentile(value, 90))
    peaks = value > detection_threshold
    peaks_per_sec = float(peaks.sum() / (len(value) / fs + 1e-6))

    # Stillness detection: low acceleration after impact peak (indicates potential unconsciousness)
    low_thresh = float(np.percentile(value, 20))
    last_peak_idx_list = np.where(peaks)[0]
    if len(last_peak_idx_list) > 0:
        last_peak_idx = last_peak_idx_list[-1]
        post_segment = value[last_peak_idx:]
        still_mask = post_segment < low_thresh
        stillness_duration = float(still_mask.sum() / fs)
        after_impact_still_flag = float(stillness_duration > 3.0)  # >3s still = concerning
    else:
        stillness_duration = 0.0
        after_impact_still_flag = 0.0

    return {
        "acc_mean": value_mean,
        "acc_std": value_std,
        "acc_max": value_max,
        "peak_per_sec": peaks_per_sec,
        "stillness_duration": stillness_duration,
        "after_impact_still_flag": after_impact_still_flag,
    }


def extract_acc(accel_xyz, fs):
    """
    Public alias. Use this everywhere.
    """
    return _extract_acc_core(accel_xyz, fs)


# Optional backwards-compat alias if you re-enable old code
extract_accel = extract_acc


def sliding_windows(sig_len, window_sec, step_sec, fs):
    win = int(window_sec * fs)
    step = int(step_sec * fs)
    for start in range(0, sig_len - win + 1, step):
        end = start + win
        yield start, end


# -----------------------------
# SYNTHETIC DATA GENERATION
# -----------------------------
def simulate_window(window_sec=20, fs_ecg=5, fs_accel=50, rng=None):
    """
    Creates one synthetic training window (ECG + acc).
    Returns (ecg_signal, acc_xyz, label, scenario_str)
    """
    if rng is None:
        rng = np.random

    scen = rng.choice(
        ["normal", "run", "trip", "fall_unconscious", "collapse", "panic"],
        p=[0.25, 0.2, 0.15, 0.15, 0.1, 0.15],
    )

    # ECG / HR-like signal
    n_ecg = window_sec * fs_ecg
    t_ecg = np.linspace(0, window_sec, n_ecg)

    if scen == "normal":
        hr_base = 70
        hr = hr_base + 3 * rng.randn(n_ecg)
    elif scen in ["run", "trip"]:
        hr_base = 120
        hr = hr_base + 5 * rng.randn(n_ecg)
    elif scen in ["fall_unconscious", "collapse"]:
        hr = 110 + 10 * rng.randn(n_ecg)
        hr -= np.linspace(0, 30, n_ecg)  # drop towards end
    elif scen == "panic":
        hr = 100 + 15 * np.sin(0.5 * t_ecg) + 10 * rng.randn(n_ecg)
    else:
        hr = 80 + 5 * rng.randn(n_ecg)

    # Accel signal
    n_accel = window_sec * fs_accel
    t_accel = np.linspace(0, window_sec, n_accel)
    accel = 0.1 * rng.randn(n_accel, 3)  # base noise

    if scen == "normal":
        accel += 0.2 * np.sin(2 * np.pi * 1.0 * t_accel)[:, None]
    elif scen == "run":
        accel += 0.7 * np.sign(np.sin(2 * np.pi * 2.0 * t_accel))[:, None]
    elif scen == "trip":
        impact_idx = int(0.5 * n_accel)
        accel[impact_idx:impact_idx + 5] += 4.0
        accel += 0.5 * np.sin(2 * np.pi * 1.5 * t_accel)[:, None]
    elif scen == "fall_unconscious":
        impact_idx = int(0.5 * n_accel)
        accel[impact_idx:impact_idx + 5] += 5.0
        accel[impact_idx + 50:] *= 0.05
    elif scen == "collapse":
        cut = int(0.3 * n_accel)
        accel[:cut] += 0.3 * np.sin(2 * np.pi * 1.0 * t_accel[:cut])[:, None]
        accel[cut:] *= 0.05
    elif scen == "panic":
        accel += 0.3 * rng.randn(n_accel, 3)

    label = int(scen in ["fall_unconscious", "collapse", "panic"])
    return hr, accel, label, scen


def build_synthetic_dataset(n_samples=1000, window_sec=20, fs_ecg=5, fs_accel=50, random_seed=42):
    """
    Build synthetic feature dataframe using your generator.
    """
    rows = []
    labels = []
    for i in range(n_samples):
        sample_rng = np.random.RandomState(random_seed + i)
        hr, accel, y, scen = simulate_window(window_sec, fs_ecg, fs_accel, rng=sample_rng)
        f_ecg = extract_ecg(hr, fs_ecg)
        f_acc = extract_acc(accel, fs_accel)
        row = {**f_ecg, **f_acc, "scenario": f"SYN_{scen}"}
        rows.append(row)
        labels.append(y)
    df = pd.DataFrame(rows)
    df["label"] = labels
    return df


# -----------------------------
# ECG MIT-BIH DATA
# -----------------------------
def load_ecg_mitbih_rows(data_root):
    """
    Expects last column as label (0 = normal, >0 = arrhythmia).
    """
    records = []
    train_path = os.path.join(data_root, "mitbih_train.csv")
    test_path = os.path.join(data_root, "mitbih_test.csv")

    if os.path.isfile(train_path):
        try:
            df_train = pd.read_csv(train_path, header=None)
            records.append(df_train)
        except Exception as e:
            print(f"WARNING: Failed to read {train_path}: {e}")
    if os.path.isfile(test_path):
        try:
            df_test = pd.read_csv(test_path, header=None)
            records.append(df_test)
        except Exception as e:
            print(f"WARNING: Failed to read {test_path}: {e}")

    if not records:
        return []

    df_all = pd.concat(records, ignore_index=True)
    ecg_windows = df_all.iloc[:, :-1].values  # except last column
    labels = df_all.iloc[:, -1].values        # last column is class label

    rows = []
    for ecg_win, lab in zip(ecg_windows, labels):
        rows.append({
            "ecg": ecg_win.astype(float),
            "fs": 125.0,           # typical sample rate used for these 187-sample beats
            "label_str": "normal" if lab == 0 else "abnormal",
        })
    return rows


def build_from_ecg_mitbih(data_root):
    rows = []
    labels = []

    records = load_ecg_mitbih_rows(data_root)
    for rec in records:
        ecg = rec["ecg"]
        fs = rec["fs"]
        label_str = rec["label_str"]

        f_ecg = extract_ecg(ecg, fs)

        # neutral acc: assume resting body
        f_acc = {
            "acc_mean": 0.1,
            "acc_std": 0.05,
            "acc_max": 0.2,
            "peak_per_sec": 0.0,
            "stillness_duration": len(ecg) / fs,
            "after_impact_still_flag": 0.0,
        }

        label = 1 if label_str != "normal" else 0
        scen = f"ECG_{label_str}"

        row = {**f_ecg, **f_acc, "scenario": scen}
        rows.append(row)
        labels.append(label)

    df = pd.DataFrame(rows)
    if not df.empty:
        df["label"] = labels
    return df


# -----------------------------
# OPTIONAL: WEDA LOADER
# -----------------------------
try:
    from load_weda import build_from_weda
except ImportError:
    build_from_weda = None


# -----------------------------
# HAZARD-WEIGHTED RISK FUSION
# -----------------------------
def compute_hazard_weighted_risk(p_sos, hazard_index, alpha=ALPHA_DEFAULT):
    """
    Fuse ML SOS probability with hazard index.
    p_sos: model probability (0..1)
    hazard_index: geo-hazard (0..1) from external API
    alpha: how strongly hazard boosts the risk
    """
    r = float(p_sos) * (1.0 + alpha * float(hazard_index))
    return float(max(0.0, min(1.0, r)))


# -----------------------------
# TRAINING PIPELINE
# -----------------------------
if __name__ == "__main__":
    dfs = []

    # ECG MITBIH
    df_ecg = build_from_ecg_mitbih(ECG_ROOT)
    if not df_ecg.empty:
        print(f"ECG MITBIH samples: {len(df_ecg)}")
        dfs.append(df_ecg)
    else:
        print("WARNING: No ECG MITBIH data loaded (check CSV).")

    # WEDA-FALL
    if build_from_weda is None:
        print("WARNING: WEDA-FALL loader not available (load_weda.py import failed).")
    else:
        weda_root = WEDA_ROOT
        df_weda = build_from_weda(
            weda_root,
            window_sec=20,
            step_sec=10,
            sensor_type="accel",
            target_fs=125.0,
            use_hr=False,
        )
        if not df_weda.empty:
            print(f"WEDA-FALL samples: {len(df_weda)}")
            dfs.append(df_weda)
        else:
            print("WARNING: No WEDA-FALL data loaded (check folders or run weda_resample.py).")

    # Synthetic
    df_synth = build_synthetic_dataset(n_samples=1000)
    print(f"Synthetic samples: {len(df_synth)}")
    dfs.append(df_synth)

    if not dfs:
        raise RuntimeError("No datasets loaded. Check dataset paths.")

    df = pd.concat(dfs, ignore_index=True)
    print(f"Total samples: {len(df)}")

    feature_cols = [c for c in df.columns if c not in ["label", "scenario"]]
    X = df[feature_cols].values
    y = df["label"].values

    # Clean NaN/inf
    if np.isnan(X).any() or np.isinf(X).any():
        print("WARNING: Found NaN or Inf values in features. Filling them.")
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    print(f"Feature columns: {feature_cols}")
    print(f"Feature matrix shape: {X.shape}")
    unique_labels, label_counts = np.unique(y, return_counts=True)
    print(f"Label distribution: {dict(zip(unique_labels, label_counts))}")

    can_stratify = len(unique_labels) > 1 and np.min(label_counts) >= 5
    if can_stratify:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
    else:
        print("WARNING: Cannot stratify split; using random split")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    # Baseline: Logistic Regression (can be used for tiny offline fallback if needed)
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    y_scores_lr = lr.predict_proba(X_test)[:, 1]
    print("\nLogistic baseline report:")
    print(classification_report(y_test, y_pred_lr))

    # LightGBM hero model
    lgbm = lgb.LGBMClassifier(
        n_estimators=200,
        num_leaves=31,
        learning_rate=0.05,
        class_weight="balanced",
        random_state=42,
    )
    lgbm.fit(X_train, y_train)
    y_pred_lgbm = lgbm.predict(X_test)
    y_scores_lgbm = lgbm.predict_proba(X_test)[:, 1]

    print("LightGBM report:")
    print(classification_report(y_test, y_pred_lgbm))

    ap_lr = average_precision_score(y_test, y_scores_lr)
    ap_lgbm = average_precision_score(y_test, y_scores_lgbm)
    print(f"Average precision LR: {ap_lr:.3f}, LGBM: {ap_lgbm:.3f}")

    # -----------------------------
    # CHOOSE THRESHOLDS FOR AGENT
    # -----------------------------
    precision, recall, thresholds = precision_recall_curve(y_test, y_scores_lgbm)

    # SOS threshold: smallest threshold where recall >= 0.90 (or fallback 0.5)
    sos_threshold = 0.5
    prealert_threshold = 0.3
    try:
        idx_90 = np.where(recall >= 0.90)[0][0]
        sos_threshold = float(thresholds[idx_90]) if idx_90 < len(thresholds) else 0.5
        print(f"Chosen SOS threshold ≈ {sos_threshold:.3f} (recall ≥ 0.90, precision ≈ {precision[idx_90]:.3f})")
    except Exception:
        print("Could not find threshold for recall ≥ 0.90; using 0.5")

    # Prealert threshold: more sensitive (higher recall), e.g. recall ≥ 0.98, else 0.5 * sos_threshold
    try:
        idx_98 = np.where(recall >= 0.98)[0][0]
        prealert_threshold = float(thresholds[idx_98]) if idx_98 < len(thresholds) else 0.5 * sos_threshold
        print(f"Chosen PREALERT threshold ≈ {prealert_threshold:.3f} (recall ≥ 0.98)")
    except Exception:
        prealert_threshold = 0.5 * sos_threshold
        print("Could not find threshold for recall ≥ 0.98; using 0.5 * sos_threshold")

    meta = {
        "features": feature_cols,
        "alpha_default": ALPHA_DEFAULT,
        "sos_threshold": sos_threshold,
        "prealert_threshold": prealert_threshold,
    }

    # Save model + meta for Grok agent
    joblib.dump(
        {"model": lgbm, "logistic": lr, "features": feature_cols, "meta": meta},
        "sos_gbt_model.joblib",
    )
    print("Saved model + meta to sos_gbt_model.joblib")
