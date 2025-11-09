import os
import glob
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, average_precision_score
import lightgbm as lgb
import joblib
import os
import json
import time
from typing import List, Dict, Optional
import requests
from dataclasses import dataclass
import anthropic  # For offline reasoning backup

# ============================================================
# GROK xAI CONFIG
# ============================================================
GROK_API_KEY = os.getenv("GROK_API_KEY", "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")
GROK_API_URL = "https://api.x.ai/v1/chat/completions"  # Grok's endpoint

# HIFD_ROOT   = "data/hifd"  
ECG_ROOT    = "data/ecg_mitbih"
WEDA_ROOT = "data/WEDA-FALL-data/125Hz"  

# Default sampling rates (adjust to match preprocessing)
FS_HR_DEFAULT     = 125    # 125 Hz HR (or HR-derived samples)
FS_ACCEL_DEFAULT  = 125   #125 Hz accelerometer

# Aliases for newer naming convention
HR_FS = FS_HR_DEFAULT
ACC_FS = FS_ACCEL_DEFAULT

#Feature extraction block
def extract_ecg(ecg_signal, fs):
    """
    ecg_signal: 1D numpy array, length~window_sec * fs
    fs: sampling frequency (Hz)
    For MVP: treat ecg_signal as filtered RR proxy, converting visual to numerical
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


def extract_accel(accel_xyz, fs):
    """
    accel_xyz: (N, 3) array of ax, ay, az
    fs: sampling frequency in Hz
    """
    a = np.asarray(accel_xyz, dtype=float)
    # Handle different input shapes
    if a.ndim == 1:
        # If 1D, try to reshape to (N, 3) if length is divisible by 3
        if len(a) % 3 == 0:
            a = a.reshape(-1, 3)
        else:
            return {
                "accel_mean": 0.0,
                "accel_std": 0.0,
                "accel_max": 0.0,
                "peaks_per_sec": 0.0,
                "stillness_duration": 0.0,
                "post_impact_still_flag": 0.0,
            }
    if a.ndim != 2 or a.shape[0] == 0 or a.shape[1] != 3:
        return {
            "accel_mean": 0.0,
            "accel_std": 0.0,
            "accel_max": 0.0,
            "peaks_per_sec": 0.0,
            "stillness_duration": 0.0,
            "post_impact_still_flag": 0.0,
        }

    mag = np.linalg.norm(a, axis=1)

    mag_mean = float(np.mean(mag))
    mag_std = float(np.std(mag))
    mag_max = float(np.max(mag))

    # SOS situation detection: peaks > threshold
    detection_threshold = float(np.percentile(mag, 90))
    peaks = mag > detection_threshold
    peaks_per_sec = float(peaks.sum() / (len(mag) / fs + 1e-6))

    # Stillness detection: low acceleration after impact peak (indicates potential unconsciousness)
    low_thresh = float(np.percentile(mag, 20))
    last_peak_idx = np.where(peaks)[0]
    if len(last_peak_idx) > 0:
        last_peak_idx = last_peak_idx[-1]
        post_segment = mag[last_peak_idx:]
        still_mask = post_segment < low_thresh
        stillness_duration = float(still_mask.sum() / fs)
        post_impact_still_flag = float(stillness_duration > 3.0)  # >3s still = potentially concerning
    else:
        stillness_duration = 0.0
        post_impact_still_flag = 0.0

    return {
        "accel_mean": mag_mean,
        "accel_std": mag_std,
        "accel_max": mag_max,
        "peaks_per_sec": peaks_per_sec,
        "stillness_duration": stillness_duration,
        "post_impact_still_flag": post_impact_still_flag,
    }


def extract_acc(accel_xyz, fs):
    """
    Alias to maintain compatibility with modules importing extract_acc.
    """
    return extract_accel(accel_xyz, fs)


def sliding_windows(sig_len, window_sec, step_sec, fs):
    win = int(window_sec * fs)
    step = int(step_sec * fs)
    for start in range(0, sig_len - win + 1, step):
        end = start + win
        yield start, end



# Synthetic data generation: create realistic fall scenarios
def simulate_window(window_sec=20, fs_ecg=5, fs_accel=50, rng=None):
    """
    Creates one synthetic training window (ECG + accel).
    Returns (ecg_signal, accel_xyz, label, scenario_str)
    """
    # Use rng (random number generator) to ensure reproducibility with same seed
    # If rng is not provided, use np.random as default
    if rng is None:
        rng = np.random
    scen = rng.choice(
        ["normal", "run", "trip", "fall_unconscious", "collapse", "panic"],
        p=[0.25, 0.2, 0.15, 0.15, 0.1, 0.15]
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
        accel[impact_idx:impact_idx+5] += 4.0
        accel += 0.5 * np.sin(2 * np.pi * 1.5 * t_accel)[:, None]
    elif scen == "fall_unconscious":
        impact_idx = int(0.5 * n_accel)
        accel[impact_idx:impact_idx+5] += 5.0
        accel[impact_idx+50:] *= 0.05
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
    Build synthetic feature dataframe using your old generator.
    """
    rows = []
    labels = []
    for i in range(n_samples):
        # Create a new random state for each sample to ensure reproducibility
        sample_rng = np.random.RandomState(random_seed + i)
        hr, accel, y, scen = simulate_window(window_sec, fs_ecg, fs_accel, rng=sample_rng)
        f_ecg = extract_ecg(hr, fs_ecg)
        f_acc = extract_accel(accel, fs_accel)
        row = {**f_ecg, **f_acc, "scenario": f"SYN_{scen}"}
        rows.append(row)
        labels.append(y)
    df = pd.DataFrame(rows)
    df["label"] = labels
    return df


# # Dataset loading: assumes preprocessed CSVs locally
# def load_hifd_records(data_root):
#     """
#     Scan data_root for CSVs with columns: 'hr', 'ax', 'ay', 'az'
#     Label is inferred from filename: filenames containing 'fall' or 'nearfall'.
#     """
#     records = []
#     if not os.path.isdir(data_root):
#         return records

#     csv_paths = glob.glob(os.path.join(data_root, "*.csv"))
#     for path in csv_paths:
#         #error handling
#         try:
#             df = pd.read_csv(path)
#         except Exception as e:
#             print(f"WARNING: Failed to read {path}: {e}")
#             continue

#         # Expect columns: hr, ax, ay, az (adjust if needed)
#         if not {"hr", "ax", "ay", "az"}.issubset(df.columns):
#             print(f"WARNING: Missing required columns in {path}")
#             continue

#         hr = df["hr"].values
#         accel = df[["ax", "ay", "az"]].values

#         fname = os.path.basename(path).lower()
#         if "fall" in fname:
#             label_str = "fall"
#         elif "near" in fname:
#             label_str = "near-fall"
#         else:
#             label_str = "adl"

#         records.append({
#             "hr": hr,
#             "fs_hr": FS_HR_DEFAULT,
#             "accel": accel,
#             "fs_accel": FS_ACCEL_DEFAULT,
#             "label_str": label_str,
#         })

#     return records


# def build_from_hifd(data_root, window_sec=20, step_sec=10):
#     rows = []
#     labels = []

#     records = load_hifd_records(data_root)
#     for rec in records:
#         hr = rec["hr"]
#         fs_hr = rec["fs_hr"]
#         accel = rec["accel"]
#         fs_accel = rec["fs_accel"]
#         label_str = rec["label_str"]

#         n_hr = len(hr)
#         n_acc = len(accel)
#         total_sec = min(n_hr / fs_hr, n_acc / fs_accel)
#         if total_sec < window_sec:
#             continue

#         for start_hr, end_hr in sliding_windows(
#             int(total_sec * fs_hr), window_sec, step_sec, fs_hr
#         ):
#             t0 = start_hr / fs_hr
#             t1 = end_hr / fs_hr
#             start_acc = int(t0 * fs_accel)
#             end_acc = int(t1 * fs_accel)
#             if end_acc > n_acc:
#                 break

#             hr_window = hr[start_hr:end_hr]
#             accel_window = accel[start_acc:end_acc]

#             f_ecg = extract_ecg(hr_window, fs_hr)
#             f_acc = extract_accel(accel_window, fs_accel)

#             label = 1 if label_str in ["fall", "near-fall"] else 0
#             scen = f"HIFD_{label_str}"

#             row = {**f_ecg, **f_acc, "scenario": scen}
#             rows.append(row)
#             labels.append(label)

#     df = pd.DataFrame(rows)
#     if not df.empty:
#         df["label"] = labels
#     return df

# ECG MIT-BIH dataset loading
def load_ecg_mitbih_rows(data_root):
    """    Expects last column as label (0 = normal, >0 = arrhythmia).
    """
    records = []
    train_path = os.path.join(data_root, "mitbih_train.csv")
    test_path = os.path.join(data_root, "mitbih_test.csv")

    #error handling
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
    labels = df_all.iloc[:, -1].values        #last column is class label

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

        # treat whole row as one window
        f_ecg = extract_ecg(ecg, fs)

        # neutral accel: assume resting body
        f_acc = {
            "accel_mean": 0.1,
            "accel_std": 0.05,
            "accel_max": 0.2,
            "peaks_per_sec": 0.0,
            "stillness_duration": len(ecg) / fs,
            "post_impact_still_flag": 0.0,
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


# WEDA loader import (placed after function definitions to avoid circular import issues)
try:
    from load_weda import build_from_weda
except ImportError:
    build_from_weda = None


# Training and evaluation
if __name__ == "__main__":
    dfs = []

    # # HIFD
    # df_hifd = build_from_hifd(HIFD_ROOT, window_sec=20, step_sec=10)
    # if not df_hifd.empty:
    #     print(f"HIFD samples: {len(df_hifd)}")
    #     dfs.append(df_hifd)
    # else:
    #     print("WARNING: No HIFD data loaded (check CSV).")

    # ECG MITBIH
    df_ecg = build_from_ecg_mitbih(ECG_ROOT)
    if not df_ecg.empty:
        print(f"ECG MITBIH samples: {len(df_ecg)}")
        dfs.append(df_ecg)
    else:
        print("WARNING: No ECG MITBIH data loaded (check CSV).")

    # 3) WEDA-FALL
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

    # 4) Synthetic
    df_synth = build_synthetic_dataset(n_samples=1000)
    print(f"Synthetic samples: {len(df_synth)}")
    dfs.append(df_synth)

    # # Optional: HIFD
    # df_hifd = build_from_hifd(HIFD_ROOT, window_sec=20, step_sec=10)
    # if not df_hifd.empty:
    #     print(f"HIFD samples: {len(df_hifd)}")
    #     dfs.append(df_hifd)
    # else:
    #     print("WARNING: No HIFD data loaded (check CSV).")

    if not dfs:
        raise RuntimeError("Check dataset paths.")

    df = pd.concat(dfs, ignore_index=True)
    print(f"Total samples: {len(df)}")

    feature_cols = [c for c in df.columns if c not in ["label", "scenario"]]
    X = df[feature_cols].values
    y = df["label"].values

    # Check for NaN/inf values
    if np.isnan(X).any() or np.isinf(X).any():
        print("WARNING: Found NaN or Inf values in features. Filling NaN with 0 and Inf with large finite values.")
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    # Validate feature consistency
    print(f"Feature columns: {feature_cols}")
    print(f"Feature matrix shape: {X.shape}")
    unique_labels, label_counts = np.unique(y, return_counts=True)
    print(f"Label distribution: {dict(zip(unique_labels, label_counts))}")

    # Check if stratification is possible (requires at least 2 samples per class in each split)
    can_stratify = len(unique_labels) > 1 and np.min(label_counts) >= 5

    if can_stratify:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
    else:
        print("WARNING: Cannot stratify split (insufficient samples per class), using random split")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    '''
    DEBUG
    # Baseline: Logistic Regression
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    y_scores_lr = lr.predict_proba(X_test)[:, 1]
    print("\nLogistic baseline report:")
    print(classification_report(y_test, y_pred_lr))

    '''
    lgbm = lgb.LGBMClassifier(
        n_estimators=400,
        num_leaves=31,
        learning_rate=0.07,
        # class_weight="balanced",
        #data set is unbalanced and performs worse on class 1
        is_unbalance=True,
        random_state=42,
    )
    lgbm.fit(X_train, y_train)
    y_pred_lgbm = lgbm.predict(X_test)
    y_scores_lgbm = lgbm.predict_proba(X_test)[:, 1]

    print("LightGBM report:")
    print(classification_report(y_test, y_pred_lgbm))

    # Precision-recall summary
    #DEBUG: ap_lr = average_precision_score(y_test, y_scores_lr)
    ap_lgbm = average_precision_score(y_test, y_scores_lgbm)
    #DEBUG: print(f"Average precision LR: {ap_lr:.3f}, LGBM: {ap_lgbm:.3f}")
    print(f"Average precision LGBM: {ap_lgbm:.3f}")
       
    # Save model + feature list for deployment
    joblib.dump({"model": lgbm, "features": feature_cols}, "sos_gbt_model.joblib")
    print("Saved model to sos_gbt_model.joblib")