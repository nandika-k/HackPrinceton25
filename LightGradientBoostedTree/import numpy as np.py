import os
import glob
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, average_precision_score
import lightgbm as lgb
import joblib

HIFD_ROOT   = "data/hifd"          
SISFALL_ROOT = "data/sisfall"      
ECG_ROOT    = "data/ecg_mitbih"    

# Default sampling rates (adjust to match preprocessing)
FS_HR_DEFAULT     = 125    # 125 Hz HR (or HR-derived samples)
FS_ACCEL_DEFAULT  = 125   #125 Hz accelerometer

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

    # linear fitting model
    t = np.arange(len(x)) / fs
    if len(t) > 1:
        slope = float(np.polyfit(t, x, 1)[0])
    else:
        slope = 0.0

    #proxy std of first differences
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
    if a.ndim != 2 or a.shape[0] == 0:
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

    #impact detection: peaks > threshold 
    impact_thresh = float(np.percentile(mag, 90))  # rough
    peaks = mag > impact_thresh
    peaks_per_sec = float(peaks.sum() / (len(mag) / fs + 1e-6))

    # stillness: absolute value of a below small threshold after prev peak
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


def sliding_windows(sig_len, window_sec, step_sec, fs):
    win = int(window_sec * fs)
    step = int(step_sec * fs)
    for start in range(0, sig_len - win + 1, step):
        end = start + win
        yield start, end



#synthetic code scenario
def simulate_window(window_sec=20, fs_ecg=5, fs_accel=50):
    """
    Creates one synthetic training window (ECG + accel).
    Returns (ecg_signal, accel_xyz, label, scenario_str)
    """
    scen = np.random.choice(
        ["normal", "run", "trip", "fall_unconscious", "collapse", "panic"],
        p=[0.25, 0.2, 0.15, 0.15, 0.1, 0.15]
    )

    # ECG / HR-like signal
    n_ecg = window_sec * fs_ecg
    t_ecg = np.linspace(0, window_sec, n_ecg)

    if scen == "normal":
        hr_base = 70
        hr = hr_base + 3 * np.random.randn(n_ecg)
    elif scen in ["run", "trip"]:
        hr_base = 120
        hr = hr_base + 5 * np.random.randn(n_ecg)
    elif scen in ["fall_unconscious", "collapse"]:
        hr = 110 + 10 * np.random.randn(n_ecg)
        hr -= np.linspace(0, 30, n_ecg)  # drop towards end
    elif scen == "panic":
        hr = 100 + 15 * np.sin(0.5 * t_ecg) + 10 * np.random.randn(n_ecg)
    else:
        hr = 80 + 5 * np.random.randn(n_ecg)

    # Accel signal
    n_accel = window_sec * fs_accel
    t_accel = np.linspace(0, window_sec, n_accel)
    accel = 0.1 * np.random.randn(n_accel, 3)  # base noise

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
        accel += 0.3 * np.random.randn(n_accel, 3)

    label = int(scen in ["fall_unconscious", "collapse", "panic"])
    return hr, accel, label, scen


def build_synthetic_dataset(n_samples=1000, window_sec=20, fs_ecg=5, fs_accel=50):
    """
    Build synthetic feature dataframe using your old generator.
    """
    rows = []
    labels = []
    for _ in range(n_samples):
        hr, accel, y, scen = simulate_window(window_sec, fs_ecg, fs_accel)
        f_ecg = extract_ecg(hr, fs_ecg)
        f_acc = extract_accel(accel, fs_accel)
        row = {**f_ecg, **f_acc, "scenario": f"SYN_{scen}"}
        rows.append(row)
        labels.append(y)
    df = pd.DataFrame(rows)
    df["label"] = labels
    return df


# dataset block assumes preprocessed CSVs locally.
def load_hifd_records(data_root):
    """
    Scan data_root for CSVs with columns: 'hr', 'ax', 'ay', 'az'
    Label is inferred from filename: filenames containing 'fall' or 'nearfall'.
    """
    records = []
    if not os.path.isdir(data_root):
        return records

    csv_paths = glob.glob(os.path.join(data_root, "*.csv"))
    for path in csv_paths:
        df = pd.read_csv(path)

        # Expect columns: hr, ax, ay, az (adjust if needed)
        if not {"hr", "ax", "ay", "az"}.issubset(df.columns):
            continue

        hr = df["hr"].values
        accel = df[["ax", "ay", "az"]].values

        fname = os.path.basename(path).lower()
        if "fall" in fname:
            label_str = "fall"
        elif "near" in fname:
            label_str = "near-fall"
        else:
            label_str = "adl"

        records.append({
            "hr": hr,
            "fs_hr": FS_HR_DEFAULT,
            "accel": accel,
            "fs_accel": FS_ACCEL_DEFAULT,
            "label_str": label_str,
        })

    return records


def build_from_hifd(data_root, window_sec=20, step_sec=10):
    rows = []
    labels = []

    records = load_hifd_records(data_root)
    for rec in records:
        hr = rec["hr"]
        fs_hr = rec["fs_hr"]
        accel = rec["accel"]
        fs_accel = rec["fs_accel"]
        label_str = rec["label_str"]

        n_hr = len(hr)
        n_acc = len(accel)
        total_sec = min(n_hr / fs_hr, n_acc / fs_accel)
        if total_sec < window_sec:
            continue

        for start_hr, end_hr in sliding_windows(
            int(total_sec * fs_hr), window_sec, step_sec, fs_hr
        ):
            t0 = start_hr / fs_hr
            t1 = end_hr / fs_hr
            start_acc = int(t0 * fs_accel)
            end_acc = int(t1 * fs_accel)
            if end_acc > n_acc:
                break

            hr_window = hr[start_hr:end_hr]
            accel_window = accel[start_acc:end_acc]

            f_ecg = extract_ecg(hr_window, fs_hr)
            f_acc = extract_accel(accel_window, fs_accel)

            label = 1 if label_str in ["fall", "near-fall"] else 0
            scen = f"HIFD_{label_str}"

            row = {**f_ecg, **f_acc, "scenario": scen}
            rows.append(row)
            labels.append(label)

    df = pd.DataFrame(rows)
    if not df.empty:
        df["label"] = labels
    return df


def load_sisfall_segments(data_root):
    records = []
    if not os.path.isdir(data_root):
        return records

    csv_paths = glob.glob(os.path.join(data_root, "*.csv"))
    for path in csv_paths:
        df = pd.read_csv(path)
        if not {"ax", "ay", "az"}.issubset(df.columns):
            continue

        accel = df[["ax", "ay", "az"]].values
        fname = os.path.basename(path).lower()
        if fname.startswith("f") or "fall" in fname:
            label_str = "FALL"
        else:
            label_str = "ADL"

        records.append({
            "accel": accel,
            "fs_accel": FS_ACCEL_DEFAULT,
            "label_str": label_str,
        })

    return records


def build_from_sisfall(data_root, window_sec=20, step_sec=10):
    rows = []
    labels = []

    records = load_sisfall_segments(data_root)
    for rec in records:
        accel = rec["accel"]
        fs_accel = rec["fs_accel"]
        label_str = rec["label_str"]
        n_acc = len(accel)
        if n_acc < window_sec * fs_accel:
            continue

        for start_acc, end_acc in sliding_windows(
            n_acc, window_sec, step_sec, fs_accel
        ):
            accel_window = accel[start_acc:end_acc]

            # Neutral ECG placeholders
            f_ecg = {
                "hr_mean": 80.0,
                "hr_std": 5.0,
                "hr_range": 15.0,
                "hr_slope": 0.0,
                "hrv_proxy": 2.0,
            }
            f_acc = extract_accel(accel_window, fs_accel)

            label = 1 if label_str.lower().startswith("f") else 0
            scen = f"SisFall_{label_str}"

            row = {**f_ecg, **f_acc, "scenario": scen}
            rows.append(row)
            labels.append(label)

    df = pd.DataFrame(rows)
    if not df.empty:
        df["label"] = labels
    return df

#ecg dataset
def load_ecg_mitbih_rows(data_root):
    """    Expects last column as label (0 = normal, >0 = arrhythmia).
    """
    records = []
    train_path = os.path.join(data_root, "mitbih_train.csv")
    test_path = os.path.join(data_root, "mitbih_test.csv")

    if os.path.isfile(train_path):
        df_train = pd.read_csv(train_path, header=None)
        records.append(df_train)
    if os.path.isfile(test_path):
        df_test = pd.read_csv(test_path, header=None)
        records.append(df_test)

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


#Training
if __name__ == "__main__":
    dfs = []

    # HIFD
    df_hifd = build_from_hifd(HIFD_ROOT, window_sec=20, step_sec=10)
    if not df_hifd.empty:
        print(f"HIFD samples: {len(df_hifd)}")
        dfs.append(df_hifd)
    else:
        print("WARNING: No HIFD data loaded (check CSV).")

    # SisFall
    df_sisfall = build_from_sisfall(SISFALL_ROOT, window_sec=20, step_sec=10)
    if not df_sisfall.empty:
        print(f"SisFall samples: {len(df_sisfall)}")
        dfs.append(df_sisfall)
    else:
        print("WARNING: No SisFall data loaded (check CSV).")

    # ECG MITBIH
    df_ecg = build_from_ecg_mitbih(ECG_ROOT)
    if not df_ecg.empty:
        print(f"ECG MITBIH samples: {len(df_ecg)}")
        dfs.append(df_ecg)
    else:
        print("WARNING: No ECG MITBIH data loaded (check CSV).")

    # Synthetic fallback
    df_synth = build_synthetic_dataset(n_samples=1000)
    print(f"Synthetic samples: {len(df_synth)}")
    dfs.append(df_synth)

    if not dfs:
        raise RuntimeError("Check dataset paths.")

    df = pd.concat(dfs, ignore_index=True)
    print(f"Total samples: {len(df)}")

    feature_cols = [c for c in df.columns if c not in ["label", "scenario"]]
    X = df[feature_cols].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Baseline: Logistic Regression
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(X_train, y_train)
    y_scores_lr = lr.predict_proba(X_test)[:, 1]
    print("\nLogistic baseline report:")
    print(classification_report(y_test, y_scores_lr > 0.5))

    # Gradient Boosted Trees (LightGBM)
    lgbm = lgb.LGBMClassifier(
        n_estimators=200,
        num_leaves=31,
        learning_rate=0.05,
        class_weight="balanced",
        random_state=42,
    )
    lgbm.fit(X_train, y_train)
    y_scores_lgbm = lgbm.predict_proba(X_test)[:, 1]

    print("LightGBM report (threshold 0.5):")
    print(classification_report(y_test, y_scores_lgbm > 0.5))

    # Precision-recall summary
    ap_lr = average_precision_score(y_test, y_scores_lr)
    ap_lgbm = average_precision_score(y_test, y_scores_lgbm)
    print(f"Average precision LR: {ap_lr:.3f}, LGBM: {ap_lgbm:.3f}")

    # Save model + feature list for deployment
    joblib.dump({"model": lgbm, "features": feature_cols}, "sos_gbt_model.joblib")
    print("Saved model to sos_gbt_model.joblib")