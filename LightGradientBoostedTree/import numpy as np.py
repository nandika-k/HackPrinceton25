import os
import glob
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, average_precision_score
import lightgbm as lgb
import joblib

SISFALL = "data/sisfall"      
ECG    = "data/ecg_mitbih"    

# Default sampling rates, fs: sampling freq (Hz)
HR_FS     = 125    # 125 Hz HR (This is for SpO2 sensor)
ACC_FS  = 125   #125 Hz accelerometer

#Feature extraction block
def extract_ecg(ecg_signal, fs):
    """For MVP: treat ecg_signal as filtered RR proxy, converting visual to numerical. 1D numpy array means l~window-sec*fs
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

def extract_acc(acc_xyz, fs):
    """
    acc_xyz: (N, 3) array of ax, ay, az
    fs: sampling frequency in Hz
    """
    a = np.asarray(acc_xyz, dtype=float)
    if a.ndim != 2 or a.shape[0] == 0:
        return {
            "acc_mean": 0.0,
            "acc_std": 0.0,
            "acc_max": 0.0,
            "peak_per_sec": 0.0,
            "stillness_duration": 0.0,
            "post_impact_still_flag": 0.0,
        }

    value = np.linalg.norm(a, axis=1)
    value_mean = float(np.mean(value))
    value_std = float(np.std(value))
    value_max = float(np.max(value))

    # SOS situation detection: peaks > threshold 
    detection_threshold = float(np.percentile(value, 90))  
    peaks = value > detection_threshold
    peak_per_sec = float(peaks.sum()/(len(value) / fs + 1e-6))

    # stillness: absolute value of a below small threshold after prev peak
    low_thresh = float(np.percentile(value, 20))
    prev_peak= np.where(peaks)[0]
    if len(prev_peak) > 0:
        prev_peak = prev_peak[-1]
        post_segment = value[prev_peak:]
        still_mask = post_segment < low_thresh
        stillness_duration = float(still_mask.sum() / fs)
        after_impact_still_flag = float(stillness_duration > 3.0)  # >3s still = potentially concerning
    else:
        stillness_duration = 0.0
        after_impact_still_flag = 0.0

    return {
        "acc_mean": value_mean,
        "acc_std": value_std,
        "acc_max": value_max,
        "peak_per_sec": peak_per_sec,
        "stillness_duration": stillness_duration,
        "after_impact_still_flag": after_impact_still_flag,
    }


def sliding_windows(sig_len, window_sec, step_sec, fs):
    win = int(window_sec * fs)
    step = int(step_sec * fs)
    for start in range(0, sig_len - win + 1, step):
        end = start + win
        yield start, end



#synthetic code scenario
def simulate_window(window_sec=20, fs_ecg=5, fs_acc=50):
    """
    Creates one synthetic training window (ECG + acc).
    Returns (ecg_signal, acc_xyz, label, scenario_str)
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

    # Acc signal
    n_acc = window_sec * fs_acc
    t_acc = np.linspace(0, window_sec, n_acc)
    acc = 0.1 * np.random.randn(n_acc, 3)  # base noise

    if scen == "normal":
        acc += 0.2 * np.sin(2 * np.pi * 1.0 * t_acc)[:, None]
    elif scen == "run":
        acc += 0.7 * np.sign(np.sin(2 * np.pi * 2.0 * t_acc))[:, None]
    elif scen == "trip":
        impact_idx = int(0.5 * n_acc)
        acc[impact_idx:impact_idx+5] += 4.0
        acc += 0.5 * np.sin(2 * np.pi * 1.5 * t_acc)[:, None]
    elif scen == "fall_unconscious":
        impact_idx = int(0.5 * n_acc)
        acc[impact_idx:impact_idx+5] += 5.0
        acc[impact_idx+50:] *= 0.05
    elif scen == "collapse":
        cut = int(0.3 * n_acc)
        acc[:cut] += 0.3 * np.sin(2 * np.pi * 1.0 * t_acc[:cut])[:, None]
        acc[cut:] *= 0.05
    elif scen == "panic":
        acc += 0.3 * np.random.randn(n_acc, 3)

    label = int(scen in ["fall_unconscious", "collapse", "panic"])
    return hr, acc, label, scen


def build_synthetic_dataset(n_samples=1000, window_sec=20, fs_ecg=5, fs_acc=50):
    """
    Build synthetic feature dataframe using your old generator.
    """
    rows = []
    labels = []
    for _ in range(n_samples):
        hr, acc, y, scen = simulate_window(window_sec, fs_ecg, fs_acc)
        f_ecg = extract_ecg(hr, fs_ecg)
        f_acc = extract_acc(acc, fs_acc)
        row = {**f_ecg, **f_acc, "scenario": f"SYN_{scen}"}
        rows.append(row)
        labels.append(y)
    df = pd.DataFrame(rows)
    df["label"] = labels
    return df


# dataset block assumes preprocessed CSVs locally.

def load_sisfall_segments(data_root):
    records = []
    if not os.path.isdir(data_root):
        return records

    csv_paths = glob.glob(os.path.join(data_root, "*.csv"))
    for path in csv_paths:
        df = pd.read_csv(path)
        if not {"ax", "ay", "az"}.issubset(df.columns):
            continue

        acc = df[["ax", "ay", "az"]].values
        fname = os.path.basename(path).lower()
        if fname.startswith("f") or "fall" in fname:
            label_str = "FALL"
        else:
            label_str = "ADL"

        records.append({
            "acc": acc,
            "fs_acc": ACC_FS,
            "label_str": label_str,
        })

    return records


def build_from_sisfall(data_root, window_sec=20, step_sec=10):
    rows = []
    labels = []

    records = load_sisfall_segments(data_root)
    for rec in records:
        acc = rec["acc"]
        fs_acc = rec["fs_acc"]
        label_str = rec["label_str"]
        n_acc = len(acc)
        if n_acc < window_sec * fs_acc:
            continue

        for start_acc, end_acc in sliding_windows(
            n_acc, window_sec, step_sec, fs_acc
        ):
            acc_window = acc[start_acc:end_acc]

            # Neutral ECG placeholders
            f_ecg = {
                "hr_mean": 80.0,
                "hr_std": 5.0,
                "hr_range": 15.0,
                "hr_slope": 0.0,
                "hrv_proxy": 2.0,
            }
            f_acc = extract_acc(acc_window, fs_acc)

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


#Training
if __name__ == "__main__":
    dfs = []

    # SisFall
    df_sisfall = build_from_sisfall(SISFALL, window_sec=20, step_sec=10)
    if not df_sisfall.empty:
        print(f"SisFall samples: {len(df_sisfall)}")
        dfs.append(df_sisfall)
    else:
        print("WARNING: No SisFall data loaded (check CSV).")

    # ECG MITBIH
    df_ecg = build_from_ecg_mitbih(ECG)
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