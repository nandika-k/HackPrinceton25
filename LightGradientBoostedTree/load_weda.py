"""
Load and process WEDA-FALL dataset for training.
Integrates WEDA-FALL data with existing LGBMAlgo.py pipeline.
"""
import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path
import shutil

# Import feature extraction functions from LGBMAlgo
from LGBMAlgo import extract_ecg, extract_accel, sliding_windows, FS_HR_DEFAULT, FS_ACCEL_DEFAULT


def load_weda_records(data_root, sensor_type='accel', target_fs=125.0):
    """
    Load WEDA-FALL records from resampled 125Hz data.
    
    Args:
        data_root: Root directory of 125Hz WEDA-FALL data
                   (e.g., 'WEDA-FALL-data-source/dataset/125Hz')
        sensor_type: Type of sensor data to load ('accel', 'gyro', 'vertical_accel')
        target_fs: Expected sampling frequency (default: 125.0 Hz)
    
    Returns:
        records: List of dictionaries containing sensor data and metadata
    """
    records = []
    
    if not os.path.isdir(data_root):
        print(f"WARNING: WEDA-FALL data directory not found: {data_root}")
        return records
    
    # Find all CSV files for the specified sensor type
    pattern = os.path.join(data_root, "**", f"*_{sensor_type}.csv")
    csv_paths = glob.glob(pattern, recursive=True)
    
    print(f"Found {len(csv_paths)} {sensor_type} files in WEDA-FALL dataset")
    
    for path in csv_paths:
        try:
            df = pd.read_csv(path)
            
            # Identify time and data columns
            time_col = None
            data_cols = []
            
            for col in df.columns:
                if 'time' in col.lower():
                    time_col = col
                elif sensor_type in col.lower() or any(axis in col.lower() for axis in ['x', 'y', 'z']):
                    data_cols.append(col)
            
            if time_col is None or len(data_cols) == 0:
                print(f"WARNING: Invalid format in {path}")
                continue
            
            # Extract sensor data
            if len(data_cols) == 3:
                # x, y, z format
                sensor_data = df[data_cols].values
            elif len(data_cols) == 1:
                # Single column (e.g., vertical_accel)
                sensor_data = df[data_cols[0]].values.reshape(-1, 1)
                # For single column, replicate to 3D (x, y, z) for compatibility
                sensor_data = np.column_stack([sensor_data, np.zeros_like(sensor_data), np.zeros_like(sensor_data)])
            else:
                print(f"WARNING: Unexpected number of data columns in {path}")
                continue
            
            # Determine label from directory structure
            # WEDA-FALL structure: .../125Hz/{ACTIVITY_CODE}/.../file.csv
            path_parts = Path(path).parts
            activity_code = None
            for part in path_parts:
                if part.startswith('F') or part.startswith('D'):
                    activity_code = part
                    break
            
            if activity_code is None:
                print(f"WARNING: Could not determine activity code from {path}")
                continue
            
            # Label: F* = fall (1), D* = ADL (0)
            label_str = "fall" if activity_code.startswith('F') else "adl"
            
            # Extract metadata from filename
            # Format: U{user_id}_R{trial}_{sensor_type}.csv
            filename = os.path.basename(path)
            user_id = None
            trial = None
            try:
                parts = filename.split('_')
                if len(parts) >= 2:
                    user_id = parts[0].replace('U', '')
                    trial = parts[1].replace('R', '')
            except:
                pass
            
            records.append({
                "sensor_data": sensor_data,
                "fs": target_fs,
                "label_str": label_str,
                "activity_code": activity_code,
                "user_id": user_id,
                "trial": trial,
                "filepath": path
            })
        
        except Exception as e:
            print(f"WARNING: Failed to load {path}: {e}")
            continue
    
    return records


def build_from_weda(data_root, window_sec=20, step_sec=10, sensor_type='accel', 
                    target_fs=125.0, use_hr=False):
    """
    Build feature dataframe from WEDA-FALL dataset.
    
    Args:
        data_root: Root directory of 125Hz WEDA-FALL data
        window_sec: Window size in seconds
        step_sec: Step size in seconds for sliding windows
        sensor_type: Type of sensor ('accel', 'gyro', 'vertical_accel')
        target_fs: Sampling frequency (default: 125.0 Hz)
        use_hr: Whether to use HR data (default: False, uses neutral HR placeholders)
    
    Returns:
        df: DataFrame with extracted features and labels
    """
    rows = []
    labels = []
    
    records = load_weda_records(data_root, sensor_type=sensor_type, target_fs=target_fs)
    
    if not records:
        print("WARNING: No WEDA-FALL records loaded")
        return pd.DataFrame()
    
    print(f"Processing {len(records)} WEDA-FALL records...")
    
    for rec in records:
        sensor_data = rec["sensor_data"]
        fs = rec["fs"]
        label_str = rec["label_str"]
        activity_code = rec["activity_code"]
        
        n_samples = len(sensor_data)
        
        # Check if data is long enough for window
        if n_samples < window_sec * fs:
            continue
        
        # Extract accelerometer features (assuming sensor_data is accel-like)
        # If it's gyro or other, we still use extract_accel function structure
        # but adapt as needed
        if sensor_data.shape[1] == 3:
            # Standard x, y, z format
            accel_window = sensor_data
        else:
            # Handle other formats
            continue
        
        # Create sliding windows
        for start_idx, end_idx in sliding_windows(n_samples, window_sec, step_sec, fs):
            window_data = accel_window[start_idx:end_idx]
            
            # Extract accelerometer features
            f_acc = extract_accel(window_data, fs)
            
            # ECG/HR features (use neutral placeholders if no HR data)
            if use_hr:
                # TODO: If WEDA-FALL has HR data, extract it here
                f_ecg = {
                    "hr_mean": 80.0,
                    "hr_std": 5.0,
                    "hr_range": 15.0,
                    "hr_slope": 0.0,
                    "hrv_proxy": 2.0,
                }
            else:
                # Neutral ECG placeholders
                f_ecg = {
                    "hr_mean": 80.0,
                    "hr_std": 5.0,
                    "hr_range": 15.0,
                    "hr_slope": 0.0,
                    "hrv_proxy": 2.0,
                }
            
            # Create label
            label = 1 if label_str == "fall" else 0
            scen = f"WEDA_{activity_code}_{label_str}"
            
            # Combine features
            row = {**f_ecg, **f_acc, "scenario": scen}
            rows.append(row)
            labels.append(label)
    
    df = pd.DataFrame(rows)
    if not df.empty:
        df["label"] = labels
    
    return df


# Example usage in main training script
if __name__ == "__main__":
    # Destination in the repo where 125Hz data should live
    dest_root = os.path.join("data", "WEDA-FALL-data", "125Hz")
    # Common source path produced by the resampling script
    src_root = os.path.join("WEDA-FALL-data-source", "dataset", "125Hz")

    def copy_weda_125hz(source_root, dest_root):
        """Copy 125Hz CSV files from source_root into dest_root.

        If dest_root already exists and contains files, copying is skipped.
        """
        if not os.path.isdir(source_root):
            return False

        os.makedirs(dest_root, exist_ok=True)
        csv_paths = glob.glob(os.path.join(source_root, "**", "*.csv"), recursive=True)
        for p in csv_paths:
            rel = os.path.relpath(p, source_root)
            dest_p = os.path.join(dest_root, rel)
            os.makedirs(os.path.dirname(dest_p), exist_ok=True)
            shutil.copy2(p, dest_p)
        return True

    # Prefer the data stored under repo/data/WEDA-FALL-data/125Hz
    if os.path.isdir(dest_root) and any(os.scandir(dest_root)):
        weda_root = dest_root
    elif os.path.isdir(src_root):
        print(f"Found source 125Hz data at {src_root}. Copying into {dest_root}...")
        copied = copy_weda_125hz(src_root, dest_root)
        if copied:
            weda_root = dest_root
        else:
            weda_root = src_root
    else:
        weda_root = dest_root

    if os.path.isdir(weda_root):
        df_weda = build_from_weda(weda_root, window_sec=20, step_sec=10, 
                                  sensor_type='accel', target_fs=125.0)

        if not df_weda.empty:
            print(f"WEDA-FALL samples: {len(df_weda)}")
            print(f"Feature columns: {df_weda.columns.tolist()}")
            print(f"Label distribution: {df_weda['label'].value_counts().to_dict()}")
        else:
            print("WARNING: No WEDA-FALL data loaded")
    else:
        print(f"WARNING: WEDA-FALL 125Hz directory not found: {weda_root}")
        print("Please run weda_resample.py first to convert data to 125Hz, or place the 125Hz CSVs under data/WEDA-FALL-data/125Hz")

