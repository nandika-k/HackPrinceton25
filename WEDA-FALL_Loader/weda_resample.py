"""
Resample WEDA-FALL dataset to 125Hz.
Converts data from various sampling rates (5Hz, 10Hz, 25Hz, 40Hz, 50Hz) to 125Hz
using interpolation.
"""
import os
import glob
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import argparse
from pathlib import Path
from tqdm import tqdm


TARGET_FS = 125.0  # Target sampling frequency in Hz
TARGET_DT = 1.0 / TARGET_FS  # Time step for 125Hz


def resample_timeseries(time_data, value_data, target_fs=125.0, method='linear'):
    """
    Resample time series data to target sampling frequency using interpolation.
    
    Args:
        time_data: 1D array of timestamps (in seconds)
        value_data: 1D or 2D array of sensor values
                   If 2D, shape should be (n_samples, n_features)
        target_fs: Target sampling frequency in Hz (default: 125.0)
        method: Interpolation method ('linear', 'cubic', 'quadratic')
    
    Returns:
        resampled_time: Uniformly spaced timestamps at target_fs
        resampled_values: Interpolated values at new timestamps
    """
    time_data = np.asarray(time_data, dtype=float)
    value_data = np.asarray(value_data, dtype=float)
    
    # Remove duplicate timestamps (keep first occurrence)
    unique_indices = np.unique(time_data, return_index=True)[1]
    time_data = time_data[unique_indices]
    
    if value_data.ndim == 1:
        value_data = value_data[unique_indices]
    else:
        value_data = value_data[unique_indices, :]
    
    # Check for valid time range
    if len(time_data) < 2:
        raise ValueError("Need at least 2 data points for interpolation")
    
    # Create uniform time grid at target sampling rate
    t_start = time_data[0]
    t_end = time_data[-1]
    n_samples = int(np.ceil((t_end - t_start) * target_fs)) + 1
    resampled_time = np.linspace(t_start, t_end, n_samples)
    
    # Interpolate values
    if value_data.ndim == 1:
        # 1D case: single sensor channel
        if method == 'linear':
            interp_func = interp1d(time_data, value_data, kind='linear', 
                                  bounds_error=False, fill_value='extrapolate')
        elif method == 'cubic':
            # Cubic requires at least 4 points
            if len(time_data) >= 4:
                interp_func = interp1d(time_data, value_data, kind='cubic',
                                      bounds_error=False, fill_value='extrapolate')
            else:
                interp_func = interp1d(time_data, value_data, kind='linear',
                                      bounds_error=False, fill_value='extrapolate')
        else:
            interp_func = interp1d(time_data, value_data, kind=method,
                                  bounds_error=False, fill_value='extrapolate')
        resampled_values = interp_func(resampled_time)
    else:
        # 2D case: multiple sensor channels (e.g., x, y, z)
        n_channels = value_data.shape[1]
        resampled_values = np.zeros((len(resampled_time), n_channels))
        
        for i in range(n_channels):
            if method == 'linear':
                interp_func = interp1d(time_data, value_data[:, i], kind='linear',
                                      bounds_error=False, fill_value='extrapolate')
            elif method == 'cubic':
                if len(time_data) >= 4:
                    interp_func = interp1d(time_data, value_data[:, i], kind='cubic',
                                          bounds_error=False, fill_value='extrapolate')
                else:
                    interp_func = interp1d(time_data, value_data[:, i], kind='linear',
                                          bounds_error=False, fill_value='extrapolate')
            else:
                interp_func = interp1d(time_data, value_data[:, i], kind=method,
                                      bounds_error=False, fill_value='extrapolate')
            resampled_values[:, i] = interp_func(resampled_time)
    
    return resampled_time, resampled_values


def resample_weda_csv(input_path, output_path, target_fs=125.0, method='linear'):
    """
    Resample a single WEDA-FALL CSV file to target sampling frequency.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to output CSV file
        target_fs: Target sampling frequency in Hz
        method: Interpolation method
    
    Returns:
        success: True if successful, False otherwise
    """
    try:
        # Read CSV file
        df = pd.read_csv(input_path)
        
        # Identify time column (usually ends with '_time_list' or 'time')
        time_col = None
        for col in df.columns:
            if 'time' in col.lower():
                time_col = col
                break
        
        if time_col is None:
            print(f"WARNING: No time column found in {input_path}")
            return False
        
        # Get time data
        time_data = df[time_col].values
        
        # Get sensor data columns (all columns except time)
        sensor_cols = [col for col in df.columns if col != time_col]
        
        if len(sensor_cols) == 0:
            print(f"WARNING: No sensor data columns found in {input_path}")
            return False
        
        # Extract sensor values
        if len(sensor_cols) == 1:
            value_data = df[sensor_cols[0]].values
        else:
            value_data = df[sensor_cols].values
        
        # Resample to target frequency
        resampled_time, resampled_values = resample_timeseries(
            time_data, value_data, target_fs=target_fs, method=method
        )
        
        # Create output dataframe
        output_data = {time_col: resampled_time}
        
        if resampled_values.ndim == 1:
            output_data[sensor_cols[0]] = resampled_values
        else:
            for i, col in enumerate(sensor_cols):
                output_data[col] = resampled_values[:, i]
        
        output_df = pd.DataFrame(output_data)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save resampled data
        output_df.to_csv(output_path, index=False)
        
        return True
    
    except Exception as e:
        print(f"ERROR processing {input_path}: {e}")
        return False


def convert_weda_dataset(source_root, output_root, source_fs='50Hz', 
                        target_fs=125.0, method='linear', sensor_types=None):
    """
    Convert entire WEDA-FALL dataset from source frequency to target frequency.
    
    Args:
        source_root: Root directory of WEDA-FALL dataset
                    (e.g., 'WEDA-FALL-data-source/dataset')
        output_root: Root directory for output (125Hz data)
        source_fs: Source frequency directory (e.g., '50Hz', '25Hz')
        target_fs: Target sampling frequency in Hz
        method: Interpolation method
        sensor_types: List of sensor types to process (e.g., ['accel', 'gyro'])
                    If None, processes all sensor types
    """
    source_dir = os.path.join(source_root, source_fs)
    output_dir = os.path.join(output_root, f"{int(target_fs)}Hz")
    
    if not os.path.isdir(source_dir):
        print(f"ERROR: Source directory not found: {source_dir}")
        return
    
    # Default sensor types if not specified
    if sensor_types is None:
        sensor_types = ['accel', 'gyro', 'orientation', 'vertical_accel']
    
    # Find all CSV files
    csv_files = []
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.csv') and not file.startswith('fall_timestamps'):
                # Check if file matches sensor types
                if any(sensor_type in file for sensor_type in sensor_types):
                    csv_files.append(os.path.join(root, file))
    
    print(f"Found {len(csv_files)} CSV files to process")
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print(f"Resampling from ~{source_fs} to {target_fs}Hz using {method} interpolation")
    print()
    
    # Process each file
    success_count = 0
    for csv_file in tqdm(csv_files, desc="Processing files"):
        # Calculate relative path from source directory
        rel_path = os.path.relpath(csv_file, source_dir)
        
        # Create output path
        output_path = os.path.join(output_dir, rel_path)
        
        # Resample file
        if resample_weda_csv(csv_file, output_path, target_fs=target_fs, method=method):
            success_count += 1
    
    print(f"\nCompleted: {success_count}/{len(csv_files)} files processed successfully")
    
    # Copy fall_timestamps.csv if it exists
    fall_timestamps_path = os.path.join(source_dir, 'fall_timestamps.csv')
    if os.path.isfile(fall_timestamps_path):
        output_timestamps_path = os.path.join(output_dir, 'fall_timestamps.csv')
        import shutil
        shutil.copy2(fall_timestamps_path, output_timestamps_path)
        print(f"Copied fall_timestamps.csv to output directory")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Resample WEDA-FALL dataset to 125Hz',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert 50Hz data to 125Hz
  python weda_resample.py --source WEDA-FALL-data-source/dataset --source-fs 50Hz
  
  # Convert 25Hz data to 125Hz with cubic interpolation
  python weda_resample.py --source WEDA-FALL-data-source/dataset --source-fs 25Hz --method cubic
  
  # Convert only accelerometer and gyroscope data
  python weda_resample.py --source WEDA-FALL-data-source/dataset --source-fs 50Hz --sensors accel gyro
        """
    )
    
    parser.add_argument(
        '--source', 
        type=str, 
        default='WEDA-FALL-data-source/dataset',
        help='Root directory of WEDA-FALL dataset (default: WEDA-FALL-data-source/dataset)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory (default: creates 125Hz directory in source parent)'
    )
    
    parser.add_argument(
        '--source-fs',
        type=str,
        default='50Hz',
        choices=['5Hz', '10Hz', '25Hz', '40Hz', '50Hz'],
        help='Source sampling frequency (default: 50Hz)'
    )
    
    parser.add_argument(
        '--target-fs',
        type=float,
        default=125.0,
        help='Target sampling frequency in Hz (default: 125.0)'
    )
    
    parser.add_argument(
        '--method',
        type=str,
        default='linear',
        choices=['linear', 'cubic', 'quadratic'],
        help='Interpolation method (default: linear)'
    )
    
    parser.add_argument(
        '--sensors',
        type=str,
        nargs='+',
        default=None,
        help='Sensor types to process (default: all: accel, gyro, orientation, vertical_accel)'
    )
    
    args = parser.parse_args()
    
    # Determine output directory
    if args.output is None:
        # Default: create in same parent directory as source
        source_parent = os.path.dirname(args.source) if os.path.dirname(args.source) else '.'
        args.output = os.path.join(source_parent, 'dataset')
    
    # Convert dataset
    convert_weda_dataset(
        source_root=args.source,
        output_root=args.output,
        source_fs=args.source_fs,
        target_fs=args.target_fs,
        method=args.method,
        sensor_types=args.sensors
    )


if __name__ == "__main__":
    main()

