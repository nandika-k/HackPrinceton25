"""
Test script to verify WEDA-FALL resampling works correctly.
Tests with a small subset of files first before processing the entire dataset.
"""
import os
import sys
import pandas as pd
import numpy as np
from weda_resample import resample_weda_csv, resample_timeseries


def test_resample_timeseries():
    """Test the core resampling function with synthetic data."""
    print("Testing resample_timeseries()...")
    
    # Create synthetic data: 50Hz -> 125Hz
    duration = 1.0  # 1 second
    source_fs = 50.0
    target_fs = 125.0
    
    # Generate time data (irregular spacing to simulate real data)
    n_samples = int(duration * source_fs)
    time_data = np.cumsum(np.random.uniform(0.018, 0.022, n_samples))  # ~50Hz but irregular
    time_data = time_data - time_data[0]  # Start at 0
    
    # Generate synthetic sensor data (sine wave with noise)
    value_data = np.sin(2 * np.pi * 2.0 * time_data) + 0.1 * np.random.randn(n_samples)
    
    try:
        resampled_time, resampled_values = resample_timeseries(
            time_data, value_data, target_fs=target_fs, method='linear'
        )
        
        # Verify output
        assert len(resampled_time) > len(time_data), "Resampled data should have more samples"
        assert np.allclose(resampled_time[1] - resampled_time[0], 1.0/target_fs, atol=1e-4), \
            "Time spacing should be uniform at target frequency"
        assert len(resampled_values) == len(resampled_time), "Time and values should have same length"
        
        print(f"  ✓ Resampled {len(time_data)} samples -> {len(resampled_time)} samples")
        print(f"  ✓ Time spacing: {resampled_time[1] - resampled_time[0]:.6f}s (expected: {1.0/target_fs:.6f}s)")
        print("  ✓ Test passed!\n")
        return True
    
    except Exception as e:
        print(f"  ✗ Test failed: {e}\n")
        return False


def test_resample_csv_file(input_file, output_file):
    """Test resampling a single CSV file."""
    print(f"Testing resample_weda_csv() with {os.path.basename(input_file)}...")
    
    if not os.path.exists(input_file):
        print(f"  ✗ File not found: {input_file}\n")
        return False
    
    try:
        # Read original file
        df_original = pd.read_csv(input_file)
        time_col = [col for col in df_original.columns if 'time' in col.lower()][0]
        original_samples = len(df_original)
        original_duration = df_original[time_col].max() - df_original[time_col].min()
        original_fs = original_samples / original_duration if original_duration > 0 else 0
        
        # Resample
        success = resample_weda_csv(input_file, output_file, target_fs=125.0, method='linear')
        
        if not success:
            print(f"  ✗ Resampling failed\n")
            return False
        
        # Verify output file exists
        if not os.path.exists(output_file):
            print(f"  ✗ Output file not created: {output_file}\n")
            return False
        
        # Read resampled file
        df_resampled = pd.read_csv(output_file)
        resampled_samples = len(df_resampled)
        resampled_duration = df_resampled[time_col].max() - df_resampled[time_col].min()
        resampled_fs = resampled_samples / resampled_duration if resampled_duration > 0 else 0
        
        # Verify
        assert resampled_samples > original_samples, "Resampled should have more samples"
        assert abs(resampled_fs - 125.0) < 5.0, f"Resampled FS should be ~125Hz, got {resampled_fs:.2f}Hz"
        
        print(f"  ✓ Original: {original_samples} samples, ~{original_fs:.1f}Hz, {original_duration:.2f}s")
        print(f"  ✓ Resampled: {resampled_samples} samples, ~{resampled_fs:.1f}Hz, {resampled_duration:.2f}s")
        print(f"  ✓ Output saved to: {output_file}")
        print("  ✓ Test passed!\n")
        return True
    
    except Exception as e:
        print(f"  ✗ Test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_single_file():
    """Test with a single real WEDA-FALL file."""
    print("=" * 60)
    print("Testing with Real WEDA-FALL File")
    print("=" * 60)
    
    # Find a test file
    test_file = "WEDA-FALL-data-source/dataset/50Hz/D07/U01_R01_accel.csv"
    
    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
        print("Skipping real file test...\n")
        return False
    
    # Create test output directory
    test_output_dir = "test_output_125Hz"
    os.makedirs(test_output_dir, exist_ok=True)
    
    output_file = os.path.join(test_output_dir, "U01_R01_accel_125Hz.csv")
    
    return test_resample_csv_file(test_file, output_file)


def main():
    """Run all tests."""
    print("=" * 60)
    print("WEDA-FALL Resampling Test Suite")
    print("=" * 60)
    print()
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Core resampling function
    tests_total += 1
    if test_resample_timeseries():
        tests_passed += 1
    
    # Test 2: Single CSV file
    tests_total += 1
    if test_single_file():
        tests_passed += 1
    
    # Summary
    print("=" * 60)
    print(f"Test Results: {tests_passed}/{tests_total} passed")
    print("=" * 60)
    
    if tests_passed == tests_total:
        print("✓ All tests passed! Ready to process full dataset.")
        return 0
    else:
        print("✗ Some tests failed. Please check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

