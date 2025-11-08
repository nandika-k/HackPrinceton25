"""
Example usage of the Fall Detection MCP Agent.
Demonstrates how to use the inference and emergency services modules.
"""
import numpy as np
from inference import FallDetectionModel
from emergency_services import EmergencyServices


def example_fall_detection():
    """Example of detecting a fall from sensor data."""
    
    # Initialize model
    print("Loading model...")
    model = FallDetectionModel("sos_gbt_model.joblib")
    
    # Simulate sensor data (replace with real sensor data)
    # ECG/HR signal: 20 seconds at 125 Hz = 2500 samples
    fs_ecg = 125.0
    window_sec = 20
    n_ecg = int(window_sec * fs_ecg)
    
    # Simulate a fall scenario: HR spikes then drops
    t = np.linspace(0, window_sec, n_ecg)
    hr_signal = 70 + 5 * np.sin(2 * np.pi * 0.1 * t)
    # Add impact: sudden spike then drop
    impact_idx = n_ecg // 2
    hr_signal[impact_idx:impact_idx+10] += 50  # Spike
    hr_signal[impact_idx+10:] -= 30  # Drop
    
    # Accelerometer data: 20 seconds at 125 Hz = 2500 samples, 3 axes
    fs_accel = 125.0
    n_accel = int(window_sec * fs_accel)
    accel_data = np.random.randn(n_accel, 3) * 0.1  # Base noise
    # Add impact
    accel_data[impact_idx:impact_idx+5] += 5.0  # Large impact
    # Add stillness after impact
    accel_data[impact_idx+50:] *= 0.05  # Very still
    
    # Make prediction
    print("\nProcessing sensor data...")
    prediction, probability, features = model.predict_from_sensors(
        hr_signal, accel_data, fs_ecg, fs_accel
    )
    
    # Display results
    print(f"\n{'='*50}")
    print(f"FALL DETECTION RESULTS")
    print(f"{'='*50}")
    print(f"Fall Detected: {'YES' if prediction == 1 else 'NO'}")
    print(f"Probability: {probability:.2%}")
    print(f"Confidence: {'HIGH' if probability >= 0.8 else 'MEDIUM' if probability >= 0.6 else 'LOW'}")
    print(f"\nKey Features:")
    print(f"  HR Mean: {features.get('hr_mean', 0):.2f} bpm")
    print(f"  HR Std: {features.get('hr_std', 0):.2f}")
    print(f"  Accel Max: {features.get('accel_max', 0):.2f} g")
    print(f"  Stillness Duration: {features.get('stillness_duration', 0):.2f} s")
    print(f"  Post-Impact Still Flag: {features.get('post_impact_still_flag', 0)}")
    print(f"{'='*50}\n")
    
    # Initialize emergency services
    print("Initializing emergency services...")
    emergency = EmergencyServices({
        "emergency_phone": "911",
        "user_phone": "+1234567890",  # Replace with actual phone
        "contact_name": "John Doe",
        "location": "123 Main St, City, State",
        "confidence_threshold": 0.7
    })
    
    # Send alert if fall detected
    if prediction == 1:
        print("Fall detected! Sending emergency alert...")
        alert_sent = emergency.send_emergency_alert(probability, features)
        if alert_sent:
            print("✓ Emergency alert sent successfully")
        else:
            print("⚠ Emergency alert failed (check configuration)")
    else:
        print("No fall detected. No action needed.")


def example_realtime_monitoring():
    """Example of real-time monitoring with sliding windows."""
    
    model = FallDetectionModel("sos_gbt_model.joblib")
    emergency = EmergencyServices()
    
    # Simulate real-time data stream
    print("Starting real-time monitoring...")
    print("(Press Ctrl+C to stop)\n")
    
    window_size = 20  # 20 second windows
    overlap = 10  # 10 second overlap
    fs = 125.0
    
    try:
        while True:
            # In real implementation, read from sensors
            # For demo: generate synthetic data
            hr_data = np.random.randn(int(window_size * fs)) * 5 + 70
            accel_data = np.random.randn(int(window_size * fs), 3) * 0.2
            
            # Make prediction
            prediction, probability, features = model.predict_from_sensors(
                hr_data, accel_data, fs, fs
            )
            
            # Display status
            status = "⚠ FALL DETECTED" if prediction == 1 else "✓ Normal"
            print(f"{status} | Probability: {probability:.2%} | "
                  f"HR: {features.get('hr_mean', 0):.1f} | "
                  f"Accel Max: {features.get('accel_max', 0):.2f}")
            
            # Send alert if needed
            if prediction == 1 and probability >= emergency.confidence_threshold:
                emergency.send_emergency_alert(probability, features)
                print("  → Emergency alert sent!")
            
            # Wait before next window (in real implementation, this would be event-driven)
            import time
            time.sleep(overlap)
    
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "realtime":
        example_realtime_monitoring()
    else:
        example_fall_detection()

