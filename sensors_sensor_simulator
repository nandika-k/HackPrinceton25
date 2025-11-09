"""
Sensor Data Simulator - Perception Layer
Simulates accelerometer, gyro, HR/vitals, and GPS data
"""
import time
import random
import numpy as np
import requests
import json
from typing import Dict, List, Optional
from datetime import datetime

API_BASE = "http://localhost:8000"


class SensorSimulator:
    """
    Simulates sensor data for testing the MVP.
    Can simulate normal activity, falls, trauma, etc.
    """
    
    def __init__(self, device_id: str = "device-001"):
        self.device_id = device_id
        self.fs_ecg = 125.0  # Sampling rate for ECG/HR
        self.fs_acc = 125.0  # Sampling rate for accelerometer
        
        # Current state
        self.latitude = 40.35  # Default: Princeton area
        self.longitude = -74.65
        self.heart_rate = 70.0
        self.is_falling = False
        
    def generate_normal_motion(self) -> Dict:
        """Generate normal walking/movement data"""
        return {
            "accel_x": 0.2 + random.gauss(0, 0.1),
            "accel_y": 0.1 + random.gauss(0, 0.1),
            "accel_z": 0.9 + random.gauss(0, 0.1),  # Gravity
            "gyro_x": random.gauss(0, 0.05),
            "gyro_y": random.gauss(0, 0.05),
            "gyro_z": random.gauss(0, 0.05),
        }
    
    def generate_fall_impact(self) -> Dict:
        """Generate fall impact data (high acceleration spike)"""
        return {
            "accel_x": random.gauss(0, 2.0) + random.choice([-4, 4]),
            "accel_y": random.gauss(0, 2.0) + random.choice([-4, 4]),
            "accel_z": 5.0 + random.gauss(0, 1.0),  # High impact
            "gyro_x": random.gauss(0, 1.0),
            "gyro_y": random.gauss(0, 1.0),
            "gyro_z": random.gauss(0, 1.0),
        }
    
    def generate_stillness(self) -> Dict:
        """Generate stillness data (post-fall unconsciousness)"""
        return {
            "accel_x": random.gauss(0, 0.01),
            "accel_y": random.gauss(0, 0.01),
            "accel_z": 0.98 + random.gauss(0, 0.01),  # Just gravity
            "gyro_x": random.gauss(0, 0.01),
            "gyro_y": random.gauss(0, 0.01),
            "gyro_z": random.gauss(0, 0.01),
        }
    
    def generate_ecg_signal(self, window_sec: int = 20, scenario: str = "normal") -> List[float]:
        """Generate ECG/HR signal for a time window"""
        n_samples = int(window_sec * self.fs_ecg)
        
        if scenario == "normal":
            hr = 70 + random.gauss(0, 3)
            signal = [hr + random.gauss(0, 2) for _ in range(n_samples)]
        elif scenario == "exercise":
            hr = 120 + random.gauss(0, 5)
            signal = [hr + random.gauss(0, 3) for _ in range(n_samples)]
        elif scenario == "trauma":
            # HR spikes then drops (shock)
            hr_base = 110
            signal = []
            for i in range(n_samples):
                if i < n_samples // 2:
                    hr = hr_base + random.gauss(0, 10)
                else:
                    hr = hr_base - 30 * (i - n_samples // 2) / (n_samples // 2) + random.gauss(0, 5)
                signal.append(max(40, hr))
        else:
            signal = [70 + random.gauss(0, 2) for _ in range(n_samples)]
        
        return signal
    
    def simulate_scenario(self, scenario: str = "normal", duration_sec: int = 60):
        """
        Simulate a scenario and send data to backend.
        Scenarios: "normal", "fall", "trauma", "exercise"
        """
        print(f"\n[Simulator] Starting scenario: {scenario}")
        print(f"[Simulator] Device: {self.device_id}")
        print(f"[Simulator] Duration: {duration_sec} seconds\n")
        
        start_time = time.time()
        sample_count = 0
        
        while time.time() - start_time < duration_sec:
            timestamp = time.time()
            
            # Generate sensor data based on scenario
            if scenario == "normal":
                motion = self.generate_normal_motion()
                ecg_signal = self.generate_ecg_signal(scenario="normal")
                heart_rate = 70 + random.gauss(0, 3)
            elif scenario == "fall":
                # First 5 seconds: normal, then fall impact, then stillness
                elapsed = time.time() - start_time
                if elapsed < 5:
                    motion = self.generate_normal_motion()
                    ecg_signal = self.generate_ecg_signal(scenario="normal")
                    heart_rate = 70 + random.gauss(0, 3)
                elif elapsed < 6:
                    motion = self.generate_fall_impact()
                    ecg_signal = self.generate_ecg_signal(scenario="trauma")
                    heart_rate = 120 + random.gauss(0, 10)
                    print(f"[Simulator] ⚠️ FALL DETECTED at {elapsed:.1f}s")
                else:
                    motion = self.generate_stillness()
                    ecg_signal = self.generate_ecg_signal(scenario="trauma")
                    heart_rate = 60 + random.gauss(0, 5)
            elif scenario == "trauma":
                motion = self.generate_normal_motion()  # Could be still or moving
                ecg_signal = self.generate_ecg_signal(scenario="trauma")
                heart_rate = 100 + random.gauss(0, 15)
            elif scenario == "exercise":
                motion = self.generate_normal_motion()
                ecg_signal = self.generate_ecg_signal(scenario="exercise")
                heart_rate = 120 + random.gauss(0, 5)
            else:
                motion = self.generate_normal_motion()
                ecg_signal = self.generate_ecg_signal(scenario="normal")
                heart_rate = 70 + random.gauss(0, 3)
            
            # Add small GPS drift (simulating movement)
            self.latitude += random.gauss(0, 0.0001)
            self.longitude += random.gauss(0, 0.0001)
            
            # Prepare payload
            payload = {
                "sensor_data": {
                    "device_id": self.device_id,
                    "timestamp": timestamp,
                    "accel_x": motion["accel_x"],
                    "accel_y": motion["accel_y"],
                    "accel_z": motion["accel_z"],
                    "gyro_x": motion["gyro_x"],
                    "gyro_y": motion["gyro_y"],
                    "gyro_z": motion["gyro_z"],
                    "heart_rate": heart_rate,
                    "ecg_signal": ecg_signal,
                    "latitude": self.latitude,
                    "longitude": self.longitude,
                    "fs_ecg": self.fs_ecg,
                    "fs_acc": self.fs_acc
                },
                "extra_context": {
                    "scenario": scenario,
                    "sample_count": sample_count
                }
            }
            
            # Send to backend
            try:
                response = requests.post(
                    f"{API_BASE}/api/analyze",
                    json=payload,
                    timeout=5.0
                )
                response.raise_for_status()
                result = response.json()
                
                # Print decision
                decision = result.get("decision", "UNKNOWN")
                risk = result.get("risk", 0.0)
                p_sos = result.get("P_SOS", 0.0)
                
                if decision in ["SOS", "PREALERT"]:
                    print(f"[ALERT] {decision} - Risk: {risk*100:.1f}%, P_SOS: {p_sos*100:.1f}%")
                elif sample_count % 10 == 0:  # Print every 10th sample
                    print(f"[Sample {sample_count}] Decision: {decision}, Risk: {risk*100:.1f}%")
                
                sample_count += 1
                
            except Exception as e:
                print(f"[ERROR] Failed to send data: {e}")
            
            # Wait before next sample (simulate 20-second windows)
            time.sleep(20)
        
        print(f"\n[Simulator] Scenario complete. Sent {sample_count} samples.\n")


def main():
    """Main function to run sensor simulator"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Sensor Data Simulator")
    parser.add_argument("--device-id", default="device-001", help="Device ID")
    parser.add_argument("--scenario", choices=["normal", "fall", "trauma", "exercise"], 
                       default="normal", help="Scenario to simulate")
    parser.add_argument("--duration", type=int, default=60, help="Duration in seconds")
    
    args = parser.parse_args()
    
    simulator = SensorSimulator(device_id=args.device_id)
    simulator.simulate_scenario(scenario=args.scenario, duration_sec=args.duration)


if __name__ == "__main__":
    main()
