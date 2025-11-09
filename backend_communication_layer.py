"""
Communication Layer: Bluetooth simulation and first responder notifications
Simulates LoRa mesh and Bluetooth broadcasts for nearby users
"""
import json
import time
from typing import Dict, List, Any
from datetime import datetime


class CommunicationLayer:
    """
    Handles communication with:
    - First responders (webhooks, SMS, etc.)
    - Nearby users via Bluetooth simulation
    - LoRa mesh simulation (for distributed autonomy)
    """
    
    def __init__(self):
        self.nearby_devices: List[Dict] = []  # Simulated nearby devices
        self.alert_history: List[Dict] = []
    
    def notify_first_responders(self, alert_data: Dict[str, Any]):
        """
        Notify first responders (911, EMS, etc.)
        In production: integrate with emergency services APIs
        """
        alert_entry = {
            "type": "FIRST_RESPONDER",
            "alert": alert_data,
            "timestamp": time.time(),
            "status": "sent"
        }
        self.alert_history.append(alert_entry)
        
        # Log the alert (in production, send to actual emergency services)
        print(f"[FIRST RESPONDER] Alert sent: {json.dumps(alert_data, indent=2)}")
        
        # Simulate sending to emergency services
        # In production: use Twilio for SMS, integrate with 911 APIs, etc.
        return alert_entry
    
    def broadcast_nearby(self, alert_data: Dict[str, Any]):
        """
        Broadcast to nearby users via Bluetooth simulation.
        In production: use actual Bluetooth Low Energy (BLE) or mesh network
        """
        alert_entry = {
            "type": "NEARBY_BROADCAST",
            "alert": alert_data,
            "timestamp": time.time(),
            "scope": alert_data.get("scope", "PREALERT"),
            "status": "broadcasted"
        }
        self.alert_history.append(alert_entry)
        
        # Simulate Bluetooth broadcast
        # In production: use BLE advertising or mesh network protocols
        print(f"[BLUETOOTH BROADCAST] Alert broadcasted to nearby devices: {json.dumps(alert_data, indent=2)}")
        
        return alert_entry
    
    def simulate_lora_mesh(self, alert_data: Dict[str, Any], hop_count: int = 3):
        """
        Simulate LoRa mesh network for distributed autonomy.
        Each node can relay alerts to nearby nodes.
        """
        mesh_entry = {
            "type": "LORA_MESH",
            "alert": alert_data,
            "hop_count": hop_count,
            "timestamp": time.time(),
            "status": "relayed"
        }
        self.alert_history.append(mesh_entry)
        
        print(f"[LORA MESH] Alert relayed through mesh (hops: {hop_count})")
        return mesh_entry
    
    def get_alert_history(self, limit: int = 100) -> List[Dict]:
        """Get recent alert history"""
        return self.alert_history[-limit:]
    
    def register_nearby_device(self, device_id: str, location: Dict[str, float]):
        """Register a nearby device (for Bluetooth simulation)"""
        device_entry = {
            "device_id": device_id,
            "location": location,
            "registered_at": time.time()
        }
        # Remove old entries for same device
        self.nearby_devices = [d for d in self.nearby_devices if d["device_id"] != device_id]
        self.nearby_devices.append(device_entry)
        
        # Keep only last 50 devices
        if len(self.nearby_devices) > 50:
            self.nearby_devices.pop(0)
    
    def get_nearby_devices(self, location: Dict[str, float], radius_km: float = 5.0) -> List[Dict]:
        """
        Get devices within radius (for Bluetooth range simulation).
        In production: use actual BLE scanning or location-based queries.
        """
        from math import radians, cos, sin, asin, sqrt
        
        def haversine(lon1, lat1, lon2, lat2):
            """Calculate distance between two points on Earth"""
            lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * asin(sqrt(a))
            r = 6371  # Earth radius in km
            return c * r
        
        nearby = []
        for device in self.nearby_devices:
            dev_loc = device["location"]
            distance = haversine(
                location["lon"], location["lat"],
                dev_loc["lon"], dev_loc["lat"]
            )
            if distance <= radius_km:
                nearby.append({**device, "distance_km": distance})
        
        return nearby
