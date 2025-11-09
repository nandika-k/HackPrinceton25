"""
Backend API Server for AI Trauma & SOS Detection System
Integrates all 6 layers of the architecture
"""
import os
import time
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import asyncio

# Import our ML components
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
try:
    from Grok.grok_agent import GrokDistressAgent
except ImportError:
    # Fallback if Grok module not available
    GrokDistressAgent = None
    print("WARNING: Grok agent not available. Some features will be disabled.")

from backend.weather_api import WeatherHazardService
from backend.communication_layer import CommunicationLayer

app = FastAPI(title="AI Trauma & SOS Detection API")

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
agent = None
weather_service = WeatherHazardService()
comm_layer = CommunicationLayer()

# In-memory storage for real-time telemetry (in production, use Redis/DB)
telemetry_store: List[Dict] = []
active_alerts: List[Dict] = []


# ============================================================
# PYDANTIC MODELS
# ============================================================
class SensorData(BaseModel):
    """Perception Layer: Sensor inputs"""
    device_id: str
    timestamp: float
    # Accelerometer/Gyro (IMU)
    accel_x: Optional[float] = None
    accel_y: Optional[float] = None
    accel_z: Optional[float] = None
    gyro_x: Optional[float] = None
    gyro_y: Optional[float] = None
    gyro_z: Optional[float] = None
    # HR/Vitals (BioTelemetry)
    heart_rate: Optional[float] = None
    ecg_signal: Optional[List[float]] = None  # Array of HR values over time
    # GPS (Orbital localization analog)
    latitude: float
    longitude: float
    altitude: Optional[float] = None
    # Sampling rates
    fs_ecg: float = 125.0
    fs_acc: float = 125.0


class AnalysisRequest(BaseModel):
    """Request for trauma analysis"""
    sensor_data: SensorData
    extra_context: Optional[Dict[str, Any]] = None


class AlertResponse(BaseModel):
    """Response from analysis"""
    device_id: str
    timestamp: float
    P_SOS: float
    hazard_index: float
    risk: float
    decision: str  # "NO_ACTION" | "PREALERT" | "SOS"
    triage: Optional[Dict[str, Any]] = None
    location: Dict[str, float]
    offline_fallback: bool


# ============================================================
# INITIALIZATION
# ============================================================
@app.on_event("startup")
async def startup_event():
    """Initialize the Grok agent on startup"""
    global agent
    if GrokDistressAgent is None:
        print("WARNING: GrokDistressAgent class not available")
        agent = None
        return
    
    try:
        model_path = os.path.join(project_root, "sos_gbt_model.joblib")
        if not os.path.exists(model_path):
            print(f"WARNING: Model file {model_path} not found. Training will be required.")
            print(f"Run: python Grok/sos_agent.py")
            agent = None
        else:
            agent = GrokDistressAgent(
                model_path=model_path,
                hazard_api_url="http://localhost:8000/api/weather/hazard",  # Internal endpoint
                first_responder_webhook="http://localhost:8000/api/alerts/first-responders",
                nearby_broadcast_url="http://localhost:8000/api/alerts/nearby-broadcast"
            )
            print("✓ GrokDistressAgent initialized")
    except Exception as e:
        print(f"ERROR initializing agent: {e}")
        import traceback
        traceback.print_exc()
        agent = None


# ============================================================
# API ENDPOINTS
# ============================================================
@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "AI Trauma & SOS Detection System",
        "version": "1.0.0"
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "agent_loaded": agent is not None,
        "timestamp": time.time()
    }


@app.post("/api/analyze", response_model=AlertResponse)
async def analyze_sensor_data(request: AnalysisRequest):
    """
    Main analysis endpoint - integrates all 6 layers:
    1. Perception Layer (sensor data)
    2. Decision Layer (ML models)
    3. Geospatial Intelligence (weather/hazard)
    4. Cognitive Layer (Grok AI triage)
    5. Communication Layer (alerts)
    6. Visualization (returns data for dashboard)
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="ML agent not initialized. Train model first.")
    
    sensor = request.sensor_data
    
    # Convert sensor data to numpy arrays
    if sensor.ecg_signal:
        ecg_signal = np.array(sensor.ecg_signal, dtype=float)
    else:
        # Fallback: generate from heart_rate if available
        if sensor.heart_rate:
            # Simulate 20-second window
            window_sec = 20
            n_samples = int(window_sec * sensor.fs_ecg)
            ecg_signal = np.full(n_samples, sensor.heart_rate) + np.random.randn(n_samples) * 2
        else:
            raise HTTPException(status_code=400, detail="Either ecg_signal or heart_rate required")
    
    # Build accelerometer array
    if sensor.accel_x is not None and sensor.accel_y is not None and sensor.accel_z is not None:
        # Single sample - expand to window
        window_sec = 20
        n_samples = int(window_sec * sensor.fs_acc)
        acc_xyz = np.array([
            [sensor.accel_x] * n_samples,
            [sensor.accel_y] * n_samples,
            [sensor.accel_z] * n_samples
        ]).T
    else:
        # Generate neutral motion
        window_sec = 20
        n_samples = int(window_sec * sensor.fs_acc)
        acc_xyz = 0.1 * np.random.randn(n_samples, 3)
    
    # Call the agent's analyze_window method
    try:
        result = agent.analyze_window(
            ecg_signal=ecg_signal,
            acc_xyz=acc_xyz,
            fs_ecg=sensor.fs_ecg,
            fs_acc=sensor.fs_acc,
            lat=sensor.latitude,
            lon=sensor.longitude,
            device_id=sensor.device_id,
            extra_context=request.extra_context or {}
        )
        
        # Store telemetry for dashboard
        telemetry_entry = {
            "device_id": sensor.device_id,
            "timestamp": sensor.timestamp,
            "P_SOS": result["P_SOS"],
            "hazard_index": result["hazard_index"],
            "risk": result["risk"],
            "decision": result["decision"],
            "location": {"lat": sensor.latitude, "lon": sensor.longitude},
            "triage": result.get("grok_triage"),
            "offline_fallback": result.get("offline_fallback", False)
        }
        telemetry_store.append(telemetry_entry)
        
        # Keep only last 1000 entries
        if len(telemetry_store) > 1000:
            telemetry_store.pop(0)
        
        # Store alerts
        if result["decision"] in ["SOS", "PREALERT"]:
            alert_entry = {
                **telemetry_entry,
                "alert_type": result["decision"],
                "created_at": datetime.now().isoformat()
            }
            active_alerts.append(alert_entry)
            # Keep only last 100 alerts
            if len(active_alerts) > 100:
                active_alerts.pop(0)
        
        return AlertResponse(
            device_id=sensor.device_id,
            timestamp=sensor.timestamp,
            P_SOS=result["P_SOS"],
            hazard_index=result["hazard_index"],
            risk=result["risk"],
            decision=result["decision"],
            triage=result.get("grok_triage"),
            location={"lat": sensor.latitude, "lon": sensor.longitude},
            offline_fallback=result.get("offline_fallback", False)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/api/telemetry")
async def get_telemetry(limit: int = 100):
    """Get recent telemetry data for dashboard"""
    return {
        "telemetry": telemetry_store[-limit:],
        "count": len(telemetry_store)
    }


@app.get("/api/alerts")
async def get_alerts(active_only: bool = True):
    """Get active alerts"""
    if active_only:
        return {
            "alerts": [a for a in active_alerts if a["decision"] in ["SOS", "PREALERT"]],
            "count": len([a for a in active_alerts if a["decision"] in ["SOS", "PREALERT"]])
        }
    return {
        "alerts": active_alerts,
        "count": len(active_alerts)
    }


@app.post("/api/alerts/first-responders")
async def first_responder_webhook(alert_data: Dict):
    """Webhook endpoint for first responder notifications"""
    print(f"[FIRST RESPONDER ALERT] {json.dumps(alert_data, indent=2)}")
    # In production: integrate with 911/EMS systems, send SMS, etc.
    comm_layer.notify_first_responders(alert_data)
    return {"status": "notified", "timestamp": time.time()}


@app.post("/api/alerts/nearby-broadcast")
async def nearby_broadcast_webhook(alert_data: Dict):
    """Webhook endpoint for nearby user broadcasts (Bluetooth simulation)"""
    print(f"[NEARBY BROADCAST] {json.dumps(alert_data, indent=2)}")
    comm_layer.broadcast_nearby(alert_data)
    return {"status": "broadcasted", "timestamp": time.time()}


@app.get("/api/weather/hazard")
async def get_weather_hazard(lat: float, lon: float, ts: Optional[int] = None):
    """Geospatial Intelligence Layer: Weather/hazard API endpoint"""
    if ts is None:
        ts = int(time.time())
    
    hazard_data = weather_service.get_hazard_index(lat, lon, ts)
    return hazard_data


@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    """WebSocket for real-time telemetry streaming"""
    await websocket.accept()
    try:
        while True:
            # Send latest telemetry
            if telemetry_store:
                latest = telemetry_store[-1]
                await websocket.send_json(latest)
            
            # Send active alerts
            active = [a for a in active_alerts if a["decision"] in ["SOS", "PREALERT"]]
            if active:
                await websocket.send_json({"type": "alerts", "data": active})
            
            await asyncio.sleep(1)  # Update every second
    except WebSocketDisconnect:
        print("WebSocket client disconnected")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    import asyncio
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
