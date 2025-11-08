"""
MCP (Model Context Protocol) Server for Fall Detection Emergency Services.
This server integrates with Dedalus Labs MCP infrastructure to provide
fall detection and emergency services notification.
"""
import os
import json
import logging
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np

# MCP server imports
try:
    from mcp.server import Server
    from mcp.types import Tool, TextContent
except ImportError:
    # Fallback if MCP SDK not available
    logging.warning("MCP SDK not installed. Install with: pip install mcp")
    Server = None

try:
    from .inference import FallDetectionModel
    from .emergency_services import EmergencyServices, EmergencyLevel
except ImportError:
    # Fallback for direct script execution
    from inference import FallDetectionModel
    from emergency_services import EmergencyServices, EmergencyLevel


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FallDetectionMCPServer:
    """MCP Server for fall detection and emergency services."""
    
    def __init__(self, model_path: str = "sos_gbt_model.joblib", config_path: Optional[str] = None):
        """
        Initialize the MCP server.
        
        Args:
            model_path: Path to trained model file
            config_path: Path to configuration file
        """
        self.model_path = model_path
        self.config = self._load_config(config_path)
        
        # Initialize model
        logger.info("Loading fall detection model...")
        self.model = FallDetectionModel(model_path)
        
        # Initialize emergency services
        logger.info("Initializing emergency services...")
        self.emergency = EmergencyServices(self.config.get("emergency", {}))
        
        # Detection history for context
        self.detection_history: List[Dict] = []
        self.max_history = 100
        
        # Server instance
        self.server = None
        if Server:
            self.server = Server("fall-detection-mcp")
            self._register_tools()
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from file or environment."""
        config = {}
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load config: {e}")
        
        # Override with environment variables
        config.setdefault("emergency", {}).update({
            "webhook_url": os.getenv("WEBHOOK_URL", ""),
            "webhook_api_key": os.getenv("WEBHOOK_API_KEY", ""),
            "emergency_contact_phone": os.getenv("EMERGENCY_CONTACT_PHONE", ""),
            "user_phone": os.getenv("USER_PHONE", ""),
            "contact_name": os.getenv("CONTACT_NAME", "User"),
            "location": os.getenv("USER_LOCATION", "Unknown"),
            "confidence_threshold": float(os.getenv("EMERGENCY_CONFIDENCE_THRESHOLD", "0.7")),
            "twilio_account_sid": os.getenv("TWILIO_ACCOUNT_SID", ""),
            "twilio_auth_token": os.getenv("TWILIO_AUTH_TOKEN", ""),
            "twilio_phone": os.getenv("TWILIO_PHONE", "")
        })
        
        return config
    
    def _register_tools(self):
        """Register MCP tools for fall detection."""
        if not self.server:
            return
        
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """List available tools."""
            return [
                Tool(
                    name="detect_fall",
                    description=(
                        "Detect if a fall has occurred based on sensor data. "
                        "Requires ECG/HR data and accelerometer data."
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "ecg_signal": {
                                "type": "array",
                                "items": {"type": "number"},
                                "description": "ECG/HR signal data (1D array)"
                            },
                            "accel_xyz": {
                                "type": "array",
                                "items": {
                                    "type": "array",
                                    "items": {"type": "number"},
                                    "minItems": 3,
                                    "maxItems": 3
                                },
                                "description": "Accelerometer data (N x 3 array: [ax, ay, az])"
                            },
                            "fs_ecg": {
                                "type": "number",
                                "default": 125.0,
                                "description": "ECG sampling frequency (Hz)"
                            },
                            "fs_accel": {
                                "type": "number",
                                "default": 125.0,
                                "description": "Accelerometer sampling frequency (Hz)"
                            },
                            "auto_alert": {
                                "type": "boolean",
                                "default": True,
                                "description": "Automatically send emergency alert if fall detected"
                            }
                        },
                        "required": ["ecg_signal", "accel_xyz"]
                    }
                ),
                Tool(
                    name="check_emergency_status",
                    description="Check the status of emergency services and recent detections",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                Tool(
                    name="update_location",
                    description="Update the user's location for emergency services",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "User's current location"
                            }
                        },
                        "required": ["location"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict) -> List[TextContent]:
            """Handle tool calls."""
            if name == "detect_fall":
                return await self._handle_detect_fall(arguments)
            elif name == "check_emergency_status":
                return await self._handle_check_status()
            elif name == "update_location":
                return await self._handle_update_location(arguments)
            else:
                raise ValueError(f"Unknown tool: {name}")
    
    async def _handle_detect_fall(self, arguments: Dict) -> List[TextContent]:
        """Handle fall detection request."""
        try:
            # Parse input data
            ecg_signal = np.array(arguments["ecg_signal"], dtype=float)
            accel_xyz = np.array(arguments["accel_xyz"], dtype=float)
            fs_ecg = arguments.get("fs_ecg", 125.0)
            fs_accel = arguments.get("fs_accel", 125.0)
            auto_alert = arguments.get("auto_alert", True)
            
            # Validate input shapes
            if ecg_signal.ndim != 1:
                raise ValueError("ECG signal must be 1D array")
            
            if accel_xyz.ndim != 2 or accel_xyz.shape[1] != 3:
                raise ValueError("Accelerometer data must be N x 3 array")
            
            # Make prediction
            prediction, probability, features = self.model.predict_from_sensors(
                ecg_signal, accel_xyz, fs_ecg, fs_accel
            )
            
            # Store in history
            detection = {
                "timestamp": datetime.now().isoformat(),
                "prediction": prediction,
                "probability": probability,
                "features": features
            }
            self.detection_history.append(detection)
            if len(self.detection_history) > self.max_history:
                self.detection_history.pop(0)
            
            # Send emergency alert if needed
            alert_sent = False
            if auto_alert and prediction == 1:
                alert_sent = self.emergency.send_emergency_alert(
                    probability, features, detection["timestamp"]
                )
            
            # Prepare response
            result = {
                "fall_detected": bool(prediction == 1),
                "probability": float(probability),
                "confidence": "high" if probability >= 0.8 else "medium" if probability >= 0.6 else "low",
                "alert_sent": alert_sent,
                "timestamp": detection["timestamp"],
                "features": {k: float(v) for k, v in features.items()}
            }
            
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
        
        except Exception as e:
            logger.error(f"Error in fall detection: {e}")
            return [TextContent(
                type="text",
                text=json.dumps({"error": str(e)}, indent=2)
            )]
    
    async def _handle_check_status(self) -> List[TextContent]:
        """Handle status check request."""
        recent_detections = self.detection_history[-10:] if self.detection_history else []
        
        status = {
            "model_loaded": self.model.model is not None,
            "emergency_services_configured": bool(
                self.emergency.has_twilio or 
                self.emergency.has_webhook or 
                self.emergency.has_emergency_contact
            ),
            "notification_methods": {
                "twilio": self.emergency.has_twilio,
                "webhook": self.emergency.has_webhook,
                "emergency_contact": self.emergency.has_emergency_contact,
                "local_logging": True  # Always available
            },
            "confidence_threshold": self.emergency.confidence_threshold,
            "location": self.emergency.location,
            "recent_detections": len(recent_detections),
            "last_detection": recent_detections[-1] if recent_detections else None
        }
        
        return [TextContent(
            type="text",
            text=json.dumps(status, indent=2)
        )]
    
    async def _handle_update_location(self, arguments: Dict) -> List[TextContent]:
        """Handle location update request."""
        location = arguments["location"]
        self.emergency.location = location
        
        return [TextContent(
            type="text",
            text=json.dumps({"status": "success", "location": location}, indent=2)
        )]
    
    async def run(self):
        """Run the MCP server."""
        if not self.server:
            logger.error("MCP Server not initialized. Install MCP SDK: pip install mcp")
            return
        
        logger.info("Starting Fall Detection MCP Server...")
        # Server run logic would go here
        # This depends on the specific MCP server implementation
        logger.info("MCP Server ready")


def main():
    """Main entry point for the MCP server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fall Detection MCP Server")
    parser.add_argument("--model-path", default="sos_gbt_model.joblib",
                       help="Path to trained model file")
    parser.add_argument("--config", default="config.json",
                       help="Path to configuration file")
    parser.add_argument("--port", type=int, default=8000,
                       help="Server port")
    
    args = parser.parse_args()
    
    server = FallDetectionMCPServer(args.model_path, args.config)
    
    # Run server
    if server.server:
        asyncio.run(server.run())
    else:
        logger.error("MCP Server SDK not available. Please install: pip install mcp")


if __name__ == "__main__":
    main()

