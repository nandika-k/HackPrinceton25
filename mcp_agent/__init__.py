"""
Fall Detection MCP Agent for Emergency Services.
"""

__version__ = "1.0.0"

from .inference import FallDetectionModel
from .emergency_services import EmergencyServices, EmergencyLevel
from .mcp_server import FallDetectionMCPServer

__all__ = [
    "FallDetectionModel",
    "EmergencyServices",
    "EmergencyLevel",
    "FallDetectionMCPServer"
]

