"""
Emergency Services Interface - Placeholder for Twilio/emergency services integration.
This interface allows easy integration with Twilio or other emergency services later.
"""
import logging
from typing import Dict, Optional, Callable, List
from datetime import datetime

logger = logging.getLogger(__name__)


class EmergencyServicesInterface:
    """
    Interface for calling emergency services.
    Provides placeholder methods that can be replaced with Twilio integration.
    """
    
    def __init__(self, emergency_callback: Optional[Callable] = None):
        """
        Initialize emergency services interface.
        
        Args:
            emergency_callback: Optional callback function to call emergency services
                               Should accept emergency_info dict and return response dict
        """
        self.emergency_callback = emergency_callback
        self.call_history: List[Dict] = []
    
    def call_emergency_services(self, emergency_info: Dict) -> Dict:
        """
        Call emergency services with emergency information.
        
        Args:
            emergency_info: Dictionary with emergency details:
                - timestamp: When the emergency was detected
                - fall_detected: Whether fall was detected
                - confidence: Detection confidence score
                - emergency_level: Severity level (low, medium, high, critical)
                - sensor_details: Sensor data features
                - location: User location
                - user_info: User information
                - agent_recommendation: Agent's recommendation
                - agent_reasoning: Agent's reasoning
        
        Returns:
            Dictionary with call status and details
        """
        logger.critical(f"EMERGENCY CALL INITIATED - Level: {emergency_info.get('emergency_level', 'unknown')}")
        
        if self.emergency_callback is not None:
            try:
                response = self.emergency_callback(emergency_info)
                self.call_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "emergency_info": emergency_info,
                    "response": response
                })
                return response
            except Exception as e:
                logger.error(f"Error calling emergency services: {e}")
                return self._placeholder_emergency_call(emergency_info)
        else:
            # Use placeholder implementation
            return self._placeholder_emergency_call(emergency_info)
    
    def _placeholder_emergency_call(self, emergency_info: Dict) -> Dict:
        """
        Placeholder emergency call implementation.
        Replace this method with actual Twilio integration.
        """
        logger.warning("Using placeholder emergency call. Implement Twilio integration.")
        
        # Prepare emergency message
        message = self._format_emergency_message(emergency_info)
        
        # Log the emergency (in real implementation, this would call Twilio)
        logger.critical("=" * 60)
        logger.critical("EMERGENCY CALL - PLACEHOLDER")
        logger.critical("=" * 60)
        logger.critical(message)
        logger.critical("=" * 60)
        logger.critical("In production, this would:")
        logger.critical("1. Call Twilio API to make phone call")
        logger.critical("2. Send SMS with emergency details")
        logger.critical("3. Provide location and sensor data to emergency services")
        logger.critical("=" * 60)
        
        # Record in history
        self.call_history.append({
            "timestamp": datetime.now().isoformat(),
            "emergency_info": emergency_info,
            "response": {"status": "logged", "placeholder": True}
        })
        
        return {
            "status": "logged",
            "message": "Emergency logged (placeholder mode - implement Twilio integration)",
            "emergency_info": emergency_info,
            "placeholder": True,
            "timestamp": datetime.now().isoformat()
        }
    
    def _format_emergency_message(self, emergency_info: Dict) -> str:
        """Format emergency information into a clear message."""
        return f"""
EMERGENCY ALERT - Fall Detection

Timestamp: {emergency_info.get('timestamp', 'Unknown')}
Emergency Level: {emergency_info.get('emergency_level', 'unknown').upper()}
Confidence: {emergency_info.get('confidence', 0):.2%}

Location: {emergency_info.get('location', 'Unknown')}
User: {emergency_info.get('user_info', 'Unknown')}

Sensor Details:
- HR Mean: {emergency_info.get('sensor_details', {}).get('hr_mean', 0):.1f} bpm
- Accel Max: {emergency_info.get('sensor_details', {}).get('accel_max', 0):.2f} g
- Stillness Duration: {emergency_info.get('sensor_details', {}).get('stillness_duration', 0):.2f} s
- Post-Impact Still: {'YES' if emergency_info.get('sensor_details', {}).get('post_impact_still_flag', 0) > 0 else 'NO'}

Agent Recommendation: {emergency_info.get('agent_recommendation', 'Unknown')}
Agent Reasoning: {emergency_info.get('agent_reasoning', 'N/A')[:200]}...
"""
    
    def set_emergency_callback(self, callback: Callable):
        """Set callback function for calling emergency services."""
        self.emergency_callback = callback
        logger.info("Emergency services callback set.")
    
    def get_call_history(self) -> List[Dict]:
        """Get history of emergency calls."""
        return self.call_history


# Example integration pattern:
"""
# To integrate with Twilio, create a callback function:

def call_emergency_with_twilio(emergency_info):
    from twilio.rest import Client
    
    # Initialize Twilio client
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    client = Client(account_sid, auth_token)
    
    # Prepare message
    message_body = format_emergency_message(emergency_info)
    location = emergency_info.get('location', 'Unknown')
    user_phone = emergency_info.get('user_phone', '911')
    
    # Make phone call
    call = client.calls.create(
        url='http://your-server.com/emergency-voice.xml',  # TwiML URL
        to=user_phone,
        from_=os.getenv("TWILIO_PHONE"),
        status_callback='http://your-server.com/status',
        status_callback_event=['initiated', 'ringing', 'answered', 'completed']
    )
    
    # Send SMS
    message = client.messages.create(
        body=message_body,
        from_=os.getenv("TWILIO_PHONE"),
        to=user_phone
    )
    
    return {
        "status": "success",
        "call_sid": call.sid,
        "message_sid": message.sid,
        "timestamp": datetime.now().isoformat()
    }

# Then set it in the interface:
emergency_interface = EmergencyServicesInterface()
emergency_interface.set_emergency_callback(call_emergency_with_twilio)
"""

