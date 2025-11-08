"""
Emergency services notification module.
Handles communication with emergency services when a fall is detected.
"""
import os
import logging
import json
from typing import Dict, Optional
from datetime import datetime
import requests
from enum import Enum


class EmergencyLevel(Enum):
    """Emergency severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EmergencyServices:
    """Handles emergency services notifications."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize emergency services handler.
        
        Args:
            config: Configuration dictionary with API keys and settings
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Custom webhook URL (optional - user can provide their own endpoint)
        # This could be a custom server, IFTTT, Zapier, or other webhook service
        self.webhook_url = os.getenv(
            "WEBHOOK_URL",
            self.config.get("webhook_url", "")
        )
        self.webhook_api_key = os.getenv(
            "WEBHOOK_API_KEY",
            self.config.get("webhook_api_key", "")
        )
        
        # Emergency contact phone numbers (for SMS/call notifications)
        # Note: Cannot directly call 911 via API in most jurisdictions
        # Instead, notify emergency contacts who can call 911 if needed
        self.emergency_contact_phone = os.getenv(
            "EMERGENCY_CONTACT_PHONE",
            self.config.get("emergency_contact_phone", "")
        )
        self.user_phone = os.getenv(
            "USER_PHONE",
            self.config.get("user_phone", "")
        )
        self.contact_name = os.getenv(
            "CONTACT_NAME",
            self.config.get("contact_name", "User")
        )
        
        # Location (should be updated via GPS or user input)
        self.location = os.getenv(
            "USER_LOCATION",
            self.config.get("location", "Unknown")
        )
        
        # Threshold for emergency notification
        self.confidence_threshold = float(os.getenv(
            "EMERGENCY_CONFIDENCE_THRESHOLD",
            self.config.get("confidence_threshold", 0.7)
        ))
        
        # Twilio configuration (RECOMMENDED - real SMS/call service)
        # Get credentials from https://www.twilio.com/
        self.twilio_account_sid = os.getenv("TWILIO_ACCOUNT_SID", "")
        self.twilio_auth_token = os.getenv("TWILIO_AUTH_TOKEN", "")
        self.twilio_phone = os.getenv("TWILIO_PHONE", "")
        
        # Notification methods status
        self.has_twilio = bool(self.twilio_account_sid and self.twilio_auth_token and self.twilio_phone)
        self.has_webhook = bool(self.webhook_url)
        self.has_emergency_contact = bool(self.emergency_contact_phone)
        
        # Log available notification methods
        if not (self.has_twilio or self.has_webhook or self.has_emergency_contact):
            self.logger.warning(
                "No emergency notification methods configured. "
                "Alerts will only be logged locally. "
                "Configure Twilio, webhook, or emergency contact phone for notifications."
            )
        else:
            methods = []
            if self.has_twilio:
                methods.append("Twilio SMS/Call")
            if self.has_webhook:
                methods.append("Webhook")
            if self.has_emergency_contact:
                methods.append("Emergency Contact")
            self.logger.info(f"Emergency notification methods: {', '.join(methods)}")
    
    def determine_emergency_level(self, probability: float, features: Dict) -> EmergencyLevel:
        """
        Determine emergency level based on prediction probability and features.
        
        Args:
            probability: Fall detection probability [0, 1]
            features: Extracted features from sensors
        
        Returns:
            EmergencyLevel enum
        """
        # Critical: High probability + stillness flag
        if probability >= 0.9 and features.get("post_impact_still_flag", 0) > 0:
            return EmergencyLevel.CRITICAL
        
        # High: High probability or stillness after impact
        if probability >= 0.8 or features.get("post_impact_still_flag", 0) > 0:
            return EmergencyLevel.HIGH
        
        # Medium: Moderate probability
        if probability >= 0.6:
            return EmergencyLevel.MEDIUM
        
        # Low: Low probability but still concerning
        return EmergencyLevel.LOW
    
    def send_emergency_alert(self, probability: float, features: Dict, 
                            timestamp: Optional[str] = None) -> bool:
        """
        Send emergency alert to services.
        
        Args:
            probability: Fall detection probability
            features: Extracted features
            timestamp: Timestamp of detection (ISO format)
        
        Returns:
            True if alert sent successfully, False otherwise
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        emergency_level = self.determine_emergency_level(probability, features)
        
        # Only send if probability exceeds threshold
        if probability < self.confidence_threshold:
            self.logger.info(
                f"Probability {probability:.3f} below threshold {self.confidence_threshold}. "
                "Not sending alert."
            )
            return False
        
        alert_data = {
            "timestamp": timestamp,
            "emergency_level": emergency_level.value,
            "probability": probability,
            "location": self.location,
            "contact_name": self.contact_name,
            "user_phone": self.user_phone,
            "features": {
                "hr_mean": features.get("hr_mean", 0),
                "hr_std": features.get("hr_std", 0),
                "accel_max": features.get("accel_max", 0),
                "stillness_duration": features.get("stillness_duration", 0),
                "post_impact_still_flag": features.get("post_impact_still_flag", 0),
            },
            "message": self._generate_alert_message(probability, emergency_level)
        }
        
        # Try multiple notification methods
        success = False
        
        # Method 1: SMS/Call via Twilio (RECOMMENDED - real service)
        if self.has_twilio:
            success = self._send_twilio_alert(alert_data) or success
        
        # Method 2: Custom webhook (user-provided endpoint)
        if self.has_webhook:
            success = self._send_webhook_alert(alert_data) or success
        
        # Method 3: Log alert (always done - works without configuration)
        self._log_alert(alert_data)
        
        return success
    
    def _generate_alert_message(self, probability: float, level: EmergencyLevel) -> str:
        """Generate human-readable alert message."""
        return (
            f"FALL DETECTION ALERT - {level.value.upper()}\n"
            f"Probability: {probability:.1%}\n"
            f"Location: {self.location}\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Contact: {self.contact_name} ({self.user_phone})"
        )
    
    def _send_webhook_alert(self, alert_data: Dict) -> bool:
        """Send alert via custom webhook endpoint."""
        try:
            headers = {
                "Content-Type": "application/json"
            }
            if self.webhook_api_key:
                headers["Authorization"] = f"Bearer {self.webhook_api_key}"
            
            response = requests.post(
                self.webhook_url,
                json=alert_data,
                headers=headers,
                timeout=10
            )
            response.raise_for_status()
            self.logger.info(f"Emergency alert sent via webhook: {response.status_code}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to send webhook alert: {e}")
            return False
    
    def _send_twilio_alert(self, alert_data: Dict) -> bool:
        """Send alert via SMS using Twilio (real service)."""
        try:
            from twilio.rest import Client
            
            client = Client(self.twilio_account_sid, self.twilio_auth_token)
            
            # Send SMS to emergency contact
            target_phone = self.emergency_contact_phone or self.user_phone
            if not target_phone:
                self.logger.warning("No target phone number configured for Twilio")
                return False
            
            message = client.messages.create(
                body=alert_data["message"],
                from_=self.twilio_phone,
                to=target_phone
            )
            
            self.logger.info(f"Twilio SMS alert sent to {target_phone}: {message.sid}")
            
            # Optional: Also make a voice call for critical emergencies
            if alert_data["emergency_level"] == EmergencyLevel.CRITICAL.value:
                try:
                    call = client.calls.create(
                        url=f"http://demo.twilio.com/docs/voice.xml",  # Replace with your TwiML
                        to=target_phone,
                        from_=self.twilio_phone
                    )
                    self.logger.info(f"Twilio voice call initiated: {call.sid}")
                except Exception as e:
                    self.logger.warning(f"Failed to initiate voice call: {e}")
            
            return True
        except ImportError:
            self.logger.warning("Twilio not installed. Install with: pip install twilio")
            return False
        except Exception as e:
            self.logger.error(f"Failed to send Twilio alert: {e}")
            return False
    
    def _log_alert(self, alert_data: Dict):
        """Log alert to file and console."""
        log_entry = {
            "timestamp": alert_data["timestamp"],
            "level": alert_data["emergency_level"],
            "probability": alert_data["probability"],
            "location": alert_data["location"],
            "message": alert_data["message"]
        }
        
        self.logger.critical(json.dumps(log_entry, indent=2))
        
        # Also write to alerts log file
        alerts_log = os.getenv("ALERTS_LOG_FILE", "alerts.log")
        try:
            with open(alerts_log, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            self.logger.error(f"Failed to write to alerts log: {e}")

