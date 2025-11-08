"""
OpenAI Agent for Fall Detection and Emergency Services.
Integrates fall detection model with OpenAI agent for intelligent emergency response.
"""
import os
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np

# OpenAI imports
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
    logging.warning("OpenAI not installed. Install with: pip install openai")

# Import fall detection service and interfaces
from agent.fall_detection_service import FallDetectionService
from agent.sensor_interface import SensorDataInterface
from agent.emergency_interface import EmergencyServicesInterface


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FallDetectionAgent:
    """
    OpenAI agent that processes sensor data, detects falls, and manages emergency responses.
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        model_path: str = "sos_gbt_model.joblib",
        confidence_threshold: float = 0.7,
        emergency_threshold: float = 0.8
    ):
        """
        Initialize the fall detection agent.
        
        Args:
            openai_api_key: OpenAI API key (default: from environment)
            model_path: Path to trained fall detection model
            confidence_threshold: Minimum confidence to trigger alert
            emergency_threshold: Minimum confidence for emergency services call
        """
        # Initialize OpenAI client
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")
        
        if OpenAI is None:
            raise ImportError("OpenAI package not installed. Install with: pip install openai")
        
        self.client = OpenAI(api_key=self.api_key)
        
        # Initialize fall detection service
        self.fall_detection = FallDetectionService(model_path=model_path)
        
        # Initialize interfaces
        self.sensor_interface = SensorDataInterface()
        self.emergency_interface = EmergencyServicesInterface()
        
        # Configuration
        self.confidence_threshold = confidence_threshold
        self.emergency_threshold = emergency_threshold
        
        # Agent system prompt
        self.system_prompt = self._create_system_prompt()
        
        # Conversation history
        self.conversation_history: List[Dict] = [
            {"role": "system", "content": self.system_prompt}
        ]
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for the OpenAI agent."""
        return """You are an intelligent fall detection and emergency response agent. Your responsibilities are:

                    1. **Analyze Sensor Data**: Review sensor data (ECG/HR and accelerometer) to detect potential falls or medical emergencies.

                    2. **Make Decisions**: Based on fall detection confidence scores and sensor features, determine:
                    - If this is a false positive (normal activity)
                    - If this requires monitoring (low confidence alert)
                    - If this is an emergency requiring immediate action (high confidence)

                    3. **Emergency Response**: When an emergency is detected:
                    - Gather all relevant sensor details (heart rate, acceleration patterns, stillness duration)
                    - Determine emergency severity (low, medium, high, critical)
                    - Provide clear information about the situation
                    - Coordinate with emergency services interface

                    4. **Communication**: Provide clear, concise summaries of:
                    - What was detected
                    - Confidence level
                    - Recommended actions
                    - Sensor data insights

                    You have access to:
                    - Fall detection model with confidence scores
                    - Sensor data analysis (HR, accelerometer patterns)
                    - Emergency services interface (for calling emergency services)
                    - Historical detection patterns

                    Always prioritize user safety. When in doubt, err on the side of caution.
                """
    
    def process_sensor_data(
        self,
        ecg_data: Optional[np.ndarray] = None,
        accel_data: Optional[np.ndarray] = None,
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Process sensor data through the agent pipeline.
        
        Args:
            ecg_data: ECG/HR signal data (1D array)
            accel_data: Accelerometer data (N x 3 array)
            metadata: Additional metadata (location, user info, etc.)
        
        Returns:
            Dictionary with detection results and agent recommendations
        """
        logger.info("Processing sensor data through agent...")
        
        # Get sensor data (from interface if not provided directly)
        if ecg_data is None or accel_data is None:
            sensor_data = self.sensor_interface.get_sensor_data()
            ecg_data = sensor_data.get("ecg")
            accel_data = sensor_data.get("accel")
            if metadata is None:
                metadata = sensor_data.get("metadata", {})
        
        if ecg_data is None or accel_data is None:
            return {
                "error": "Sensor data not available",
                "status": "error"
            }
        
        # Run fall detection
        detection_result = self.fall_detection.detect_fall(
            ecg_data=ecg_data,
            accel_data=accel_data,
            metadata=metadata
        )
        
        # Prepare context for agent
        agent_context = self._prepare_agent_context(detection_result, metadata)
        
        # Get agent decision
        agent_decision = self._get_agent_decision(agent_context)
        
        # Combine results
        result = {
            "timestamp": datetime.now().isoformat(),
            "detection": detection_result,
            "agent_decision": agent_decision,
            "metadata": metadata
        }
        
        # Handle emergency if detected
        if detection_result.get("fall_detected") and detection_result.get("probability", 0) >= self.emergency_threshold:
            emergency_response = self._handle_emergency(result)
            result["emergency_response"] = emergency_response
        
        return result
    
    def _prepare_agent_context(self, detection_result: Dict, metadata: Dict) -> str:
        """Prepare context string for the agent."""
        context = f"""Fall Detection Results:
- Fall Detected: {detection_result.get('fall_detected', False)}
- Confidence: {detection_result.get('probability', 0):.2%}
- Emergency Level: {detection_result.get('emergency_level', 'unknown')}

Sensor Features:
- HR Mean: {detection_result.get('features', {}).get('hr_mean', 0):.1f} bpm
- HR Std: {detection_result.get('features', {}).get('hr_std', 0):.1f}
- Accel Max: {detection_result.get('features', {}).get('accel_max', 0):.2f} g
- Stillness Duration: {detection_result.get('features', {}).get('stillness_duration', 0):.2f} s
- Post-Impact Still Flag: {detection_result.get('features', {}).get('post_impact_still_flag', 0)}

Metadata:
- Location: {metadata.get('location', 'Unknown')}
- User: {metadata.get('user_name', 'Unknown')}
- Timestamp: {detection_result.get('timestamp', 'Unknown')}
"""
        return context
    
    def _get_agent_decision(self, context: str) -> Dict[str, Any]:
        """Get decision from OpenAI agent."""
        try:
            # Add user message with context
            user_message = f"""Analyze this fall detection scenario and provide recommendations:

{context}

Please provide:
1. Your assessment of the situation
2. Recommended action (monitor, alert, emergency)
3. Reasoning for your decision
4. Any additional context or concerns
"""
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # or "gpt-4" for better reasoning
                messages=self.conversation_history + [{"role": "user", "content": user_message}],
                temperature=0.3,  # Lower temperature for more consistent decisions
                max_tokens=500
            )
            
            agent_response = response.choices[0].message.content
            
            # Parse agent response (simple extraction)
            decision = self._parse_agent_response(agent_response)
            
            # Update conversation history (keep last 10 messages)
            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": agent_response})
            if len(self.conversation_history) > 21:  # 1 system + 10 exchanges
                self.conversation_history = [self.conversation_history[0]] + self.conversation_history[-20:]
            
            return {
                "recommendation": decision.get("action", "monitor"),
                "reasoning": agent_response,
                "parsed_decision": decision
            }
        
        except Exception as e:
            logger.error(f"Error getting agent decision: {e}")
            return {
                "recommendation": "error",
                "reasoning": f"Error: {str(e)}",
                "parsed_decision": {}
            }
    
    def _parse_agent_response(self, response: str) -> Dict[str, str]:
        """Parse agent response to extract structured decision."""
        response_lower = response.lower()
        
        # Extract action
        if "emergency" in response_lower or "call" in response_lower or "911" in response_lower:
            action = "emergency"
        elif "alert" in response_lower or "notify" in response_lower:
            action = "alert"
        else:
            action = "monitor"
        
        return {
            "action": action,
            "raw_response": response
        }
    
    def _handle_emergency(self, result: Dict) -> Dict[str, Any]:
        """Handle emergency situation."""
        logger.critical("EMERGENCY DETECTED - Handling emergency response...")
        
        # Prepare emergency information
        emergency_info = {
            "timestamp": result["timestamp"],
            "fall_detected": result["detection"]["fall_detected"],
            "confidence": result["detection"]["probability"],
            "emergency_level": result["detection"].get("emergency_level", "high"),
            "sensor_details": result["detection"]["features"],
            "location": result["metadata"].get("location", "Unknown"),
            "user_info": result["metadata"].get("user_name", "Unknown"),
            "agent_recommendation": result["agent_decision"]["recommendation"],
            "agent_reasoning": result["agent_decision"]["reasoning"]
        }
        
        # Call emergency services interface
        emergency_response = self.emergency_interface.call_emergency_services(emergency_info)
        
        return emergency_response
    
    def get_agent_summary(self, result: Dict) -> str:
        """Get human-readable summary from agent."""
        detection = result["detection"]
        decision = result["agent_decision"]
        
        summary = f"""
=== Fall Detection Agent Summary ===
Timestamp: {result['timestamp']}

Detection Results:
- Fall Detected: {'YES' if detection['fall_detected'] else 'NO'}
- Confidence: {detection.get('probability', 0):.2%}
- Emergency Level: {detection.get('emergency_level', 'unknown')}

Agent Decision:
- Recommendation: {decision.get('recommendation', 'unknown')}
- Reasoning: {decision.get('reasoning', 'N/A')[:200]}...

"""
        
        if "emergency_response" in result:
            summary += f"Emergency Response: {result['emergency_response'].get('status', 'unknown')}\n"
        
        return summary


def main():
    """Example usage of the fall detection agent."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fall Detection Agent")
    parser.add_argument("--openai-key", type=str, help="OpenAI API key")
    parser.add_argument("--model-path", type=str, default="sos_gbt_model.joblib",
                       help="Path to trained model")
    parser.add_argument("--test", action="store_true", help="Run test with sample data")
    
    args = parser.parse_args()
    
    # Initialize agent
    agent = FallDetectionAgent(
        openai_api_key=args.openai_key,
        model_path=args.model_path,
        confidence_threshold=0.7,
        emergency_threshold=0.8
    )
    
    if args.test:
        # Test with sample data
        print("Running test with sample sensor data...")
        
        # Generate sample data (simulate fall scenario)
        fs = 125.0
        window_sec = 20
        n_samples = int(window_sec * fs)
        
        # Simulate fall: HR spike then drop, high accel impact
        t = np.linspace(0, window_sec, n_samples)
        hr_data = 70 + 5 * np.sin(2 * np.pi * 0.1 * t)
        hr_data[n_samples//2:n_samples//2+10] += 50  # Spike
        hr_data[n_samples//2+10:] -= 30  # Drop
        
        accel_data = np.random.randn(n_samples, 3) * 0.1
        accel_data[n_samples//2:n_samples//2+5] += 5.0  # Impact
        accel_data[n_samples//2+50:] *= 0.05  # Stillness
        
        metadata = {
            "location": "123 Main St, City, State",
            "user_name": "Test User",
            "user_phone": "+1234567890"
        }
        
        # Process through agent
        result = agent.process_sensor_data(
            ecg_data=hr_data,
            accel_data=accel_data,
            metadata=metadata
        )
        
        # Print summary
        print(agent.get_agent_summary(result))
        
        # Print full result (for debugging)
        print("\nFull Result:")
        print(json.dumps(result, indent=2, default=str))
    else:
        print("Fall Detection Agent initialized.")
        print("Use process_sensor_data() method to process sensor data.")
        print("Run with --test flag to test with sample data.")


if __name__ == "__main__":
    main()

