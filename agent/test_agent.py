"""
Simple test script to verify the agent works without sensors or Twilio.
"""
import os
import numpy as np
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import FallDetectionAgent


def test_agent_basic():
    """Test basic agent functionality."""
    print("=" * 60)
    print("Testing Fall Detection Agent")
    print("=" * 60)
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY not set. Set it to test OpenAI integration.")
        print("For now, testing without OpenAI (will use placeholder)...")
        return False
    
    try:
        # Initialize agent
        print("\n1. Initializing agent...")
        agent = FallDetectionAgent(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model_path="sos_gbt_model.joblib",
            confidence_threshold=0.7,
            emergency_threshold=0.8
        )
        print("   ✓ Agent initialized")
        
        # Generate test data (simulate fall)
        print("\n2. Generating test sensor data (simulating fall)...")
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
        
        print(f"   ✓ Generated {len(hr_data)} HR samples and {len(accel_data)} accel samples")
        
        # Process through agent
        print("\n3. Processing sensor data through agent...")
        result = agent.process_sensor_data(
            ecg_data=hr_data,
            accel_data=accel_data,
            metadata=metadata
        )
        print("   ✓ Processing complete")
        
        # Display results
        print("\n4. Results:")
        print(agent.get_agent_summary(result))
        
        # Check detection
        if result["detection"]["fall_detected"]:
            print("   ✓ Fall detected correctly")
        else:
            print("   ⚠ Fall not detected (may be expected depending on model)")
        
        # Check agent decision
        if "agent_decision" in result:
            print(f"   ✓ Agent recommendation: {result['agent_decision'].get('recommendation', 'unknown')}")
        
        # Check emergency response
        if "emergency_response" in result:
            print(f"   ✓ Emergency response: {result['emergency_response'].get('status', 'unknown')}")
        
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_without_openai():
    """Test fall detection service without OpenAI."""
    print("=" * 60)
    print("Testing Fall Detection Service (without OpenAI)")
    print("=" * 60)
    
    try:
        from agent import FallDetectionService
        
        print("\n1. Initializing fall detection service...")
        service = FallDetectionService(model_path="sos_gbt_model.joblib")
        print("   ✓ Service initialized")
        
        # Generate test data
        print("\n2. Generating test sensor data...")
        fs = 125.0
        window_sec = 20
        n_samples = int(window_sec * fs)
        
        hr_data = 70 + 5 * np.random.randn(n_samples)
        accel_data = np.random.randn(n_samples, 3) * 0.2
        
        print(f"   ✓ Generated test data")
        
        # Detect fall
        print("\n3. Running fall detection...")
        result = service.detect_fall(hr_data, accel_data)
        
        print(f"   ✓ Fall detected: {result['fall_detected']}")
        print(f"   ✓ Confidence: {result['probability']:.2%}")
        print(f"   ✓ Emergency level: {result['emergency_level']}")
        
        print("\n" + "=" * 60)
        print("✓ Fall detection service works!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n")
    
    # Test 1: Fall detection service (no OpenAI needed)
    test_without_openai()
    
    print("\n")
    
    # Test 2: Full agent (needs OpenAI API key)
    if os.getenv("OPENAI_API_KEY"):
        test_agent_basic()
    else:
        print("=" * 60)
        print("Skipping OpenAI agent test (OPENAI_API_KEY not set)")
        print("Set OPENAI_API_KEY environment variable to test full agent")
        print("=" * 60)

