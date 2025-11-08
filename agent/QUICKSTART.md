# Quick Start: Fall Detection Agent

## Setup (5 minutes)

### 1. Install Dependencies

```bash
cd agent
pip install -r requirements.txt
```

### 2. Set OpenAI API Key

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

Or create a `.env` file:
```
OPENAI_API_KEY=your-openai-api-key
```

### 3. Ensure Model Exists

Make sure `sos_gbt_model.joblib` exists (train with `LGBMAlgo.py` if needed).

## Test It Works

### Quick Test

```bash
python -m agent.fall_detection_agent --test
```

This will:
- Test the agent with sample data
- Show detection results
- Display agent decision
- Show emergency response (placeholder)

**Expected Output:**
```
=== Fall Detection Agent Summary ===
Timestamp: 2024-01-01T12:00:00

Detection Results:
- Fall Detected: YES
- Confidence: 85.23%
- Emergency Level: high

Agent Decision:
- Recommendation: emergency
- Reasoning: Based on the sensor data analysis...
```

## Basic Usage

```python
from agent import FallDetectionAgent
import numpy as np

# Initialize agent
agent = FallDetectionAgent()

# Process sensor data
ecg_data = np.array([...])  # Your ECG data
accel_data = np.array([...])  # Your accelerometer data

result = agent.process_sensor_data(
    ecg_data=ecg_data,
    accel_data=accel_data,
    metadata={
        "location": "123 Main St",
        "user_name": "John Doe",
        "user_phone": "+1234567890"
    }
)

print(agent.get_agent_summary(result))
```

## Integration Points

### 1. Sensor Data (Your Job Later)

**File:** `agent/sensor_interface.py`

**What to do:**
1. Implement `get_my_sensor_data()` function
2. Return dict with `ecg`, `accel`, and `metadata`
3. Set as callback: `sensor_interface.set_data_callback(get_my_sensor_data)`

**Example:**
```python
def get_my_sensor_data():
    ecg_data = read_ecg_sensor()  # Your code
    accel_data = read_accel_sensor()  # Your code
    return {
        "ecg": ecg_data,
        "accel": accel_data,
        "metadata": {"location": "...", "user_name": "..."}
    }
```

### 2. Twilio Integration (Your Job Later)

**File:** `agent/emergency_interface.py`

**What to do:**
1. Implement `call_emergency_with_twilio()` function
2. Use Twilio API to make calls/send SMS
3. Set as callback: `emergency_interface.set_emergency_callback(call_emergency_with_twilio)`

**Example:**
```python
def call_emergency_with_twilio(emergency_info):
    from twilio.rest import Client
    client = Client(account_sid, auth_token)
    
    # Make call
    call = client.calls.create(...)
    
    # Send SMS
    message = client.messages.create(...)
    
    return {"status": "success", "call_sid": call.sid}
```

## Architecture

```
Your Sensors → Sensor Interface → Fall Detection → OpenAI Agent → Emergency Interface → Twilio
                (placeholder)        (trained)      (GPT-4)         (placeholder)     (your code)
```

## What's Ready Now

✅ **Fall Detection**: Uses your trained model  
✅ **OpenAI Agent**: Makes intelligent decisions  
✅ **Interfaces**: Placeholders ready for your code  
✅ **Testing**: Can test without sensors/Twilio  

## What You Need to Add

🔲 **Sensor Integration**: Implement `get_my_sensor_data()`  
🔲 **Twilio Integration**: Implement `call_emergency_with_twilio()`  

## Next Steps

1. **Test the agent**: `python -m agent.fall_detection_agent --test`
2. **Review interfaces**: Check `sensor_interface.py` and `emergency_interface.py`
3. **Implement sensors**: Add your sensor code when ready
4. **Implement Twilio**: Add your Twilio code when ready
5. **Deploy**: Set up environment and deploy

## Files Overview

- `fall_detection_agent.py` - Main agent (uses OpenAI)
- `fall_detection_service.py` - Wraps your trained model
- `sensor_interface.py` - **Placeholder for your sensor code**
- `emergency_interface.py` - **Placeholder for your Twilio code**
- `example_integration.py` - Shows integration pattern

## Support

See `README.md` for detailed documentation.

