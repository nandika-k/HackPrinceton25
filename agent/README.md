# Fall Detection Agent with OpenAI

Intelligent fall detection agent that uses OpenAI for decision-making and integrates with sensor data and emergency services.

## Architecture

```
┌─────────────────┐
│  Sensor Data    │ (Your sensor integration)
│  Interface      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Fall Detection  │ (Trained ML model)
│    Service      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  OpenAI Agent   │ (Decision making)
│                 │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Emergency     │ (Twilio integration)
│   Interface     │
└─────────────────┘
```

## Setup

### 1. Install Dependencies

```bash
cd agent
pip install -r requirements.txt
```

### 2. Set Environment Variables

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

### 3. Ensure Model is Trained

Make sure `sos_gbt_model.joblib` exists (train with `LGBMAlgo.py`).

## Usage

### Basic Usage

```python
from agent import FallDetectionAgent
import numpy as np

# Initialize agent
agent = FallDetectionAgent(
    openai_api_key="your-key",  # Or set OPENAI_API_KEY env var
    model_path="sos_gbt_model.joblib"
)

# Process sensor data
ecg_data = np.array([...])  # Your ECG/HR data
accel_data = np.array([...])  # Your accelerometer data

metadata = {
    "location": "123 Main St, City, State",
    "user_name": "John Doe",
    "user_phone": "+1234567890"
}

result = agent.process_sensor_data(
    ecg_data=ecg_data,
    accel_data=accel_data,
    metadata=metadata
)

# Get summary
print(agent.get_agent_summary(result))
```

### Test Mode

```bash
python -m agent.fall_detection_agent --test
```

## Integration Points

### 1. Sensor Data Integration

Replace the placeholder in `sensor_interface.py`:

```python
from agent import SensorDataInterface

def get_my_sensor_data():
    # Your sensor code here
    ecg_data = read_ecg_sensor()
    accel_data = read_accel_sensor()
    
    return {
        "ecg": ecg_data,
        "accel": accel_data,
        "metadata": {
            "location": get_gps_location(),
            "user_name": "User Name",
            "user_phone": "+1234567890"
        },
        "timestamp": datetime.now().isoformat()
    }

# Set in agent
sensor_interface = SensorDataInterface()
sensor_interface.set_data_callback(get_my_sensor_data)

# Use in agent
agent = FallDetectionAgent(...)
agent.sensor_interface = sensor_interface
```

### 2. Twilio Integration

Replace the placeholder in `emergency_interface.py`:

```python
from agent import EmergencyServicesInterface
from twilio.rest import Client
import os

def call_emergency_with_twilio(emergency_info):
    client = Client(
        os.getenv("TWILIO_ACCOUNT_SID"),
        os.getenv("TWILIO_AUTH_TOKEN")
    )
    
    # Make call
    call = client.calls.create(
        url='http://your-server.com/emergency-voice.xml',
        to=emergency_info.get('user_phone', '911'),
        from_=os.getenv("TWILIO_PHONE")
    )
    
    # Send SMS
    message = client.messages.create(
        body=format_emergency_message(emergency_info),
        from_=os.getenv("TWILIO_PHONE"),
        to=emergency_info.get('user_phone', '911')
    )
    
    return {
        "status": "success",
        "call_sid": call.sid,
        "message_sid": message.sid
    }

# Set in agent
emergency_interface = EmergencyServicesInterface()
emergency_interface.set_emergency_callback(call_emergency_with_twilio)

# Use in agent
agent = FallDetectionAgent(...)
agent.emergency_interface = emergency_interface
```

## Configuration

### Thresholds

```python
agent = FallDetectionAgent(
    confidence_threshold=0.7,  # Minimum confidence to trigger alert
    emergency_threshold=0.8    # Minimum confidence for emergency call
)
```

### OpenAI Model

Change the model in `fall_detection_agent.py`:

```python
response = self.client.chat.completions.create(
    model="gpt-4",  # or "gpt-4o-mini" for faster/cheaper
    ...
)
```

## Agent Decision Making

The agent uses OpenAI to:
1. **Analyze** sensor data and detection results
2. **Reason** about the situation (false positive, real fall, etc.)
3. **Recommend** actions (monitor, alert, emergency)
4. **Coordinate** emergency response

The agent considers:
- Detection confidence
- Sensor patterns (HR, acceleration, stillness)
- Historical context
- User location and metadata

## Emergency Response Flow

1. **Sensor Data** → Fall Detection Service
2. **Detection Result** → OpenAI Agent (analysis)
3. **Agent Decision** → Emergency Interface
4. **Emergency Call** → Twilio (your implementation)

## Testing

### Test with Sample Data

```bash
python -m agent.fall_detection_agent --test
```

### Test Integration Points

```python
# Test sensor interface
from agent import SensorDataInterface
sensor = SensorDataInterface()
data = sensor.get_sensor_data()
print(data)

# Test emergency interface
from agent import EmergencyServicesInterface
emergency = EmergencyServicesInterface()
result = emergency.call_emergency_services({
    "emergency_level": "high",
    "location": "Test Location",
    ...
})
print(result)
```

## Next Steps

1. ✅ **Sensor Integration**: Implement `get_my_sensor_data()` in `sensor_interface.py`
2. ✅ **Twilio Integration**: Implement `call_emergency_with_twilio()` in `emergency_interface.py`
3. ✅ **Deploy**: Set up environment variables and deploy agent
4. ✅ **Monitor**: Track agent decisions and emergency calls

## Files

- `fall_detection_agent.py` - Main agent with OpenAI integration
- `fall_detection_service.py` - Wraps trained model
- `sensor_interface.py` - Placeholder for sensor integration
- `emergency_interface.py` - Placeholder for Twilio integration
- `requirements.txt` - Dependencies

## Notes

- The agent uses placeholder implementations for sensors and Twilio
- Replace placeholders with actual implementations when ready
- All interfaces are designed for easy integration
- Agent decisions are logged for monitoring and improvement

