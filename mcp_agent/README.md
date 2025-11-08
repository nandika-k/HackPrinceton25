# Fall Detection MCP Agent for Emergency Services

This MCP (Model Context Protocol) agent integrates with Dedalus Labs to provide fall detection and automatic emergency services notification.

## Features

- **Fall Detection**: Uses trained LightGBM model to detect falls from ECG/HR and accelerometer data
- **Emergency Services Integration**: Automatically alerts emergency services when falls are detected
- **MCP Server**: Provides MCP-compatible API for integration with Dedalus Labs infrastructure
- **Configurable Alerts**: Supports multiple notification methods (API, SMS via Twilio)
- **Real-time Monitoring**: Supports continuous monitoring with sliding windows

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

First, ensure you have a trained model:

```bash
cd ../LightGradientBoostedTree
python LGBMAlgo.py
```

This will create `sos_gbt_model.joblib` in the current directory.

### 3. Configure Emergency Services

**Note:** There is no universal emergency services API. Configure one of these options:

#### Option 1: Twilio (Recommended - Real SMS/Call Service)

Get a Twilio account at https://www.twilio.com/ (free trial available)

```bash
# Twilio configuration (for SMS/calls to emergency contacts)
export TWILIO_ACCOUNT_SID="your-account-sid"
export TWILIO_AUTH_TOKEN="your-auth-token"
export TWILIO_PHONE="+1234567890"  # Your Twilio phone number

# Emergency contact (person to notify - they can call 911 if needed)
export EMERGENCY_CONTACT_PHONE="+1987654321"
export USER_PHONE="+1234567890"
export CONTACT_NAME="John Doe"
export USER_LOCATION="123 Main St, City, State"
```

#### Option 2: Custom Webhook

If you have your own server/endpoint (IFTTT, Zapier, custom API):

```bash
export WEBHOOK_URL="https://your-server.com/emergency-alert"
export WEBHOOK_API_KEY="your-api-key"  # Optional
```

#### Option 3: Local Logging Only (Default - No Configuration Needed)

If no services are configured, alerts will be logged to `alerts.log` file. This is useful for testing.

```bash
# Detection threshold
export EMERGENCY_CONFIDENCE_THRESHOLD="0.7"
```

**Important:** Cannot directly call 911 via API in most jurisdictions. The system notifies emergency contacts who can call 911 if needed.

### 4. Run the MCP Server

```bash
python mcp_server.py --model-path ../sos_gbt_model.joblib --config config.json
```

## Usage

### Basic Fall Detection

```python
from inference import FallDetectionModel
import numpy as np

# Load model
model = FallDetectionModel("sos_gbt_model.joblib")

# Prepare sensor data
ecg_signal = np.array([...])  # Your ECG/HR data
accel_data = np.array([...])  # Your accelerometer data (N x 3)

# Make prediction
prediction, probability, features = model.predict_from_sensors(
    ecg_signal, accel_data, fs_ecg=125.0, fs_accel=125.0
)

if prediction == 1:
    print(f"Fall detected with {probability:.2%} confidence!")
```

### Emergency Services Integration

```python
from emergency_services import EmergencyServices

# Initialize emergency services
emergency = EmergencyServices({
    "emergency_phone": "911",
    "user_phone": "+1234567890",
    "location": "123 Main St"
})

# Send alert
if prediction == 1:
    emergency.send_emergency_alert(probability, features)
```

### Example Script

Run the example:

```bash
python example_usage.py
```

For real-time monitoring:

```bash
python example_usage.py realtime
```

## MCP Server API

The MCP server provides the following tools:

### `detect_fall`

Detect if a fall has occurred based on sensor data.

**Input:**
- `ecg_signal`: ECG/HR signal data (1D array)
- `accel_xyz`: Accelerometer data (N x 3 array)
- `fs_ecg`: ECG sampling frequency (Hz, default: 125.0)
- `fs_accel`: Accelerometer sampling frequency (Hz, default: 125.0)
- `auto_alert`: Automatically send emergency alert if fall detected (default: true)

**Output:**
- `fall_detected`: Boolean indicating if fall was detected
- `probability`: Confidence score [0, 1]
- `alert_sent`: Whether emergency alert was sent
- `features`: Extracted features

### `check_emergency_status`

Check the status of emergency services and recent detections.

**Output:**
- `model_loaded`: Whether model is loaded
- `emergency_services_configured`: Whether emergency services are configured
- `confidence_threshold`: Current confidence threshold
- `location`: Current user location
- `recent_detections`: Number of recent detections

### `update_location`

Update the user's location for emergency services.

**Input:**
- `location`: User's current location (string)

## Deployment with Dedalus Labs

### 1. Create Dedalus Labs Account

1. Sign up at [Dedalus Labs](https://dedaluslabs.ai/)
2. Generate an API key in your dashboard
3. Set `DEDALUS_API_KEY` environment variable

### 2. Deploy MCP Server

```bash
# Install Dedalus CLI
pip install dedalus-labs

# Deploy server
dedalus deploy . --name "fall-detection-mcp"
```

### 3. Configure Environment Variables

Set environment variables in the Dedalus UI:
- `EMERGENCY_API_URL`
- `EMERGENCY_API_KEY`
- `USER_PHONE`
- `USER_LOCATION`
- `TWILIO_ACCOUNT_SID` (if using Twilio)
- `TWILIO_AUTH_TOKEN` (if using Twilio)

## Architecture

```
┌─────────────────┐
│  Sensor Data    │ (ECG/HR, Accelerometer)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Feature        │ (extract_ecg, extract_accel)
│  Extraction     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Fall Detection │ (LightGBM Model)
│  Model          │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Emergency      │ (API, SMS, Logging)
│  Services       │
└─────────────────┘
```

## Safety Considerations

⚠️ **Important**: This system is for demonstration purposes. For production use:

1. **Verify Accuracy**: Ensure model accuracy meets safety requirements
2. **False Positive Handling**: Implement confirmation mechanisms to reduce false alarms
3. **Compliance**: Ensure compliance with local regulations for automated emergency communications
4. **Testing**: Thoroughly test in controlled environments before deployment
5. **Redundancy**: Implement backup notification systems
6. **Monitoring**: Continuously monitor system performance and alert accuracy

## License

See main project license.

