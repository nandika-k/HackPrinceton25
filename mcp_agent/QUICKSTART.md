# Quick Start Guide - Emergency Services Setup

## ⚠️ Important Note

**There is NO universal emergency services API.** The fake URL `https://api.emergency.services/alert` was just a placeholder example.

## Real Options for Emergency Notifications

### Option 1: Twilio (Recommended - Works Immediately)

Twilio is a real service for SMS and phone calls. Perfect for notifying emergency contacts.

1. **Sign up for Twilio** (free trial): https://www.twilio.com/
2. **Get your credentials** from the Twilio dashboard
3. **Configure**:

```bash
export TWILIO_ACCOUNT_SID="ACxxxxx..."
export TWILIO_AUTH_TOKEN="your-auth-token"
export TWILIO_PHONE="+1234567890"  # Your Twilio number
export EMERGENCY_CONTACT_PHONE="+1987654321"  # Who to notify
export USER_LOCATION="123 Main St, City, State"
```

4. **Install Twilio**:
```bash
pip install twilio
```

### Option 2: Custom Webhook (Your Own Server)

If you have your own server, IFTTT, Zapier, or any webhook endpoint:

```bash
export WEBHOOK_URL="https://your-server.com/emergency-alert"
export WEBHOOK_API_KEY="optional-api-key"
```

### Option 3: Local Logging Only (No Configuration Needed)

If nothing is configured, alerts are automatically logged to `alerts.log`. This works immediately for testing:

```bash
# No configuration needed - just run!
python example_usage.py
```

Check `alerts.log` for any detected falls.

## Testing Without Configuration

You can test the system immediately without any API configuration:

```bash
cd mcp_agent
python example_usage.py
```

This will:
- Load the model
- Process sensor data
- Detect falls
- Log alerts to `alerts.log` (no external services needed)

## What About 911?

**Cannot directly call 911 via API** in most jurisdictions due to regulations. Instead:

1. System detects fall
2. System sends SMS/call to emergency contact (family, friend, caregiver)
3. Emergency contact calls 911 if needed

This is the standard approach for automated fall detection systems.

## Next Steps

1. **For testing**: Use local logging (no configuration needed)
2. **For production**: Set up Twilio (recommended) or your own webhook
3. **For deployment**: Configure environment variables in your deployment platform

