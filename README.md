# PulseGuard: Real-Time Health Monitoring & Emergency Alert System

**PulseGuard** is a real-time health monitoring application built for rapid cardiac emergency detection and response. It collects and visualizes ECG and accelerometer data from an external sensor (e.g. Arduino), processes the stream using a machine learning model, and triggers alerts with a CPR guide when abnormal patterns are detected. Built with Python, Flask, React, Tailwind CSS, and Socket.IO, the system runs fully in-browser and is optimized for quick response.

> Built at HackPrinceton '25 | Inspired by wearable health tech & real-world vitals monitoring systems

---

## Features

- **Real-Time Data Streaming** via Serial + WebSockets (ECG & Accelerometer)
- **Live ECG & Accelerometer Charts** (React + Chart.js)
- **Machine Learning Model** for cardiac anomaly detection (e.g., arrhythmia risk, fall detection)
- **Grok Integration** (for intelligent feedback or summaries)
- **Instant Emergency Alert Page** if threshold crossed
- **Step-by-step CPR Guide** (accessible anytime)
- **Cross-device compatible** (phone, tablet, laptop)
- **Modern UI** with animated transitions & mobile responsiveness

---

## System Architecture
```
[Sensor Device (e.g. Arduino)]
|
| Serial Data (ECG, ax, ay, az)
↓
[Python Flask Backend]
|
| Socket.IO Streaming
↓
[React Frontend] ←→ [ML / Grok Decision Layer]
|
| Triggers Alert Page or CPR Guide
↓
[User Interface]
```


---

## Project Structure

```
PulseGuard/
│
├── backend/
│   ├── pulseguard_stream.py     # Serial reader, WebSocket streamer
│   ├── model.py                 # ML logic (placeholder or final model)
│   ├── CardiacHero.py           # Legacy or test script
│   ├── requirements.txt         # Backend dependencies
│
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Monitor.jsx       # Main dashboard with vitals and charts
│   │   │   ├── Alert.jsx         # Emergency alert screen
│   │   │   └── CPRGuide.jsx      # Step-by-step guide
│   │   ├── components/
│   │   │   ├── LiveECGChart.jsx
│   │   │   ├── LiveAccelChart.jsx
│   │   │   └── VitalsCard.jsx
│   └── package.json
│
├── README.md
└── .gitignore
```

### Installation & Usage
Backend (Python + Flask + ML)
Install Dependencies
```
    cd backend
    python -m venv .venv
    source .venv/bin/activate  # or .venv\Scripts\activate on Windows
    pip install -r requirements.txt
```
## Run Backend Server
```
python pulseguard_stream.py
```

Ensure the correct serial port is connected and update /dev/tty.usbmodemXXXX as needed in the script.

## Frontend (React + Tailwind + Chart.js)
Install Dependencies
```
cd frontend
npm install
```

Start Development Server
```
npm run dev -- --host
```
This allows local network access (e.g., phone connected to same Wi-Fi)

### ML Model Integration

Replace the placeholder logic in pulseguard_stream.py or model.py with your custom anomaly detection model (e.g., logistic regression, XGBoost, neural nets)

Inputs: ecg, ax, ay, az

Output: A prediction value (e.g., 0–1 risk score or binary)

Trigger emergency logic if prediction > threshold (e.g., 0.7)

### Real-Time Visualizations

ECG: Line chart (updates every 125 Hz)
Accelerometer: Three-axis live chart
WebSockets ensure low-latency UI updates
Responsive mobile-friendly layout
Grok AI Integration (Optional)
If using Grok, pipe sensor_data through a separate ML/Grok process and return:
Risk score
Alert flag
Context-aware summary or recommendation (e.g., “Possible fainting event”)

### Example Output
```
ECG=+0.189 | ax=+0.021 ay=+0.098 az=-0.026 | pred=0.81 | SOS=⚠️
ECG=+0.140 | ax=+0.024 ay=+0.091 az=-0.024 | pred=0.12 | SOS=OK
```

Triggers alert if pred >= 0.70

### Roadmap
- Add user authentication & profiles
- Save vitals data to MongoDB
- Export session summaries (CSV, PDF)
- Notifications via SMS/email
- Deploy with Docker or Render

### Built With

Frontend: React.js, Tailwind CSS, Chart.js, Vite
Backend: Python, Flask, Socket.IO, PySerial
ML: Scikit-learn, Pandas, NumPy
Dev Tools: VSCode, Git, GitHub

### Contributors
- Asifur, Zach, Naiessha, Naandita

### License
MIT License — open to contributions, improvements, and customization.

### Why Rescuerar?

Because every second counts in a cardiac emergency. PulseGuard empowers rapid response through real-time data, smart alerts, and life-saving guidance — right in your browser.

“Be the pulse between data and decision.”