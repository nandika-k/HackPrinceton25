from flask import Flask
from flask_socketio import SocketIO
import serial
import threading
import time

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

SERIAL_PORT = '/dev/tty.usbmodem21101'  # Change to your port
BAUD_RATE = 9600

def read_serial():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        ser.reset_input_buffer()

        while True:
            line = ser.readline().decode('utf-8').strip()
            if line and ',' in line:
                try:
                    ecg, ax, ay, az = map(float, line.split(','))
                    socketio.emit('sensor_data', {
                        'ecg': ecg,
                        'ax': ax,
                        'ay': ay,
                        'az': az,
                        'timestamp': time.time()
                    })
                except Exception as e:
                    print(f"Parse error: {e}")
    except Exception as e:
        print(f"Serial connection error: {e}")

# Background thread to read serial
@socketio.on('connect')
def handle_connect():
    print("Client connected!")

if __name__ == '__main__':
    threading.Thread(target=read_serial).start()
    socketio.run(app, host='0.0.0.0', port=5100)
