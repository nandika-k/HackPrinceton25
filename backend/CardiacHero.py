"""
Real-time Data Acquisition for ECG and Accelerometer
采集 Arduino 发送的 ECG 和 MMA7660FC 加速度数据并实时显示
"""

import serial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import time

# 配置参数
SERIAL_PORT = '/dev/tty.usbmodem11101'  # 根据你的实际端口修改（Windows: COM3, Mac: /dev/tty.usbmodem*, Linux: /dev/ttyACM0）
BAUD_RATE = 9600
SAMPLE_RATE = 125  # Hz
WINDOW_SIZE = 500  # 显示最近 500 个点（约 4 秒）

class RealtimeDAQ:
    def __init__(self, port, baud_rate, window_size):
        """初始化 DAQ 系统"""
        self.port = port
        self.baud_rate = baud_rate
        self.window_size = window_size
        
        # 初始化数据缓冲区
        self.ecg_buffer = deque(maxlen=window_size)
        self.ax_buffer = deque(maxlen=window_size)
        self.ay_buffer = deque(maxlen=window_size)
        self.az_buffer = deque(maxlen=window_size)
        self.time_buffer = deque(maxlen=window_size)
        
        # 初始化串口
        try:
            self.ser = serial.Serial(port, baud_rate, timeout=1)
            print(f"成功连接到 {port}")
            time.sleep(2)  # 等待 Arduino 初始化
            # 清空初始化信息
            self.ser.reset_input_buffer()
        except Exception as e:
            print(f"无法连接到串口: {e}")
            raise
        
        # 时间戳
        self.start_time = time.time()
        
    def read_data(self):
        """从串口读取一行数据并解析"""
        try:
            line = self.ser.readline().decode('utf-8').strip()
            
            # 跳过初始化信息
            if "Initializing" in line or "Sensor ready" in line or line == "":
                return None
            
            # 解析数据: ECG, ax, ay, az
            values = line.split(',')
            if len(values) == 4:
                ecg = float(values[0])
                ax = float(values[1])
                ay = float(values[2])
                az = float(values[3])
                
                # 计算相对时间
                current_time = time.time() - self.start_time
                
                return ecg, ax, ay, az, current_time
            
        except Exception as e:
            print(f"数据解析错误: {e}")
            return None
        
        return None
    
    def update_buffers(self, data):
        """更新数据缓冲区"""
        if data is not None:
            ecg, ax, ay, az, t = data
            self.ecg_buffer.append(ecg)
            self.ax_buffer.append(ax)
            self.ay_buffer.append(ay)
            self.az_buffer.append(az)
            self.time_buffer.append(t)
    
    def close(self):
        """关闭串口连接"""
        if hasattr(self, 'ser') and self.ser.is_open:
            self.ser.close()
            print("串口已关闭")


class RealtimePlotter:
    def __init__(self, daq):
        """初始化实时绘图"""
        self.daq = daq
        
        # 创建图形窗口
        self.fig, self.axes = plt.subplots(4, 1, figsize=(12, 10))
        self.fig.suptitle('Real-time ECG and Accelerometer Data', fontsize=14, fontweight='bold')
        
        # 初始化四个子图
        self.lines = []
        titles = ['ECG Signal', 'Acceleration X (g)', 'Acceleration Y (g)', 'Acceleration Z (g)']
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        
        for i, (ax, title, color) in enumerate(zip(self.axes, titles, colors)):
            line, = ax.plot([], [], color=color, linewidth=1.5)
            self.lines.append(line)
            
            ax.set_title(title, fontsize=11, pad=5)
            ax.set_xlabel('Time (s)', fontsize=9)
            ax.set_ylabel('Amplitude', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, self.daq.window_size / SAMPLE_RATE)
            
            # 设置 Y 轴范围
            if i == 0:  # ECG
                ax.set_ylim(-400, 400)  # 根据实际信号幅度调整
            else:  # 加速度
                ax.set_ylim(-2, 2)  # ±2g
        
        plt.tight_layout()
        
    def init_animation(self):
        """动画初始化"""
        for line in self.lines:
            line.set_data([], [])
        return self.lines
    
    def update_animation(self, frame):
        """更新动画帧"""
        # 读取新数据
        data = self.daq.read_data()
        self.daq.update_buffers(data)
        
        # 更新图形
        if len(self.daq.time_buffer) > 0:
            time_array = np.array(self.daq.time_buffer)
            
            # 更新 ECG
            self.lines[0].set_data(time_array, np.array(self.daq.ecg_buffer))
            
            # 更新加速度
            self.lines[1].set_data(time_array, np.array(self.daq.ax_buffer))
            self.lines[2].set_data(time_array, np.array(self.daq.ay_buffer))
            self.lines[3].set_data(time_array, np.array(self.daq.az_buffer))
            
            # 动态调整 X 轴范围
            if time_array[-1] > self.daq.window_size / SAMPLE_RATE:
                for ax in self.axes:
                    ax.set_xlim(time_array[-1] - self.daq.window_size / SAMPLE_RATE, 
                               time_array[-1])
        
        return self.lines
    
    def start(self):
        """启动实时绘图"""
        anim = FuncAnimation(
            self.fig, 
            self.update_animation,
            init_func=self.init_animation,
            interval=1000/SAMPLE_RATE,  # 更新间隔（毫秒）
            blit=True,
            cache_frame_data=False
        )
        plt.show()


def main():
    """主函数"""
    print("=" * 50)
    print("Real-time ECG and Accelerometer DAQ System")
    print("=" * 50)
    print(f"串口: {SERIAL_PORT}")
    print(f"波特率: {BAUD_RATE}")
    print(f"采样率: {SAMPLE_RATE} Hz")
    print(f"显示窗口: {WINDOW_SIZE} 个点 ({WINDOW_SIZE/SAMPLE_RATE:.1f} 秒)")
    print("=" * 50)
    print("\n提示: 关闭图形窗口以停止采集\n")
    
    try:
        # 初始化 DAQ
        daq = RealtimeDAQ(SERIAL_PORT, BAUD_RATE, WINDOW_SIZE)
        
        # 初始化绘图
        plotter = RealtimePlotter(daq)
        
        # 开始实时绘图
        plotter.start()
        
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"\n错误: {e}")
    finally:
        if 'daq' in locals():
            daq.close()
        print("程序退出")


if __name__ == "__main__":
    main()