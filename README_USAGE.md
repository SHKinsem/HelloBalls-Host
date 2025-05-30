# 🎾 HelloBalls Tennis Ball Retrieval Robot

A multi-threaded robot control system that uses computer vision to detect and retrieve tennis balls, and deliver them to people.

## 🎯 Features

- **Real-time Ball Detection**: Uses YOLO11 for accurate tennis ball detection
- **Person Recognition**: Detects people for ball delivery
- **PID Motor Control**: Smooth and responsive robot movement
- **Multi-threading Architecture**: Separate threads for CV, PID control, serial communication, and UI
- **ESP32 Integration**: 50Hz communication with microcontroller
- **Live Preview**: Real-time camera feed with detection overlays

## 🤖 Robot Operation Modes

### State 0: STOP
- Robot is idle
- Motors stopped
- Ready to receive commands

### State 1: CHASE_BALL
- Actively tracking and approaching detected balls
- PID control for smooth navigation
- Adjusts speed based on ball distance

### State 2: RETURN_HOME
- Navigation/return mode
- Used for returning to base position

### State 3: DELIVER_BALL
- Person detection mode
- Tilt servo control for ball delivery
- Adjustable angle based on person position

## 🔧 Hardware Requirements

- Camera (USB webcam or CSI camera)
- ESP32 microcontroller
- Motor driver and wheels
- Servo motor for ball delivery mechanism
- Serial connection (USB or UART)

## 📦 Software Dependencies

```bash
# Python packages
pip install opencv-python numpy pyserial

# System requirements
- Python 3.8+
- OpenCV 4.x
- YOLO11 model files
- Built yolo11_api module
```

## 🚀 Quick Start

### 1. Test System Components
```bash
python3 test_system.py
```

### 2. Build YOLO API (if not already built)
```bash
./build.sh
```

### 3. Run the Robot System
```bash
# Basic usage
python3 main.py

# With options
python3 main.py --camera 1 --serial-port /dev/ttyUSB0 --person-mode
```

## ⚙️ Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--no-preview` | Disable camera preview window | Preview enabled |
| `--serial-port` | Specify serial port | Auto-detect |
| `--camera` | Camera ID to use | 0 |
| `--ball-mode` | Start in ball detection mode | Default |
| `--person-mode` | Start in person detection mode | Off |
| `--resolution` | Camera resolution (720p/712p) | 720p |

## 🎮 Runtime Controls

### Preview Window Controls
- **Q**: Quit the application
- **S**: Switch camera
- **R**: Toggle resolution (720p ↔ 712p)
- **F**: Toggle fullscreen mode

### System States
The robot automatically transitions between states based on detection:
- Ball detected → State 1 (CHASE_BALL)
- No ball → State 0 (STOP)
- Person detected (in person mode) → State 3 (DELIVER_BALL)

## 📡 ESP32 Communication Protocol

The system sends commands to ESP32 at 50Hz in the format:
```
(state, left_motor_speed, right_motor_speed, tilt_servo_angle)
```

### Example Commands
```python
(0, 0, 0, 0)      # Stop
(1, 1000, 800, 0) # Chase ball: turn right
(3, 0, 0, 45)     # Deliver ball: tilt servo to 45°
```

## 🔍 System Monitoring

The system provides real-time status monitoring:

```
[2025-05-27 10:30:15.123] [STATUS] System Status: CV=True(29.8fps), PID=True(50.0fps), Serial=True(50.0fps) | Camera=0(1280x720) | Ball=YES(87%) | State=CHASE_BALL | Motors=L:1200,R:800
```

## 🧮 PID Control Configuration

The system uses two PID controllers:

### X-Axis Controller (Steering)
```python
x_pid = PIDController(
    kp=500,    # Proportional gain
    ki=10.0,   # Integral gain  
    kd=100.0,  # Derivative gain
    setpoint=0,           # Target: center of frame
    output_limits=(-1000, 1000)  # Steering limits
)
```

### Y-Axis Controller (Speed)
```python
y_pid = PIDController(
    kp=3000,   # Higher gain for speed control
    ki=10.0,   # Integral gain
    kd=100.0,  # Derivative gain
    setpoint=0.75,        # Target: 75% down frame
    output_limits=(-2000, 2000)  # Speed limits
)
```

## 🛠️ Troubleshooting

### No Camera Found
```bash
# List available cameras
ls /dev/video*

# Test camera manually
python3 -c "import cv2; print('Camera 0:', cv2.VideoCapture(0).isOpened())"
```

### Serial Connection Issues
```bash
# List serial ports
python3 -c "from scripts.HelloBalls_Serial import SerialComm; print(SerialComm.list_available_ports())"

# Check permissions
sudo usermod -a -G dialout $USER
# (logout and login again)
```

### YOLO Model Missing
```bash
# Check model files
ls -la ptq_models/

# Rebuild if necessary
./build.sh
```

## 📁 Project Structure

```
HelloBalls-Host/
├── main.py                    # Main control program
├── test_system.py            # System test script
├── scripts/
│   ├── HelloBalls_CV.py      # Computer vision module
│   └── HelloBalls_Serial.py  # Serial communication module
├── build/
│   └── yolo11_api.so         # Compiled YOLO API
├── ptq_models/               # YOLO model files
└── yolo11_demo/              # YOLO demo and examples
```

## 🎯 Ball Detection Algorithm

The system uses two ball selection strategies:

1. **Bottom Edge Priority**: Selects the ball closest to the bottom of frame (nearest to robot)
2. **Center Proximity**: Selects the ball closest to horizontal center

## 🤝 Contributing

To modify the system:

1. **CV Parameters**: Edit `HelloBalls_CV.py` for detection thresholds
2. **PID Tuning**: Adjust PID parameters in `main.py`
3. **Serial Protocol**: Modify commands in `HelloBalls_Serial.py`
4. **States**: Add new robot states in the state machine

## 📄 License

This project is part of the HelloBalls robot system. See individual module licenses for details.

---

**Happy Ball Hunting! 🎾🤖**
