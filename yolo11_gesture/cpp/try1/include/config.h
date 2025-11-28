#pragma once

// Path of D-Robotics *.bin model.
#define MODEL_PATH "../../ptq_models/converted_model.bin"

// Camera device ID
#define CAMERA_ID 0

// Preprocessing method selection
#define RESIZE_TYPE 0
#define LETTERBOX_TYPE 1
#define PREPROCESS_TYPE LETTERBOX_TYPE

// Model parameters
#define CLASSES_NUM 80
#define NMS_THRESHOLD 0.3
#define SCORE_THRESHOLD 0.2  // Default threshold for most classes
#define SPORTS_BALL_THRESHOLD 0.3  // Higher threshold for sports balls (class 32)
#define NMS_TOP_K 300
#define REG 16  // Discretization level of regression part

// Drawing parameters
#define FONT_SIZE 1.0
#define FONT_THICKNESS 1.0
#define LINE_SIZE 2.0

// FPS calculation
#define FPS_SMOOTH_FACTOR 0.9

// PID Controller Parameters (initial values, can be tuned)
#define PID_KP 0.05
#define PID_KI 0.001
#define PID_KD 0.01
#define PID_MAX_OUTPUT 100  // Maximum motor speed
#define PID_MIN_OUTPUT -100 // Minimum motor speed (reverse)

// Default camera resolution
#define DEFAULT_CAM_WIDTH 1280
#define DEFAULT_CAM_HEIGHT 720

// Serial communication
#define BAUD_RATE 115200
#define COMMAND_FORMAT "0,%d,%d"  // Format: "0,speed1,speed2"
#define MAX_SERIAL_PORTS 20       // Maximum number of serial ports to check
#define RECONNECT_INTERVAL 2      // Seconds to wait before attempting reconnection
