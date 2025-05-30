#!/usr/bin/env python3
# HelloBalls_CV_Robot.py - Enhanced CV module with robot control for HelloBalls robot
# Combines computer vision with serial communication for autonomous robot behavior

import os
import sys
import numpy as np
import cv2
import time
import glob
import argparse
import threading
import queue
from enum import Enum

# Import path handling to find the YOLO API module
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
build_dir = os.path.join(parent_dir, "build")
sys.path.append(build_dir)

# Import our modules
from HelloBalls_Serial import SerialComm

# Try to import the YOLO API module
try:
    import yolo11_api # type: ignore
except ImportError as e:
    print(f"Error importing yolo11_api: {e}")
    print(f"Searched in: {build_dir}")
    print("Make sure the module is compiled and available in the build directory")
    sys.exit(1)

# Constants
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
CONFIDENCE_THRESHOLD = 0.4
SPORTS_BALL_CLASS = 32  # Sports ball class ID in COCO dataset
PERSON_CLASS = 0  # Person class ID in COCO dataset

# Detection modes
MODE_BALL_DETECTION = 0
MODE_PERSON_DETECTION = 1
MODE_NAMES = ["Ball Detection", "Person Detection"]

# Ball selection algorithms
BALL_SELECTION_BOTTOM_EDGE = 0  # Select ball with lowest bottom edge (closest to robot)
BALL_SELECTION_CENTER_PROXIMITY = 1  # Select ball closest to horizontal center
BALL_SELECTION_MODES = ["Bottom Edge Priority", "Center Proximity"]

# Robot states
ROBOT_STATE_STOP = 0
ROBOT_STATE_MANUAL = 1
ROBOT_STATE_AUTO = 2
ROBOT_STATE_SHOOT = 3

# Robot behavior states
class RobotBehavior(Enum):
    SEARCHING = "searching"
    APPROACHING = "approaching"
    ALIGNING = "aligning"
    SHOOTING = "shooting"
    STOPPED = "stopped"

# PID controller for ball tracking
class SimplePID:
    def __init__(self, kp=1.0, ki=0.0, kd=0.0, setpoint=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.prev_error = 0.0
        self.integral = 0.0
        self.prev_time = time.time()
        
    def update(self, current_value):
        current_time = time.time()
        dt = current_time - self.prev_time
        
        if dt <= 0.0:
            return 0.0
            
        error = self.setpoint - current_value
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term
        self.integral += error * dt
        i_term = self.ki * self.integral
        
        # Derivative term
        derivative = (error - self.prev_error) / dt
        d_term = self.kd * derivative
        
        # Calculate output
        output = p_term + i_term + d_term
        
        # Store values for next iteration
        self.prev_error = error
        self.prev_time = current_time
        
        return output
    
    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0
        self.prev_time = time.time()

class SimpleFpsCounter:
    """Simple FPS counter for performance monitoring"""
    def __init__(self):
        self.prev_time = time.time()
        self.frames = 0
        self.fps = 0
        self.last_console_print = time.time()
        
    def update(self, print_to_console=False):
        self.frames += 1
        current_time = time.time()
        elapsed = current_time - self.prev_time
        
        if elapsed >= 1.0:
            self.fps = self.frames / elapsed
            self.frames = 0
            self.prev_time = current_time
            
            # If print_to_console is True, print FPS every 2 seconds
            if print_to_console and (current_time - self.last_console_print >= 2.0):
                print(f"FPS: {self.fps:.1f}")
                self.last_console_print = current_time
            
        return self.fps

class HelloBallsRobot:
    """Enhanced CV class with robot control for the HelloBalls robot"""
    
    def __init__(self, show_preview=True, detection_mode=MODE_BALL_DETECTION, serial_port=None):
        """Initialize the robot system
        
        Args:
            show_preview (bool): Whether to show a preview window
            detection_mode (int): Initial detection mode (ball or person)
            serial_port (str): Serial port for ESP32 communication
        """
        self.show_preview = show_preview
        self.detection_mode = detection_mode
        self.ball_selection_mode = BALL_SELECTION_BOTTOM_EDGE
        self.camera = None
        self.camera_id = None
        self.frame_width = 1280
        self.frame_height = 712
        self.model_initialized = False
        self.fps_counter = SimpleFpsCounter()
        
        # Detection results
        self.detected_objects = []
        self.best_target = None
        self.detection_confidence = 0
        
        # Preview window settings
        self.window_name = "HelloBalls Robot - Detection & Control"
        self.is_fullscreen = False
        self.is720p = True
        
        # Console output configuration
        self.print_fps_to_console = not show_preview
        
        # Serial communication
        self.serial_comm = SerialComm(port=serial_port, auto_reconnect=True, timeout=0.01)
        self.serial_connected = False
        
        # Robot control
        self.robot_state = ROBOT_STATE_STOP
        self.behavior_state = RobotBehavior.STOPPED
        self.autonomous_mode = False
        
        # PID controllers for navigation
        self.steering_pid = SimplePID(kp=0.8, ki=0.1, kd=0.2, setpoint=0.0)  # For horizontal alignment
        self.approach_pid = SimplePID(kp=0.5, ki=0.05, kd=0.1, setpoint=200.0)  # For distance control
        
        # Robot parameters
        self.base_speed = 400
        self.max_speed = 800
        self.min_speed = 200
        self.turn_threshold = 50  # Pixel threshold for center alignment
        self.approach_distance = 150  # Target ball size for approach
        self.shoot_distance = 200   # Ball size threshold for shooting
        
        # Behavior timers
        self.behavior_start_time = time.time()
        self.last_detection_time = time.time()
        self.detection_timeout = 2.0  # Seconds without detection before searching
        
        # Shooting parameters
        self.tilt_angle = 0
        self.shoot_duration = 2.0  # Duration to maintain shooting position
        self.shoot_start_time = 0
        
    def find_available_camera(self):
        """Find an available camera
        
        Returns:
            int: Camera ID if found, None otherwise
        """
        # Try common camera indices
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"Found working camera at index {i}")
                    cap.release()
                    return i
                cap.release()
        
        # If no camera found with standard indices, try device paths (Linux)
        if os.path.exists("/dev/"):
            video_devices = glob.glob("/dev/video*")
            for device in video_devices:
                try:
                    # Extract number from device path
                    device_num = int(device.replace("/dev/video", ""))
                    cap = cv2.VideoCapture(device_num)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            print(f"Found working camera at {device} (index {device_num})")
                            cap.release()
                            return device_num
                    cap.release()
                except Exception as e:
                    print(f"Error checking {device}: {e}")
        
        print("No working camera found")
        return None

    def initialize(self):
        """Initialize the robot system (camera, model, and serial)
        
        Returns:
            bool: True if initialization was successful
        """
        # Find and open camera
        self.camera_id = self.find_available_camera()
        if self.camera_id is None:
            print("Error: No camera found")
            return False
        
        self.camera = cv2.VideoCapture(self.camera_id)
        if not self.camera.isOpened():
            print(f"Error: Failed to open camera {self.camera_id}")
            return False
        
        # Set camera properties
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer for reduced latency
        
        # Get actual resolution
        actual_width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"Camera opened with resolution: {actual_width}x{actual_height}")
        
        # Load YOLO model
        print("Initializing YOLO detection model...")
        try:
            self.model_initialized = yolo11_api.initialize_model()
            if not self.model_initialized:
                print("Error: Failed to initialize YOLO model")
                return False
            print("YOLO model initialized successfully")
        except Exception as e:
            print(f"Error initializing model: {e}")
            return False
            
        # Initialize serial communication
        print("Initializing serial communication...")
        try:
            self.serial_connected = self.serial_comm.connect()
            if self.serial_connected:
                print("Serial communication initialized successfully")
                # Send initial stop command
                self.send_robot_command(ROBOT_STATE_STOP, 0, 0, 0)
            else:
                print("Warning: Serial communication failed, running in preview-only mode")
        except Exception as e:
            print(f"Warning: Serial initialization error: {e}")
            print("Continuing in preview-only mode")
            
        # Setup preview window if enabled
        if self.show_preview:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            # Set window size to 80% of detected or default screen size
            try:
                # Try to get screen resolution using xrandr (Linux)
                import subprocess
                output = subprocess.check_output('xrandr | grep "\*" | cut -d" " -f4', shell=True).decode('utf-8').strip()
                screen_w, screen_h = map(int, output.split('x'))
            except:
                # Fallback to a common resolution
                screen_w, screen_h = 1920, 1080
                
            # Get actual camera resolution
            if self.camera and self.camera.isOpened():
                actual_width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
                actual_height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
            else:
                actual_width, actual_height = self.frame_width, self.frame_height
                
            # Calculate window size
            window_w = int(screen_w * 0.8)
            window_h = int(window_w * actual_height / actual_width)
            
            if window_h > screen_h * 0.8:
                window_h = int(screen_h * 0.8)
                window_w = int(window_h * actual_width / actual_height)
                
            cv2.resizeWindow(self.window_name, window_w, window_h)
            
            # Position window in center of screen
            win_x = (screen_w - window_w) // 2
            win_y = (screen_h - window_h) // 2
            cv2.moveWindow(self.window_name, win_x, win_y)
            
            print(f"Preview window initialized at ({win_x}, {win_y}) with size {window_w}x{window_h}")
            print("Controls:")
            print("  q: Quit")
            print("  a: Toggle autonomous mode")
            print("  s: Emergency stop")
            print("  m: Switch detection mode")
            print("  r: Toggle camera resolution")
            print("  f: Toggle fullscreen")
            print("  p: Toggle preview")
            
        return True
        
    def preprocess_image_letterbox(self, frame):
        """Preprocess image with letterboxing to maintain aspect ratio
        
        Args:
            frame: Input BGR image
            
        Returns:
            tuple: Preprocessed image and scale factors
        """
        # Calculate scale to maintain aspect ratio
        x_scale = min(INPUT_HEIGHT / frame.shape[0], INPUT_WIDTH / frame.shape[1])
        y_scale = x_scale
        
        # Calculate new dimensions
        new_w = int(frame.shape[1] * x_scale)
        x_shift = int((INPUT_WIDTH - new_w) / 2)
        
        new_h = int(frame.shape[0] * y_scale)
        y_shift = int((INPUT_HEIGHT - new_h) / 2)
        
        # Resize the image while maintaining aspect ratio
        resized = cv2.resize(frame, (new_w, new_h))
        
        # Create a canvas with gray background
        canvas = np.ones((INPUT_HEIGHT, INPUT_WIDTH, 3), dtype=np.uint8) * 127
        
        # Paste the resized image onto the canvas
        canvas[y_shift:y_shift+new_h, x_shift:x_shift+new_w] = resized
        
        return canvas, x_scale, y_scale, x_shift, y_shift
        
    def send_robot_command(self, state, motor_speed_1, motor_speed_2, tilt_angle=0):
        """Send command to the robot via serial communication
        
        Args:
            state (int): Robot state
            motor_speed_1 (int): Left motor speed
            motor_speed_2 (int): Right motor speed
            tilt_angle (int): Tilt angle for shooting
            
        Returns:
            bool: True if command sent successfully
        """
        if not self.serial_connected:
            return False
            
        try:
            success = self.serial_comm.send_command(state, motor_speed_1, motor_speed_2, tilt_angle)
            if not success:
                print("Failed to send robot command")
            return success
        except Exception as e:
            print(f"Error sending robot command: {e}")
            return False
    
    def calculate_motor_speeds(self, steering_output, forward_speed=None):
        """Calculate individual motor speeds based on steering output
        
        Args:
            steering_output (float): PID steering output
            forward_speed (int): Base forward speed
            
        Returns:
            tuple: (left_speed, right_speed)
        """
        if forward_speed is None:
            forward_speed = self.base_speed
            
        # Limit steering output
        max_steering = self.max_speed - self.min_speed
        steering_output = max(-max_steering, min(max_steering, steering_output))
        
        # Calculate motor speeds
        if steering_output > 0:  # Turn right
            left_speed = forward_speed + abs(steering_output)
            right_speed = forward_speed - abs(steering_output)
        else:  # Turn left
            left_speed = forward_speed - abs(steering_output)
            right_speed = forward_speed + abs(steering_output)
            
        # Ensure speeds are within bounds
        left_speed = max(self.min_speed, min(self.max_speed, left_speed))
        right_speed = max(self.min_speed, min(self.max_speed, right_speed))
        
        return int(left_speed), int(right_speed)
    
    def update_robot_behavior(self):
        """Update robot behavior based on detection results"""
        if not self.autonomous_mode:
            return
            
        current_time = time.time()
        
        # Check if we have a recent detection
        if self.best_target is not None:
            self.last_detection_time = current_time
            
        # Determine behavior based on detection mode and target
        if self.detection_mode == MODE_BALL_DETECTION:
            self.update_ball_following_behavior()
        elif self.detection_mode == MODE_PERSON_DETECTION:
            self.update_person_tracking_behavior()
    
    def update_ball_following_behavior(self):
        """Update behavior for ball following mode"""
        current_time = time.time()
        
        if self.best_target is None:
            # No ball detected
            if current_time - self.last_detection_time > self.detection_timeout:
                # Lost the ball, start searching
                self.behavior_state = RobotBehavior.SEARCHING
                self.search_for_ball()
            return
        
        # Extract ball information
        cls_id, x, y, w, h, conf = self.best_target
        if cls_id != SPORTS_BALL_CLASS:
            return
            
        # Calculate ball properties
        ball_center_x = x + w / 2
        ball_center_y = y + h / 2
        ball_size = w * h  # Use area as distance indicator
        
        frame_center_x = self.frame_width / 2 if self.camera else 640
        error_x = ball_center_x - frame_center_x
        
        # Determine behavior based on ball position and size
        if ball_size > self.shoot_distance:
            # Ball is close enough, switch to person detection for shooting
            self.behavior_state = RobotBehavior.SHOOTING
            self.switch_detection_mode()
            self.robot_state = ROBOT_STATE_SHOOT
            self.send_robot_command(ROBOT_STATE_STOP, 0, 0, 0)  # Stop for mode switch
            
        elif ball_size > self.approach_distance:
            # Ball is close, align for final approach
            self.behavior_state = RobotBehavior.ALIGNING
            if abs(error_x) < self.turn_threshold:
                # Aligned, move forward slowly
                self.send_robot_command(ROBOT_STATE_AUTO, self.min_speed, self.min_speed, 0)
            else:
                # Need to align
                steering = self.steering_pid.update(error_x)
                left_speed, right_speed = self.calculate_motor_speeds(steering, self.min_speed)
                self.send_robot_command(ROBOT_STATE_AUTO, left_speed, right_speed, 0)
                
        else:
            # Ball is far, approach while steering
            self.behavior_state = RobotBehavior.APPROACHING
            steering = self.steering_pid.update(error_x)
            left_speed, right_speed = self.calculate_motor_speeds(steering, self.base_speed)
            self.send_robot_command(ROBOT_STATE_AUTO, left_speed, right_speed, 0)
    
    def update_person_tracking_behavior(self):
        """Update behavior for person tracking mode (shooting phase)"""
        current_time = time.time()
        
        if self.best_target is None:
            # No person detected, search briefly then return to ball mode
            if current_time - self.last_detection_time > 1.0:  # Shorter timeout for person
                self.behavior_state = RobotBehavior.SEARCHING
                self.search_for_person()
            return
        
        # Extract person information
        cls_id, x, y, w, h, conf = self.best_target
        if cls_id != PERSON_CLASS:
            return
            
        # Calculate person center
        person_center_x = x + w / 2
        frame_center_x = self.frame_width / 2 if self.camera else 640
        error_x = person_center_x - frame_center_x
        
        if abs(error_x) < self.turn_threshold:
            # Person is aligned, initiate shooting sequence
            self.behavior_state = RobotBehavior.SHOOTING
            if self.shoot_start_time == 0:
                self.shoot_start_time = current_time
                
            # Execute shooting sequence
            self.execute_shooting_sequence()
        else:
            # Align with person
            self.behavior_state = RobotBehavior.ALIGNING
            steering = self.steering_pid.update(error_x)
            left_speed, right_speed = self.calculate_motor_speeds(steering, 0)  # Turn in place
            self.send_robot_command(ROBOT_STATE_AUTO, left_speed, right_speed, 0)
    
    def search_for_ball(self):
        """Search for ball by rotating in place"""
        # Rotate slowly to search for ball
        search_speed = self.min_speed
        self.send_robot_command(ROBOT_STATE_AUTO, -search_speed, search_speed, 0)
    
    def search_for_person(self):
        """Search for person by small adjustments"""
        # Make small turning movements to find person
        search_speed = self.min_speed // 2
        self.send_robot_command(ROBOT_STATE_AUTO, -search_speed, search_speed, 0)
    
    def execute_shooting_sequence(self):
        """Execute the ball shooting sequence"""
        current_time = time.time()
        elapsed_time = current_time - self.shoot_start_time
        
        if elapsed_time < self.shoot_duration:
            # Maintain shooting position with tilt
            self.send_robot_command(ROBOT_STATE_SHOOT, 0, 0, self.tilt_angle)
        else:
            # Shooting complete, return to ball detection mode
            self.behavior_state = RobotBehavior.SEARCHING
            self.switch_detection_mode()  # Back to ball detection
            self.robot_state = ROBOT_STATE_AUTO
            self.shoot_start_time = 0
            # Reset PID controllers
            self.steering_pid.reset()
            self.approach_pid.reset()
    
    def toggle_autonomous_mode(self):
        """Toggle autonomous robot behavior"""
        self.autonomous_mode = not self.autonomous_mode
        
        if self.autonomous_mode:
            print("Autonomous mode ENABLED")
            self.robot_state = ROBOT_STATE_AUTO
            self.behavior_state = RobotBehavior.SEARCHING
            self.behavior_start_time = time.time()
            # Ensure we start in ball detection mode
            if self.detection_mode != MODE_BALL_DETECTION:
                self.switch_detection_mode()
        else:
            print("Autonomous mode DISABLED")
            self.robot_state = ROBOT_STATE_STOP
            self.behavior_state = RobotBehavior.STOPPED
            self.send_robot_command(ROBOT_STATE_STOP, 0, 0, 0)
            
        return self.autonomous_mode
    
    def emergency_stop(self):
        """Emergency stop - disable autonomous mode and stop robot"""
        self.autonomous_mode = False
        self.robot_state = ROBOT_STATE_STOP
        self.behavior_state = RobotBehavior.STOPPED
        self.send_robot_command(ROBOT_STATE_STOP, 0, 0, 0)
        print("EMERGENCY STOP - Robot stopped, autonomous mode disabled")
    
    def switch_detection_mode(self):
        """Switch between ball and person detection"""
        # Switch detection mode
        self.detection_mode = (self.detection_mode + 1) % len(MODE_NAMES)
        
        # Reset detected objects and best target
        self.detected_objects = []
        self.best_target = None
        
        # Reset PID controllers when switching modes
        self.steering_pid.reset()
        self.approach_pid.reset()
        
        print(f"Switched to {MODE_NAMES[self.detection_mode]} mode")
        return self.detection_mode
    
    def switch_ball_selection_mode(self):
        """Switch between ball selection algorithms"""
        self.ball_selection_mode = (self.ball_selection_mode + 1) % len(BALL_SELECTION_MODES)
        print(f"Switched to {BALL_SELECTION_MODES[self.ball_selection_mode]} algorithm")
        return self.ball_selection_mode
        
    def process_frame(self):
        """Process a single frame
        
        Returns:
            tuple: (success, frame with annotations)
        """
        if not self.camera or not self.camera.isOpened():
            return False, None
            
        # Capture frame
        ret, frame = self.camera.read()
        if not ret or frame is None:
            print("Error: Failed to capture frame")
            return False, None
            
        # Get frame dimensions
        height, width = frame.shape[:2]
        
        # Preprocess the frame
        preprocessed_frame, x_scale, y_scale, x_shift, y_shift = self.preprocess_image_letterbox(frame)
        
        # Run detection
        detection_results = yolo11_api.inference(preprocessed_frame)
        
        # Reset detection results
        self.detected_objects = []
        self.best_target = None
        closest_to_center_distance = float('inf')
        
        # Process detection results
        if detection_results and len(detection_results.class_ids) > 0:
            for cls_id, boxes, confs in zip(detection_results.class_ids, 
                                         detection_results.bboxes, 
                                         detection_results.scores):
                                         
                # Skip low confidence detections
                if confs < CONFIDENCE_THRESHOLD:
                    continue
                    
                # Convert bounding box to original frame coordinates
                x = (boxes[0] - x_shift) / x_scale
                y = (boxes[1] - y_shift) / y_scale
                w = boxes[2] / x_scale
                h = boxes[3] / y_scale
                
                # Store all detections
                self.detected_objects.append({
                    'class_id': cls_id,
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h,
                    'confidence': confs
                })
                
                # Only process target objects for the current mode
                if ((self.detection_mode == MODE_BALL_DETECTION and cls_id == SPORTS_BALL_CLASS) or
                    (self.detection_mode == MODE_PERSON_DETECTION and cls_id == PERSON_CLASS)):
                    
                    # Ball selection algorithms
                    if self.ball_selection_mode == BALL_SELECTION_BOTTOM_EDGE:
                        # Get the ball closest to the bottom of the frame
                        ball_bottom_y = y + h
                        
                        if self.best_target is None or ball_bottom_y > closest_to_center_distance:
                            closest_to_center_distance = ball_bottom_y
                            self.best_target = (cls_id, x, y, w, h, confs)
                            
                    elif self.ball_selection_mode == BALL_SELECTION_CENTER_PROXIMITY:
                        # Get the ball closest to the horizontal center
                        ball_center_x = x + w / 2
                        distance_to_center = abs(ball_center_x - width / 2)
                        
                        if self.best_target is None or distance_to_center < closest_to_center_distance:
                            closest_to_center_distance = distance_to_center
                            self.best_target = (cls_id, x, y, w, h, confs)
        
        # Update detection confidence if we have a target
        if self.best_target:
            self.detection_confidence = self.best_target[5]  # confs is at index 5
        else:
            self.detection_confidence = 0
        
        # Update robot behavior
        self.update_robot_behavior()
        
        # Update FPS counter and optionally print to console
        fps = self.fps_counter.update(print_to_console=self.print_fps_to_console)
        
        # If preview is enabled, draw UI elements and detections
        if self.show_preview:
            frame = self.draw_ui(frame)
            
        return True, frame
        
    def draw_ui(self, frame):
        """Draw UI elements on the frame
        
        Args:
            frame: Input BGR image
            
        Returns:
            image: Frame with UI elements
        """
        height, width = frame.shape[:2]
        
        # Draw semi-transparent black background for info panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Display current mode with mode-appropriate color
        mode_color = (0, 0, 255) if self.detection_mode == MODE_BALL_DETECTION else (0, 255, 0)
        cv2.putText(frame, f"Mode: {MODE_NAMES[self.detection_mode]}", (20, 35), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
        
        # Display FPS
        fps = self.fps_counter.update()
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 65), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display autonomous mode status
        auto_color = (0, 255, 0) if self.autonomous_mode else (0, 0, 255)
        auto_text = "AUTO: ON" if self.autonomous_mode else "AUTO: OFF"
        cv2.putText(frame, auto_text, (20, 95), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, auto_color, 2)
        
        # Display behavior state
        cv2.putText(frame, f"Behavior: {self.behavior_state.value}", (20, 125), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Display serial connection status
        serial_color = (0, 255, 0) if self.serial_connected else (0, 0, 255)
        serial_text = "SERIAL: ON" if self.serial_connected else "SERIAL: OFF"
        cv2.putText(frame, serial_text, (20, 155), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, serial_color, 2)
                  
        # Display detection status if we have a best target
        if self.best_target:
            cls_id = self.best_target[0]
            if self.detection_mode == MODE_BALL_DETECTION and cls_id == SPORTS_BALL_CLASS:
                status = f"Ball detected: {int(self.detection_confidence * 100)}%"
                status_color = (0, 0, 255)  # Red for balls
            elif self.detection_mode == MODE_PERSON_DETECTION and cls_id == PERSON_CLASS:
                status = f"Person detected: {int(self.detection_confidence * 100)}%"
                status_color = (0, 255, 0)  # Green for people
            else:
                status = "Target detected"
                status_color = (255, 255, 255)
                
            cv2.putText(frame, status, (220, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        else:
            # No detection message with appropriate color
            if self.detection_mode == MODE_BALL_DETECTION:
                cv2.putText(frame, "No ball detected", (220, 35), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "No person detected", (220, 35), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Get the target class ID for the current detection mode
        target_class_id = SPORTS_BALL_CLASS if self.detection_mode == MODE_BALL_DETECTION else PERSON_CLASS
        
        # Draw only detected objects that match the current detection mode
        for obj in self.detected_objects:
            # Skip low-confidence detections
            if obj['confidence'] < CONFIDENCE_THRESHOLD:
                continue
                
            cls_id = obj['class_id']
            
            # Only draw boxes for objects that match the current detection mode
            if cls_id == target_class_id:
                x, y, w, h = obj['x'], obj['y'], obj['width'], obj['height']
                
                # Choose color based on class
                if cls_id == SPORTS_BALL_CLASS:
                    color = (0, 0, 255)  # Red for balls
                    label = f"Ball: {int(obj['confidence'] * 100)}%"
                elif cls_id == PERSON_CLASS:
                    color = (0, 255, 0)  # Green for people
                    label = f"Person: {int(obj['confidence'] * 100)}%"
                
                # Draw bounding box
                cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
                
                # Add label
                cv2.putText(frame, label, (int(x), int(y - 10)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw additional visualization for best target
        if self.best_target:
            cls_id, x, y, w, h, confs = self.best_target
            
            if self.detection_mode == MODE_BALL_DETECTION:
                # Ball tracking visualization
                ball_center_x = x + w / 2
                ball_center_y = y + h / 2
                
                # Draw center point of the ball
                cv2.circle(frame, (int(ball_center_x), int(ball_center_y)), 5, (0, 255, 255), -1)
                
                # Calculate target position (bottom center of frame)
                target_x = width / 2
                target_y = height * 0.9  # 90% down the frame
                
                # Draw target position
                cv2.circle(frame, (int(target_x), int(target_y)), 10, (255, 255, 0), 2)
                cv2.line(frame, (int(target_x - 15), int(target_y)), 
                        (int(target_x + 15), int(target_y)), (255, 255, 0), 2)
                cv2.line(frame, (int(target_x), int(target_y - 15)), 
                        (int(target_x), int(target_y + 15)), (255, 255, 0), 2)
                
                # Draw line from ball to target
                cv2.line(frame, (int(ball_center_x), int(ball_center_y)), 
                        (int(target_x), int(target_y)), (0, 255, 255), 2)
                
                # Show ball size for distance estimation
                ball_size = w * h
                cv2.putText(frame, f"Size: {int(ball_size)}", (int(x), int(y + h + 20)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
            elif self.detection_mode == MODE_PERSON_DETECTION:
                # Person tracking visualization
                person_center_x = x + w / 2
                person_center_y = y + h / 2
                
                # Draw center point of the person
                cv2.circle(frame, (int(person_center_x), int(person_center_y)), 5, (255, 150, 0), -1)
                
                # Calculate center of frame
                frame_center_x = width / 2
                
                # Draw a vertical line at frame center for reference
                cv2.line(frame, (int(frame_center_x), 0), (int(frame_center_x), height), 
                        (0, 150, 255), 1, cv2.LINE_AA)
                
                # Draw line from person to center line
                cv2.line(frame, (int(person_center_x), int(person_center_y)), 
                        (int(frame_center_x), int(person_center_y)), (0, 255, 255), 2)
        
        return frame
    
    def toggle_preview(self):
        """Toggle preview window on/off"""
        self.show_preview = not self.show_preview
        
        if self.show_preview:
            # Create window if we're turning on preview
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            print("Preview window enabled")
            self.print_fps_to_console = False
        else:
            # Close window if we're turning off preview
            cv2.destroyWindow(self.window_name)
            print("Preview window disabled")
            self.print_fps_to_console = True
            
        return self.show_preview
        
    def run(self):
        """Main processing loop
        
        Returns:
            bool: True if completed successfully
        """
        if not self.model_initialized or not self.camera or not self.camera.isOpened():
            print("Error: Robot system not properly initialized")
            return False
            
        print("Starting robot control loop...")
        print("Press 'a' to toggle autonomous mode, 's' for emergency stop")
        
        try:
            while True:
                # Process a frame
                success, frame = self.process_frame()
                
                if not success:
                    print("Error processing frame")
                    break
                    
                # Show frame if preview is enabled
                if self.show_preview and frame is not None:
                    cv2.imshow(self.window_name, frame)
                    
                    # Check for key presses
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('a'):
                        self.toggle_autonomous_mode()
                    elif key == ord('s'):
                        self.emergency_stop()
                    elif key == ord('m'):
                        self.switch_detection_mode()
                    elif key == ord('b'):
                        self.switch_ball_selection_mode()
                    elif key == ord('p'):
                        self.toggle_preview()
                    elif key == ord('r'):
                        # Toggle resolution (simplified)
                        print("Resolution toggle not implemented in robot mode")
                    elif key == ord('f'):
                        # Toggle fullscreen (simplified)
                        print("Fullscreen toggle not implemented in robot mode")
                else:
                    # When not showing preview, still check for keyboard input
                    # but at a lower frequency to reduce CPU usage
                    key = cv2.waitKey(100) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('a'):
                        self.toggle_autonomous_mode()
                    elif key == ord('s'):
                        self.emergency_stop()
                    elif key == ord('p'):
                        self.toggle_preview()
                        
        except KeyboardInterrupt:
            print("Processing stopped by user")
        except Exception as e:
            print(f"Error in processing loop: {e}")
        finally:
            # Clean up
            self.cleanup()
                
        return True
        
    def get_detection_results(self):
        """Get the latest detection results
        
        Returns:
            dict: Detection results including best target info and robot status
        """
        results = {
            'objects': self.detected_objects,
            'best_target': None,
            'fps': self.fps_counter.fps,
            'mode': MODE_NAMES[self.detection_mode],
            'autonomous_mode': self.autonomous_mode,
            'behavior_state': self.behavior_state.value,
            'robot_state': self.robot_state,
            'serial_connected': self.serial_connected
        }
        
        # Include best target details if available
        if self.best_target:
            cls_id, x, y, w, h, conf = self.best_target
            
            # Calculate center point of the target
            center_x = x + w/2
            center_y = y + h/2
            
            # Calculate error from center of frame (for external PID use)
            if self.camera and self.camera.isOpened():
                frame_width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
                frame_center_x = frame_width / 2
                error_x = center_x - frame_center_x
            else:
                error_x = 0
            
            results['best_target'] = {
                'class_id': cls_id,
                'x': x,
                'y': y,
                'width': w,
                'height': h,
                'confidence': conf,
                'center_x': center_x,
                'center_y': center_y,
                'error_x': error_x,
                'size': w * h  # For distance estimation
            }
            
        return results
        
    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up robot resources...")
        
        # Stop the robot
        if self.serial_connected:
            self.send_robot_command(ROBOT_STATE_STOP, 0, 0, 0)
            self.serial_comm.disconnect()
            
        if self.camera:
            self.camera.release()
            
        if self.show_preview:
            cv2.destroyAllWindows()
            
        try:
            yolo11_api.cleanup_model()
            print("Model resources released")
        except Exception as e:
            print(f"Error cleaning up model: {e}")

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HelloBalls Robot Control System')
    parser.add_argument('--no-preview', action='store_true', help='Disable preview window')
    parser.add_argument('--mode', type=int, default=0, choices=[0, 1], 
                     help='Detection mode: 0=Ball Detection, 1=Person Detection')
    parser.add_argument('--serial-port', type=str, help='Serial port for ESP32 communication')
    parser.add_argument('--auto-start', action='store_true', help='Start in autonomous mode')
    args = parser.parse_args()
    
    # Create robot instance
    robot = HelloBallsRobot(
        show_preview=not args.no_preview, 
        detection_mode=args.mode,
        serial_port=args.serial_port
    )
    
    # Initialize robot system
    if robot.initialize():
        # Start in autonomous mode if requested
        if args.auto_start:
            robot.toggle_autonomous_mode()
            
        # Run the processing loop
        robot.run()
        
        # Clean up
        robot.cleanup()
    else:
        print("Failed to initialize robot system")
