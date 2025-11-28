#!/usr/bin/env python3
# HelloBalls_CV.py - CV module for HelloBalls robot
# Provides camera interface and object detection functionality

import os
import sys
import numpy as np
import cv2
import time
import glob
import argparse
import subprocess
# Import serial communication
from scripts.HelloBalls_Serial_tmp import SerialComm

# Import path handling to find the YOLO API module
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
build_dir = os.path.join(parent_dir, "build")
sys.path.append(build_dir)

'''
# Try to import the YOLO API module
try:
    import yolo11_api  # type: ignore
except ImportError as e:
    print(f"Error importing yolo11_api: {e}")
    print(f"Searched in: {build_dir}")
    print("Make sure the module is compiled and available in the build directory")
    sys.exit(1)
'''

# Try to import the Gesture API module
try:
    import gesture_api  # type: ignore
    print("Successfully imported gesture_api module",gesture_api.__file__)
except ImportError as e:
    print(f"Error importing gesture_api: {e}")
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

# Robot states
ROBOT_STATE_STOP = 0
ROBOT_STATE_CHASE_BALL = 1
ROBOT_STATE_RETURN_HOME = 2
ROBOT_STATE_DELIVER_BALL = 3
ROBOT_STATE_SEARCH = 4
ROBOT_STATE_NAMES = ["STOP", "CHASE_BALL", "RETURN_HOME", "DELIVER_BALL", "SEARCH"]

# Ball selection algorithms
BALL_SELECTION_BOTTOM_EDGE = 0  # Select ball with lowest bottom edge (closest to robot)
BALL_SELECTION_CENTER_PROXIMITY = 1  # Select ball closest to horizontal center
BALL_SELECTION_MODES = ["Bottom Edge Priority", "Center Proximity"]


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

            if print_to_console and (current_time - self.last_console_print >= 2.0):
                print(f"Current FPS: {self.fps:.1f}")
                self.last_console_print = current_time

        return self.fps


class PIDController:
    """Simple PID controller for robot movement"""

    def __init__(self, kp=1.0, ki=0.0, kd=0.1, max_output=800, setpoint=0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_output = max_output
        self.setpoint = setpoint

        self.prev_error = 0
        self.integral = 0
        self.last_time = time.time()

    def compute(self, process_variable):
        """Compute PID output from process variable (calculates error internally)"""
        current_time = time.time()
        dt = current_time - self.last_time

        if dt <= 0:
            dt = 0.01

        # Calculate error from setpoint
        error = self.setpoint - process_variable

        # Proportional term
        proportional = self.kp * error

        # Integral term
        self.integral += error * dt
        integral = self.ki * self.integral

        # Derivative term
        derivative = self.kd * (error - self.prev_error) / dt

        # Compute output
        output = proportional + integral + derivative

        # Clamp output
        output = max(-self.max_output, min(self.max_output, output))

        # Update for next iteration
        self.prev_error = error
        self.last_time = current_time

        return output

    def reset(self):
        """Reset PID controller"""
        self.prev_error = 0
        self.integral = 0
        self.last_time = time.time()


class HelloBallsCV:
    """Main CV class for the HelloBalls robot"""

    def __init__(self, show_preview=True, detection_mode=MODE_BALL_DETECTION, serial_port='/dev/ttyS1'):
        """Initialize the CV system"""
        self.show_preview = show_preview
        self.detection_mode = detection_mode
        self.ball_selection_mode = BALL_SELECTION_BOTTOM_EDGE
        self.camera = None
        self.camera_id = None
        self.frame_width = 1280
        self.frame_height = 712
        self.model_initialized = False
        self.fps_counter = SimpleFpsCounter()
        self.serial_port = serial_port

        # Detection results
        self.detected_objects = []
        self.best_target = None
        self.detection_confidence = 0

        # Preview window settings
        self.window_name = "HelloBalls - Detection Preview"
        self.is_fullscreen = False
        self.is720p = True

        # Console output configuration
        self.print_fps_to_console = not show_preview        # Robot control
        self.robot_state = ROBOT_STATE_STOP
        self.serial_comm = SerialComm(port=self.serial_port, auto_reconnect=True, timeout=0.01)
        # Create separate PID controllers for X and Y like main.py
        self.x_pid = PIDController(kp=500, ki=10.0, kd=100.0, max_output=1000,
                                   setpoint=0)  # Steering control - target center
        self.y_pid = PIDController(kp=3000, ki=10.0, kd=100.0, max_output=2000,
                                   setpoint=0.75)  # Speed control - target 75% down

        # Tilt angle control for search mode
        self.tilt_angle = 0  # Current tilt angle (0-35 degrees)
        self.min_tilt_angle = -5
        self.max_tilt_angle = 35

        # Search mode speed control
        self.search_left_speed = 0   # Left motor speed in search mode
        self.search_right_speed = 0  # Right motor speed in search mode
        self.search_turn_speed = 100 # Low speed for turning in search mode
        
        # Person centering control in search mode
        self.auto_person_centering = False  # Flag to enable automatic person centering
        self.person_centering_base_speed = 120  # Base speed for person centering turns
        self.manual_override_time = 0       # Time when manual control was last used
        self.manual_override_duration = 3.0 # Seconds to wait before re-enabling auto centering

        # Motor output scaling
        self.motor_output_scale = 1.0
        self.max_motor_command = 3000
        self.last_serial_time = 0
        self.serial_interval = 0.02  # 50Hz serial communication

        # Forward boost mode variables
        self.boost_start_time = 0
        self.boost_active = False

        # Robot control callback
        self.robot_command_callback = None

    def find_available_camera(self):
        """Find an available camera"""
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
        """Initialize the CV system (camera and model)"""
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
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Load YOLO model
        try:
            #self.model_initialized = yolo11_api.initialize_model()
            self.model_initialized=gesture_api.initialize_model()
            if not self.model_initialized:
                print("Error: Failed to initialize YOLO model")
                return False
        except Exception as e:
            print(f"Error initializing model: {e}")
            return False

        # Initialize serial communication
        if not self.serial_comm.connect():
            print("Warning: Failed to connect to robot. CV will work without robot control.")

        # Setup preview window if enabled
        if self.show_preview:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            # Set window size to 80% of detected or default screen size
            try:
                # Try to get screen resolution using xrandr (Linux)
                import subprocess
                output = subprocess.check_output('xrandr | grep "\*" | cut -d" " -f4', shell=True).decode(
                    'utf-8').strip()
                screen_w, screen_h = map(int, output.split('x'))
            except:
                # Fallback to a common resolution
                screen_w, screen_h = 1920, 1080

            if self.camera and self.camera.isOpened():
                actual_width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
                actual_height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
            else:
                actual_width, actual_height = self.frame_width, self.frame_height

            window_w = int(screen_w * 0.8)
            window_h = int(window_w * actual_height / actual_width)

            if window_h > screen_h * 0.8:
                window_h = int(screen_h * 0.8)
                window_w = int(window_h * actual_width / actual_height)            
                cv2.resizeWindow(self.window_name, window_w, window_h)

            win_x = (screen_w - window_w) // 2
            win_y = (screen_h - window_h) // 2
            cv2.moveWindow(self.window_name, win_x, win_y)

            print(f"Preview window initialized at ({win_x}, {win_y}) with size {window_w}x{window_h}")
            print("Press 'q' to quit, 'r' to toggle resolution, 'f' to toggle fullscreen, "
                  "'m' to switch detection mode, 'b' to switch ball selection algorithm, 'p' to toggle preview")
            print("Robot control: '0' STOP, '1' CHASE_BALL, '2' RETURN_HOME, '3' DELIVER_BALL, '4' SEARCH")
            print("Search mode: 'w' tilt up, 's' tilt down")

        return True

    def preprocess_image_letterbox(self, frame):
        """Preprocess image with letterboxing to maintain aspect ratio"""
        x_scale = min(INPUT_HEIGHT / frame.shape[0], INPUT_WIDTH / frame.shape[1])
        y_scale = x_scale

        new_w = int(frame.shape[1] * x_scale)
        x_shift = int((INPUT_WIDTH - new_w) / 2)

        new_h = int(frame.shape[0] * y_scale)
        y_shift = int((INPUT_HEIGHT - new_h) / 2)

        resized = cv2.resize(frame, (new_w, new_h))

        canvas = np.ones((INPUT_HEIGHT, INPUT_WIDTH, 3), dtype=np.uint8) * 127
        canvas[y_shift:y_shift + new_h, x_shift:x_shift + new_w] = resized

        return canvas, x_scale, y_scale, x_shift, y_shift

    def toggle_resolution(self):
        """Toggle camera resolution

        Returns:
            bool: New resolution state
        """
        if not self.camera or not self.camera.isOpened():
            return self.is720p

        # Store current camera settings to preserve state
        original_buffersize = self.camera.get(cv2.CAP_PROP_BUFFERSIZE)
        
        # Temporarily set buffer size for smoother transition
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if self.is720p:
            # Switch to 712p
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 712)
            print("\r\nResolution changed to 1280x712")
        else:
            # Switch to 720p
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            print("\r\nResolution changed to 1280x720")

        # Gentle buffer flush - only one frame to avoid state disruption
        try:
            ret, _ = self.camera.read()
            if not ret:
                print("Warning: Could not read frame after resolution change")
        except Exception as e:
            print(f"Warning: Error reading frame after resolution change: {e}")

        # Restore original buffer size to maintain camera stability
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, original_buffersize)

        # Get actual resolution
        actual_width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"\r\nActual resolution: {actual_width}x{actual_height}")

        self.is720p = not self.is720p
        return self.is720p

    def toggle_fullscreen(self):
        """Toggle fullscreen mode for preview window

        Returns:
            bool: New fullscreen state
        """
        if not self.show_preview:
            print("\r\nFullscreen mode requires preview window to be enabled")
            return False

        self.is_fullscreen = not self.is_fullscreen

        if self.is_fullscreen:
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            print("\r\nSwitched to fullscreen mode")
        else:
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            print("\r\nExited fullscreen mode")

        return self.is_fullscreen

    def toggle_preview(self):
        """Toggle preview window on/off

        Returns:
            bool: New preview state
        """
        self.show_preview = not self.show_preview

        if self.show_preview:
            # Create window if we're turning on preview
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

            # Set window size to 80% of detected or default screen size
            try:
                # Try to get screen resolution using xrandr (Linux)
                import subprocess
                output = subprocess.check_output('xrandr | grep "\*" | cut -d" " -f4', shell=True).decode(
                    'utf-8').strip()
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

            print(f"\r\nPreview window enabled")
            print("\r\nPress 'q' to quit, 'r' to toggle resolution, 'f' to toggle fullscreen, "
                  "'m' to switch detection mode, 'b' to switch ball selection algorithm, 'p' to toggle preview")

            # Turn off console FPS printing when preview is on
            self.print_fps_to_console = False
        else:
            # Close window if we're turning off preview
            cv2.destroyWindow(self.window_name)
            print("\r\nPreview window disabled, FPS will be printed to console every 2 seconds")

            # Turn on console FPS printing when preview is off
            self.print_fps_to_console = True

        return self.show_preview

    def switch_detection_mode(self):
        """Switch between ball and person detection"""
        self.detection_mode = (self.detection_mode + 1) % len(MODE_NAMES)
        self.detected_objects = []
        self.best_target = None
        self.x_pid.reset()  # Reset both PID controllers when switching modes
        self.y_pid.reset()
        self.person_centering_pid.reset()  # Reset person centering PID as well
        
        # Enable automatic person centering when switching to person detection in SEARCH mode
        if self.detection_mode == MODE_PERSON_DETECTION and self.robot_state == ROBOT_STATE_SEARCH:
            self.auto_person_centering = True
            print(f"\r\nSwitched to {MODE_NAMES[self.detection_mode]} mode - Automatic person centering ENABLED")
        else:
            self.auto_person_centering = False
            print(f"\r\nSwitched to {MODE_NAMES[self.detection_mode]} mode")
        
        return self.detection_mode

    def switch_ball_selection_mode(self):
        """Switch between ball selection algorithms"""
        self.ball_selection_mode = (self.ball_selection_mode + 1) % len(BALL_SELECTION_MODES)
        print(f"\r\nSwitched to {BALL_SELECTION_MODES[self.ball_selection_mode]} algorithm")
        return self.ball_selection_mode


    # def process_frame(self):
    #     """Process a single frame"""

    #     print("ffffffffffffffffffffframe")
    #     if not self.camera or not self.camera.isOpened():
    #         return False, None

    #     print("frameeeeeeeeeeeee")

    #     # Capture frame
    #     ret, frame = self.camera.read()
    #     print(ret)
    #     if not ret or frame is None:
    #         print("Error: Failed to capture frame")
    #         return False, None

    #     # Get frame dimensions
    #     height, width = frame.shape[:2]

    #     # Preprocess the frame
    #     preprocessed_frame, x_scale, y_scale, x_shift, y_shift = self.preprocess_image_letterbox(frame)

    #     # Run detection
    #     detection_results = yolo11_api.inference(preprocessed_frame)

    #     # Reset detection results
    #     self.detected_objects = []
    #     self.best_target = None
    #     closest_to_center_distance = float('inf')

    #     # Process detection results
    #     if detection_results and len(detection_results.class_ids) > 0:
    #         for cls_id, boxes, confs in zip(detection_results.class_ids,
    #                                         detection_results.bboxes,
    #                                         detection_results.scores):

    #             if confs < CONFIDENCE_THRESHOLD:
    #                 continue

    #             # Convert bounding box to original frame coordinates
    #             x = (boxes[0] - x_shift) / x_scale
    #             y = (boxes[1] - y_shift) / y_scale
    #             w = boxes[2] / x_scale
    #             h = boxes[3] / y_scale

    #             # Store all detections
    #             self.detected_objects.append({
    #                 'class_id': cls_id,
    #                 'x': x,
    #                 'y': y,
    #                 'width': w,
    #                 'height': h,
    #                 'confidence': confs
    #             })

    #             #modify: print bbox , confs
    #             #print(f"[BBOX] Class:{cls_id},x:{x:.1f},y:{y:.1f},w:{w:.1f},h:{h:.1f},confidence::{confs:.2f}")
    #             #Only process target objects for the current mode
    #             if ((self.detection_mode == MODE_BALL_DETECTION and cls_id == SPORTS_BALL_CLASS) or
    #                     (self.detection_mode == MODE_PERSON_DETECTION and cls_id == PERSON_CLASS)):

    #                 # Ball selection algorithms
    #                 if self.ball_selection_mode == BALL_SELECTION_BOTTOM_EDGE:
    #                     ball_bottom_y = y + h

    #                     if self.best_target is None or ball_bottom_y > closest_to_center_distance:
    #                         closest_to_center_distance = ball_bottom_y
    #                         self.best_target = (cls_id, x, y, w, h, confs)

    #                 elif self.ball_selection_mode == BALL_SELECTION_CENTER_PROXIMITY:
    #                     ball_center_x = x + w / 2
    #                     distance_to_center = abs(ball_center_x - width / 2)

    #                     if self.best_target is None or distance_to_center < closest_to_center_distance:
    #                         closest_to_center_distance = distance_to_center
    #                         self.best_target = (cls_id, x, y, w, h, confs)

    #     # Update detection confidence
    #     if self.best_target:
    #         self.detection_confidence = self.best_target[5]
    #     else:
    #         self.detection_confidence = 0

    #     # Control robot based on current state and detections
    #     self.control_robot()

    #     # Update FPS counter
    #     fps = self.fps_counter.update(print_to_console=self.print_fps_to_console)

    #     # Draw UI if preview is enabled
    #     if self.show_preview:
    #         frame = self.draw_ui(frame)

    #     return True, frame

    def draw_ui(self, frame):
        """Draw UI elements on the frame"""
        height, width = frame.shape[:2]

        # Draw semi-transparent black background for info panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 130), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Display current mode with mode-appropriate color
        mode_color = (0, 0, 255) if self.detection_mode == MODE_BALL_DETECTION else (0, 255, 0)
        cv2.putText(frame, f"Mode: {MODE_NAMES[self.detection_mode]}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)

        # Display robot state
        state_color = (0, 255, 255) if self.robot_state != ROBOT_STATE_STOP else (128, 128, 128)
        cv2.putText(frame, f"Robot: {ROBOT_STATE_NAMES[self.robot_state]}", (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)

        # Display FPS
        fps = self.fps_counter.update()
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Display detection status
        if self.best_target:
            cls_id = self.best_target[0]
            if self.detection_mode == MODE_BALL_DETECTION and cls_id == SPORTS_BALL_CLASS:
                status = f"Ball detected: {int(self.detection_confidence * 100)}%"
                status_color = (0, 0, 255)
            elif self.detection_mode == MODE_PERSON_DETECTION and cls_id == PERSON_CLASS:
                status = f"Person detected: {int(self.detection_confidence * 100)}%"
                status_color = (0, 255, 0)
            else:
                status = "Target detected"
                status_color = (255, 255, 255)

            cv2.putText(frame, status, (20, 125),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        else:
            if self.detection_mode == MODE_BALL_DETECTION:
                cv2.putText(frame, "No ball detected", (20, 125),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "No person detected", (20, 125),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw detection boxes and visualizations
        target_class_id = SPORTS_BALL_CLASS if self.detection_mode == MODE_BALL_DETECTION else PERSON_CLASS

        for obj in self.detected_objects:
            if obj['confidence'] < CONFIDENCE_THRESHOLD:
                continue

            cls_id = obj['class_id']

            if cls_id == target_class_id:
                x, y, w, h = obj['x'], obj['y'], obj['width'], obj['height']

                if cls_id == SPORTS_BALL_CLASS:
                    color = (0, 0, 255)
                    label = f"Ball: {int(obj['confidence'] * 100)}%"
                elif cls_id == PERSON_CLASS:
                    color = (0, 255, 0)
                    label = f"Person: {int(obj['confidence'] * 100)}%"

                cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
                cv2.putText(frame, label, (int(x), int(y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw tracking visualization for best target
        if self.best_target:
            cls_id, x, y, w, h, confs = self.best_target

            if self.detection_mode == MODE_BALL_DETECTION:
                # Ball tracking visualization
                ball_center_x = x + w / 2
                ball_center_y = y + h / 2

                cv2.circle(frame, (int(ball_center_x), int(ball_center_y)), 5, (0, 255, 255), -1)

                target_x = width / 2
                target_y = height * 0.9

                cv2.circle(frame, (int(target_x), int(target_y)), 10, (255, 255, 0), 2)
                cv2.line(frame, (int(target_x - 15), int(target_y)),
                         (int(target_x + 15), int(target_y)), (255, 255, 0), 2)
                cv2.line(frame, (int(target_x), int(target_y - 15)),
                         (int(target_x), int(target_y + 15)), (255, 255, 0), 2)

                cv2.line(frame, (int(ball_center_x), int(ball_center_y)),
                         (int(target_x), int(target_y)), (0, 255, 255), 2)

            elif self.detection_mode == MODE_PERSON_DETECTION:
                # Person tracking visualization
                person_center_x = x + w / 2
                person_center_y = y + h / 2

                cv2.circle(frame, (int(person_center_x), int(person_center_y)), 5, (255, 150, 0), -1)

                frame_center_x = width / 2

                # Draw center reference line
                cv2.line(frame, (int(frame_center_x), 0), (int(frame_center_x), height),
                         (0, 150, 255), 1, cv2.LINE_AA)

                # Draw error line
                cv2.line(frame, (int(person_center_x), int(person_center_y)),
                         (int(frame_center_x), int(person_center_y)), (0, 255, 255), 2)                # Show centering status
                error_x = person_center_x - frame_center_x
                if abs(error_x) < 30:
                    cv2.putText(frame, "CENTERED", (int(frame_center_x - 50), 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Show automatic person centering status in SEARCH mode
                if self.robot_state == ROBOT_STATE_SEARCH and self.auto_person_centering:
                    # Check if manual override is active
                    current_time = time.time()
                    manual_override_active = (self.manual_override_time > 0 and 
                                            current_time - self.manual_override_time < self.manual_override_duration)
                    
                    if manual_override_active:
                        status_text = "MANUAL OVERRIDE"
                        status_color = (0, 165, 255)  # Orange for manual override
                        remaining_time = self.manual_override_duration - (current_time - self.manual_override_time)
                        cv2.putText(frame, status_text, (20, 155),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                        cv2.putText(frame, f"Auto in {remaining_time:.1f}s", (20, 185),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
                    else:
                        status_text = "AUTO-CENTERING"
                        status_color = (0, 255, 255)  # Yellow for active auto-centering
                        cv2.putText(frame, status_text, (20, 155),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                        
                        # Show error value
                        error_text = f"Error: {int(error_x)}px"
                        cv2.putText(frame, error_text, (20, 185),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        return frame

    def show_help(self):
        """Display help information"""
        print("\n" + "=" * 50)
        print("HelloBalls CV System - Keyboard Controls")
        print("=" * 50)
        print("  'q' - Quit the application")
        print("  'r' - Toggle camera resolution (720p/712p)")
        print("  'f' - Toggle fullscreen mode (preview window)")
        print("  'm' - Switch detection mode (Ball/Person)")
        print("  'b' - Switch ball selection algorithm")
        print("  'p' - Toggle preview window on/off")
        print("  'h' - Show this help message")
        print("  '0' - Robot STOP state")
        print("  '1' - Robot CHASE_BALL state")
        print("  '2' - Robot RETURN_HOME state")
        print("  '3' - Robot DELIVER_BALL state")
        print("  '4' - Robot SEARCH state")
        print("  'i' - Tilt camera up (SEARCH mode only)")
        print("  'k' - Tilt camera down (SEARCH mode only)")
        print("  'w' - Move forward (SEARCH mode only)")
        print("  's' - Move backward (SEARCH mode only)")
        print("  'a' - Turn left (SEARCH mode only)")
        print("  'd' - Turn right (SEARCH mode only)")
        print("  'space' - Stop movement (SEARCH mode only)")
        print("")
        print("SEARCH Mode Features:")
        print("  • Manual movement controls ('w'/'s'/'a'/'d')")
        print("  • Camera tilt control ('i'/'k')")
        print("  • Auto person centering (press 'm' for person mode)")
        print("=" * 50)
        print(f"Current mode: {MODE_NAMES[self.detection_mode]}")
        print(f"Ball selection: {BALL_SELECTION_MODES[self.ball_selection_mode]}")
        print(f"Preview: {'On' if self.show_preview else 'Off'}")
        print(f"FPS: {self.fps_counter.fps:.1f}")
        if self.robot_state == ROBOT_STATE_SEARCH:
            print(f"Tilt angle: {self.tilt_angle}° (range: {self.min_tilt_angle}-{self.max_tilt_angle}°)")
            if self.auto_person_centering:
                print(f"Auto person centering: ENABLED")
            else:
                print(f"Auto person centering: DISABLED")
        print("=" * 50)

    def show_system_info(self):
        """Display detailed system information"""
        print("\n" + "=" * 60)
        print("HelloBalls CV System - System Information")
        print("=" * 60)

        # Camera information
        if self.camera and self.camera.isOpened():
            actual_width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
            print(f"Camera ID: {self.camera_id}")
            print(f"Camera Resolution: {actual_width}x{actual_height}")
            print(f"Resolution Mode: {'720p' if self.is720p else '712p'}")
        else:
            print("Camera: Not initialized")

        # Model information
        print(f"YOLO Model: {'Initialized' if self.model_initialized else 'Not initialized'}")

        # Detection settings
        print(f"Detection Mode: {MODE_NAMES[self.detection_mode]}")
        print(f"Ball Selection Algorithm: {BALL_SELECTION_MODES[self.ball_selection_mode]}")
        print(f"Confidence Threshold: {CONFIDENCE_THRESHOLD}")

        # Preview window information
        if self.show_preview:
            try:
                import subprocess
                output = subprocess.check_output('xrandr | grep "\*" | cut -d" " -f4', shell=True).decode(
                    'utf-8').strip()
                screen_w, screen_h = map(int, output.split('x'))
            except:
                screen_w, screen_h = 1920, 1080

            if self.camera and self.camera.isOpened():
                actual_width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
                actual_height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
            else:
                actual_width, actual_height = self.frame_width, self.frame_height

            window_w = int(screen_w * 0.8)
            window_h = int(window_w * actual_height / actual_width)

            if window_h > screen_h * 0.8:
                window_h = int(screen_h * 0.8)
                window_w = int(window_h * actual_width / actual_height)

            win_x = (screen_w - window_w) // 2
            win_y = (screen_h - window_h) // 2

            print(f"Preview Window: Enabled")
            print(f"Window Position: ({win_x}, {win_y})")
            print(f"Window Size: {window_w}x{window_h}")
            print(f"Screen Resolution: {screen_w}x{screen_h}")
            print(f"Fullscreen Mode: {'On' if self.is_fullscreen else 'Off'}")
        else:
            print("Preview Window: Disabled")
            print("Console FPS Printing: Enabled")

        # Performance information
        print(f"Current FPS: {self.fps_counter.fps:.1f}")
        print(f"Terminal Input: {'Enabled' if self.terminal_input_enabled else 'Disabled'}")

        # Detection results
        if self.best_target:
            cls_id, x, y, w, h, conf = self.best_target
            print(f"Current Target: {MODE_NAMES[self.detection_mode]} at ({x:.1f}, {y:.1f})")
            print(f"Target Confidence: {conf * 100:.1f}%")
            print(f"Target Size: {w:.1f}x{h:.1f}")
        else:
            print("Current Target: None detected")

        print(f"Total Objects Detected: {len(self.detected_objects)}")
        print("=" * 60)

    # def setup_terminal_input(self):
    #     """Setup terminal for non-blocking input"""
    #     try:
    #         # Save original terminal settings
    #         self.old_terminal_settings = termios.tcgetattr(sys.stdin)
    #
    #         # Set terminal to raw mode for immediate key detection
    #         tty.setraw(sys.stdin.fileno())
    #
    #         print("\nTerminal input enabled. Press keys:")
    #         print("  'q' - Quit  |  'r' - Resolution  |  'f' - Fullscreen")
    #         print("  'm' - Mode  |  'b' - Ball Algo   |  'p' - Preview")
    #         print("  'i' - Info  |  'h' - Help")
    #         print("-" * 50)
    #
    #     except Exception as e:
    #         print(f"Warning: Could not setup terminal input: {e}")
    #         self.terminal_input_enabled = False

    # def restore_terminal(self):
    #     """Restore original terminal settings"""
    #     if self.old_terminal_settings is not None:
    #         try:
    #             termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_terminal_settings)
    #         except Exception as e:
    #             print(f"Warning: Could not restore terminal settings: {e}")

    # def check_terminal_input(self):
    #     """Check for terminal keyboard input (non-blocking)
    #
    #     Returns:
    #         str or None: Key pressed, or None if no input
    #     """
    #     if not self.terminal_input_enabled:
    #         return None
    #
    #     try:
    #         # Check if input is available without blocking
    #         if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
    #             key = sys.stdin.read(1)
    #             return key
    #     except Exception:
    #         pass
    #
    #     return None

    def handle_keyboard_input(self, key):
        """Handle keyboard input from both preview window and terminal

        Args:
            key (str or int): Key pressed

        Returns:
            bool: True to continue processing, False to quit
        """
        # Convert to string if it's an integer (from cv2.waitKey)
        if isinstance(key, int):
            if key == 255 or key == -1:  # No key pressed
                return True
            key = chr(key)

        # Handle different key commands
        if key == 'q' or key == 'Q':
            print("\nQuitting...")
            return False
        elif key == 'r' or key == 'R':
            self.toggle_resolution()
        elif key == 'f' or key == 'F':
            self.toggle_fullscreen()
        elif key == 'm' or key == 'M':
            self.switch_detection_mode()
        elif key == 'b' or key == 'B':
            self.switch_ball_selection_mode()
        elif key == 'p' or key == 'P':
            self.toggle_preview()
        elif key == 'h' or key == 'H':
            self.show_help()
        # Robot control commands
        elif key in ['0', '1', '2', '3', '4']:
            state = int(key)
            self.robot_state = state
            state_names = ["STOP", "CHASE_BALL", "RETURN_HOME", "DELIVER_BALL", "SEARCH"]
            print(f"\nRobot state set to {state_names[state]}")
            # Reset PID controllers when changing states
            self.x_pid.reset()
            self.y_pid.reset()
            # Reset boost state when changing states
            self.boost_active = False
            # Reset search mode speeds when entering any state
            self.search_left_speed = 0
            self.search_right_speed = 0
            
            # Handle automatic person centering state
            if state == ROBOT_STATE_SEARCH:
                # Enable auto person centering if in person detection mode
                if self.detection_mode == MODE_PERSON_DETECTION:
                    self.auto_person_centering = True
                    print(f"SEARCH mode activated - Automatic person centering enabled")
                else:
                    self.auto_person_centering = False
            else:
                # Disable auto person centering when leaving SEARCH mode
                self.auto_person_centering = False
            
            # Immediately send command to trigger connection (this will trigger auto-reconnect if needed)
            if state == ROBOT_STATE_SEARCH:
                self.serial_comm.send_command(self.robot_state, self.search_left_speed, self.search_right_speed, self.tilt_angle)
            else:
                self.serial_comm.send_command(self.robot_state, 0, 0, 0)
        # Tilt control for search mode
        elif key == 'i' or key == 'I':
            if self.robot_state == ROBOT_STATE_SEARCH:
                if self.tilt_angle < self.max_tilt_angle:
                    self.tilt_angle += 3
                    print(f"\nTilt angle increased to {self.tilt_angle}°")
                    self.serial_comm.send_command(self.robot_state, self.search_left_speed, self.search_right_speed, self.tilt_angle)
                else:
                    print(f"\nTilt angle already at maximum ({self.max_tilt_angle}°)")
            else:
                print(f"\nTilt control only available in SEARCH mode (current mode: {ROBOT_STATE_NAMES[self.robot_state]})")
        elif key == 'k' or key == 'K':
            if self.robot_state == ROBOT_STATE_SEARCH:
                if self.tilt_angle > self.min_tilt_angle:
                    self.tilt_angle -= 3
                    print(f"\nTilt angle decreased to {self.tilt_angle}°")
                    self.serial_comm.send_command(self.robot_state, self.search_left_speed, self.search_right_speed, self.tilt_angle)
                else:
                    print(f"\nTilt angle already at minimum ({self.min_tilt_angle}°)")
            else:
                print(f"\nTilt control only available in SEARCH mode (current mode: {ROBOT_STATE_NAMES[self.robot_state]})")
        # Forward and backward movement for search mode
        elif key == 'w' or key == 'W':
            if self.robot_state == ROBOT_STATE_SEARCH:
                # Move forward: both motors forward
                self.search_left_speed = self.search_turn_speed
                self.search_right_speed = self.search_turn_speed
                print(f"\nMoving forward (speed: {self.search_turn_speed})")
                self.serial_comm.send_command(self.robot_state, self.search_left_speed, self.search_right_speed, self.tilt_angle)
            else:
                print(f"\nMovement control only available in SEARCH mode (current mode: {ROBOT_STATE_NAMES[self.robot_state]})")
        elif key == 's' or key == 'S':
            if self.robot_state == ROBOT_STATE_SEARCH:
                # Move backward: both motors reverse
                self.search_left_speed = -self.search_turn_speed
                self.search_right_speed = -self.search_turn_speed
                print(f"\nMoving backward (speed: {self.search_turn_speed})")
                self.serial_comm.send_command(self.robot_state, self.search_left_speed, self.search_right_speed, self.tilt_angle)
            else:
                print(f"\nMovement control only available in SEARCH mode (current mode: {ROBOT_STATE_NAMES[self.robot_state]})")
        # Turning control for search mode
        elif key == 'a' or key == 'A':
            if self.robot_state == ROBOT_STATE_SEARCH:
                # Turn left: left motor forward, right motor slower/reverse
                self.search_left_speed = -self.search_turn_speed
                self.search_right_speed = self.search_turn_speed
                
                # Temporarily disable auto person centering when manual control is used
                if self.auto_person_centering:
                    self.manual_override_time = time.time()
                    print(f"\nTurning left (speed: {self.search_turn_speed}) - Manual override active")
                else:
                    print(f"\nTurning left (speed: {self.search_turn_speed})")
                    
                self.serial_comm.send_command(self.robot_state, self.search_left_speed, self.search_right_speed, self.tilt_angle)
            else:
                print(f"\nTurn control only available in SEARCH mode (current mode: {ROBOT_STATE_NAMES[self.robot_state]})")
        elif key == 'd' or key == 'D':
            if self.robot_state == ROBOT_STATE_SEARCH:
                # Turn right: left motor slower/reverse, right motor forward
                self.search_left_speed = self.search_turn_speed
                self.search_right_speed = -self.search_turn_speed
                
                # Temporarily disable auto person centering when manual control is used
                if self.auto_person_centering:
                    self.manual_override_time = time.time()
                    print(f"\nTurning right (speed: {self.search_turn_speed}) - Manual override active")
                else:
                    print(f"\nTurning right (speed: {self.search_turn_speed})")
                    
                self.serial_comm.send_command(self.robot_state, self.search_left_speed, self.search_right_speed, self.tilt_angle)
            else:
                print(f"\nTurn control only available in SEARCH mode (current mode: {ROBOT_STATE_NAMES[self.robot_state]})")
        elif key == ' ':  # Spacebar to stop movement in search mode
            if self.robot_state == ROBOT_STATE_SEARCH:
                self.search_left_speed = 0
                self.search_right_speed = 0
                
                # Reset manual override timer when explicitly stopping
                self.manual_override_time = 0
                
                print(f"\nStopped movement")
                self.serial_comm.send_command(self.robot_state, self.search_left_speed, self.search_right_speed, self.tilt_angle)
            else:
                print(f"\nMovement control only available in SEARCH mode (current mode: {ROBOT_STATE_NAMES[self.robot_state]})")

        return True

    def initialize(self):
        """Initialize the CV system (camera and model)

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

        # Load YOLO model
        try:
            #self.model_initialized = yolo11_api.initialize_model()
            self.model_initialized= gesture_api.initialize_model()
            if not self.model_initialized:
                print("Error: Failed to initialize YOLO model")
                return False
        except Exception as e:
            print(f"Error initializing model: {e}")
            return False

        # Setup preview window if enabled
        if self.show_preview:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            # Set window size to 80% of detected or default screen size
            try:
                # Try to get screen resolution using xrandr (Linux)
                import subprocess
                output = subprocess.check_output('xrandr | grep "\*" | cut -d" " -f4', shell=True).decode(
                    'utf-8').strip()
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
            print("Press 'q' to quit, 'r' to toggle resolution, 'f' to toggle fullscreen, "
                  "'m' to switch detection mode, 'b' to switch ball selection algorithm, 'p' to toggle preview")

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
        canvas[y_shift:y_shift + new_h, x_shift:x_shift + new_w] = resized

        return canvas, x_scale, y_scale, x_shift, y_shift

    def toggle_resolution(self):
        """Toggle camera resolution

        Returns:
            bool: New resolution state
        """
        if not self.camera or not self.camera.isOpened():
            return self.is720p

        # Store current camera settings to preserve state
        original_buffersize = self.camera.get(cv2.CAP_PROP_BUFFERSIZE)
        
        # Temporarily set buffer size for smoother transition
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if self.is720p:
            # Switch to 712p
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 712)
            print("\r\nResolution changed to 1280x712")
        else:
            # Switch to 720p
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            print("\r\nResolution changed to 1280x720")

        # Gentle buffer flush - only one frame to avoid state disruption
        try:
            ret, _ = self.camera.read()
            if not ret:
                print("Warning: Could not read frame after resolution change")
        except Exception as e:
            print(f"Warning: Error reading frame after resolution change: {e}")

        # Restore original buffer size to maintain camera stability
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, original_buffersize)

        # Get actual resolution
        actual_width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"\r\nActual resolution: {actual_width}x{actual_height}")

        self.is720p = not self.is720p
        return self.is720p

    def toggle_fullscreen(self):
        """Toggle fullscreen mode for preview window

        Returns:
            bool: New fullscreen state
        """
        if not self.show_preview:
            print("\r\nFullscreen mode requires preview window to be enabled")
            return False

        self.is_fullscreen = not self.is_fullscreen

        if self.is_fullscreen:
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            print("\r\nSwitched to fullscreen mode")
        else:
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            print("\r\nExited fullscreen mode")

        return self.is_fullscreen

    def toggle_preview(self):
        """Toggle preview window on/off

        Returns:
            bool: New preview state
        """
        self.show_preview = not self.show_preview

        if self.show_preview:
            # Create window if we're turning on preview
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

            # Set window size to 80% of detected or default screen size
            try:
                # Try to get screen resolution using xrandr (Linux)
                import subprocess
                output = subprocess.check_output('xrandr | grep "\*" | cut -d" " -f4', shell=True).decode(
                    'utf-8').strip()
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

            print(f"\r\nPreview window enabled")
            print("\r\nPress 'q' to quit, 'r' to toggle resolution, 'f' to toggle fullscreen, "
                  "'m' to switch detection mode, 'b' to switch ball selection algorithm, 'p' to toggle preview")

            # Turn off console FPS printing when preview is on
            self.print_fps_to_console = False
        else:
            # Close window if we're turning off preview
            cv2.destroyWindow(self.window_name)
            print("\r\nPreview window disabled, FPS will be printed to console every 2 seconds")

            # Turn on console FPS printing when preview is off
            self.print_fps_to_console = True

        return self.show_preview

    def switch_detection_mode(self):
        """Switch between ball and person detection

        Returns:
            int: New detection mode
        """
        # Switch detection mode
        self.detection_mode = (self.detection_mode + 1) % len(MODE_NAMES)

        # Reset detected objects and best target
        self.detected_objects = []
        self.best_target = None

        # Reset PID controllers when switching detection modes to ensure clean state
        self.x_pid.reset()
        self.y_pid.reset()

        # Enable automatic person centering when switching to person detection in SEARCH mode
        if self.detection_mode == MODE_PERSON_DETECTION and self.robot_state == ROBOT_STATE_SEARCH:
            self.auto_person_centering = True
            print(f"\r\nSwitched to {MODE_NAMES[self.detection_mode]} mode - Automatic person centering ENABLED")
        else:
            self.auto_person_centering = False
            print(f"\r\nSwitched to {MODE_NAMES[self.detection_mode]} mode")

        return self.detection_mode

    def switch_ball_selection_mode(self):
        """Switch between ball selection algorithms

        Returns:
            int: New ball selection mode
        """
        self.ball_selection_mode = (self.ball_selection_mode + 1) % len(BALL_SELECTION_MODES)
        print(f"\r\nSwitched to {BALL_SELECTION_MODES[self.ball_selection_mode]} algorithm")
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
        #detection_results = yolo11_api.inference(preprocessed_frame)
        detection_results = gesture_api.inference(preprocessed_frame)

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
                #modify:print bbox & conf level
                #print(f"[BBOX] Class:{cls_id},x:{x:.1f},y:{y:.1f},w:{w:.1f},h:{h:.1f},confidence::{confs:.2f}")
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

        # Update FPS counter and optionally print to console
        fps = self.fps_counter.update(print_to_console=self.print_fps_to_console)

        # Control robot based on current state and detection results
        self.control_robot()

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
        cv2.rectangle(overlay, (10, 10), (320, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Display current mode with mode-appropriate color
        mode_color = (0, 0, 255) if self.detection_mode == MODE_BALL_DETECTION else (0, 255, 0)
        cv2.putText(frame, f"Mode: {MODE_NAMES[self.detection_mode]}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)

        # Display FPS
        fps = self.fps_counter.update()
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

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

            cv2.putText(frame, status, (20, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        else:
            # No detection message with appropriate color
            if self.detection_mode == MODE_BALL_DETECTION:
                cv2.putText(frame, "No ball detected", (20, 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "No person detected", (20, 95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

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

    def run(self):
        """Main processing loop"""
        if not self.model_initialized or not self.camera or not self.camera.isOpened():
            print("Error: CV system not properly initialized")
            return False

        print("Starting CV processing loop with robot control...")
        # Check actual connection status before displaying
        connection_status = "Connected" if self.serial_comm.connected else "Auto-reconnect enabled"
        print(f"Serial connection: {connection_status}")

        try:
            while True:
                success, frame = self.process_frame()

                if not success:
                    print("Error processing frame")
                    break

                if self.show_preview and frame is not None:
                    cv2.imshow(self.window_name, frame)

                    key = cv2.waitKey(1) & 0xFF
                    if not self.handle_keyboard_input(key):
                        break
                else:
                    key = cv2.waitKey(10) & 0xFF
                    if not self.handle_keyboard_input(key):
                        break

        except KeyboardInterrupt:
            print("\nProcessing stopped by user (Ctrl+C)")
        except Exception as e:
            print(f"Error in processing loop: {e}")
        finally:
            # Send stop command before cleanup
            if self.serial_comm.connected:
                print("Sending stop command to robot...")
                for _ in range(3):  # Send multiple stop commands
                    self.serial_comm.send_command(ROBOT_STATE_STOP, 0, 0, 0)
                    time.sleep(0.1)
                self.serial_comm.disconnect()

            if self.camera:
                self.camera.release()

            if self.show_preview:
                cv2.destroyAllWindows()

            try:
                #yolo11_api.cleanup_model()
                gesture_api.cleanup_model()
                print("Model resources released")
            except Exception as e:
                print(f"Error cleaning up model: {e}")

        return True

    def get_detection_results(self):
        """Get the latest detection results

        Returns:
            dict: Detection results including best target info
        """
        results = {
            'objects': self.detected_objects,
            'best_target': None,
            'fps': self.fps_counter.fps,
            'mode': MODE_NAMES[self.detection_mode]
        }

        # Include best target details if available
        if self.best_target:
            cls_id, x, y, w, h, conf = self.best_target

            # Calculate center point of the target
            center_x = x + w / 2
            center_y = y + h / 2

            # Calculate error from center of frame (for external PID use)
            if self.camera and self.camera.isOpened():
                frame_width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
                frame_center_x = frame_width / 2
                error_x = center_x - frame_center_x
            else:
                error_x = 0

            results['best_target'] = {
                'y': y,
                'class_id': cls_id,
                'x': x,
                'width': w,
                'height': h,
                'confidence': conf,
                'center_x': center_x,
                'center_y': center_y,
                'error_x': error_x  # For external PID use
            }

        return results

    def cleanup(self):
        """Clean up resources"""
        if self.serial_comm.connected:
            print("Sending stop command to robot...")
            for _ in range(3):
                self.serial_comm.send_command(ROBOT_STATE_STOP, 0, 0, 0)
                time.sleep(0.1)
            self.serial_comm.disconnect()

        if self.camera:
            self.camera.release()

        if self.show_preview:
            cv2.destroyAllWindows()

        try:
            #yolo11_api.cleanup_model()
            gesture_api.cleanup_model()
            print("Model resources released")
        except Exception as e:
            print(f"Error cleaning up model: {e}")

    def control_robot(self):
        """Control robot based on current state and detection results"""
        current_time = time.time()

        # Check if it's time to send serial command (50Hz)
        if current_time - self.last_serial_time < self.serial_interval:
            return

        self.last_serial_time = current_time

        # Initialize variables
        left_speed = 0
        right_speed = 0

        # Determine what command to send based on robot state
        if self.robot_state == ROBOT_STATE_CHASE_BALL:
            # Only proceed with ball chasing if we have a detected ball
            if (self.best_target and
                    self.detection_mode == MODE_BALL_DETECTION and
                    self.best_target[0] == SPORTS_BALL_CLASS):

                # Get ball information
                cls_id, x, y, w, h, confidence = self.best_target

                # Get frame dimensions for normalization
                if hasattr(self, 'camera') and self.camera:
                    frame_width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
                    frame_height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
                else:
                    frame_width = self.frame_width
                    frame_height = self.frame_height

                if frame_width > 0 and frame_height > 0:
                    # Calculate ball center
                    ball_center_x = x + w / 2
                    ball_center_y = y + h / 2

                    # Normalize coordinates similar to main.py
                    normalized_x = (ball_center_x - frame_width / 2) / (frame_width / 2)  # -1 (left) to 1 (right)
                    normalized_y = ball_center_y / frame_height  # 0 (top) to 1 (bottom)

                    # Check if ball is well-centered (for boost mode)
                    x_error_threshold = 0.1  # Ball is within 10% of center horizontally
                    y_error_threshold = 0.1  # Ball is within 10% of target vertically
                    y_target_error = abs(normalized_y - self.y_pid.setpoint)

                    ball_centered = (abs(normalized_x) < x_error_threshold and
                                     y_target_error < y_error_threshold)

                    # Special "forward boost" mode when ball is centered
                    if ball_centered and not self.boost_active:
                        # Start boost mode - pure sprint towards ball
                        print("Ball centered! Activating forward boost sprint")
                        self.boost_start_time = current_time
                        self.boost_active = True

                    # Check if we're in boost mode and it hasn't expired
                    if self.boost_active:
                        boost_duration = 3.0  # 8 second forward boost sprint - extended for reliable reach
                        if current_time - self.boost_start_time < boost_duration:
                            # Apply forward boost: high speed sprint towards ball
                            boost_speed = 1500  # Even higher forward speed during boost
                            left_speed = boost_speed
                            right_speed = boost_speed
                            print(f"Forward boost sprint: {current_time - self.boost_start_time:.2f}s / {boost_duration}s")
                        else:
                            # Boost duration expired, return to normal PID control
                            self.boost_active = False
                            print(f"Forward boost sprint completed after {boost_duration}s - returning to PID control")

                    # Normal PID control when not in boost mode
                    if not self.boost_active:
                        steering = self.x_pid.compute(normalized_x)  # Output: -1000 to 1000
                        base_speed = self.y_pid.compute(normalized_y)  # Output: -2000 to 2000

                        # Combine components with scaling like main.py
                        scaled_left_speed = (base_speed - steering) * self.motor_output_scale
                        scaled_right_speed = (base_speed + steering) * self.motor_output_scale

                        left_speed = int(scaled_left_speed)
                        right_speed = int(scaled_right_speed)

                        # Limit motor speeds to max command values
                        left_speed = max(min(left_speed, self.max_motor_command), -self.max_motor_command)
                        right_speed = max(min(right_speed, self.max_motor_command), -self.max_motor_command)
            else:                # Handle ball lost during chase
                if self.boost_active:
                    # Don't cancel boost immediately - allow some time for ball detection recovery
                    # Ball might be temporarily lost due to motion blur or occlusion during sprint
                    boost_immunity_time = 3.0  # Allow 3 seconds of immunity before canceling boost
                    if current_time - self.boost_start_time > boost_immunity_time:
                        self.boost_active = False
                        print("Forward boost sprint canceled: ball lost for too long")
                    else:
                        print(f"Ball temporarily lost during boost - continuing sprint ({current_time - self.boost_start_time:.1f}s)")
                        # Continue with last boost speeds even if ball is temporarily lost
                        left_speed = 800
                        right_speed = 800

        elif self.robot_state == ROBOT_STATE_DELIVER_BALL:
            # State 3: DELIVER_BALL - Explicitly stop motors
            # This ensures wheels stop spinning when in ball delivery mode
            left_speed = 0
            right_speed = 0

        elif self.robot_state == ROBOT_STATE_RETURN_HOME:
            # State 2: RETURN_HOME - Stop motors (could be extended for homing behavior)
            left_speed = 0
            right_speed = 0

        elif self.robot_state == ROBOT_STATE_STOP:
            # State 0: STOP - Explicitly stop motors
            left_speed = 0
            right_speed = 0

        elif self.robot_state == ROBOT_STATE_SEARCH:
            # State 4: SEARCH - Manual controls + automatic person centering
            
            # Check if manual override is still active
            current_time = time.time()
            manual_override_active = (self.manual_override_time > 0 and 
                                    current_time - self.manual_override_time < self.manual_override_duration)
            
            # Check if automatic person centering should be active
            if (self.auto_person_centering and 
                self.detection_mode == MODE_PERSON_DETECTION and 
                self.best_target is not None and
                not manual_override_active):
                
                # PERSON CENTERING IMPLEMENTATION:
                # Uses the same normalized coordinate system and x_pid controller as ball tracking
                # for consistent behavior and tuning across detection modes. Person-specific
                # scaling (0.7x) provides gentler movement appropriate for human interaction.
                
                # Extract person information
                cls_id, x, y, w, h, conf = self.best_target
                
                if cls_id == PERSON_CLASS:
                    # Calculate person center coordinates
                    person_center_x = x + w / 2
                    person_center_y = y + h / 2
                    
                    # Get frame dimensions for normalization
                    if self.camera and self.camera.isOpened():
                        frame_width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
                        frame_height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    else:
                        frame_width = self.frame_width
                        frame_height = self.frame_height
                    
                    if frame_width > 0 and frame_height > 0:
                        # Normalize coordinates exactly like ball detection
                        normalized_x = (person_center_x - frame_width / 2) / (frame_width / 2)  # -1 (left) to 1 (right)
                        normalized_y = person_center_y / frame_height  # 0 (top) to 1 (bottom)
                        
                        # Define centering threshold in normalized coordinates (like ball detection)
                        x_error_threshold = 0.08  # Within 8% of center horizontally
                        
                        if abs(normalized_x) > x_error_threshold:
                            # Person is not centered - use differential steering to turn in place
                            # NO forward movement, only left/right turning at low speed
                            
                            # Calculate steering output using x_pid controller
                            steering = self.x_pid.compute(normalized_x)  # Output: -1000 to 1000
                            
                            # Scale down for gentle turning
                            turn_scale = 0.1  # Much smaller scale for slow turning
                            turn_speed = int(abs(steering) * turn_scale)
                            
                            # Ensure minimum turn speed but cap at max
                            min_turn_speed = 70   # Minimum speed to overcome friction
                            max_turn_speed = 100  # Maximum turning speed for safety
                            
                            if turn_speed < min_turn_speed:
                                turn_speed = min_turn_speed
                            elif turn_speed > max_turn_speed:
                                turn_speed = max_turn_speed
                            
                            # Differential steering: opposite motor directions to turn in place
                            if normalized_x > 0:
                                # Person is to the right, turn right: right motor faster, left motor slower
                                left_speed = turn_speed
                                right_speed = -turn_speed
                                direction = "right"
                            else:
                                # Person is to the left, turn left: left motor faster, right motor slower
                                left_speed = -turn_speed
                                right_speed = turn_speed
                                direction = "left"
                            
                            self.search_left_speed = left_speed
                            self.search_right_speed = right_speed
                            
                            print(f"Turning to face person: norm_x={normalized_x:.3f}, turning {direction}, speed={turn_speed}, L={left_speed}, R={right_speed}")
                        else:
                            # Person is centered horizontally, stop motors
                            self.search_left_speed = 0
                            self.search_right_speed = 0
                            print("Person centered - robot is facing person")
                else:
                    # No person detected or wrong target type, stop automatic control
                    self.search_left_speed = 0
                    self.search_right_speed = 0
                    print("Auto-centering: No person detected - stopping motors")
            elif manual_override_active:
                # Manual override is active, show remaining time
                remaining_time = self.manual_override_duration - (current_time - self.manual_override_time)
                print(f"Manual override active - auto centering resumes in {remaining_time:.1f}s")
            
            # If not using automatic centering, use manual search mode speeds
            left_speed = self.search_left_speed
            right_speed = self.search_right_speed

        # Always send command to robot (this will trigger auto-reconnect if needed)
        # Use the current robot state regardless of connection status
        # For SEARCH mode and DELIVER_BALL mode, send tilt angle; for other modes, use default 0
        if self.robot_state == ROBOT_STATE_SEARCH:
            self.serial_comm.send_command(self.robot_state, left_speed, right_speed, self.tilt_angle)
        elif self.robot_state == ROBOT_STATE_DELIVER_BALL:
            self.serial_comm.send_command(self.robot_state, left_speed, right_speed, self.tilt_angle)
        else:
            self.serial_comm.send_command(self.robot_state, left_speed, right_speed, 0)


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HelloBalls Computer Vision System with Robot Control')
    parser.add_argument('--no-preview', action='store_true', help='Disable preview window')
    parser.add_argument('--mode', type=int, default=0, choices=[0, 1],
                        help='Detection mode: 0=Ball Detection, 1=Person Detection')
    parser.add_argument('--serial-port', type=str, default='/dev/ttyS1',
                        help='Serial port for robot communication (default: /dev/ttyS1)')
    args = parser.parse_args()

    cv_system = HelloBallsCV(show_preview=not args.no_preview, detection_mode=args.mode, serial_port=args.serial_port)

    if cv_system.initialize():
        cv_system.run()
        cv_system.cleanup()
    else:
        print("Failed to initialize CV system")