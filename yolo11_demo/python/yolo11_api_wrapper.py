#!/usr/bin/env python3
"""
YOLOv11 API Wrapper for the C++ implementation.

This module provides a clean Python interface to the C++ YOLOv11 implementation
for object detection and tracking.
"""

import os
import sys
import numpy as np
import cv2
import time
import glob  # For searching files

# Add the current directory to the Python path to find yolo11_api.so
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Change working directory to match the C++ executable's expected location
# This is critical for the C++ code to find the model file
cpp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cpp")
os.chdir(cpp_dir)
print(f"Changed working directory to: {os.getcwd()}")

try:
    import yolo11_api
except ImportError as e:
    print(f"Error: Failed to import yolo11_api module: {e}")
    print("Make sure the C++ extension is properly compiled.")
    sys.exit(1)

# Constants that match those in config.h
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
CONFIDENCE_THRESHOLD = 0.3
SPORTS_BALL_CLASS = 32  # Sports ball class ID in COCO dataset
PERSON_CLASS = 0        # Person class ID in COCO dataset

# Detection modes
MODE_BALL_DETECTION = 0
MODE_PERSON_DETECTION = 1
MODE_NAMES = ["Ball Detection", "Person Detection"]

class FPSCounter:
    """Simple FPS counter class to measure performance"""
    def __init__(self):
        self.prev_time = time.time()
        self.frames = 0
        self.fps = 0
        
    def update(self):
        """Update and return the current FPS value"""
        self.frames += 1
        current_time = time.time()
        elapsed = current_time - self.prev_time
        
        if elapsed >= 1.0:
            self.fps = self.frames / elapsed
            self.frames = 0
            self.prev_time = current_time
            
        return self.fps

class YoloDetector:
    """Main class for YOLOv11 detection using the C++ backend"""
    
    def __init__(self):
        """Initialize the YOLOv11 detector"""
        self.model_initialized = False
        self.pid_controller = None
        
    def initialize(self):
        """Initialize the YOLO model"""
        try:
            # Find the model file (for informational purposes)
            model_path = self.find_model_file()
            if model_path:
                print(f"Found model at: {model_path}")
            
            # Initialize the model
            self.model_initialized = yolo11_api.initialize_model()
            if not self.model_initialized:
                print("Warning: Model initialization returned False")
            return self.model_initialized
        except Exception as e:
            print(f"Error initializing model: {e}")
            return False
            
    def find_model_file(self):
        """Find the YOLO model file using the same path as C++"""
        # Get base directory (parent of python directory)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_filename = "yolo11m_detect_bayese_640x640_nv12_modified.bin"
        
        # Check standard locations
        search_paths = [
            os.path.join(base_dir, "ptq_models", model_filename),
            os.path.join(base_dir, "..", "ptq_models", model_filename),
            os.path.join(os.path.dirname(base_dir), "ptq_models", model_filename),
            "/home/sunrise/Documents/HelloBalls-Host/yolo11_demo/ptq_models/" + model_filename
        ]
        
        # Try environment variable path first
        env_model_path = os.environ.get('YOLO11_MODEL_PATH')
        if env_model_path and os.path.exists(env_model_path):
            return env_model_path
            
        # Check all other paths
        for path in search_paths:
            if os.path.exists(path):
                return path
                
        return None

    def create_pid_controller(self, kp=0.05, ki=0.001, kd=0.01):
        """Create a PID controller with the specified parameters"""
        self.pid_controller = yolo11_api.PIDController(kp, ki, kd)
        return self.pid_controller
    
    def reset_pid_controller(self):
        """Reset the internal state of the PID controller"""
        if self.pid_controller:
            self.pid_controller.reset()
        else:
            self.pid_controller = self.create_pid_controller()
    
    def preprocess_image(self, frame, input_width=INPUT_WIDTH, input_height=INPUT_HEIGHT):
        """
        Preprocess an image using letterbox method to maintain aspect ratio
        
        Args:
            frame (numpy.ndarray): Input BGR image
            input_width (int): Target width for the model
            input_height (int): Target height for the model
            
        Returns:
            tuple: (processed_image, x_scale, y_scale, x_shift, y_shift)
        """
        # Calculate scale to maintain aspect ratio
        x_scale = min(input_height / frame.shape[0], input_width / frame.shape[1])
        y_scale = x_scale
        
        # Calculate new dimensions
        new_w = int(frame.shape[1] * x_scale)
        x_shift = int((input_width - new_w) / 2)
        
        new_h = int(frame.shape[0] * y_scale)
        y_shift = int((input_height - new_h) / 2)
        
        # Resize the image while maintaining aspect ratio
        resized = cv2.resize(frame, (new_w, new_h))
        
        # Create a canvas with gray background (127,127,127)
        canvas = np.ones((input_height, input_width, 3), dtype=np.uint8) * 127
        
        # Paste the resized image onto the canvas
        canvas[y_shift:y_shift+new_h, x_shift:x_shift+new_w] = resized
        
        return canvas, x_scale, y_scale, x_shift, y_shift
    
    def detect(self, frame):
        """
        Perform object detection on an image frame
        
        Args:
            frame (numpy.ndarray): Input BGR image
            
        Returns:
            dict: Detection results with keys:
                'bboxes': list of (x, y, width, height) tuples
                'scores': list of confidence scores
                'class_ids': list of class IDs
                'scale_info': tuple of (x_scale, y_scale, x_shift, y_shift)
        """
        if not self.model_initialized:
            if not self.initialize():
                return None
        
        # Preprocess the frame
        preprocessed_frame, x_scale, y_scale, x_shift, y_shift = self.preprocess_image(frame)
        
        try:
            # Run inference
            detection_results = yolo11_api.inference(preprocessed_frame)
            
            # Convert from C++ output struct to dictionary format
            results = {
                'bboxes': detection_results.bboxes,
                'scores': detection_results.scores,
                'class_ids': detection_results.class_ids,
                'scale_info': (x_scale, y_scale, x_shift, y_shift)
            }
            return results
        except Exception as e:
            print(f"Error during detection: {e}")
            return None
    
    def process_ball_detection(self, frame, detection, pid_controller=None):
        """
        Process ball detection and calculate movement values
        
        Args:
            frame (numpy.ndarray): Input frame
            detection (tuple): (x, y, w, h, score, class_id) detection data
            pid_controller (PIDController): PID controller instance
            
        Returns:
            dict: Processing results with control values and visualization info
        """
        if not pid_controller:
            if not self.pid_controller:
                self.create_pid_controller()
            pid_controller = self.pid_controller
            
        x, y, w, h, conf, cls_id = detection
        
        # Frame dimensions
        frame_width = frame.shape[1]
        frame_height = frame.shape[0]
        
        # Calculate ball center
        ball_center_x = x + w / 2
        ball_center_y = y + h / 2
        
        # Calculate target position (bottom center of frame)
        target_x = frame_width / 2
        target_y = frame_height * 0.9  # 90% down the frame
        
        # Calculate error (distance from target position)
        x_error = ball_center_x - target_x
        
        # Calculate control output using PID controller
        steering = pid_controller.calculate(x_error)
        
        # Convert steering to motor speeds
        base_speed = 50  # Base forward speed for ball chasing
        if steering > 0:
            # Ball is to the right, turn right
            left_speed = base_speed + abs(steering)
            right_speed = base_speed - abs(steering)
        else:
            # Ball is to the left, turn left
            left_speed = base_speed - abs(steering)
            right_speed = base_speed + abs(steering)
        
        # Ensure speeds are within bounds (-100 to 100)
        left_speed = max(min(left_speed, 100), -100)
        right_speed = max(min(right_speed, 100), -100)
        
        # Return control values and visualization points
        return {
            'left_speed': left_speed,
            'right_speed': right_speed,
            'error': x_error,
            'steering': steering,
            'ball_center': (ball_center_x, ball_center_y),
            'target_point': (target_x, target_y),
            'confidence': conf,
            'message': f"Ball tracking: {int(conf * 100)}% confident"
        }
    
    def process_person_detection(self, frame, detection, pid_controller=None):
        """
        Process person detection and calculate movement values
        
        Args:
            frame (numpy.ndarray): Input frame
            detection (tuple): (x, y, w, h, score, class_id) detection data
            pid_controller (PIDController): PID controller instance
            
        Returns:
            dict: Processing results with control values and visualization info
        """
        if not pid_controller:
            if not self.pid_controller:
                self.create_pid_controller()
            pid_controller = self.pid_controller
            
        x, y, w, h, conf, cls_id = detection
        
        # Frame dimensions
        frame_width = frame.shape[1]
        frame_height = frame.shape[0]
        
        # Calculate person center
        person_center_x = x + w / 2
        person_center_y = y + h / 2
        
        # Calculate frame center
        frame_center_x = frame_width / 2
        
        # Calculate error (distance from center)
        x_error = person_center_x - frame_center_x
        
        # Calculate control output using PID controller
        steering = pid_controller.calculate(x_error)
        
        # Convert steering to motor speeds - slower for person following
        base_speed = 30  # Lower base speed for safety
        if steering > 0:
            # Person is to the right, turn right
            left_speed = base_speed + abs(steering)
            right_speed = base_speed - abs(steering)
        else:
            # Person is to the left, turn left
            left_speed = base_speed - abs(steering)
            right_speed = base_speed + abs(steering)
        
        # Ensure speeds are within bounds (-100 to 100)
        left_speed = max(min(left_speed, 100), -100)
        right_speed = max(min(right_speed, 100), -100)
        
        # Return control values and visualization points
        return {
            'left_speed': left_speed,
            'right_speed': right_speed,
            'error': x_error,
            'steering': steering,
            'person_center': (person_center_x, person_center_y),
            'frame_center': (frame_center_x, frame_height/2),
            'confidence': conf,
            'message': f"Person tracking: {int(conf * 100)}% confident"
        }
    
    def toggle_camera_resolution(self, cap, is720p):
        """
        Toggle camera resolution between 1280x720 and 1280x712
        
        Args:
            cap: OpenCV video capture object
            is720p (bool): Current state (True if 720p, False if 712p)
            
        Returns:
            bool: New state
        """
        # Set properties that affect switching delay
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if is720p:
            # Switch to 712p
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 712)
            
            # Flush the buffer by grabbing a few frames
            for _ in range(2):
                cap.grab()
                
            print("Resolution changed to 1280x712")
        else:
            # Switch to 720p
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            # Flush the buffer by grabbing a few frames
            for _ in range(2):
                cap.grab()
                
            print("Resolution changed to 1280x720")
        
        # Get actual resolution after change
        actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"Actual resolution: {actual_width}x{actual_height}")
        
        return not is720p
    
    def cleanup(self):
        """Clean up resources used by the detector"""
        if self.model_initialized:
            try:
                yolo11_api.cleanup_model()
                self.model_initialized = False
                return True
            except Exception as e:
                print(f"Error cleaning up model: {e}")
                return False
        return True
        
# Visualization functions

def draw_detection_visualization(frame, detection_type, detection_info):
    """
    Draw visualization elements for detection on the frame
    
    Args:
        frame (numpy.ndarray): Frame to draw on
        detection_type (str): 'ball' or 'person'
        detection_info (dict): Detection info from process_ball/person_detection
        
    Returns:
        numpy.ndarray: Frame with visualization
    """
    if detection_type == 'ball':
        # Draw ball detection visualization
        ball_center = detection_info['ball_center']
        target_point = detection_info['target_point']
        
        # Draw ball center point
        cv2.circle(frame, (int(ball_center[0]), int(ball_center[1])), 5, (0, 255, 255), -1)
        
        # Draw target position
        tx, ty = target_point
        cv2.circle(frame, (int(tx), int(ty)), 10, (255, 255, 0), 2)
        cv2.line(frame, (int(tx - 15), int(ty)), (int(tx + 15), int(ty)), (255, 255, 0), 2)
        cv2.line(frame, (int(tx), int(ty - 15)), (int(tx), int(ty + 15)), (255, 255, 0), 2)
        
        # Draw line from ball to target
        cv2.line(frame, (int(ball_center[0]), int(ball_center[1])), 
                (int(target_point[0]), int(target_point[1])), (0, 255, 255), 2)
        
    elif detection_type == 'person':
        # Draw person detection visualization
        person_center = detection_info['person_center']
        frame_center = detection_info['frame_center']
        
        # Draw person center point
        cv2.circle(frame, (int(person_center[0]), int(person_center[1])), 5, (255, 150, 0), -1)
        
        # Draw vertical line at frame center
        fx = frame_center[0]
        cv2.line(frame, (int(fx), 0), (int(fx), frame.shape[0]), (0, 150, 255), 1, cv2.LINE_AA)
        
        # Draw line from person to center line
        cv2.line(frame, (int(person_center[0]), int(person_center[1])), 
                (int(fx), int(person_center[1])), (0, 255, 255), 2)
    
    # Draw info panel
    draw_info_panel(frame, detection_info)
    
    return frame

def draw_info_panel(frame, info):
    """Draw information panel in the corner of the frame"""
    # Semi-transparent black background for info panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (320, 140), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Display message
    cv2.putText(frame, info['message'], (20, 35), 
              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Display motor speeds
    cv2.putText(frame, f"Motors L:{int(info['left_speed'])} R:{int(info['right_speed'])}", 
              (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Display error and PID info
    cv2.putText(frame, f"Error: {int(info['error'])} PID: {int(info['steering'])}", 
              (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return frame

def draw_fps(frame, fps):
    """Draw FPS counter on the frame"""
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 125), 
              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame