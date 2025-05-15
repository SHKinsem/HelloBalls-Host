#!/usr/bin/env python3
"""
Demo application for YOLOv11 object detection using the yolo11_api_wrapper module.
This script demonstrates real-time ball and person detection using a webcam.

Features:
- Ball tracking mode: Tracks and follows tennis balls
- Person tracking mode: Follows people
- PID controller for motor speed calculation
- On-screen visualization of detection and tracking data

Usage:
    python3 demo_yolo11.py

Keyboard shortcuts:
    'q' - Quit the application
    'r' - Toggle camera resolution (1280x720 ⟷ 1280x712)
    'm' - Switch between ball detection and person detection modes
    'f' - Toggle fullscreen mode
"""

import os
import sys
import cv2
import time
import numpy as np
import subprocess

# Import the API wrapper
from yolo11_api_wrapper import (
    YoloDetector, FPSCounter,
    draw_detection_visualization, draw_fps,
    SPORTS_BALL_CLASS, PERSON_CLASS,
    MODE_BALL_DETECTION, MODE_PERSON_DETECTION, MODE_NAMES
)

def position_window(window_name, width, height, screen_w=None, screen_h=None):
    """Position window in the center of the screen with the given dimensions"""
    if screen_w is None or screen_h is None:
        try:
            # Try to get screen resolution using xrandr (Linux)
            output = subprocess.check_output('xrandr | grep "\\*" | cut -d" " -f4', shell=True).decode('utf-8').strip()
            screen_w, screen_h = map(int, output.split('x'))
        except Exception:
            # Fallback to a common resolution
            screen_w, screen_h = 1920, 1080
    
    # Calculate window position
    win_x = (screen_w - width) // 2
    win_y = (screen_h - height) // 2
    
    cv2.resizeWindow(window_name, width, height)
    cv2.moveWindow(window_name, win_x, win_y)
    
    return win_x, win_y, screen_w, screen_h

def setup_camera(camera_id=0, width=1280, height=720):
    """Initialize the camera with the given resolution"""
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return None
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    # Get actual resolution
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Camera opened with resolution: {actual_width}x{actual_height}")
    
    return cap

def find_best_detection(detections, detection_mode):
    """Find the best detection based on the current mode"""
    if not detections:
        return None
    
    target_class = SPORTS_BALL_CLASS if detection_mode == MODE_BALL_DETECTION else PERSON_CLASS
    
    # Find the best detection with highest confidence for the target class
    best_detection = None
    highest_confidence = 0
    
    for i, (bbox, score, class_id) in enumerate(zip(
            detections['bboxes'], detections['scores'], detections['class_ids'])):
        
        if class_id == target_class and score > highest_confidence:
            highest_confidence = score
            best_detection = (*bbox, score, class_id)
    
    return best_detection

def main():
    """Main entry point for the demo application"""
    print("YOLOv11 Object Detection Demo")
    print("-" * 40)
    
    # Change working directory to match the C++ executable's expected location
    # This is critical for the C++ code to find the model file
    cpp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cpp")
    os.chdir(cpp_dir)
    print(f"Changed working directory to: {os.getcwd()}")
    
    # Create detector instance
    detector = YoloDetector()
    
    # Initialize the model
    if not detector.initialize():
        print("Failed to initialize YOLO model. Exiting.")
        return 1
    
    # Create PID controller
    pid_controller = detector.create_pid_controller()
    
    # Create a named window
    window_name = "YOLOv11 Object Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Setup camera
    cap = setup_camera()
    if cap is None:
        return 1
    
    # Get actual camera resolution for window sizing
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    # Calculate window size (80% of screen size)
    win_x, win_y, screen_w, screen_h = position_window(
        window_name, 
        int(actual_width), 
        int(actual_height)
    )
    
    # Create FPS counter
    fps_counter = FPSCounter()
    
    # Detection mode state
    detection_mode = MODE_BALL_DETECTION
    
    # Camera resolution state
    is720p = True
    
    # Fullscreen state
    is_fullscreen = False
    
    print(f"Window positioned at ({win_x}, {win_y}) with size {actual_width}x{actual_height}")
    print("Keyboard shortcuts:")
    print("  'q' - Quit the application")
    print("  'r' - Toggle camera resolution")
    print("  'm' - Switch detection mode")
    print("  'f' - Toggle fullscreen")
    print(f"Current detection mode: {MODE_NAMES[detection_mode]}")
    
    try:
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Error: Failed to capture frame")
                break
            
            # Run detection
            detections = detector.detect(frame)
            
            # Find the best detection for current mode
            best_detection = find_best_detection(detections, detection_mode)
            
            # Process detection based on mode
            detection_info = None
            if best_detection:
                x, y, w, h, conf, cls_id = best_detection
                
                # Apply scale correction from letterbox preprocessing
                x_scale, y_scale, x_shift, y_shift = detections['scale_info']
                x = (x - x_shift) / x_scale
                y = (y - y_shift) / y_scale
                w = w / x_scale
                h = h / y_scale
                
                # Process the detection based on the mode
                if detection_mode == MODE_BALL_DETECTION and cls_id == SPORTS_BALL_CLASS:
                    detection_info = detector.process_ball_detection(
                        frame, (x, y, w, h, conf, cls_id), pid_controller
                    )
                    draw_detection_visualization(frame, 'ball', detection_info)
                    
                elif detection_mode == MODE_PERSON_DETECTION and cls_id == PERSON_CLASS:
                    detection_info = detector.process_person_detection(
                        frame, (x, y, w, h, conf, cls_id), pid_controller
                    )
                    draw_detection_visualization(frame, 'person', detection_info)
                    
                # Draw bounding box
                color = (0, 0, 255) if cls_id == SPORTS_BALL_CLASS else (0, 255, 0)
                cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
            else:
                # When no detection, draw an empty info panel
                overlay = frame.copy()
                cv2.rectangle(overlay, (10, 10), (320, 140), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                
                mode_text = f"Mode: {MODE_NAMES[detection_mode]}"
                cv2.putText(frame, mode_text, (20, 35), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                status_text = f"No {MODE_NAMES[detection_mode].lower()[:-11]} detected"
                cv2.putText(frame, status_text, (20, 65), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Calculate and display FPS
            fps = fps_counter.update()
            draw_fps(frame, fps)
            
            # Show the frame
            cv2.imshow(window_name, frame)
            
            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                is720p = detector.toggle_camera_resolution(cap, is720p)
            elif key == ord('f'):
                # Toggle fullscreen mode
                is_fullscreen = not is_fullscreen
                if is_fullscreen:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    print("Switched to fullscreen mode")
                else:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(window_name, int(actual_width), int(actual_height))
                    cv2.moveWindow(window_name, win_x, win_y)
                    print("Exited fullscreen mode")
            elif key == ord('m'):
                # Switch detection mode
                detection_mode = (detection_mode + 1) % len(MODE_NAMES)
                print(f"Switched to {MODE_NAMES[detection_mode]} mode")
                # Reset PID controller when switching modes
                detector.reset_pid_controller()
                
    except KeyboardInterrupt:
        print("\nDetection stopped by user")
    except Exception as e:
        print(f"Error in detection loop: {e}")
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        detector.cleanup()
        print("Resources released. Demo completed.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())