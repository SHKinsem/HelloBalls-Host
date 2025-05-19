#!/usr/bin/env python3
# HelloBalls-Host main.py
# Main controller program for HelloBalls robot with multithreading support

import os
import sys
import time
import threading
import queue
import numpy as np
import cv2  # Added explicit import
from datetime import datetime
import argparse
import signal
import glob

# Add script directory to path to find modules
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, "scripts"))

# Import our custom modules
try:
    from scripts.HelloBalls_CV import HelloBallsCV
    from scripts.HelloBalls_Serial import SerialComm
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure you are running this script from the HelloBalls-Host directory")
    sys.exit(1)

# Global variables for inter-thread communication
# These will be protected by locks when accessed
class SharedState:
    def __init__(self):
        # CV thread outputs
        self.ball_detected = False
        self.ball_x = 0          # x position of ball in image coordinates
        self.ball_y = 0          # y position of ball in image coordinates
        self.ball_width = 0      # width of ball in pixels
        self.ball_height = 0     # height of ball in pixels
        self.ball_confidence = 0 # detection confidence score
        self.frame_width = 0     # frame width for normalizing coordinates
        self.frame_height = 0    # frame height for normalizing coordinates
        self.current_frame = None # The most recent processed frame
        self.new_frame_available = False # Flag to indicate a new frame is ready
        self.camera_id = 0       # Current camera ID
        self.switch_camera_requested = False  # Flag to request camera switch
        self.toggle_resolution_requested = False  # Flag to request resolution toggle
        self.is720p = True       # Current resolution state
        
        # PID thread outputs
        self.left_motor_speed = 0
        self.right_motor_speed = 0
        self.robot_state = 0     # 0 = stop, 1 = chase ball, 2 = return home
        
        # Status flags
        self.cv_running = False
        self.pid_running = False
        self.serial_running = False
        
        # Thread control flags
        self.shutdown_requested = False
        
        # Create mutex locks for thread safety
        self.cv_lock = threading.Lock()
        self.pid_lock = threading.Lock()
        self.serial_lock = threading.Lock()
        self.frame_lock = threading.Lock()
        
        # Statistics and diagnostics
        self.cv_fps = 0
        self.pid_iterations_per_second = 0
        self.serial_messages_per_second = 0
        self.last_stats_time = time.time()
        self.stats_count = {
            'cv': 0,
            'pid': 0,
            'serial': 0
        }

# Create global shared state
shared_state = SharedState()

# Create queue for logging messages across threads
log_queue = queue.Queue()

def log_message(level, message):
    """Thread-safe logging function"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    log_queue.put(f"[{timestamp}] [{level}] {message}")
    
    # Also print to console immediately
    print(f"[{timestamp}] [{level}] {message}")

def update_statistics(thread_name):
    """Update performance statistics for a given thread"""
    with threading.Lock():
        shared_state.stats_count[thread_name] += 1
        
        current_time = time.time()
        elapsed = current_time - shared_state.last_stats_time
        
        if elapsed >= 1.0:
            # Calculate rates per second
            shared_state.cv_fps = shared_state.stats_count['cv'] / elapsed
            shared_state.pid_iterations_per_second = shared_state.stats_count['pid'] / elapsed
            shared_state.serial_messages_per_second = shared_state.stats_count['serial'] / elapsed
            
            # Reset counters
            shared_state.stats_count = {
                'cv': 0,
                'pid': 0,
                'serial': 0
            }
            shared_state.last_stats_time = current_time

def find_available_cameras():
    """Find all available cameras and return their indices"""
    available_cameras = []
    
    # Try common camera indices (0-9)
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                # This is a working camera
                resolution = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                             int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                available_cameras.append((i, resolution))
            cap.release()
    
    # If no camera found with standard indices, try device paths (Linux)
    if os.path.exists("/dev/"):
        video_devices = glob.glob("/dev/video*")
        for device in video_devices:
            try:
                # Extract number from device path
                device_num = int(device.replace("/dev/video", ""))
                # Skip if we already found this camera
                if any(cam[0] == device_num for cam in available_cameras):
                    continue
                    
                cap = cv2.VideoCapture(device_num)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        resolution = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                                     int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                        available_cameras.append((device_num, resolution))
                cap.release()
            except Exception as e:
                print(f"Error checking {device}: {e}")
    
    return available_cameras

def toggle_resolution(camera):
    """Toggle camera resolution between 720p and 712p
    
    Args:
        camera: OpenCV camera object
        
    Returns:
        bool: New resolution state (True for 720p, False for 712p)
    """
    is720p = shared_state.is720p
    
    if not camera or not camera.isOpened():
        return is720p
        
    # Set properties that affect switching delay
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if is720p:
        # Switch to 712p
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 712)
        
        # Flush the buffer
        for _ in range(2):
            camera.grab()
            
        log_message("INFO", "Resolution changed to 1280x712")
    else:
        # Switch to 720p
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Flush the buffer
        for _ in range(2):
            camera.grab()
            
        log_message("INFO", "Resolution changed to 1280x720")
    
    # Get actual resolution
    actual_width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
    log_message("INFO", f"Actual resolution: {actual_width}x{actual_height}")
    
    # Return the new state
    return not is720p

class PIDController:
    """Custom PID controller for robot movement"""
    def __init__(self, kp, ki, kd, setpoint=0, output_limits=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.output_limits = output_limits
        
        self.prev_error = 0
        self.integral = 0
        self.last_time = time.time()
        
    def calculate(self, process_variable):
        """Calculate PID output based on measured process variable"""
        current_time = time.time()
        dt = current_time - self.last_time
        
        # Avoid division by zero or very small dt
        if dt < 0.001:
            dt = 0.001
            
        # Calculate error
        error = self.setpoint - process_variable
        
        # Proportional term
        p_out = self.kp * error
        
        # Integral term
        self.integral += error * dt
        i_out = self.ki * self.integral
        
        # Derivative term (using error, not measurement to avoid derivative kick)
        d_out = self.kd * (error - self.prev_error) / dt
        
        # Calculate total output
        output = p_out + i_out + d_out
        
        # Apply output limits if specified
        if self.output_limits is not None:
            output = max(min(output, self.output_limits[1]), self.output_limits[0])
        
        # Store current error for next iteration
        self.prev_error = error
        self.last_time = current_time
        
        return output
    
    def reset(self):
        """Reset the controller's internal state"""
        self.prev_error = 0
        self.integral = 0

def cv_thread(show_preview=True):
    """Computer Vision processing thread"""
    log_message("INFO", "Starting CV thread")
    
    # Initialize CV system with preview turned off - we'll handle display in a separate UI thread
    cv_system = None
    camera_id = shared_state.camera_id
    
    # Create and initialize the CV system
    try:
        cv_system = HelloBallsCV(show_preview=False)  # Always turn off internal preview
        if not cv_system.initialize():
            log_message("ERROR", "Failed to initialize CV system")
            return
    except Exception as e:
        log_message("ERROR", f"Exception initializing CV system: {e}")
        return
    
    # Update shared state that CV thread is running
    with shared_state.cv_lock:
        shared_state.cv_running = True
        shared_state.frame_width = cv_system.frame_width
        shared_state.frame_height = cv_system.frame_height
    
    try:
        # Main CV processing loop
        while not shared_state.shutdown_requested:
            # Check if camera switch was requested
            with shared_state.cv_lock:
                switch_requested = shared_state.switch_camera_requested
                toggle_res_requested = shared_state.toggle_resolution_requested
                
                if switch_requested:
                    # Reset the flag
                    shared_state.switch_camera_requested = False
                    camera_id = shared_state.camera_id
                
                if toggle_res_requested:
                    # Reset the flag
                    shared_state.toggle_resolution_requested = False
            
            # Handle camera switch request
            if switch_requested:
                log_message("INFO", f"Switching to camera {camera_id}")
                # Clean up current CV system
                if cv_system:
                    cv_system.cleanup()
                
                # Create and initialize a new CV system with the new camera
                try:
                    cv_system = HelloBallsCV(show_preview=False, camera_id=camera_id)
                    if not cv_system.initialize():
                        log_message("ERROR", f"Failed to initialize CV system with camera {camera_id}")
                        # Try to fallback to previous camera
                        with shared_state.cv_lock:
                            shared_state.camera_id = 0  # Fallback to default camera
                            
                        cv_system = HelloBallsCV(show_preview=False)
                        if not cv_system.initialize():
                            log_message("ERROR", "Failed to initialize CV system with fallback camera")
                            return
                
                    # Update frame dimensions in shared state
                    with shared_state.cv_lock:
                        shared_state.frame_width = cv_system.frame_width
                        shared_state.frame_height = cv_system.frame_height
                        
                    log_message("INFO", f"Camera {camera_id} initialized successfully with resolution {cv_system.frame_width}x{cv_system.frame_height}")
                except Exception as e:
                    log_message("ERROR", f"Exception switching cameras: {e}")
                    # Try to fallback to default camera
                    try:
                        cv_system = HelloBallsCV(show_preview=False)
                        if not cv_system.initialize():
                            log_message("ERROR", "Failed to initialize CV system with fallback camera")
                            return
                    except Exception as e:
                        log_message("ERROR", f"Failed to initialize fallback camera: {e}")
                        return
            
            # Handle resolution toggle request
            if toggle_res_requested:
                log_message("INFO", "Toggling camera resolution")
                if cv_system and cv_system.camera and cv_system.camera.isOpened():
                    # Toggle resolution using the function from HelloBalls_CV
                    new_is720p = toggle_resolution(cv_system.camera)
                    with shared_state.cv_lock:
                        shared_state.is720p = new_is720p
                        shared_state.frame_width = cv_system.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
                        shared_state.frame_height = cv_system.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
            
            # Process a single frame
            success, frame = cv_system.process_frame()
            
            if success and frame is not None:
                # Get detection results
                results = cv_system.get_detection_results()
                
                # Update shared state with detection results
                with shared_state.cv_lock:
                    if results['best_target'] is not None:
                        target = results['best_target']
                        shared_state.ball_detected = True
                        shared_state.ball_x = target['center_x']
                        shared_state.ball_y = target['center_y']
                        shared_state.ball_width = target['width']
                        shared_state.ball_height = target['height']
                        shared_state.ball_confidence = target['confidence']
                    else:
                        shared_state.ball_detected = False
                    
                    # Update FPS
                    shared_state.cv_fps = results['fps']
                
                # Update shared frame for UI thread - using a separate lock for frame access
                with shared_state.frame_lock:
                    shared_state.current_frame = frame.copy()
                    shared_state.new_frame_available = True
                    
                # Update statistics
                update_statistics('cv')
            else:
                log_message("WARNING", "Failed to process CV frame")
            
            # Sleep for a tiny amount to prevent CPU overload
            # The CV system has its own internal timing
            time.sleep(0.001)
            
    except Exception as e:
        log_message("ERROR", f"Exception in CV thread: {e}")
    finally:
        # Clean up resources
        if cv_system:
            cv_system.cleanup()
        
        # Update shared state
        with shared_state.cv_lock:
            shared_state.cv_running = False
        
        log_message("INFO", "CV thread stopped")

def ui_thread(window_name="HelloBalls Preview"):
    """Dedicated UI thread to handle preview window and UI events"""
    log_message("INFO", "Starting UI thread for preview window")
    
    # Display available cameras
    available_cameras = find_available_cameras()
    if available_cameras:
        log_message("INFO", f"Available cameras: {len(available_cameras)}")
        for i, (cam_id, res) in enumerate(available_cameras):
            log_message("INFO", f"  [{i}] Camera {cam_id}: {res[0]}x{res[1]}")
    else:
        log_message("WARNING", "No cameras found")
    
    # Create window for preview
    if not shared_state.shutdown_requested:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Set initial window size
        try:
            # Try to get screen resolution using xrandr (Linux)
            try:
                import subprocess
                output = subprocess.check_output('xrandr | grep "\*" | cut -d" " -f4', shell=True).decode('utf-8').strip()
                screen_w, screen_h = map(int, output.split('x'))
            except:
                # Fallback to a common resolution
                screen_w, screen_h = 1920, 1080
                
            # Calculate window size (80% of screen size)
            window_w = int(screen_w * 0.8)
            window_h = int(screen_h * 0.8)
                
            cv2.resizeWindow(window_name, window_w, window_h)
            
            # Position window in center of screen
            win_x = (screen_w - window_w) // 2
            win_y = (screen_h - window_h) // 2
            cv2.moveWindow(window_name, win_x, win_y)
        except:
            # Fallback size
            cv2.resizeWindow(window_name, 1024, 768)
            
        log_message("INFO", "Preview window created")
        log_message("INFO", "Press 'q' to quit, 's' to switch camera, 'r' to toggle resolution, 'f' to toggle fullscreen")
    
    # Track fullscreen state
    is_fullscreen = False
    
    try:
        # Main UI loop
        while not shared_state.shutdown_requested:
            # Check if we have a new frame to show
            frame_to_show = None
            
            with shared_state.frame_lock:
                if shared_state.new_frame_available and shared_state.current_frame is not None:
                    frame_to_show = shared_state.current_frame.copy()
                    shared_state.new_frame_available = False
            
            # If we have a valid frame, show it
            if frame_to_show is not None:
                # Add any additional UI elements here if needed
                
                # Add info about camera switching and resolution
                resolution_text = "1280x720" if shared_state.is720p else "1280x712"
                cv2.putText(frame_to_show, f"Resolution: {resolution_text}", 
                          (10, frame_to_show.shape[0] - 40), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Add info about camera switching
                cv2.putText(frame_to_show, "Press 's' to switch camera, 'r' to toggle resolution", 
                          (10, frame_to_show.shape[0] - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Show the frame
                cv2.imshow(window_name, frame_to_show)
            
            # Handle UI events - this is crucial for preventing window freezing
            # waitKey processes events and allows the window to update
            key = cv2.waitKey(1) & 0xFF
            
            # Check for key presses
            if key == ord('q'):
                log_message("INFO", "Quit requested from UI")
                shared_state.shutdown_requested = True
                break
            elif key == ord('s'):
                # Switch camera
                with shared_state.cv_lock:
                    available_cameras = find_available_cameras()
                    if available_cameras:
                        # Find index of current camera in available list
                        current_idx = -1
                        for i, (cam_id, _) in enumerate(available_cameras):
                            if cam_id == shared_state.camera_id:
                                current_idx = i
                                break
                        
                        # Select next camera
                        next_idx = (current_idx + 1) % len(available_cameras)
                        next_cam_id = available_cameras[next_idx][0]
                        
                        # Request camera switch
                        shared_state.camera_id = next_cam_id
                        shared_state.switch_camera_requested = True
                        log_message("INFO", f"Camera switch requested to camera {next_cam_id}")
                    else:
                        log_message("WARNING", "No cameras available to switch to")
            elif key == ord('r'):
                # Toggle resolution
                with shared_state.cv_lock:
                    shared_state.toggle_resolution_requested = True
                    log_message("INFO", "Resolution toggle requested")
            elif key == ord('f'):
                # Toggle fullscreen
                is_fullscreen = not is_fullscreen
                if is_fullscreen:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            
            # Avoid unnecessary CPU usage with a small sleep
            time.sleep(0.01)
            
    except Exception as e:
        log_message("ERROR", f"Exception in UI thread: {e}")
    finally:
        # Clean up resources
        cv2.destroyAllWindows()
        log_message("INFO", "UI thread stopped")

def pid_thread(control_rate=50):
    """PID control thread - runs at fixed frequency"""
    log_message("INFO", "Starting PID control thread")

    MOTOR_OUTPUT_SCALE = 10.0
    MAX_MOTOR_COMMAND = 1500  # Max absolute value for a single motor command
    
    # Initialize PID controllers
    # X controller (horizontal position) - for steering
    x_pid = PIDController(
        kp=40,            # Proportional gain (reverted to original, for pre-scaled output)
        ki=0.05,           # Integral gain (reverted to original, for pre-scaled output)
        kd=10.0,            # Derivative gain (reverted to original, for pre-scaled output)
        setpoint=0,         # Target is center of frame (will be converted to normalized coordinates)
        output_limits=(-50, 50) # Pre-scale steering output (-50*10 = -500 to 50*10 = 500)
    )
    
    # Y controller (vertical position) - for speed
    y_pid = PIDController(
        kp=200,            # Proportional gain (reverted to original, for pre-scaled output)
        ki=0.05,           # Integral gain (reverted to original, for pre-scaled output)
        kd=0.1,            # Derivative gain (reverted to original, for pre-scaled output)
        setpoint=0.75,      # Target position is 75% down the frame (normalized 0-1)
        output_limits=(0, 100) # Pre-scale speed output (0*10 = 0 to 100*10 = 1000)
    )
    
    # Calculate control period
    control_period = 1.0 / control_rate
    
    # Update shared state that PID thread is running
    with shared_state.pid_lock:
        shared_state.pid_running = True
    
    try:
        last_ball_detected = False
        consecutive_no_ball_frames = 0
        max_no_ball_frames = control_rate * 2  # 2 seconds without detection
        
        # Main PID control loop
        while not shared_state.shutdown_requested:
            start_time = time.time()
            
            # Get current ball position from shared state
            with shared_state.cv_lock:
                ball_detected = shared_state.ball_detected
                ball_x = shared_state.ball_x
                ball_y = shared_state.ball_y
                frame_width = shared_state.frame_width
                frame_height = shared_state.frame_height
            
            # Add safety check for frame dimensions
            if frame_width == 0 or frame_height == 0:
                log_message("WARNING", f"Frame dimensions are zero (w:{frame_width}, h:{frame_height}) in pid_thread. Setting motors to 0.")
                left_speed = 0
                right_speed = 0
                robot_state = 0
                with shared_state.pid_lock:
                    shared_state.left_motor_speed = left_speed
                    shared_state.right_motor_speed = right_speed
                    shared_state.robot_state = robot_state
                
                update_statistics('pid') # Still update stats for the iteration
                elapsed = time.time() - start_time
                sleep_time = max(0, control_period - elapsed)
                time.sleep(sleep_time)
                continue # Skip the rest of this loop iteration

            # Detect transitions from ball detected to lost
            if not ball_detected and last_ball_detected:
                log_message("INFO", "Ball lost")
                # Reset boost state when ball is lost
                if hasattr(pid_thread, "boost_active") and pid_thread.boost_active:
                    pid_thread.boost_active = False
                    log_message("INFO", "Forward boost canceled: ball lost")
                
            # Detect transitions from ball lost to detected
            if ball_detected and not last_ball_detected:
                log_message("INFO", "Ball detected")
                # Reset PID controller states on new detection
                x_pid.reset()
                y_pid.reset()
            
            last_ball_detected = ball_detected
            
            # Calculate motor speeds based on PID outputs if ball is detected
            if ball_detected:
                # Reset consecutive frame counter
                consecutive_no_ball_frames = 0
                
                # Normalize coordinates to (-1, 1) for X axis and (0, 1) for Y axis
                normalized_x = (ball_x - frame_width/2) / (frame_width/2)  # -1 (left) to 1 (right)
                normalized_y = ball_y / frame_height  # 0 (top) to 1 (bottom)
                
                # Define thresholds for "ball is centered" condition
                x_error_threshold = 0.1  # Ball is within 10% of center horizontally
                y_error_threshold = 0.1  # Ball is within 10% of target vertically
                y_target_error = abs(normalized_y - y_pid.setpoint)
                
                # Check if ball is well-centered in both axes
                ball_centered = (abs(normalized_x) < x_error_threshold and 
                                 y_target_error < y_error_threshold)
                
                # Special "forward boost" mode when ball is centered
                current_time = time.time()
                if not hasattr(pid_thread, "boost_start_time"):
                    pid_thread.boost_start_time = 0
                if not hasattr(pid_thread, "boost_active"):
                    pid_thread.boost_active = False
                
                if ball_centered and not pid_thread.boost_active:
                    # Start boost mode
                    log_message("INFO", "Ball centered! Activating forward boost")
                    pid_thread.boost_start_time = current_time
                    pid_thread.boost_active = True
                    
                # Check if we're in boost mode and it hasn't expired
                if pid_thread.boost_active:
                    boost_duration = 1.0  # 1 second forward boost
                    if current_time - pid_thread.boost_start_time < boost_duration:
                        # Apply forward boost: equal power to both motors
                        boost_speed = 1300  # High forward speed during boost
                        left_speed = boost_speed
                        right_speed = boost_speed
                        robot_state = 3  # Special state for boost mode
                        log_message("INFO", f"Forward boost active: {current_time - pid_thread.boost_start_time:.2f}s")
                    else:
                        # Boost duration expired, return to normal control
                        pid_thread.boost_active = False
                        log_message("INFO", "Forward boost completed")
                        
                        # Reset PID controllers after boost
                        x_pid.reset()
                        y_pid.reset()
                        
                        # Calculate normal PID outputs
                        steering = x_pid.calculate(normalized_x)    # Output: -50 to 50
                        base_speed = y_pid.calculate(normalized_y)  # Output: 0 to 100
                        
                        # Apply normal control
                        scaled_left_speed = (base_speed + steering) * MOTOR_OUTPUT_SCALE
                        scaled_right_speed = (base_speed - steering) * MOTOR_OUTPUT_SCALE
                        
                        left_speed = int(scaled_left_speed)
                        right_speed = int(scaled_right_speed)
                        
                        # Set robot state back to chase ball
                        robot_state = 1
                else:
                    # Normal PID control when not in boost mode
                    steering = x_pid.calculate(normalized_x)    # Output: -50 to 50
                    base_speed = y_pid.calculate(normalized_y)  # Output: 0 to 100
                    
                    # Combine components and apply scaling
                    scaled_left_speed = (base_speed + steering) * MOTOR_OUTPUT_SCALE
                    scaled_right_speed = (base_speed - steering) * MOTOR_OUTPUT_SCALE
                    
                    left_speed = int(scaled_left_speed)
                    right_speed = int(scaled_right_speed)
                    
                    # Set robot state to chase ball
                    robot_state = 1
                
                # Ensure speeds are within absolute limits
                left_speed = max(min(left_speed, MAX_MOTOR_COMMAND), -MAX_MOTOR_COMMAND)
                right_speed = max(min(right_speed, MAX_MOTOR_COMMAND), -MAX_MOTOR_COMMAND)
            else:
                # Increment consecutive no-ball frame counter
                consecutive_no_ball_frames += 1
                
                if consecutive_no_ball_frames > max_no_ball_frames:
                    # If we haven't seen a ball for a while, stop motors
                    left_speed = 0
                    right_speed = 0
                    robot_state = 0
                else:
                    # Continue with last known speeds for a short period
                    # We'll get these from the shared state
                    with shared_state.pid_lock:
                        left_speed = shared_state.left_motor_speed
                        right_speed = shared_state.right_motor_speed
                        robot_state = shared_state.robot_state
                    
                    # Gradually reduce speed if ball is lost
                    deceleration_factor = 0.9  # 10% reduction per frame
                    left_speed = int(left_speed * deceleration_factor)
                    right_speed = int(right_speed * deceleration_factor)
            
            # Update shared state with motor speeds
            with shared_state.pid_lock:
                shared_state.left_motor_speed = left_speed
                shared_state.right_motor_speed = right_speed
                shared_state.robot_state = robot_state
            
            # Update statistics
            update_statistics('pid')
            
            # Calculate sleep time to maintain fixed control rate
            elapsed = time.time() - start_time
            sleep_time = max(0, control_period - elapsed)
            time.sleep(sleep_time)
            
    except Exception as e:
        log_message("ERROR", f"Exception in PID thread: {e}")
    finally:
        # Update shared state
        with shared_state.pid_lock:
            shared_state.pid_running = False
            # Ensure motors are stopped on thread exit
            shared_state.left_motor_speed = 0
            shared_state.right_motor_speed = 0
            shared_state.robot_state = 0
        
        log_message("INFO", "PID control thread stopped")

def serial_thread(port=None, message_rate=50):
    """Serial communication thread"""
    log_message("INFO", "Starting serial communication thread")
    
    # Initialize serial communication
    serial_comm = SerialComm(port=port, auto_reconnect=True)
    connected = serial_comm.connect()
    
    if not connected:
        log_message("ERROR", "Failed to establish serial connection")
        
    # Calculate message period
    message_period = 1.0 / message_rate
    
    # Update shared state that serial thread is running
    with shared_state.serial_lock:
        shared_state.serial_running = True
    
    try:
        # Main serial communication loop
        while not shared_state.shutdown_requested:
            start_time = time.time()
            
            # Try to reconnect if connection was lost
            if not serial_comm.connected and serial_comm.auto_reconnect:
                if serial_comm.connect():
                    log_message("INFO", "Serial connection re-established")
                else:
                    # Sleep a bit longer if connection failed
                    time.sleep(1)
                    continue
                    
            # Get motor speeds from shared state
            with shared_state.pid_lock:
                state = shared_state.robot_state
                left_speed = shared_state.left_motor_speed
                right_speed = shared_state.right_motor_speed
            
            # Send command to MCU
            if serial_comm.connected:
                success = serial_comm.send_command(state, left_speed, right_speed)
                if not success:
                    log_message("WARNING", "Failed to send serial command")
                else:
                    # Update statistics on successful send
                    update_statistics('serial')
            
            # Calculate sleep time to maintain fixed message rate
            elapsed = time.time() - start_time
            sleep_time = max(0, message_period - elapsed)
            time.sleep(sleep_time)
            
    except Exception as e:
        log_message("ERROR", f"Exception in serial thread: {e}")
    finally:
        # Ensure motors are stopped before exit
        if serial_comm.connected:
            serial_comm.send_command(0, 0, 0)
            serial_comm.disconnect()
        
        # Update shared state
        with shared_state.serial_lock:
            shared_state.serial_running = False
        
        log_message("INFO", "Serial communication thread stopped")

def status_monitoring_thread(update_rate=1):
    """Thread to monitor system status and output information periodically"""
    log_message("INFO", "Starting status monitoring thread")
    
    try:
        while not shared_state.shutdown_requested:
            # Get current status from shared state
            with shared_state.cv_lock:
                cv_running = shared_state.cv_running
                cv_fps = shared_state.cv_fps
                ball_detected = shared_state.ball_detected
                ball_confidence = shared_state.ball_confidence
                camera_id = shared_state.camera_id
                is720p = shared_state.is720p
                resolution = "1280x720" if is720p else "1280x712"
            
            with shared_state.pid_lock:
                pid_running = shared_state.pid_running
                left_speed = shared_state.left_motor_speed
                right_speed = shared_state.right_motor_speed
                state = shared_state.robot_state
            
            with shared_state.serial_lock:
                serial_running = shared_state.serial_running
            
            # Log status information
            state_dict = {0: "STOP", 1: "CHASE", 2: "RETURN", 3: "BOOST"}
            state_name = state_dict.get(state, str(state))
            
            status = (
                f"Status: [CV: {'✓' if cv_running else '✗'}] "
                f"[PID: {'✓' if pid_running else '✗'}] "
                f"[Serial: {'✓' if serial_running else '✗'}] | "
                f"Camera: {camera_id} ({resolution}) | "
                f"Ball: {'Detected' if ball_detected else 'Not detected'} "
                f"({ball_confidence:.2f}) | "
                f"Motors: L={left_speed} R={right_speed} | "
                f"State: {state_name} | "
                f"FPS: {cv_fps:.1f}"
            )
            
            log_message("STATUS", status)
            
            # Sleep for specified period
            time.sleep(update_rate)
            
    except Exception as e:
        log_message("ERROR", f"Exception in status thread: {e}")
    finally:
        log_message("INFO", "Status monitoring thread stopped")

def signal_handler(signum, frame):
    """Handle termination signals gracefully"""
    log_message("INFO", f"Received signal {signum}, initiating shutdown...")
    shared_state.shutdown_requested = True

def main():
    """Main program entry point"""
    parser = argparse.ArgumentParser(description='HelloBalls Robot Controller')
    parser.add_argument('--no-preview', action='store_true', help='Disable CV preview window')
    parser.add_argument('--serial-port', type=str, help='Serial port for MCU communication')
    parser.add_argument('--ball-mode', action='store_true', help='Start in ball detection mode (default)')
    parser.add_argument('--person-mode', action='store_true', help='Start in person detection mode')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID to use (default: 0)')
    parser.add_argument('--resolution', choices=['720p', '712p'], default='720p', 
                      help='Initial camera resolution (default: 720p)')
    args = parser.parse_args()
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    log_message("INFO", "Starting HelloBalls Robot Controller")
    
    # Set initial camera ID and resolution
    shared_state.camera_id = args.camera
    shared_state.is720p = args.resolution == '720p'
    
    # Create thread handles
    threads = []
    
    # Start CV thread first
    cv_thread_handle = threading.Thread(
        target=cv_thread,
        args=(not args.no_preview,),
        name="CV-Thread"
    )
    threads.append(cv_thread_handle)
    cv_thread_handle.start()
    time.sleep(0.5)  # Small delay to allow CV system to initialize first
    
    # Start UI thread for preview display if preview is enabled
    if not args.no_preview:
        ui_thread_handle = threading.Thread(
            target=ui_thread,
            name="UI-Thread"
        )
        threads.append(ui_thread_handle)
        ui_thread_handle.start()
    
    # Start PID thread
    pid_thread_handle = threading.Thread(
        target=pid_thread,
        args=(50,),  # 50Hz control rate
        name="PID-Thread"
    )
    threads.append(pid_thread_handle)
    pid_thread_handle.start()
    
    # Start serial thread
    serial_thread_handle = threading.Thread(
        target=serial_thread,
        args=(args.serial_port, 50),  # 50Hz message rate
        name="Serial-Thread"
    )
    threads.append(serial_thread_handle)
    serial_thread_handle.start()
    
    # Start status monitoring thread
    status_thread_handle = threading.Thread(
        target=status_monitoring_thread,
        args=(1,),  # 1Hz status update rate
        name="Status-Thread"
    )
    threads.append(status_thread_handle)
    status_thread_handle.start()
    
    try:
        # Wait for all threads to complete (which would happen after Ctrl+C)
        for thread in threads:
            thread.join()
    except KeyboardInterrupt:
        # Handle keyboard interrupt
        log_message("INFO", "KeyboardInterrupt received, initiating shutdown...")
        shared_state.shutdown_requested = True
    finally:
        # Ensure everything is properly shut down
        log_message("INFO", "Waiting for threads to terminate...")
        
        # Add a timeout in case threads don't terminate properly
        timeout = 5.0  # seconds
        start_time = time.time()
        
        for thread in threads:
            remaining_time = max(0, timeout - (time.time() - start_time))
            if thread.is_alive():
                thread.join(remaining_time)
        
        # Check if any threads are still running
        still_alive = [t.name for t in threads if t.is_alive()]
        if still_alive:
            log_message("WARNING", f"Some threads did not terminate gracefully: {', '.join(still_alive)}")
        else:
            log_message("INFO", "All threads terminated successfully")
    
    log_message("INFO", "HelloBalls Robot Controller shutdown complete")

if __name__ == "__main__":
    main()