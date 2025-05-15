import os
import sys
import numpy as np
import cv2
import time
import glob  # For searching files

# Add the current directory to the Python path to find yolo11_api.so
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the module
import yolo11_api # type: ignore

# Change working directory to match the C++ executable's expected location
cpp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cpp")
os.chdir(cpp_dir)
print(f"Changed working directory to: {os.getcwd()}")

# Constants that match those in config.h
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
CONFIDENCE_THRESHOLD = 0.4
SPORTS_BALL_CLASS = 32  # Sports ball class ID in COCO dataset

# Person class ID in COCO dataset
PERSON_CLASS = 0

# Detection modes
MODE_BALL_DETECTION = 0
MODE_PERSON_DETECTION = 1
MODE_NAMES = ["Ball Detection", "Person Detection"]

def find_model_file():
    """Find the YOLO model file using the exact same path as in C++"""
    # First, check if the model exists in the standard location
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_filename = "yolo11m_detect_bayese_640x640_nv12_modified.bin"
    
    # Absolute path to the model in the ptq_models directory
    absolute_path = os.path.join(base_dir, "ptq_models", model_filename)
    
    if os.path.exists(absolute_path):
        print(f"Found model at: {absolute_path}")
        return absolute_path
    
    # If not found, try the path relative to where we changed the working directory (cpp dir)
    cpp_relative_path = "../../ptq_models/" + model_filename
    cpp_absolute_path = os.path.normpath(os.path.join(cpp_dir, cpp_relative_path))
    
    print(f"Looking for model at C++ relative path: {cpp_relative_path}")
    print(f"Which resolves to: {cpp_absolute_path}")
    
    # Check if the file exists at the path the C++ code would use
    if os.path.exists(cpp_absolute_path):
        print(f"Found model at: {cpp_absolute_path}")
        return cpp_absolute_path
    
    # If model still not found, search in other common locations
    search_paths = [
        os.path.join(base_dir, "ptq_models", model_filename),
        os.path.join(base_dir, "..", "ptq_models", model_filename),
        os.path.join(os.path.dirname(base_dir), "ptq_models", model_filename),
        "/home/sunrise/Documents/HelloBalls-Host/yolo11_demo/ptq_models/" + model_filename
    ]
    
    for path in search_paths:
        if os.path.exists(path):
            print(f"Found model at: {path}")
            return path
    
    # If model not found, provide helpful error and suggestions
    print("\nERROR: YOLO model file not found!")
    print(f"Expected model file: {model_filename}")
    
    # Check if other model files exist that could be used
    ptq_dir = os.path.join(base_dir, "ptq_models")
    if os.path.exists(ptq_dir):
        available_models = [f for f in os.listdir(ptq_dir) if f.endswith('.bin')]
        if available_models:
            print("\nAvailable model files:")
            for model in available_models:
                print(f"  - {model}")
            
            # If we have the yolo11s model but need yolo11m, suggest copying/renaming
            if "yolo11s_detect_bayese_640x640_nv12_modified.bin" in available_models:
                print(f"\nFound yolo11s model but C++ code needs yolo11m model.")
                print(f"To use the model, you could copy and rename the existing model:")
                print(f"cp {os.path.join(ptq_dir, 'yolo11s_detect_bayese_640x640_nv12_modified.bin')} "
                      f"{os.path.join(ptq_dir, model_filename)}")
    
    # Create an environment variable to specify the model path
    print("\nTIP: You can also specify the model path using an environment variable:")
    print("export YOLO11_MODEL_PATH=/path/to/your/model.bin")
    
    # Try to read from environment variable as a last resort
    env_model_path = os.environ.get('YOLO11_MODEL_PATH')
    if env_model_path and os.path.exists(env_model_path):
        print(f"Found model at environment variable path: {env_model_path}")
        return env_model_path
        
    return None

def preprocess_image_letterbox(frame, input_width, input_height):
    """
    Python implementation matching the C++ preprocess function with letterbox
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

def toggle_resolution_cpp_style(cap, is720p):
    """
    Python implementation matching the C++ toggleResolution function
    with optimizations to reduce delay when switching resolutions
    """
    # Set properties that affect switching delay
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer size to reduce latency
    
    # Store current position in case we need to use a frame grab trick
    current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
    
    # Release and reopen approach is sometimes faster than just setting properties
    if is720p:
        # Quick toggle to the other resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 712)
        
        # Flush the buffer by grabbing a few frames
        for _ in range(2):
            cap.grab()
            
        print("Resolution changed to 1280x712")
    else:
        # Quick toggle to the other resolution
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

def test_pid_controller():
    """Test the PIDController class from the extension"""
    print("Testing PIDController...")
    
    # Create a PID controller instance with Kp=1.0, Ki=0.1, Kd=0.05
    pid = yolo11_api.PIDController(1.0, 0.1, 0.05)
    
    # Test the calculate method
    error = 5.0
    output = pid.calculate(error)
    print(f"PID output for error={error}: {output}")
    
    # Test multiple iterations
    print("Testing PID control over 10 iterations with decreasing error:")
    for i in range(10):
        err = 10.0 - i
        out = pid.calculate(err)
        print(f"Iteration {i+1}: Error={err:.2f}, Output={out:.4f}")
    
    return True

def test_model_basics():
    """Test basic model operations without loading an actual model"""
    print("\nTesting ModelOutput structure...")
    
    # Create a ModelOutput object
    output = yolo11_api.ModelOutput()
    
    # Test that we can set and access class_ids
    output.class_ids = [1, 2, 3]
    print(f"Class IDs: {output.class_ids}")
    
    # Test that we can set and access confidences
    output.scores = [0.95, 0.85, 0.75]
    print(f"Confidences: {output.scores}")
    
    return True

def process_person_detection(frame, x, y, w, h, confs, pid_controller):
    """
    Handle person detection mode operations when a person is detected
    """
    # Calculate person center
    person_center_x = x + w / 2
    person_center_y = y + h / 2
    frame_width = frame.shape[1]
    
    # Draw center point of the person
    cv2.circle(frame, (int(person_center_x), int(person_center_y)), 5, (255, 150, 0), -1)
    
    # Calculate distance from center of frame horizontally
    frame_center_x = frame_width / 2
    x_error = person_center_x - frame_center_x
    
    # Draw a vertical line at frame center for reference
    cv2.line(frame, (int(frame_center_x), 0), (int(frame_center_x), frame.shape[0]), 
            (0, 150, 255), 1, cv2.LINE_AA)
    
    # Draw line from person to center line
    cv2.line(frame, (int(person_center_x), int(person_center_y)), 
            (int(frame_center_x), int(person_center_y)), (0, 255, 255), 2)
    
    # Calculate PID control for person tracking
    steering = pid_controller.calculate(x_error)
    
    # Convert steering to motor speeds - slower base speed for person following
    base_speed = 30  # Lower base speed for safety when following people
    left_speed = base_speed
    right_speed = base_speed
    
    if steering > 0:
        # Person is to the right, need to turn right
        left_speed = base_speed + abs(steering)
        right_speed = base_speed - abs(steering)
    else:
        # Person is to the left, need to turn left
        left_speed = base_speed - abs(steering)
        right_speed = base_speed + abs(steering)
    
    # Ensure speeds are within bounds
    left_speed = max(min(left_speed, 100), -100)
    right_speed = max(min(right_speed, 100), -100)
    
    # Display motor speeds on frame
    motor_text = f"Motors L:{int(left_speed)} R:{int(right_speed)}"
    cv2.putText(frame, motor_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Display error and PID output
    error_text = f"Error: {int(x_error)} PID: {int(steering)}"
    cv2.putText(frame, error_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Display tracking status
    confidence_text = f"Person tracking: {int(confs * 100)}% confident"
    cv2.putText(frame, confidence_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return True

def process_ball_detection(frame, x, y, w, h, confs, pid_controller):
    """
    Handle ball detection mode operations when a ball is detected
    """
    # Calculate ball center
    ball_center_x = x + w / 2
    ball_center_y = y + h / 2
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]
    
    # Draw center point of the ball
    cv2.circle(frame, (int(ball_center_x), int(ball_center_y)), 5, (0, 255, 255), -1)
    
    # Calculate target position (bottom center of frame)
    target_x = frame_width / 2
    target_y = frame_height * 0.9  # 90% down the frame
    
    # Draw target position
    cv2.circle(frame, (int(target_x), int(target_y)), 10, (255, 255, 0), 2)
    cv2.line(frame, (int(target_x - 15), int(target_y)), 
            (int(target_x + 15), int(target_y)), (255, 255, 0), 2)
    cv2.line(frame, (int(target_x), int(target_y - 15)), 
            (int(target_x), int(target_y + 15)), (255, 255, 0), 2)
    
    # Draw line from ball to target
    cv2.line(frame, (int(ball_center_x), int(ball_center_y)), 
            (int(target_x), int(target_y)), (0, 255, 255), 2)
    
    # Calculate error (distance from target position)
    x_error = ball_center_x - target_x
    
    # Use PID controller to calculate steering value
    steering = pid_controller.calculate(x_error)
    
    # Convert steering to motor speeds - higher base speed for ball chasing
    base_speed = 50  # Base forward speed for ball chasing
    left_speed = base_speed
    right_speed = base_speed
    
    if steering > 0:
        # Ball is to the right, need to turn right
        left_speed = base_speed + abs(steering)
        right_speed = base_speed - abs(steering)
    else:
        # Ball is to the left, need to turn left
        left_speed = base_speed - abs(steering)
        right_speed = base_speed + abs(steering)
    
    # Ensure speeds are within bounds
    left_speed = max(min(left_speed, 100), -100)
    right_speed = max(min(right_speed, 100), -100)
    
    # Display motor speeds on frame
    motor_text = f"Motors L:{int(left_speed)} R:{int(right_speed)}"
    cv2.putText(frame, motor_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Display error and PID output
    error_text = f"Error: {int(x_error)} PID: {int(steering)}"
    cv2.putText(frame, error_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Display tracking status
    confidence_text = f"Ball tracking: {int(confs * 100)}% confident"
    cv2.putText(frame, confidence_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return True

def find_cpp_function_name(module, possible_names):
    """Find the actual function name in the module that might match one of several C++ style names"""
    for name in possible_names:
        if hasattr(module, name):
            return name
    return None

def load_yolo_model():
    """Load YOLO model using the existing API functions"""
    print("Loading YOLO detection model...")
    
    # Find the model file path (for informational purposes only)
    model_path = find_model_file()
    if model_path:
        print(f"Found model at: {model_path}")
    
    # The model path is likely hardcoded in the API, similar to MODEL_PATH in C++
    try:
        # Initialize the model without parameters - it likely uses a hardcoded path internally
        model_initialized = yolo11_api.initialize_model()
        if model_initialized:
            print(f"Model initialized successfully")
            return True  # Return True instead of a handle
        else:
            print("Model initialization returned False")
            return None
    except Exception as e:
        print(f"Error initializing model: {e}")
        return None

def run_camera_detection():
    """Use the C++ implementation through Python bindings to run camera-based detection"""
    print("\nStarting camera-based detection using C++ implementation...")
    
    # Load the model using our updated function
    model_initialized = load_yolo_model()
    
    if not model_initialized:
        print("Error: Failed to load model")
        return False
    
    # Create a named window first, so we can position it
    cv2.namedWindow("YOLOv11m Object Detection", cv2.WINDOW_NORMAL)
    
    # Position the window in the center of the screen
    # Get screen dimensions using an alternative method
    try:
        # Try to get screen resolution using xrandr (Linux)
        import subprocess
        output = subprocess.check_output('xrandr | grep "\*" | cut -d" " -f4', shell=True).decode('utf-8').strip()
        screen_w, screen_h = map(int, output.split('x'))
    except:
        # Fallback to a common resolution if we can't detect it
        screen_w, screen_h = 1920, 1080
        print(f"Could not detect screen resolution, using default: {screen_w}x{screen_h}")
    
    # Open camera
    camera_id = 0  # Default camera ID
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return False
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Get actual resolution
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Camera opened with resolution: {actual_width}x{actual_height}")
    
    # Calculate window size (80% of screen size)
    window_w = int(screen_w * 0.8)
    window_h = int(window_w * actual_height / actual_width)  # Maintain aspect ratio
    
    if window_h > screen_h * 0.8:
        # If too tall, scale by height instead
        window_h = int(screen_h * 0.8)
        window_w = int(window_h * actual_width / actual_height)
    
    # Resize window to desired size
    cv2.resizeWindow("YOLOv11m Object Detection", window_w, window_h)
    
    # Position window in center of screen
    win_x = (screen_w - window_w) // 2
    win_y = (screen_h - window_h) // 2
    cv2.moveWindow("YOLOv11m Object Detection", win_x, win_y)
    
    # Create a PID controller for ball tracking
    pid_controller = yolo11_api.PIDController(0.05, 0.001, 0.01)
    
    # Create FPS counter
    fps_counter = SimpleFpsCounter()
    
    # Track resolution state
    is720p = True
    
    # Track detection mode
    detection_mode = MODE_BALL_DETECTION
    
    # Variables to display information in the upper left
    motor_left_speed = 0
    motor_right_speed = 0
    error_value = 0
    pid_output = 0
    detection_confidence = 0
    
    print(f"Window positioned at ({win_x}, {win_y}) with size {window_w}x{window_h}")
    print("Press 'q' to quit, 'r' to toggle resolution, 'f' to toggle fullscreen, 'm' to switch detection mode")
    print(f"Current detection mode: {MODE_NAMES[detection_mode]}")
    
    # Track fullscreen state
    is_fullscreen = False
    
    try:
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Error: Failed to capture frame")
                break
            
            # Get frame dimensions for display purposes
            height, width = frame.shape[:2]
            
            # Preprocess the frame
            preprocessed_frame, x_scale, y_scale, x_shift, y_shift = preprocess_image_letterbox(
                frame, INPUT_WIDTH, INPUT_HEIGHT)
            
            # Run detection using the C++ API
            detection_results = yolo11_api.inference(preprocessed_frame)
            
            # Clear detection processed flag
            detection_processed = False
            highest_confidence = 0
            best_detection = None
            
            # First pass: find the best target for current mode
            if detection_results:
                for cls_id, boxes, confs in zip(detection_results.class_ids, 
                                               detection_results.bboxes, 
                                               detection_results.scores):
                    # Only process target objects for the current mode
                    if ((detection_mode == MODE_BALL_DETECTION and cls_id == SPORTS_BALL_CLASS) or
                        (detection_mode == MODE_PERSON_DETECTION and cls_id == PERSON_CLASS)):
                        
                        # Keep track of highest confidence detection
                        if confs > highest_confidence:
                            highest_confidence = confs
                            best_detection = (cls_id, boxes, confs)
            
            # Draw info panel in the upper left
            # Semi-transparent black background for info panel
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (320, 140), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            # Display current mode in top-left corner with mode-appropriate color
            mode_color = (0, 0, 255) if detection_mode == MODE_BALL_DETECTION else (0, 255, 0)
            cv2.putText(frame, f"Mode: {MODE_NAMES[detection_mode]}", (20, 35), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
            
            # Display FPS in the upper left
            fps = fps_counter.update()
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 65), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                      
            # Display motor, error and PID info
            cv2.putText(frame, f"Motors L:{int(motor_left_speed)} R:{int(motor_right_speed)}", 
                      (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Error: {int(error_value)} PID: {int(pid_output)}", 
                      (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
            # Process the best detection if found
            if best_detection:
                cls_id, boxes, confs = best_detection
                
                # Scale box coordinates back to original frame
                x = (boxes[0] - x_shift) / x_scale
                y = (boxes[1] - y_shift) / y_scale
                w = boxes[2] / x_scale
                h = boxes[3] / y_scale
                
                # Draw bounding box with different colors based on detection mode
                if detection_mode == MODE_BALL_DETECTION and cls_id == SPORTS_BALL_CLASS:
                    # Ball detection mode
                    box_color = (0, 0, 255)  # Red for balls
                    label = f"Ball: {int(confs * 100)}%"
                    
                    # Process ball detection
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
                    
                    # Calculate error (distance from target position)
                    error_value = ball_center_x - target_x
                    
                elif detection_mode == MODE_PERSON_DETECTION and cls_id == PERSON_CLASS:
                    # Person detection mode
                    box_color = (0, 255, 0)  # Green for people
                    label = f"Person: {int(confs * 100)}%"
                    
                    # Process person detection
                    person_center_x = x + w / 2
                    person_center_y = y + h / 2
                    
                    # Draw center point of the person
                    cv2.circle(frame, (int(person_center_x), int(person_center_y)), 5, (255, 150, 0), -1)
                    
                    # Calculate distance from center of frame horizontally
                    frame_center_x = width / 2
                    
                    # Draw a vertical line at frame center for reference
                    cv2.line(frame, (int(frame_center_x), 0), (int(frame_center_x), height), 
                            (0, 150, 255), 1, cv2.LINE_AA)
                    
                    # Draw line from person to center line
                    cv2.line(frame, (int(person_center_x), int(person_center_y)), 
                            (int(frame_center_x), int(person_center_y)), (0, 255, 255), 2)
                    
                    # Calculate error (distance from center)
                    error_value = person_center_x - frame_center_x
                
                # Draw bounding box
                cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), box_color, 3)
                
                # Add label
                cv2.putText(frame, label, (int(x), int(y - 10)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
                
                # Use PID controller to calculate steering value
                pid_output = pid_controller.calculate(error_value)
                
                # Convert steering to motor speeds
                base_speed = 50 if detection_mode == MODE_BALL_DETECTION else 30
                
                if pid_output > 0:
                    # Turn right
                    motor_left_speed = base_speed + abs(pid_output)
                    motor_right_speed = base_speed - abs(pid_output)
                else:
                    # Turn left
                    motor_left_speed = base_speed - abs(pid_output)
                    motor_right_speed = base_speed + abs(pid_output)
                
                # Ensure speeds are within bounds
                motor_left_speed = max(min(motor_left_speed, 100), -100)
                motor_right_speed = max(min(motor_right_speed, 100), -100)
                
                detection_processed = True
                detection_confidence = confs
            
            # Display message if no target object is detected
            if not detection_processed:
                # if detection_mode == MODE_BALL_DETECTION:
                #     cv2.putText(frame, "No ball detected", (width//2 - 100, height//2), 
                #                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                # else:
                #     cv2.putText(frame, "No person detected", (width//2 - 120, height//2), 
                #                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                
                # # Reset control values when nothing detected
                motor_left_speed = 0
                motor_right_speed = 0
                error_value = 0
                pid_output = 0
            
            # Show the frame
            cv2.imshow("YOLOv11m Object Detection", frame)
            
            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                is720p = toggle_resolution_cpp_style(cap, is720p)
            elif key == ord('f'):
                # Toggle fullscreen mode
                is_fullscreen = not is_fullscreen
                if is_fullscreen:
                    cv2.setWindowProperty("YOLOv11m Object Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    print("Switched to fullscreen mode")
                else:
                    cv2.setWindowProperty("YOLOv11m Object Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("YOLOv11m Object Detection", window_w, window_h)
                    cv2.moveWindow("YOLOv11m Object Detection", win_x, win_y)
                    print("Exited fullscreen mode")
            elif key == ord('m'):
                # Switch detection mode
                detection_mode = (detection_mode + 1) % len(MODE_NAMES)
                print(f"Switched to {MODE_NAMES[detection_mode]} mode")
                # Reset PID controller when switching modes to avoid carry-over
                pid_controller = yolo11_api.PIDController(0.05, 0.001, 0.01)
    
    except KeyboardInterrupt:
        print("Detection stopped by user")
    except Exception as e:
        print(f"Error in detection loop: {e}")
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        try:
            yolo11_api.cleanup_model()
            print("Model resources released")
        except Exception as e:
            print(f"Error cleaning up model: {e}")
    
    return True

# Simple FPS counter fallback
class SimpleFpsCounter:
    def __init__(self):
        self.prev_time = time.time()
        self.frames = 0
        self.fps = 0
        
    def update(self):
        self.frames += 1
        current_time = time.time()
        elapsed = current_time - self.prev_time
        
        if elapsed >= 1.0:
            self.fps = self.frames / elapsed
            self.frames = 0
            self.prev_time = current_time
            
        return self.fps

if __name__ == "__main__":
    print("Testing YOLO11 API Python Extension")
    print("-" * 50)
    
    # Check if mandatory libraries are available
    try:
        # import yolo11_api
        print(f"Successfully imported yolo11_api module, version: {getattr(yolo11_api, '__version__', 'unknown')}")
        
        # Display available functions for debugging
        functions = [name for name in dir(yolo11_api) if not name.startswith('_') and callable(getattr(yolo11_api, name))]
        print(f"Available API functions: {', '.join(functions)}")
        
    except ImportError as e:
        print(f"Error: Failed to import yolo11_api module: {e}")
        print("Please make sure it's properly installed and compiled.")
        sys.exit(1)
    
    # Run basic tests
    pid_success = test_pid_controller()
    model_basics_success = test_model_basics()
    
    # Run camera detection directly (no user input needed)
    camera_success = run_camera_detection()
    
    print("\nTest Results:")
    print(f"PID Controller: {'PASS' if pid_success else 'FAIL'}")
    print(f"Model Basics: {'PASS' if model_basics_success else 'FAIL'}")
    print(f"Camera Detection: {'PASS' if camera_success else 'FAIL'}")