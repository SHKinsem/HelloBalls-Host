import os
import sys
import numpy as np
import cv2
import time
import glob  # For searching files

# Add the current directory to the Python path to find yolo11_api.so
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the module
import yolo11_api

# Change working directory to match the C++ executable's expected location
cpp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cpp")
os.chdir(cpp_dir)
print(f"Changed working directory to: {os.getcwd()}")

# Constants that match those in config.h
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
CONFIDENCE_THRESHOLD = 0.4
SPORTS_BALL_CLASS = 32  # Sports ball class ID in COCO dataset

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
    """
    if is720p:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 712)
        print("Resolution changed to 1280x712")
    else:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
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

def process_ball_detection_python(frame, bboxes, scores, indices, x_scale, y_scale, x_shift, y_shift, pid_controller):
    """Python implementation of processBallDetection to use when the C++ version isn't available"""
    # Check for sports ball class (usually 32 in COCO dataset)
    if SPORTS_BALL_CLASS >= len(bboxes) or not indices[SPORTS_BALL_CLASS]:
        return False
    
    # Get the highest confidence ball detection
    best_ball_idx = indices[SPORTS_BALL_CLASS][0]
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]
    
    # Convert bounding box to actual frame coordinates
    width = bboxes[SPORTS_BALL_CLASS][best_ball_idx].width / x_scale
    height = bboxes[SPORTS_BALL_CLASS][best_ball_idx].height / y_scale
    y1 = (bboxes[SPORTS_BALL_CLASS][best_ball_idx].y - y_shift) / y_scale
    x1 = (bboxes[SPORTS_BALL_CLASS][best_ball_idx].x - x_shift) / x_scale
    
    # Calculate ball center
    ball_center_x = x1 + width / 2
    ball_center_y = y1 + height / 2
    
    # Calculate target position (bottom center of frame)
    target_x = frame_width / 2
    target_y = frame_height * 0.9  # 90% down the frame
    
    # Draw target position
    cv2.circle(frame, (int(target_x), int(target_y)), 10, (255, 255, 0), 2)
    cv2.line(frame, (int(target_x - 15), int(target_y)), (int(target_x + 15), int(target_y)), (255, 255, 0), 2)
    cv2.line(frame, (int(target_x), int(target_y - 15)), (int(target_x), int(target_y + 15)), (255, 255, 0), 2)
    
    # Draw line from ball to target
    cv2.line(frame, (int(ball_center_x), int(ball_center_y)), (int(target_x), int(target_y)), (0, 255, 255), 2)
    
    # Calculate error (distance from target position)
    x_error = ball_center_x - target_x
    
    # Use PID controller to calculate steering value
    steering = pid_controller.calculate(x_error)
    
    # Convert steering to motor speeds
    base_speed = 50  # Base forward speed
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
    
    # Create a PID controller for ball tracking
    pid_controller = yolo11_api.PIDController(0.05, 0.001, 0.01)
    
    # Create FPS counter
    fps_counter = SimpleFpsCounter()
    
    # Track resolution state
    is720p = True
    
    print("Press 'q' to quit, 'r' to toggle resolution")
    
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
            
            if detection_results:
                # Draw detections and process ball tracking
                for cls_id, boxes, confs in zip(detection_results.class_ids, 
                                               detection_results.bboxes, 
                                               detection_results.scores):
                    if cls_id == SPORTS_BALL_CLASS or cls_id == 0:  # Sports ball or person
                        # Scale box coordinates back to original frame
                        x = (boxes[0] - x_shift) / x_scale
                        y = (boxes[1] - y_shift) / y_scale
                        w = boxes[2] / x_scale
                        h = boxes[3] / y_scale
                        
                        # Draw bounding box
                        color = (0, 0, 255) if cls_id == SPORTS_BALL_CLASS else (0, 255, 0)
                        cv2.rectangle(frame, (int(x), int(y)), 
                                     (int(x + w), int(y + h)), color, 2)
                        
                        # Add label
                        label = f"Ball: {int(confs * 100)}%" if cls_id == SPORTS_BALL_CLASS else f"Person: {int(confs * 100)}%"
                        cv2.putText(frame, label, (int(x), int(y - 5)), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        
                        # For balls, calculate center and draw tracking info
                        if cls_id == SPORTS_BALL_CLASS:
                            # Calculate ball center
                            ball_center_x = x + w / 2
                            ball_center_y = y + h / 2
                            
                            # Draw center point
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
                            x_error = ball_center_x - target_x
                            
                            # Use PID controller to calculate steering value
                            steering = pid_controller.calculate(x_error)
                            
                            # Convert steering to motor speeds
                            base_speed = 50  # Base forward speed
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
                            cv2.putText(frame, motor_text, (10, 90), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                            
                            # Display error and PID output
                            error_text = f"Error: {int(x_error)} PID: {int(steering)}"
                            cv2.putText(frame, error_text, (10, 120), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display fps
            fps = fps_counter.update()
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            
            # Show the frame
            cv2.imshow("YOLOv11m Object Detection", frame)
            
            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                is720p = toggle_resolution_cpp_style(cap, is720p)
    
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
        import yolo11_api
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