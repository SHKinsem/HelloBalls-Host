import os
import sys
import numpy as np
import cv2

# Add the current directory to the Python path to find yolo11_api.so
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the module
import yolo11_api

def test_pid_controller():
    """Test the PIDController class from the extension"""
    print("Testing PIDController...")
    
    # Create a PID controller instance with Kp=1.0, Ki=0.1, Kd=0.05
    pid = yolo11_api.PIDController(1.0, 0.1, 0.05)
    
    # Test the calculate method (not compute)
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

if __name__ == "__main__":
    print("Testing YOLO11 API Python Extension")
    print("-" * 50)
    
    pid_success = test_pid_controller()
    model_basics_success = test_model_basics()
    
    print("\nTest Results:")
    print(f"PID Controller: {'PASS' if pid_success else 'FAIL'}")
    print(f"Model Basics: {'PASS' if model_basics_success else 'FAIL'}")