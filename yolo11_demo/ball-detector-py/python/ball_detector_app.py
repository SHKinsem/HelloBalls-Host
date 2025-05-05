import cv2
import numpy as np
import sys
import time
from pathlib import Path

# Add the cpp_api directory to the path
sys.path.append(str(Path(__file__).parent.parent / 'cpp_api'))

try:
    import ball_detector_cpp
    _has_cpp_module = True
except ImportError:
    print("Warning: C++ module not found. Make sure you've built the C++ bindings.")
    _has_cpp_module = False

class BallDetectorApp:
    def __init__(self):
        self.model_loaded = False
        if _has_cpp_module:
            self.model_loaded = ball_detector_cpp.load_model()
            if self.model_loaded:
                print("Model loaded successfully")
            else:
                print("Failed to load model")

    def detect_in_frame(self, frame):
        """
        Detect balls in a frame and return their coordinates
        
        Args:
            frame: OpenCV BGR image
            
        Returns:
            A tuple containing (x_coordinates, y_coordinates, confidences)
        """
        if not _has_cpp_module or not self.model_loaded:
            # Return empty results if the module isn't available
            return ([], [], [])
            
        # Call C++ function to detect balls
        return ball_detector_cpp.detect_balls(frame)
    
    def detect_from_camera(self, camera_id=0, display=True):
        """
        Run ball detection on camera feed
        
        Args:
            camera_id: Camera device ID
            display: Whether to display the video feed with detection results
            
        Returns:
            Generator that yields (frame, x_coords, y_coords, confidences) for each frame
        """
        cap = cv2.VideoCapture(camera_id)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Process frame
                start_time = time.time()
                x_coords, y_coords, confidences = self.detect_in_frame(frame)
                processing_time = time.time() - start_time
                
                # Draw results on frame if display is enabled
                if display:
                    for i in range(len(x_coords)):
                        x, y = int(x_coords[i]), int(y_coords[i])
                        conf = confidences[i]
                        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                        cv2.putText(frame, f"{conf:.2f}", (x+10, y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    # Display FPS
                    fps = 1.0 / processing_time if processing_time > 0 else 0
                    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    cv2.imshow("Ball Detection", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                yield (frame, x_coords, y_coords, confidences)
                
        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()
    
    def __del__(self):
        if _has_cpp_module and self.model_loaded:
            ball_detector_cpp.cleanup_model()