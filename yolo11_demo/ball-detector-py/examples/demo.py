import sys
import os
from pathlib import Path

# Add the path to the ball_detector module
sys.path.append(str(Path(__file__).parent.parent / 'python'))

from ball_detector_app import BallDetectorApp

def main():
    detector = BallDetectorApp()
    
    # Example of processing video feed with results displayed
    for frame, x_coords, y_coords, confidences in detector.detect_from_camera(camera_id=0):
        if x_coords:  # If balls detected
            print(f"Detected {len(x_coords)} balls:")
            for i in range(len(x_coords)):
                print(f"  Ball at ({x_coords[i]:.1f}, {y_coords[i]:.1f}) with confidence {confidences[i]:.2f}")
    
if __name__ == "__main__":
    main()