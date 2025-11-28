#!/usr/bin/env python3
# combined_cv_serial_demo.py - Demonstration of integrated CV and Serial functionality
# Shows how to use the HelloBalls_CV.py and HelloBalls_Serial.py modules together

import os
import sys
import time
import threading
import queue
import signal

# Add script directory to path to find modules
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

# Import our modules
from HelloBalls_CV import HelloBallsCV
from scripts.HelloBalls_Serial_tmp import SerialComm

class CombinedCVSerial:
    """Combined CV and Serial controller for HelloBalls robot"""
    
    def __init__(self, show_preview=True, serial_port=None):
        """Initialize the combined system
        
        Args:
            show_preview (bool): Whether to show CV preview
            serial_port (str): Serial port for communication
        """
        self.show_preview = show_preview
        self.running = False
        
        # Initialize CV system
        self.cv_system = HelloBallsCV(show_preview=show_preview, detection_mode=0)  # Start with ball detection
        
        # Initialize Serial system
        self.serial_comm = SerialComm(port=serial_port, auto_reconnect=True, timeout=0.01)
        self.serial_connected = False
        
        # Control parameters
        self.autonomous_mode = False
        self.base_speed = 400
        self.max_speed = 800
        self.min_speed = 200
        self.turn_threshold = 50  # Pixel threshold for being "centered"
        
        # Simple PID-like control
        self.last_error = 0
        self.error_history = []
        self.max_history = 5
        
        # Threading
        self.cv_thread = None
        self.control_thread = None
        self.detection_results = {}
        self.results_lock = threading.Lock()
        
    def initialize(self):
        """Initialize both CV and Serial systems"""
        print("Initializing CV system...")
        if not self.cv_system.initialize():
            print("Failed to initialize CV system")
            return False
            
        print("Initializing Serial communication...")
        try:
            self.serial_connected = self.serial_comm.connect()
            if self.serial_connected:
                print("Serial communication initialized successfully")
                # Send initial stop command
                self.send_robot_command(0, 0, 0, 0)
            else:
                print("Warning: Serial communication failed, running in CV-only mode")
        except Exception as e:
            print(f"Warning: Serial initialization error: {e}")
            
        return True
    
    def send_robot_command(self, state, motor1, motor2, tilt=0):
        """Send command to robot via serial"""
        if not self.serial_connected:
            return False
            
        try:
            return self.serial_comm.send_command(state, motor1, motor2, tilt)
        except Exception as e:
            print(f"Error sending command: {e}")
            return False
    
    def calculate_motor_speeds(self, error_x, frame_width):
        """Calculate motor speeds based on ball position error"""
        if not self.autonomous_mode:
            return 0, 0
            
        # Normalize error to percentage of frame width
        error_percent = error_x / (frame_width / 2)
        
        # Simple proportional control
        steering = error_percent * 200  # Adjust this gain as needed
        
        # Limit steering
        steering = max(-300, min(300, steering))
        
        # Calculate motor speeds
        if abs(error_x) < self.turn_threshold:
            # Move forward when centered
            left_speed = self.base_speed
            right_speed = self.base_speed
        else:
            # Turn based on error
            if steering > 0:  # Ball is to the right, turn right
                left_speed = self.base_speed + abs(steering)
                right_speed = self.base_speed - abs(steering)
            else:  # Ball is to the left, turn left
                left_speed = self.base_speed - abs(steering)
                right_speed = self.base_speed + abs(steering)
        
        # Ensure speeds are within bounds
        left_speed = max(self.min_speed, min(self.max_speed, left_speed))
        right_speed = max(self.min_speed, min(self.max_speed, right_speed))
        
        return int(left_speed), int(right_speed)
    
    def cv_thread_function(self):
        """CV processing thread function"""
        print("CV thread started")
        
        while self.running:
            try:
                # Process a frame
                success, frame = self.cv_system.process_frame()
                
                if success:
                    # Get detection results
                    results = self.cv_system.get_detection_results()
                    
                    # Update shared results
                    with self.results_lock:
                        self.detection_results = results.copy()
                        
                    # Show frame if preview is enabled
                    if self.show_preview and frame is not None:
                        # Add some extra info to the frame
                        if self.autonomous_mode:
                            cv2.putText(frame, "AUTO MODE", (10, frame.shape[0] - 30), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        cv2.imshow(self.cv_system.window_name, frame)
                        
                        # Check for key presses
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            self.running = False
                            break
                        elif key == ord('a'):
                            self.toggle_autonomous_mode()
                        elif key == ord('s'):
                            self.emergency_stop()
                        elif key == ord('m'):
                            self.cv_system.switch_detection_mode()
                        elif key == ord('b'):
                            self.cv_system.switch_ball_selection_mode()
                        
                else:
                    print("CV processing failed")
                    time.sleep(0.1)
                    
            except Exception as e:
                print(f"Error in CV thread: {e}")
                time.sleep(0.1)
                
        print("CV thread ended")
    
    def control_thread_function(self):
        """Robot control thread function"""
        print("Control thread started")
        
        while self.running:
            try:
                if self.autonomous_mode and self.serial_connected:
                    # Get latest detection results
                    with self.results_lock:
                        results = self.detection_results.copy()
                    
                    # Process results for robot control
                    if results.get('best_target'):
                        target = results['best_target']
                        
                        # Only act on ball detections in autonomous mode
                        if (results.get('mode') == 'Ball Detection' and 
                            target.get('class_id') == 32):  # Sports ball class
                            
                            error_x = target.get('error_x', 0)
                            
                            # Calculate motor speeds
                            left_speed, right_speed = self.calculate_motor_speeds(
                                error_x, 640)  # Assume 640 width for calculation
                            
                            # Send command to robot
                            self.send_robot_command(2, left_speed, right_speed, 0)  # State 2 for auto mode
                            
                            # Debug output
                            if abs(error_x) < self.turn_threshold:
                                print(f"Centered! Moving forward: L={left_speed}, R={right_speed}")
                            else:
                                direction = "right" if error_x > 0 else "left"
                                print(f"Ball to {direction}, error={error_x:.1f}: L={left_speed}, R={right_speed}")
                        
                        elif results.get('mode') == 'Person Detection':
                            # In person detection mode, just rotate to align
                            target = results['best_target']
                            error_x = target.get('error_x', 0)
                            
                            if abs(error_x) < self.turn_threshold:
                                # Person is centered, prepare to shoot
                                print("Person centered - ready to shoot!")
                                self.send_robot_command(3, 0, 0, 45)  # State 3 with tilt angle
                                time.sleep(2)  # Hold shooting position
                                
                                # Switch back to ball detection
                                self.cv_system.switch_detection_mode()
                            else:
                                # Rotate to center person
                                turn_speed = self.min_speed
                                if error_x > 0:
                                    self.send_robot_command(2, -turn_speed, turn_speed, 0)
                                else:
                                    self.send_robot_command(2, turn_speed, -turn_speed, 0)
                    else:
                        # No target detected, search
                        if results.get('mode') == 'Ball Detection':
                            # Rotate to search for ball
                            search_speed = self.min_speed // 2
                            self.send_robot_command(2, -search_speed, search_speed, 0)
                        
                # Sleep to avoid overwhelming the serial connection
                time.sleep(0.05)  # 20Hz control loop
                
            except Exception as e:
                print(f"Error in control thread: {e}")
                time.sleep(0.1)
                
        print("Control thread ended")
    
    def toggle_autonomous_mode(self):
        """Toggle autonomous mode"""
        self.autonomous_mode = not self.autonomous_mode
        
        if self.autonomous_mode:
            print("AUTONOMOUS MODE ENABLED")
            # Ensure we're in ball detection mode
            if self.cv_system.detection_mode != 0:
                self.cv_system.switch_detection_mode()
        else:
            print("AUTONOMOUS MODE DISABLED")
            self.emergency_stop()
            
        return self.autonomous_mode
    
    def emergency_stop(self):
        """Emergency stop"""
        self.autonomous_mode = False
        if self.serial_connected:
            self.send_robot_command(0, 0, 0, 0)  # Stop
        print("EMERGENCY STOP!")
    
    def run(self):
        """Run the combined system"""
        self.running = True
        
        # Start threads
        self.cv_thread = threading.Thread(target=self.cv_thread_function, daemon=True)
        self.control_thread = threading.Thread(target=self.control_thread_function, daemon=True)
        
        self.cv_thread.start()
        self.control_thread.start()
        
        print("Combined CV-Serial system started")
        print("Controls:")
        print("  q: Quit")
        print("  a: Toggle autonomous mode")
        print("  s: Emergency stop")
        print("  m: Switch detection mode")
        print("  b: Switch ball selection algorithm")
        
        try:
            # Main thread just waits
            while self.running:
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\\nShutdown requested...")
            
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("Cleaning up...")
        self.running = False
        
        # Stop robot
        if self.serial_connected:
            self.send_robot_command(0, 0, 0, 0)
            self.serial_comm.disconnect()
        
        # Wait for threads
        if self.cv_thread and self.cv_thread.is_alive():
            self.cv_thread.join(timeout=2)
        if self.control_thread and self.control_thread.is_alive():
            self.control_thread.join(timeout=2)
            
        # Cleanup CV system
        self.cv_system.cleanup()
        
        print("Cleanup complete")

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    print(f"\\nReceived signal {signum}, shutting down...")
    global combined_system
    if combined_system:
        combined_system.cleanup()
    sys.exit(0)

# Global variable for signal handler
combined_system = None

if __name__ == "__main__":
    import argparse
    import cv2
    
    parser = argparse.ArgumentParser(description='Combined CV-Serial HelloBalls Controller')
    parser.add_argument('--no-preview', action='store_true', help='Disable CV preview window')
    parser.add_argument('--serial-port', type=str, help='Serial port for robot communication')
    parser.add_argument('--auto-start', action='store_true', help='Start in autonomous mode')
    args = parser.parse_args()
    
    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create combined system
    combined_system = CombinedCVSerial(
        show_preview=not args.no_preview,
        serial_port=args.serial_port
    )
    
    # Initialize system
    if combined_system.initialize():
        # Start autonomous mode if requested
        if args.auto_start:
            combined_system.toggle_autonomous_mode()
            
        # Run the system
        combined_system.run()
    else:
        print("Failed to initialize combined system")
