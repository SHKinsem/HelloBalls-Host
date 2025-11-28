#!/usr/bin/env python3
"""
Test script for HelloBalls person centering functionality
This script helps validate the person centering implementation in SEARCH mode
"""

import os
import sys
import time
import argparse

# Add the scripts directory to the path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from HelloBalls_CV import HelloBallsCV, MODE_PERSON_DETECTION, ROBOT_STATE_SEARCH

def test_person_centering():
    """Test the person centering functionality"""
    print("🎾 HelloBalls Person Centering Test")
    print("=" * 50)
    
    # Initialize CV system in person detection mode
    cv_system = HelloBallsCV(
        show_preview=True, 
        detection_mode=MODE_PERSON_DETECTION,
        serial_port='/dev/ttyS1'  # Adjust if needed
    )
    
    if not cv_system.initialize():
        print("❌ Failed to initialize CV system")
        return False
    
    print("✅ CV system initialized successfully")
    print("\nTest Instructions:")
    print("1. The robot will start in STOP mode")
    print("2. Press 'Enter' to switch to SEARCH mode")
    print("3. Stand in front of the camera and move left/right")
    print("4. Observe the robot's turning behavior")
    print("5. Press 'q' to quit\n")
    
    # Wait for user to be ready
    input("Press Enter when ready to start testing...")
    
    try:
        # Switch to SEARCH mode and enable person centering
        cv_system.robot_state = ROBOT_STATE_SEARCH
        cv_system.auto_person_centering = True
        
        print("\n🔍 SEARCH mode activated - Person centering ENABLED")
        print("Expected behavior:")
        print("  - Robot should turn toward detected persons")
        print("  - Left motor forward + right motor reverse = turn right")
        print("  - Left motor reverse + right motor forward = turn left")
        print("  - Robot should stop when person is centered")
        print("\nWatch the console output for debugging info...")
        
        frame_count = 0
        last_status_time = time.time()
        
        while True:
            success, frame = cv_system.process_frame()
            if not success:
                print("❌ Failed to process frame")
                break
            
            # Show frame if preview is enabled
            if cv_system.show_preview and frame is not None:
                cv_system.display_frame(frame)
            
            # Control robot based on current state
            cv_system.control_robot()
            
            # Print status every 2 seconds
            frame_count += 1
            current_time = time.time()
            if current_time - last_status_time >= 2.0:
                print(f"\n📊 Status Update (Frame {frame_count}):")
                print(f"  Robot State: {cv_system.robot_state}")
                print(f"  Detection Mode: {'Person Detection' if cv_system.detection_mode == MODE_PERSON_DETECTION else 'Ball Detection'}")
                print(f"  Auto Centering: {cv_system.auto_person_centering}")
                print(f"  Person Detected: {cv_system.best_target is not None}")
                
                if cv_system.best_target:
                    cls_id, x, y, w, h, conf = cv_system.best_target
                    person_center_x = x + w / 2
                    frame_width = cv_system.camera.get(cv_system.camera.CAP_PROP_FRAME_WIDTH) if cv_system.camera else 640
                    frame_center_x = frame_width / 2
                    error_x = person_center_x - frame_center_x
                    print(f"  Person Center X: {person_center_x:.1f}")
                    print(f"  Frame Center X: {frame_center_x:.1f}")
                    print(f"  Error X: {error_x:.1f} pixels")
                    print(f"  Motor Speeds: L={cv_system.search_left_speed}, R={cv_system.search_right_speed}")
                
                last_status_time = current_time
            
            # Handle keyboard input
            if cv_system.show_preview:
                key = cv_system.cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n👋 Quitting test...")
                    break
                elif key == ord(' '):
                    # Toggle auto centering with spacebar
                    cv_system.auto_person_centering = not cv_system.auto_person_centering
                    status = "ENABLED" if cv_system.auto_person_centering else "DISABLED"
                    print(f"\n🔄 Auto person centering {status}")
    
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
    
    finally:
        # Ensure robot is stopped
        cv_system.robot_state = 0  # STOP
        cv_system.control_robot()
        cv_system.cleanup()
        print("✅ Test completed - Robot stopped")
    
    return True

def test_pid_response():
    """Test PID controller response to simulated errors"""
    print("\n🧮 Testing PID Controller Response")
    print("=" * 40)
    
    # Create CV system to access PID controller
    cv_system = HelloBallsCV(show_preview=False, detection_mode=MODE_PERSON_DETECTION)
    
    # Test different error values
    test_errors = [-200, -100, -50, -25, 0, 25, 50, 100, 200]
    
    print("Error (px) -> PID Output -> Expected Behavior")
    print("-" * 45)
    
    for error in test_errors:
        pid_output = cv_system.person_centering_pid.compute(error)
        
        if error > 0:
            behavior = "Turn RIGHT (L+, R-)"
        elif error < 0:
            behavior = "Turn LEFT (L-, R+)"
        else:
            behavior = "CENTERED (Stop)"
        
        print(f"{error:8} -> {pid_output:8.1f} -> {behavior}")
        
        # Reset PID for next test
        cv_system.person_centering_pid.reset()
    
    print("\nPID Parameters:")
    print(f"  Kp: {cv_system.person_centering_pid.kp}")
    print(f"  Ki: {cv_system.person_centering_pid.ki}")
    print(f"  Kd: {cv_system.person_centering_pid.kd}")
    print(f"  Max Output: {cv_system.person_centering_pid.max_output}")

def main():
    parser = argparse.ArgumentParser(description='Test HelloBalls person centering functionality')
    parser.add_argument('--pid-only', action='store_true', help='Only test PID controller response')
    parser.add_argument('--no-preview', action='store_true', help='Disable camera preview')
    args = parser.parse_args()
    
    if args.pid_only:
        test_pid_response()
    else:
        print("🚨 SAFETY REMINDER:")
        print("- Ensure robot is on a stable surface")
        print("- Keep clear of robot's movement area")
        print("- Be ready to use emergency stop if needed")
        print("- Test in a safe, controlled environment")
        
        confirm = input("\nDo you want to proceed with the test? (y/N): ")
        if confirm.lower() in ['y', 'yes']:
            test_person_centering()
            if not args.no_preview:
                test_pid_response()
        else:
            print("Test cancelled")

if __name__ == "__main__":
    main()
