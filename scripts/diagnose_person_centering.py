#!/usr/bin/env python3
"""
Diagnostic script for person centering functionality in HelloBalls robot.
This script helps debug why wheels don't move in SEARCH mode with person detection.
"""

import sys
import os
import time

# Add the scripts directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

# Import the HelloBalls CV system
try:
    from HelloBalls_CV import HelloBallsCV, MODE_PERSON_DETECTION, ROBOT_STATE_SEARCH, PERSON_CLASS
    print("✅ Successfully imported HelloBalls_CV")
except ImportError as e:
    print(f"❌ Failed to import HelloBalls_CV: {e}")
    sys.exit(1)

def test_person_centering_logic():
    """Test the person centering logic step by step"""
    print("\n🔍 Testing Person Centering Logic")
    print("=" * 50)
    
    # Create CV system instance
    cv_system = HelloBallsCV(show_preview=False, detection_mode=MODE_PERSON_DETECTION)
    
    # Set SEARCH state
    cv_system.robot_state = ROBOT_STATE_SEARCH
    print(f"Robot state: {cv_system.robot_state} (SEARCH)")
    
    # Check detection mode
    print(f"Detection mode: {cv_system.detection_mode} (0=Ball, 1=Person)")
    
    # Check auto_person_centering flag
    print(f"Auto person centering enabled: {cv_system.auto_person_centering}")
    
    # Enable auto person centering manually for testing
    cv_system.auto_person_centering = True
    print(f"Manually enabled auto person centering: {cv_system.auto_person_centering}")
    
    # Simulate a person detection
    frame_width = 1280
    person_x = 200  # Person at left side of frame
    person_w = 100
    person_center_x = person_x + person_w / 2  # 250
    frame_center_x = frame_width / 2  # 640
    error_x = person_center_x - frame_center_x  # 250 - 640 = -390 (person to left)
    
    print(f"\n📊 Simulated Person Detection:")
    print(f"Frame width: {frame_width}")
    print(f"Person center X: {person_center_x}")
    print(f"Frame center X: {frame_center_x}")
    print(f"Error X: {error_x} pixels")
    
    # Simulate best_target
    cv_system.best_target = (PERSON_CLASS, person_x, 100, person_w, 150, 0.8)  # (cls_id, x, y, w, h, conf)
    print(f"Best target set: {cv_system.best_target}")
    
    # Test PID controller
    print(f"\n🎛️ Testing PID Controller:")
    print(f"PID setpoint: {cv_system.person_centering_pid.setpoint}")
    print(f"PID max_output: {cv_system.person_centering_pid.max_output}")
    
    # Calculate PID output
    pid_output = cv_system.person_centering_pid.compute(error_x)
    print(f"PID output for error {error_x}: {pid_output}")
    
    # Test the control logic manually
    print(f"\n🎮 Testing Control Logic:")
    
    # Check centering threshold
    centering_threshold = 50
    print(f"Centering threshold: {centering_threshold} pixels")
    print(f"Error exceeds threshold: {abs(error_x) > centering_threshold}")
    
    if abs(error_x) > centering_threshold:
        steering_output = cv_system.person_centering_pid.compute(error_x)
        steering_scaled = int(steering_output)
        
        print(f"Steering output: {steering_output}")
        print(f"Steering scaled: {steering_scaled}")
        
        # Calculate motor speeds
        if error_x > 0:
            # Person is to the right, turn right
            left_speed = abs(steering_scaled)
            right_speed = -abs(steering_scaled)
            direction = "right"
        else:
            # Person is to the left, turn left
            left_speed = -abs(steering_scaled)
            right_speed = abs(steering_scaled)
            direction = "left"
        
        print(f"Direction to turn: {direction}")
        print(f"Raw motor speeds - Left: {left_speed}, Right: {right_speed}")
        
        # Apply minimum speed constraint
        min_turn_speed = 80
        if abs(left_speed) < min_turn_speed:
            left_speed = min_turn_speed if left_speed > 0 else -min_turn_speed
        if abs(right_speed) < min_turn_speed:
            right_speed = min_turn_speed if right_speed > 0 else -min_turn_speed
        
        print(f"After min speed constraint ({min_turn_speed}) - Left: {left_speed}, Right: {right_speed}")
        
        # Apply maximum speed limit
        max_centering_speed = 150
        left_speed = max(min(left_speed, max_centering_speed), -max_centering_speed)
        right_speed = max(min(right_speed, max_centering_speed), -max_centering_speed)
        
        print(f"Final motor speeds - Left: {left_speed}, Right: {right_speed}")
        
        # Check if speeds are reasonable
        if left_speed != 0 or right_speed != 0:
            print("✅ Motor speeds are non-zero - should move!")
        else:
            print("❌ Motor speeds are zero - won't move!")
    
    return cv_system

def test_auto_centering_activation():
    """Test when auto centering gets activated"""
    print("\n🚀 Testing Auto Centering Activation")
    print("=" * 50)
    
    # Start with ball detection mode
    cv_system = HelloBallsCV(show_preview=False, detection_mode=0)  # Ball mode
    print(f"Initial detection mode: {cv_system.detection_mode} (Ball)")
    print(f"Initial auto_person_centering: {cv_system.auto_person_centering}")
    
    # Switch to SEARCH state
    cv_system.robot_state = ROBOT_STATE_SEARCH
    print(f"Set robot state to SEARCH: {cv_system.robot_state}")
    print(f"Auto_person_centering after SEARCH: {cv_system.auto_person_centering}")
    
    # Switch to person detection mode
    print("\nSwitching to person detection mode...")
    cv_system.switch_detection_mode()
    print(f"Detection mode after switch: {cv_system.detection_mode} (Person)")
    print(f"Auto_person_centering after mode switch: {cv_system.auto_person_centering}")
    
    # Check the condition in handle_keyboard_input for state 4
    print("\n🔍 Testing keyboard state 4 activation:")
    if cv_system.detection_mode == MODE_PERSON_DETECTION:
        cv_system.auto_person_centering = True
        print("✅ Auto person centering would be enabled for SEARCH mode")
    else:
        cv_system.auto_person_centering = False
        print("❌ Auto person centering would NOT be enabled")
    
    return cv_system

def main():
    """Run all diagnostic tests"""
    print("🤖 HelloBalls Person Centering Diagnostics")
    print("=" * 60)
    
    try:
        # Test 1: Person centering logic
        cv_system1 = test_person_centering_logic()
        
        # Test 2: Auto centering activation
        cv_system2 = test_auto_centering_activation()
        
        print("\n" + "=" * 60)
        print("🎯 DIAGNOSTIC SUMMARY")
        print("=" * 60)
        
        print("\n💡 PROBABLE CAUSES FOR NON-MOVING WHEELS:")
        print("1. Auto person centering flag not enabled")
        print("2. No person detected (best_target is None)")
        print("3. Person detection class ID mismatch")
        print("4. Manual override preventing auto centering")
        print("5. PID output too small (below minimum threshold)")
        print("6. Serial communication issues")
        
        print("\n🔧 DEBUGGING STEPS:")
        print("1. Ensure you're in SEARCH mode (press '4')")
        print("2. Switch to person detection (press 'm')")
        print("3. Check console output for 'Auto-centering person' messages")
        print("4. Verify person is being detected in preview window")
        print("5. Check serial connection status")
        
        print("\n📝 NEXT ACTIONS:")
        print("1. Add debug prints to control_robot method")
        print("2. Test with actual person detection")
        print("3. Verify serial communication is working")
        print("4. Check if manual override timeout is interfering")
        
    except Exception as e:
        print(f"❌ Error during diagnostics: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
