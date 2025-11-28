#!/usr/bin/env python3
"""
Test script to verify that SEARCH mode (robot state 4) continues to work
after pressing 'r' to toggle camera resolution.

This test simulates the problematic scenario:
1. Set robot to SEARCH mode (state 4)
2. Toggle camera resolution with 'r' key
3. Verify SEARCH mode keyboard controls ('w'/'s') still work
"""

import sys
import os

# Add the scripts directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

try:
    from HelloBalls_CV import HelloBallsCV, ROBOT_STATE_SEARCH
    import cv2
    import time
    
    def test_resolution_toggle_with_search_mode():
        """Test that SEARCH mode works after resolution toggle"""
        print("Testing SEARCH mode functionality after resolution toggle...")
        print("=" * 60)
        
        # Initialize CV system
        cv_system = HelloBallsCV(show_preview=False)  # No preview for automated test
        
        if not cv_system.initialize():
            print("ERROR: Failed to initialize CV system")
            return False
            
        print("✓ CV system initialized successfully")
        
        # Set robot to SEARCH mode
        cv_system.robot_state = ROBOT_STATE_SEARCH
        cv_system.tilt_angle = 15  # Set initial tilt angle
        print(f"✓ Robot set to SEARCH mode (state {ROBOT_STATE_SEARCH})")
        print(f"✓ Initial tilt angle: {cv_system.tilt_angle}°")
        
        # Test initial SEARCH mode controls
        print("\nTesting initial SEARCH mode controls...")
        
        # Test 'w' key (tilt up)
        original_tilt = cv_system.tilt_angle
        cv_system.handle_keyboard_input('w')
        if cv_system.tilt_angle > original_tilt:
            print(f"✓ 'w' key works: tilt increased from {original_tilt}° to {cv_system.tilt_angle}°")
        else:
            print(f"✗ 'w' key failed: tilt remained at {cv_system.tilt_angle}°")
            return False
            
        # Test 's' key (tilt down)
        original_tilt = cv_system.tilt_angle
        cv_system.handle_keyboard_input('s')
        if cv_system.tilt_angle < original_tilt:
            print(f"✓ 's' key works: tilt decreased from {original_tilt}° to {cv_system.tilt_angle}°")
        else:
            print(f"✗ 's' key failed: tilt remained at {cv_system.tilt_angle}°")
            return False
        
        # Now toggle camera resolution (this was causing the issue)
        print("\nToggling camera resolution...")
        original_resolution = cv_system.is720p
        new_resolution = cv_system.toggle_resolution()
        
        if new_resolution != original_resolution:
            resolution_name = "720p" if new_resolution else "712p"
            print(f"✓ Resolution toggled successfully to {resolution_name}")
        else:
            print("✗ Resolution toggle failed")
            return False
            
        # Test that SEARCH mode controls still work after resolution toggle
        print("\nTesting SEARCH mode controls AFTER resolution toggle...")
        
        # Test 'w' key again
        original_tilt = cv_system.tilt_angle
        cv_system.handle_keyboard_input('w')
        if cv_system.tilt_angle > original_tilt:
            print(f"✓ 'w' key still works: tilt increased from {original_tilt}° to {cv_system.tilt_angle}°")
        else:
            print(f"✗ 'w' key broken after resolution toggle: tilt remained at {cv_system.tilt_angle}°")
            return False
            
        # Test 's' key again
        original_tilt = cv_system.tilt_angle
        cv_system.handle_keyboard_input('s')
        if cv_system.tilt_angle < original_tilt:
            print(f"✓ 's' key still works: tilt decreased from {original_tilt}° to {cv_system.tilt_angle}°")
        else:
            print(f"✗ 's' key broken after resolution toggle: tilt remained at {cv_system.tilt_angle}°")
            return False
            
        # Test a few frames can be processed correctly
        print("\nTesting frame processing after resolution toggle...")
        success_count = 0
        for i in range(5):
            ret, frame = cv_system.process_frame()
            if ret:
                success_count += 1
                time.sleep(0.1)  # Brief pause between frames
                
        if success_count >= 4:  # Allow for one potential failure
            print(f"✓ Frame processing works: {success_count}/5 frames processed successfully")
        else:
            print(f"✗ Frame processing issues: only {success_count}/5 frames processed successfully")
            return False
            
        # Clean up
        cv_system.cleanup()
        print("\n" + "=" * 60)
        print("✓ All tests passed! SEARCH mode works correctly after resolution toggle.")
        return True
        
    if __name__ == "__main__":
        try:
            success = test_resolution_toggle_with_search_mode()
            if success:
                print("\nSUCCESS: The resolution toggle fix works correctly!")
                exit(0)
            else:
                print("\nFAILURE: Issues detected with the fix.")
                exit(1)
        except Exception as e:
            print(f"\nERROR during testing: {e}")
            import traceback
            traceback.print_exc()
            exit(1)
            
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this from the HelloBalls-Host directory")
    exit(1)
