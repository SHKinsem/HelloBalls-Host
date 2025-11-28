# HelloBalls Person Centering Testing Guide

## 🎯 Testing Objectives
Verify that the robot properly turns to face detected persons in SEARCH mode using the improved differential steering implementation.

## 🧪 Test Scenarios

### 1. PID Controller Response Test
```bash
cd /home/sunrise/Documents/HelloBalls-Host/scripts
python3 test_person_centering.py --pid-only
```

**Expected Results:**
- Negative errors (person left) → Negative PID output → Turn left
- Positive errors (person right) → Positive PID output → Turn right
- Zero error → Zero output → Stop turning

### 2. Live Person Centering Test
```bash
cd /home/sunrise/Documents/HelloBalls-Host/scripts
python3 test_person_centering.py
```

**Expected Behaviors:**
- Robot starts in STOP mode
- Switch to SEARCH mode enables person centering
- Robot turns toward detected persons
- Robot stops when person is centered (±50 pixels)
- Smooth differential steering (not jerky movements)

### 3. Motor Direction Verification

#### When Person is to the RIGHT (positive error):
- **Left Motor**: Forward (positive speed)
- **Right Motor**: Reverse (negative speed)
- **Result**: Robot turns right (clockwise)

#### When Person is to the LEFT (negative error):
- **Left Motor**: Reverse (negative speed)
- **Right Motor**: Forward (positive speed)
- **Result**: Robot turns left (counter-clockwise)

### 4. Speed and Control Verification

#### Minimum Speed Requirements:
- Minimum turning speed: 80 (to overcome static friction)
- Maximum centering speed: ±150 (for safe operation)
- PID output range: ±150 (controlled by max_output)

#### Control Responsiveness:
- Robot should respond within 1-2 frames of person detection
- Smooth acceleration/deceleration (no sudden jerks)
- Stable centering (minimal oscillation when person is centered)

## 🔧 Troubleshooting Guide

### Problem: Robot doesn't turn at all
**Possible Causes:**
- Serial communication issues
- Motor speeds below minimum threshold
- Robot not in SEARCH mode
- Auto person centering disabled

**Solutions:**
1. Check serial connection: `ls /dev/tty*`
2. Verify robot state: Should be `ROBOT_STATE_SEARCH` (4)
3. Ensure `auto_person_centering = True`
4. Check motor command output in console

### Problem: Robot turns wrong direction
**Possible Causes:**
- Motor wiring reversed
- Incorrect differential steering logic
- Sign error in PID calculation

**Solutions:**
1. Verify motor control logic in `control_robot()` method
2. Test with simple manual commands
3. Check error calculation: `error_x = person_center_x - frame_center_x`

### Problem: Robot oscillates around person
**Possible Causes:**
- PID gains too high
- Centering threshold too small
- No derivative damping

**Solutions:**
1. Reduce Kp gain (currently 200)
2. Increase centering threshold (currently 50 pixels)
3. Increase Kd gain for damping (currently 50)

### Problem: Robot too slow to respond
**Possible Causes:**
- PID gains too low
- Minimum speed threshold too high
- Processing lag

**Solutions:**
1. Increase Kp gain
2. Reduce minimum turn speed (currently 80)
3. Check frame processing rate

## 📊 Performance Metrics

### Success Criteria:
- [ ] Robot turns toward person within 2 seconds
- [ ] Robot centers person within ±50 pixels
- [ ] No continuous oscillation when centered
- [ ] Smooth motor control (no jerky movements)
- [ ] Proper direction (right turn for person on right)
- [ ] Motor speeds stay within safe limits (±150)

### Timing Benchmarks:
- Detection to response: < 200ms
- Centering completion: < 5 seconds (for 90° turn)
- Position holding: ±25 pixels steady state

## 🛠️ Configuration Parameters

### Current PID Settings:
```python
person_centering_pid = PIDController(
    kp=200,      # Proportional gain
    ki=5.0,      # Integral gain  
    kd=50.0,     # Derivative gain
    max_output=150,  # Speed limit
    setpoint=0   # Target: centered
)
```

### Control Thresholds:
```python
centering_threshold = 50     # Pixels for "centered"
min_turn_speed = 80         # Minimum effective speed
max_centering_speed = 150   # Maximum safe speed
```

## 🔄 Manual Override Testing

### Test Manual Control:
1. Enter SEARCH mode
2. Use manual controls (WASD keys)
3. Verify manual override temporarily disables auto-centering
4. Confirm auto-centering resumes after override timeout

### Expected Behavior:
- Manual input creates 3-second override window
- Auto-centering pauses during override
- Console shows "Manual override active" message
- Auto-centering resumes automatically

## 📝 Test Log Template

```
Date: ___________
Tester: __________

[ ] PID Response Test Passed
[ ] Live Centering Test Passed
[ ] Motor Direction Correct
[ ] Speed Control Appropriate
[ ] No Oscillation Issues
[ ] Manual Override Works

Issues Found:
_________________________________
_________________________________
_________________________________

Recommended Adjustments:
_________________________________
_________________________________
_________________________________
```

## 🚀 Next Steps After Testing

1. **If tests pass**: Ready for real-world scenarios
2. **If minor issues**: Adjust PID parameters
3. **If major issues**: Review differential steering logic
4. **Performance optimization**: Fine-tune thresholds and gains
