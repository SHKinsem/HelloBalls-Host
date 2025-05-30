# State 3 Transition Flow - Seamless Operation

## Overview
When transitioning to state 3, the robot maintains continuous operation by using the previous state until tilt angle input is complete.

## Transition Flow Diagram

```
Initial State: State 1 (or any other state)
│
│ User presses '3'
│
├─ Display: "State changed to: 3"
├─ Display: "Continuing with previous state (1) while waiting for tilt input..."
├─ Start tilt input thread (non-blocking)
│
├─ Main Loop Continues (50Hz):
│  │
│  ├─ Commands sent with: state=1, tilt=0  ← Previous state maintained
│  ├─ Robot responds normally to WASD keys
│  ├─ Status: "Using prev state (1) - waiting for tilt input..."
│  │
│  └─ Check for tilt input completion (non-blocking)
│
├─ User enters tilt angle (e.g., "45")
│
├─ Tilt input complete:
│  ├─ Display: "Tilt input complete. Now using state 3 with tilt angle 45"
│  └─ Switch to actual state 3 operation
│
└─ Final State: State 3 with tilt angle
   │
   └─ Commands sent with: state=3, tilt=45
```

## Key Benefits

### ✅ **Seamless Operation**
- No interruption in robot control during state transition
- 50Hz communication rate maintained throughout
- Previous state behavior continues until ready

### ✅ **User Experience**
- Robot remains responsive during configuration
- Clear status messages show what's happening
- No "dead time" where robot stops responding

### ✅ **Technical Robustness**
- Threading prevents blocking operations
- Queue-based communication ensures thread safety
- Proper terminal handling and cleanup
- Graceful error handling for all scenarios

## Example Session

```
Action: STOP | Command: state=1, m1=0, m2=0, tilt=0 | Input: 

[User presses '3']
State changed to: 3
Continuing with previous state (1) while waiting for tilt input...

[Input thread starts, prompt appears]
Enter tilt angle for state 3 (current: 0): 

Action: STOP | Command: state=1, m1=0, m2=0, tilt=0 | Using prev state (1) - waiting for tilt input... | Input: 

[User presses 'w' - robot still responds normally]
Action: FORWARD | Command: state=1, m1=800, m2=800, tilt=0 | Using prev state (1) - waiting for tilt input... | Input: 

[User types "45" and presses Enter]
Tilt angle set to: 45
Returning to keyboard control mode...
Tilt input complete. Now using state 3 with tilt angle 45

[Robot now operates in state 3]
Action: FORWARD | Command: state=3, m1=800, m2=800, tilt=45 | Tilt: 45 | Input: 
```

## State Behavior Summary

| Scenario | Effective State | Tilt Angle | Description |
|----------|----------------|------------|-------------|
| Normal Operation (State 0-2) | Current State | 0 | Standard operation |
| Normal Operation (State 3) | 3 | User-defined | Using configured tilt angle |
| Transitioning to State 3 | Previous State | 0 | Maintains previous operation during input |
| State 3 Input Complete | 3 | User-defined | Switches to state 3 with new tilt |

This design ensures that the robot never stops operating and users can configure tilt angles without any service interruption.
