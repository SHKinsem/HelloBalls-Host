#pragma once

#include "config.h" // Include config.h first for PID constants
#include <iostream>
#include <ctime>

class PIDController {
private:
    float kp;             // Proportional gain
    float ki;             // Integral gain
    float kd;             // Derivative gain
    float maxOutput;      // Maximum output value
    float minOutput;      // Minimum output value
    float integral;       // Accumulated error
    float previousError;  // Error from previous calculation
    float previousTime;   // Time of previous calculation

public:
    // Constructor with default values
    PIDController(float kp = PID_KP, float ki = PID_KI, float kd = PID_KD, 
                 float maxOutput = PID_MAX_OUTPUT, float minOutput = PID_MIN_OUTPUT);
    
    // Reset the PID controller
    void reset();
    
    // Calculate the control output based on the error
    float calculate(float error);
    
    // Getter and setter methods for PID parameters
    void setKp(float value) { kp = value; }
    void setKi(float value) { ki = value; }
    void setKd(float value) { kd = value; }
    void setMaxOutput(float value) { maxOutput = value; }
    void setMinOutput(float value) { minOutput = value; }
    
    float getKp() const { return kp; }
    float getKi() const { return ki; }
    float getKd() const { return kd; }
};