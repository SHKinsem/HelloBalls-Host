#include "pid_controller.h"
#include <algorithm> // For std::max/min
#include <iostream>
#include <sys/time.h>

// Get current time in seconds
static double getTimeInSeconds() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

PIDController::PIDController(float kp, float ki, float kd, float maxOutput, float minOutput)
    : kp(kp), ki(ki), kd(kd), maxOutput(maxOutput), minOutput(minOutput) {
    reset();
}

void PIDController::reset() {
    integral = 0.0f;
    previousError = 0.0f;
    previousTime = getTimeInSeconds();
}

float PIDController::calculate(float error) {
    // Get current time and calculate time difference
    float currentTime = getTimeInSeconds();
    float deltaTime = currentTime - previousTime;
    
    // Avoid division by zero
    if (deltaTime <= 0.0f) {
        deltaTime = 0.001f;  // 1ms minimum time step
    }
    
    // Calculate integral and derivative terms
    integral += error * deltaTime;
    float derivative = (error - previousError) / deltaTime;
    
    // Calculate output
    float output = kp * error + ki * integral + kd * derivative;
    
    // Clamp output to limits
    output = std::max(std::min(output, maxOutput), minOutput);
    
    // Save current error and time for next calculation
    previousError = error;
    previousTime = currentTime;
    
    return output;
}