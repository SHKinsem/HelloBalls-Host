/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
Copyright (c) 2024，WuChao D-Robotics.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

// Note: This program runs on RDK board for real-time camera inference.

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

#include "include/config.h"
#include "include/model.h"
#include "include/utils.h"
#include "include/preprocess.h"
#include "include/postprocess.h"
#include "include/pid_controller.h"

// Function to find ball positions and update motor speeds
bool processBallDetection(
    const std::vector<std::vector<cv::Rect2d>>& bboxes,
    const std::vector<std::vector<float>>& scores,
    const std::vector<std::vector<int>>& indices,
    float x_scale, float y_scale, int x_shift, int y_shift,
    PIDController& pidController,
    cv::Mat& frame);

int main(int argc, char* argv[]) {
    // Process command line arguments for serial port
    // std::string portName = "";
    // float kp = PID_KP, ki = PID_KI, kd = PID_KD;
    
    // for (int i = 1; i < argc; i++) {
    //     std::string arg = argv[i];
    //     if (arg == "--port" && i + 1 < argc) {
    //         portName = argv[++i];
    //     } else if (arg == "--kp" && i + 1 < argc) {
    //         kp = std::stof(argv[++i]);
    //     } else if (arg == "--ki" && i + 1 < argc) {
    //         ki = std::stof(argv[++i]);
    //     } else if (arg == "--kd" && i + 1 < argc) {
    //         kd = std::stof(argv[++i]);
    //     }
    // }
    
    // Initialize PID controller
    // PIDController pidController(kp, ki, kd);
    // std::cout << "PID Parameters: Kp=" << kp << ", Ki=" << ki << ", Kd=" << kd << std::endl;
    
    // Initialize serial communication
    // SerialComm serialComm;
    // if (!portName.empty()) {
    //     std::cout << "Attempting to connect to specified port: " << portName << std::endl;
    //     if (!serialComm.connect(portName)) {
    //         std::cout << "Failed to connect to " << portName << ", will try auto-discovery..." << std::endl;
    //         serialComm.connect();
    //     }
    // } else {
    //     std::cout << "No port specified, trying auto-discovery..." << std::endl;
    //     serialComm.connect();
    // }
    
    // 1. Model loading and initialization
    hbPackedDNNHandle_t packed_dnn_handle;
    hbDNNHandle_t dnn_handle;
    int input_H, input_W;
    int order[6] = {0, 1, 2, 3, 4, 5}; // Default order
    
    if (loadModel(packed_dnn_handle, dnn_handle, input_H, input_W, order) != 0) {
        std::cerr << "Failed to load model" << std::endl;
        return -1;
    }
    
    // Get output count
    int32_t output_count = 0;
    if (hbDNNGetOutputCount(&output_count, dnn_handle) != 0) {
        std::cerr << "Failed to get output count" << std::endl;
        return -1;
    }

    // 2. Allocate tensors for input and output
    hbDNNTensor input;
    hbDNNTensor* output = nullptr;
    if (prepareTensors(input, output, dnn_handle, output_count, input_H, input_W) != 0) {
        std::cerr << "Failed to prepare tensors" << std::endl;
        return -1;
    }

    // 3. Open camera
    cv::VideoCapture cap(CAMERA_ID);
    if (!cap.isOpened()) {
        std::cerr << "Error: Unable to open camera" << std::endl;
        cleanupModel(packed_dnn_handle, input, output, output_count);
        return -1;
    }
    
    // Set camera resolution
    cap.set(cv::CAP_PROP_FRAME_WIDTH, DEFAULT_CAM_WIDTH);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, DEFAULT_CAM_HEIGHT);
    int actual_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int actual_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    std::cout << "Camera resolution: " << actual_width << "x" << actual_height << std::endl;

    std::cout << "Camera opened successfully." << std::endl;
    std::cout << "Press 'q' to quit, 'r' to toggle resolution, 'p' to reset PID." << std::endl;

    // Track resolution state
    bool is720p = true;

    // Create FPS calculator
    FPSCalculator fpsCalc;

    // Main loop
    cv::Mat frame;
    while (true) {
        // Get a frame from camera
        if (!cap.read(frame)) {
            std::cerr << "Error: Could not read frame from camera" << std::endl;
            break;
        }
        
        if (frame.empty()) {
            std::cerr << "Error: Empty frame received" << std::endl;
            continue;
        }

        // Pre-process the frame
        cv::Mat img_nv12;
        float x_scale = 1.0, y_scale = 1.0;
        int x_shift = 0, y_shift = 0;
        preprocess(frame, img_nv12, input_H, input_W, x_scale, y_scale, x_shift, y_shift);
        
        // Copy to input tensor
        copyToInputTensor(img_nv12, input);
        
        // Run inference
        hbDNNTaskHandle_t task_handle = nullptr;
        if (runInference(dnn_handle, input, output, task_handle) != 0) {
            std::cerr << "Inference failed" << std::endl;
            hbDNNReleaseTask(task_handle);
            continue;
        }

        // Post-process the results
        std::vector<std::vector<cv::Rect2d>> bboxes;
        std::vector<std::vector<float>> scores;
        postprocess(output, order, input_H, input_W, bboxes, scores);
        
        // Apply NMS
        std::vector<std::vector<int>> indices;
        applyNMS(bboxes, scores, indices);
        
        // Draw detections
        drawDetections(frame, bboxes, scores, indices, x_scale, y_scale, x_shift, y_shift);
        
        // Process ball detection and update motor speeds
        // processBallDetection(bboxes, scores, indices, x_scale, y_scale, 
        //                    x_shift, y_shift, pidController, frame);
        
        // Calculate and display FPS
        float fps = fpsCalc.update();
        std::string fps_text = "FPS: " + std::to_string(static_cast<int>(fps));
        cv::putText(frame, fps_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);

        // Display the image
        cv::imshow("YOLO11 Ball Tracking", frame);

        // Release task
        hbDNNReleaseTask(task_handle);

        // Check for key press
        int key = cv::waitKey(1);
        if (key == 'q') {
            std::cout << "Exiting..." << std::endl;
            break;
        } else if (key == 'r') {
            toggleResolution(cap, is720p);}
        // } else if (key == 'p') {
        //     std::cout << "Resetting PID controller..." << std::endl;
        //     pidController.reset();
        // }
    }

    // Cleanup resources
    cleanupModel(packed_dnn_handle, input, output, output_count);
    
    // Release camera
    cap.release();
    cv::destroyAllWindows();

    std::cout << "Program ended successfully" << std::endl;
    return 0;
}

bool processBallDetection(
    const std::vector<std::vector<cv::Rect2d>>& bboxes,
    const std::vector<std::vector<float>>& scores,
    const std::vector<std::vector<int>>& indices,
    float x_scale, float y_scale, int x_shift, int y_shift,
    PIDController& pidController,
    cv::Mat& frame) {
    
    // Check for sports ball class (usually 32 in COCO dataset)
    const int ballClassId = 32;  // sports ball in COCO
    
    // if (ballClassId >= bboxes.size() || indices[ballClassId].empty()) {
    //     // No ball detected, stop motors
    //     return false;
    // }
    
    // Get the highest confidence ball detection
    int bestBallIdx = indices[ballClassId][0];
    int frameWidth = frame.cols;
    int frameHeight = frame.rows;
    
    // Convert bounding box to actual frame coordinates
    float width = bboxes[ballClassId][bestBallIdx].width / x_scale;
    float height = bboxes[ballClassId][bestBallIdx].height / y_scale;
    float y1 = (bboxes[ballClassId][bestBallIdx].y - y_shift) / y_scale; // Remove inversion
    float x1 = (bboxes[ballClassId][bestBallIdx].x - x_shift) / x_scale;
    // Calculate ball center
    float ball_center_x = x1 + width / 2;
    float ball_center_y = y1 + height / 2; // Adjust for non-inverted y-coordinate
    float correct_y = frameHeight - y1; // Correct for inverted y-coordinate
    
    // Get frame dimensions
    
    // Calculate target position (bottom center of frame)
    float target_x = frameWidth / 2;
    float target_y = frameHeight * 0.9;  // 90% down the frame (bottom area)
    
    // Draw target position
    cv::circle(frame, cv::Point(target_x, target_y), 10, cv::Scalar(255, 255, 0), 2);
    cv::line(frame, cv::Point(target_x - 15, target_y), cv::Point(target_x + 15, target_y), cv::Scalar(255, 255, 0), 2);
    cv::line(frame, cv::Point(target_x, target_y - 15), cv::Point(target_x, target_y + 15), cv::Scalar(255, 255, 0), 2);
    
    // Draw line from ball to target
    cv::line(frame, cv::Point(ball_center_x, ball_center_y), cv::Point(target_x, target_y), cv::Scalar(0, 255, 255), 2);
    
    // Calculate error (distance from target position)
    float x_error = ball_center_x - target_x;
    
    // Use PID controller to calculate steering value
    float steering = pidController.calculate(x_error);
    
    // Convert steering to motor speeds
    // Positive error (ball right of center) -> turn right -> left motor faster
    // Negative error (ball left of center) -> turn left -> right motor faster
    int baseSpeed = 50;  // Base forward speed
    int leftSpeed = baseSpeed;
    int rightSpeed = baseSpeed;
    
    if (steering > 0) {
        // Ball is to the right, need to turn right
        leftSpeed = baseSpeed + abs(steering);
        rightSpeed = baseSpeed - abs(steering);
    } else {
        // Ball is to the left, need to turn left
        leftSpeed = baseSpeed - abs(steering);
        rightSpeed = baseSpeed + abs(steering);
    }
    
    // Ensure speeds are within bounds
    leftSpeed = std::max(std::min(leftSpeed, PID_MAX_OUTPUT), PID_MIN_OUTPUT);
    rightSpeed = std::max(std::min(rightSpeed, PID_MAX_OUTPUT), PID_MIN_OUTPUT);
    
    // Display motor speeds on frame
    std::string motorText = "Motors L:" + std::to_string(leftSpeed) + " R:" + std::to_string(rightSpeed);
    cv::putText(frame, motorText, cv::Point(10, 90), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    
    // Display error and PID output
    std::string errorText = "Error: " + std::to_string(int(x_error)) + " PID: " + std::to_string(int(steering));
    cv::putText(frame, errorText, cv::Point(10, 120), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    
    return true;
}