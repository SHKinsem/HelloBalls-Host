#pragma once

#include <iostream>
#include <chrono>
#include <string>
#include <opencv2/opencv.hpp>

#include "config.h"

// Error checking macro for RDK functions
#define RDK_CHECK_SUCCESS(value, errmsg)                                         \
    do                                                                           \
    {                                                                            \
        auto ret_code = value;                                                   \
        if (ret_code != 0)                                                       \
        {                                                                        \
            std::cout << "[ERROR] " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::cout << errmsg << ", error code:" << ret_code << std::endl;     \
            return ret_code;                                                     \
        }                                                                        \
    } while (0);

// Class names for COCO dataset
extern std::vector<std::string> object_names;

// FPS calculation class
class FPSCalculator {
private:
    std::chrono::time_point<std::chrono::system_clock> lastTime;
    float fps;
    float smoothFactor;

public:
    FPSCalculator(float smooth = FPS_SMOOTH_FACTOR);
    float update();
    float getFPS() const;
};

// Camera utility functions
void toggleResolution(cv::VideoCapture &cap, bool &is720p);

// Helper for detecting target classes (person/sports ball)
bool isTargetClass(int cls_id);