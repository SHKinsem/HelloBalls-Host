#include "utils.h"
#include "config.h"

// COCO Names
std::vector<std::string> object_names = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", 
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", 
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", 
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", 
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", 
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", 
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", 
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", 
    "scissors", "teddy bear", "hair drier", "toothbrush"
};

FPSCalculator::FPSCalculator(float smooth) : fps(0.0f), smoothFactor(smooth) {
    lastTime = std::chrono::system_clock::now();
}

float FPSCalculator::update() {
    auto currentTime = std::chrono::system_clock::now();
    float newFPS = 1000.0f / std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - lastTime).count();
    
    // Use smoothing factor for weighted average
    if (fps == 0.0f) {
        fps = newFPS;
    } else {
        fps = smoothFactor * fps + (1.0f - smoothFactor) * newFPS;
    }
    
    lastTime = currentTime;
    return fps;
}

float FPSCalculator::getFPS() const {
    return fps;
}

void toggleResolution(cv::VideoCapture &cap, bool &is720p) {
    if (is720p) {
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 712);
        std::cout << "Resolution changed to 1280x712" << std::endl;
    } else {
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
        std::cout << "Resolution changed to 1280x720" << std::endl;
    }
    is720p = !is720p;
}

bool isTargetClass(int cls_id) {
    return cls_id == 0 || // person
           cls_id == 32;  // sports ball
}