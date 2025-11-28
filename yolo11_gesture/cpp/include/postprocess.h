#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
#include "dnn/hb_dnn.h"
#include "dnn/hb_sys.h"
#include "config.h"

// Performs YOLO11 detection postprocessing
void postprocess(hbDNNTensor* output, int order[6],
                int input_H, int input_W,
                std::vector<std::vector<cv::Rect2d>>& bboxes,
                std::vector<std::vector<float>>& scores);

// Performs NMS and drawing on the frame
void drawDetections(cv::Mat& frame,
                   const std::vector<std::vector<cv::Rect2d>>& bboxes,
                   const std::vector<std::vector<float>>& scores,
                   const std::vector<std::vector<int>>& indices,
                   float x_scale, float y_scale, int x_shift, int y_shift);

// Apply Non-Maximum Suppression
void applyNMS(const std::vector<std::vector<cv::Rect2d>>& bboxes,
             const std::vector<std::vector<float>>& scores,
             std::vector<std::vector<int>>& indices);