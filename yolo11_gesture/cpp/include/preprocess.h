#pragma once

#include <opencv2/opencv.hpp>
#include "dnn/hb_dnn.h"
#include "dnn/hb_sys.h"

// Preprocess image for model input
void preprocess(const cv::Mat& frame, cv::Mat& img_nv12, 
               int input_H, int input_W, 
               float& x_scale, float& y_scale, 
               int& x_shift, int& y_shift);

// Copy preprocessed data to tensor
void copyToInputTensor(const cv::Mat& img_nv12, hbDNNTensor& input);