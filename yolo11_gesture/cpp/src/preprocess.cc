#include "preprocess.h"
#include "config.h"
#include <iostream>
#include <string.h>

void preprocess(const cv::Mat& frame, cv::Mat& img_nv12, 
               int input_H, int input_W, 
               float& x_scale, float& y_scale, 
               int& x_shift, int& y_shift) {
    cv::Mat resize_img;
    
    if (PREPROCESS_TYPE == LETTERBOX_TYPE) { // letter box
        x_scale = std::min(1.0 * input_H / frame.rows, 1.0 * input_W / frame.cols);
        y_scale = x_scale;
        if (x_scale <= 0 || y_scale <= 0) {
            std::cerr << "Invalid scale factor." << std::endl;
            return;
        }

        int new_w = frame.cols * x_scale;
        x_shift = (input_W - new_w) / 2;
        int x_other = input_W - new_w - x_shift;

        int new_h = frame.rows * y_scale;
        y_shift = (input_H - new_h) / 2;
        int y_other = input_H - new_h - y_shift;

        cv::Size targetSize(new_w, new_h);
        cv::resize(frame, resize_img, targetSize);
        cv::copyMakeBorder(resize_img, resize_img, y_shift, y_other, x_shift, x_other, 
                          cv::BORDER_CONSTANT, cv::Scalar(127, 127, 127));
    } else if (PREPROCESS_TYPE == RESIZE_TYPE) { // resize
        cv::Size targetSize(input_W, input_H);
        cv::resize(frame, resize_img, targetSize);

        y_scale = 1.0 * input_H / frame.rows;
        x_scale = 1.0 * input_W / frame.cols;
        y_shift = 0;
        x_shift = 0;
    }

    // Convert to YUV420SP (NV12) format
    cv::Mat yuv_mat;
    cv::cvtColor(resize_img, yuv_mat, cv::COLOR_BGR2YUV_I420);
    uint8_t *yuv = yuv_mat.ptr<uint8_t>();
    img_nv12 = cv::Mat(input_H * 3 / 2, input_W, CV_8UC1);
    uint8_t *ynv12 = img_nv12.ptr<uint8_t>();
    int uv_height = input_H / 2;
    int uv_width = input_W / 2;
    int y_size = input_H * input_W;
    
    memcpy(ynv12, yuv, y_size);
    uint8_t *nv12 = ynv12 + y_size;
    uint8_t *u_data = yuv + y_size;
    uint8_t *v_data = u_data + uv_height * uv_width;
    
    for (int i = 0; i < uv_width * uv_height; i++) {
        *nv12++ = *u_data++;
        *nv12++ = *v_data++;
    }
}

void copyToInputTensor(const cv::Mat& img_nv12, hbDNNTensor& input) {
    memcpy(input.sysMem[0].virAddr, img_nv12.ptr<uint8_t>(), img_nv12.total());
    hbSysFlushMem(&input.sysMem[0], HB_SYS_MEM_CACHE_CLEAN);
}