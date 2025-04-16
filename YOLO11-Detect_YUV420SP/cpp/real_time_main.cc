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

// 注意: 此程序在RDK板端端运行，用于实时摄像头推理
// Attention: This program runs on RDK board for real-time camera inference.

// D-Robotics *.bin 模型路径
// Path of D-Robotics *.bin model.
#define MODEL_PATH "../../ptq_models/yolo11n_detect_bayese_640x640_nv12_int16softmax_modified.bin"

// 摄像头设备ID，默认为0
// Camera device ID, default is 0
#define CAMERA_ID 0

// 前处理方式选择, 0:Resize, 1:LetterBox
// Preprocessing method selection, 0: Resize, 1: LetterBox
#define RESIZE_TYPE 0
#define LETTERBOX_TYPE 1
#define PREPROCESS_TYPE LETTERBOX_TYPE

// 模型的类别数量, 默认80
// Number of classes in the model, default is 80
#define CLASSES_NUM 80

// NMS的阈值, 默认0.45
// Non-Maximum Suppression (NMS) threshold, default is 0.45
#define NMS_THRESHOLD 0.7

// 分数阈值, 默认0.25
// Score threshold, default is 0.25
#define SCORE_THRESHOLD 0.25

// NMS选取的前K个框数, 默认300
// Number of top-K boxes selected by NMS, default is 300
#define NMS_TOP_K 300

// 控制回归部分离散化程度的超参数, 默认16
// A hyperparameter that controls the discretization level of the regression part, default is 16
#define REG 16

// 绘制标签的字体尺寸, 默认1.0
// Font size for drawing labels, default is 1.0.
#define FONT_SIZE 1.0

// 绘制标签的字体粗细, 默认 1.0
// Font thickness for drawing labels, default is 1.0.
#define FONT_THICKNESS 1.0

// 绘制矩形框的线宽, 默认2.0
// Line width for drawing bounding boxes, default is 2.0.
#define LINE_SIZE 2.0

// 帧率计算的平滑因子（值越大越平滑）
// Smoothing factor for FPS calculation (higher value, smoother)
#define FPS_SMOOTH_FACTOR 0.9

// C/C++ Standard Librarys
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <string>

// Thrid Party Librarys
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

// RDK BPU libDNN API
#include "dnn/hb_dnn.h"
#include "dnn/hb_dnn_ext.h"
#include "dnn/plugin/hb_dnn_layer.h"
#include "dnn/plugin/hb_dnn_plugin.h"
#include "dnn/hb_sys.h"

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

// COCO Names
std::vector<std::string> object_names = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};

// 用于存储和计算帧率的类
class FPSCalculator {
private:
    std::chrono::time_point<std::chrono::system_clock> lastTime;
    float fps;
    float smoothFactor;

public:
    FPSCalculator(float smooth = FPS_SMOOTH_FACTOR) : fps(0.0f), smoothFactor(smooth) {
        lastTime = std::chrono::system_clock::now();
    }

    float update() {
        auto currentTime = std::chrono::system_clock::now();
        float newFPS = 1000.0f / std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - lastTime).count();
        
        // 使用平滑因子进行加权平均
        if (fps == 0.0f) {
            fps = newFPS;
        } else {
            fps = smoothFactor * fps + (1.0f - smoothFactor) * newFPS;
        }
        
        lastTime = currentTime;
        return fps;
    }

    float getFPS() const {
        return fps;
    }
};

int main()
{
    // 0. 加载bin模型
    auto begin_time = std::chrono::system_clock::now();
    hbPackedDNNHandle_t packed_dnn_handle;
    const char *model_file_name = MODEL_PATH;
    RDK_CHECK_SUCCESS(
        hbDNNInitializeFromFiles(&packed_dnn_handle, &model_file_name, 1),
        "hbDNNInitializeFromFiles failed");
    std::cout << "\033[31m Load D-Robotics Quantize model time = " << std::fixed << std::setprecision(2) << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - begin_time).count() / 1000.0 << " ms\033[0m" << std::endl;

    // 1. 打印相关版本信息
    std::cout << "[INFO] OpenCV Version: " << CV_VERSION << std::endl;
    std::cout << "[INFO] MODEL_PATH: " << MODEL_PATH << std::endl;
    std::cout << "[INFO] CLASSES_NUM: " << CLASSES_NUM << std::endl;
    std::cout << "[INFO] NMS_THRESHOLD: " << NMS_THRESHOLD << std::endl;
    std::cout << "[INFO] SCORE_THRESHOLD: " << SCORE_THRESHOLD << std::endl;
    std::cout << "[INFO] CAMERA_ID: " << CAMERA_ID << std::endl;

    // 2. 打印模型信息

    // 2.1 模型名称
    const char **model_name_list;
    int model_count = 0;
    RDK_CHECK_SUCCESS(
        hbDNNGetModelNameList(&model_name_list, &model_count, packed_dnn_handle),
        "hbDNNGetModelNameList failed");

    // 如果这个bin模型有多个打包，则只使用第一个，一般只有一个
    if (model_count > 1) {
        std::cout << "This model file have more than 1 model, only use model 0.";
    }
    const char *model_name = model_name_list[0];
    std::cout << "[model name]: " << model_name << std::endl;

    // 2.2 获得Packed模型的第一个模型的handle
    hbDNNHandle_t dnn_handle;
    RDK_CHECK_SUCCESS(
        hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle, model_name),
        "hbDNNGetModelHandle failed");

    // 2.3 模型输入检查
    int32_t input_count = 0;
    RDK_CHECK_SUCCESS(
        hbDNNGetInputCount(&input_count, dnn_handle),
        "hbDNNGetInputCount failed");

    hbDNNTensorProperties input_properties;
    RDK_CHECK_SUCCESS(
        hbDNNGetInputTensorProperties(&input_properties, dnn_handle, 0),
        "hbDNNGetInputTensorProperties failed");

    // 2.3.1 D-Robotics YOLO11 *.bin 模型应该为单输入
    if (input_count > 1) {
        std::cout << "Your Model have more than 1 input, please check!" << std::endl;
        return -1;
    }

    // 2.3.2 D-Robotics YOLO11 *.bin 模型输入Tensor类型应为nv12
    if (input_properties.tensorType == HB_DNN_IMG_TYPE_NV12) {
        std::cout << "input tensor type: HB_DNN_IMG_TYPE_NV12" << std::endl;
    } else {
        std::cout << "input tensor type is not HB_DNN_IMG_TYPE_NV12, please check!" << std::endl;
        return -1;
    }

    // 2.3.3 D-Robotics YOLO11 *.bin 模型输入Tensor数据排布应为NCHW
    if (input_properties.tensorLayout == HB_DNN_LAYOUT_NCHW) {
        std::cout << "input tensor layout: HB_DNN_LAYOUT_NCHW" << std::endl;
    } else {
        std::cout << "input tensor layout is not HB_DNN_LAYOUT_NCHW, please check!" << std::endl;
        return -1;
    }

    // 2.3.4 D-Robotics YOLO11 *.bin 模型输入Tensor数据的valid shape应为(1,3,H,W)
    int32_t input_H, input_W;
    if (input_properties.validShape.numDimensions == 4) {
        input_H = input_properties.validShape.dimensionSize[2];
        input_W = input_properties.validShape.dimensionSize[3];
        std::cout << "input tensor valid shape: (" << input_properties.validShape.dimensionSize[0];
        std::cout << ", " << input_properties.validShape.dimensionSize[1];
        std::cout << ", " << input_H;
        std::cout << ", " << input_W << ")" << std::endl;
    } else {
        std::cout << "input tensor validShape.numDimensions is not 4 such as (1,3,640,640), please check!" << std::endl;
        return -1;
    }

    // 2.4 模型输出检查
    int32_t output_count = 0;
    RDK_CHECK_SUCCESS(
        hbDNNGetOutputCount(&output_count, dnn_handle),
        "hbDNNGetOutputCount failed");

    // 2.4.1 D-Robotics YOLO11 *.bin 模型应该有6个输出
    if (output_count == 6) {
        for (int i = 0; i < 6; i++) {
            hbDNNTensorProperties output_properties;
            RDK_CHECK_SUCCESS(
                hbDNNGetOutputTensorProperties(&output_properties, dnn_handle, i),
                "hbDNNGetOutputTensorProperties failed");
            std::cout << "output[" << i << "] ";
            std::cout << "valid shape: (" << output_properties.validShape.dimensionSize[0];
            std::cout << ", " << output_properties.validShape.dimensionSize[1];
            std::cout << ", " << output_properties.validShape.dimensionSize[2];
            std::cout << ", " << output_properties.validShape.dimensionSize[3] << "), ";
            if (output_properties.quantiType == SHIFT)
                std::cout << "QuantiType: SHIFT" << std::endl;
            if (output_properties.quantiType == SCALE)
                std::cout << "QuantiType: SCALE" << std::endl;
            if (output_properties.quantiType == NONE)
                std::cout << "QuantiType: NONE" << std::endl;
        }
    } else {
        std::cout << "Your Model's outputs num is not 6, please check!" << std::endl;
        return -1;
    }

    // 2.4.2 调整输出头顺序的映射
    int order[6] = {0, 1, 2, 3, 4, 5};
    int32_t H_8 = input_H / 8;
    int32_t H_16 = input_H / 16;
    int32_t H_32 = input_H / 32;
    int32_t W_8 = input_W / 8;
    int32_t W_16 = input_W / 16;
    int32_t W_32 = input_W / 32;
    int32_t order_we_want[6][3] = {
        {H_8, W_8, CLASSES_NUM},   // output[order[3]]: (1, H // 8,  W // 8,  CLASSES_NUM)
        {H_8, W_8, 64},            // output[order[0]]: (1, H // 8,  W // 8,  64)
        {H_16, W_16, CLASSES_NUM}, // output[order[4]]: (1, H // 16, W // 16, CLASSES_NUM)
        {H_16, W_16, 64},          // output[order[1]]: (1, H // 16, W // 16, 64)
        {H_32, W_32, CLASSES_NUM}, // output[order[5]]: (1, H // 32, W // 32, CLASSES_NUM)
        {H_32, W_32, 64},          // output[order[2]]: (1, H // 32, W // 32, 64)
    };
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            hbDNNTensorProperties output_properties;
            RDK_CHECK_SUCCESS(
                hbDNNGetOutputTensorProperties(&output_properties, dnn_handle, j),
                "hbDNNGetOutputTensorProperties failed");
            int32_t h = output_properties.validShape.dimensionSize[1];
            int32_t w = output_properties.validShape.dimensionSize[2];
            int32_t c = output_properties.validShape.dimensionSize[3];
            if (h == order_we_want[i][0] && w == order_we_want[i][1] && c == order_we_want[i][2]) {
                order[i] = j;
                break;
            }
        }
    }

    // 2.4.3 打印并检查调整后的输出头顺序的映射
    if (order[0] + order[1] + order[2] + order[3] + order[4] + order[5] == 0 + 1 + 2 + 3 + 4 + 5) {
        std::cout << "Outputs order check SUCCESS, continue." << std::endl;
        std::cout << "order = {";
        for (int i = 0; i < 6; i++) {
            std::cout << order[i] << ", ";
        }
        std::cout << "}" << std::endl;
    } else {
        std::cout << "Outputs order check FAILED, use default" << std::endl;
        for (int i = 0; i < 6; i++)
            order[i] = i;
    }

    // 3. 打开摄像头
    cv::VideoCapture cap(CAMERA_ID);
    if (!cap.isOpened()) {
        std::cerr << "Error: Unable to open camera" << std::endl;
        return -1;
    }
    
    // 设置摄像头分辨率（如果支持）
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    
    std::cout << "Camera opened successfully." << std::endl;
    std::cout << "Press 'q' to quit." << std::endl;

    // 4. 准备模型输出数据的空间
    hbDNNTensor *output = new hbDNNTensor[output_count];
    for (int i = 0; i < 6; i++) {
        hbDNNTensorProperties &output_properties = output[i].properties;
        hbDNNGetOutputTensorProperties(&output_properties, dnn_handle, i);
        int out_aligned_size = output_properties.alignedByteSize;
        hbSysMem &mem = output[i].sysMem[0];
        hbSysAllocCachedMem(&mem, out_aligned_size);
    }

    // 5. 预先分配内存给输入张量
    hbDNNTensor input;
    input.properties = input_properties;
    hbSysAllocCachedMem(&input.sysMem[0], int(3 * input_H * input_W / 2));

    // 6. 创建FPS计算器
    FPSCalculator fpsCalc;

    // 主循环
    cv::Mat frame;
    while (true) {
        // 从摄像头获取一帧
        if (!cap.read(frame)) {
            std::cerr << "Error: Could not read frame from camera" << std::endl;
            break;
        }
        
        if (frame.empty()) {
            std::cerr << "Error: Empty frame received" << std::endl;
            continue;
        }

        // 前处理
        float y_scale = 1.0;
        float x_scale = 1.0;
        int x_shift = 0;
        int y_shift = 0;
        cv::Mat resize_img;
        
        if (PREPROCESS_TYPE == LETTERBOX_TYPE) { // letter box
            x_scale = std::min(1.0 * input_H / frame.rows, 1.0 * input_W / frame.cols);
            y_scale = x_scale;
            if (x_scale <= 0 || y_scale <= 0) {
                std::cerr << "Invalid scale factor." << std::endl;
                continue;
            }

            int new_w = frame.cols * x_scale;
            x_shift = (input_W - new_w) / 2;
            int x_other = input_W - new_w - x_shift;

            int new_h = frame.rows * y_scale;
            y_shift = (input_H - new_h) / 2;
            int y_other = input_H - new_h - y_shift;

            cv::Size targetSize(new_w, new_h);
            cv::resize(frame, resize_img, targetSize);
            cv::copyMakeBorder(resize_img, resize_img, y_shift, y_other, x_shift, x_other, cv::BORDER_CONSTANT, cv::Scalar(127, 127, 127));
        } else if (PREPROCESS_TYPE == RESIZE_TYPE) { // resize
            cv::Size targetSize(input_W, input_H);
            cv::resize(frame, resize_img, targetSize);

            y_scale = 1.0 * input_H / frame.rows;
            x_scale = 1.0 * input_W / frame.cols;
            y_shift = 0;
            x_shift = 0;
        }

        // 转换为YUV420SP (NV12)格式
        cv::Mat img_nv12;
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

        // 将准备好的输入数据放入hbDNNTensor
        memcpy(input.sysMem[0].virAddr, ynv12, int(3 * input_H * input_W / 2));
        hbSysFlushMem(&input.sysMem[0], HB_SYS_MEM_CACHE_CLEAN);

        // 推理
        hbDNNTaskHandle_t task_handle = nullptr;
        hbDNNInferCtrlParam infer_ctrl_param;
        HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&infer_ctrl_param);
        hbDNNInfer(&task_handle, &output, &input, dnn_handle, &infer_ctrl_param);

        // 等待任务结束
        hbDNNWaitTaskDone(task_handle, 0);

        // YOLO11-Detect 后处理
        float CONF_THRES_RAW = -log(1 / SCORE_THRESHOLD - 1);
        std::vector<std::vector<cv::Rect2d>> bboxes(CLASSES_NUM);
        std::vector<std::vector<float>> scores(CLASSES_NUM);

        // 小目标特征图
        if (output[order[0]].properties.quantiType != NONE || output[order[1]].properties.quantiType != SCALE) {
            std::cerr << "Invalid output quantization type" << std::endl;
            hbDNNReleaseTask(task_handle);
            continue;
        }
        
        hbSysFlushMem(&(output[order[0]].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
        hbSysFlushMem(&(output[order[1]].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);

        auto *s_cls_raw = reinterpret_cast<float *>(output[order[0]].sysMem[0].virAddr);
        auto *s_bbox_raw = reinterpret_cast<int32_t *>(output[order[1]].sysMem[0].virAddr);
        auto *s_bbox_scale = reinterpret_cast<float *>(output[order[1]].properties.scale.scaleData);
        for (int h = 0; h < H_8; h++) {
            for (int w = 0; w < W_8; w++) {
                float *cur_s_cls_raw = s_cls_raw;
                int32_t *cur_s_bbox_raw = s_bbox_raw;

                int cls_id = 0;
                for (int i = 1; i < CLASSES_NUM; i++) {
                    if (cur_s_cls_raw[i] > cur_s_cls_raw[cls_id]) {
                        cls_id = i;
                    }
                }

                if (cur_s_cls_raw[cls_id] < CONF_THRES_RAW) {
                    s_cls_raw += CLASSES_NUM;
                    s_bbox_raw += REG * 4;
                    continue;
                }

                float score = 1 / (1 + std::exp(-cur_s_cls_raw[cls_id]));

                float ltrb[4], sum, dfl;
                for (int i = 0; i < 4; i++) {
                    ltrb[i] = 0.;
                    sum = 0.;
                    for (int j = 0; j < REG; j++) {
                        int index_id = REG * i + j;
                        dfl = std::exp(float(cur_s_bbox_raw[index_id]) * s_bbox_scale[index_id]);
                        ltrb[i] += dfl * j;
                        sum += dfl;
                    }
                    ltrb[i] /= sum;
                }

                if (ltrb[2] + ltrb[0] <= 0 || ltrb[3] + ltrb[1] <= 0) {
                    s_cls_raw += CLASSES_NUM;
                    s_bbox_raw += REG * 4;
                    continue;
                }

                float x1 = (w + 0.5 - ltrb[0]) * 8.0;
                float y1 = (h + 0.5 - ltrb[1]) * 8.0;
                float x2 = (w + 0.5 + ltrb[2]) * 8.0;
                float y2 = (h + 0.5 + ltrb[3]) * 8.0;

                bboxes[cls_id].push_back(cv::Rect2d(x1, y1, x2 - x1, y2 - y1));
                scores[cls_id].push_back(score);

                s_cls_raw += CLASSES_NUM;
                s_bbox_raw += REG * 4;
            }
        }

        // 中目标特征图
        if (output[order[2]].properties.quantiType != NONE || output[order[3]].properties.quantiType != SCALE) {
            std::cerr << "Invalid output quantization type" << std::endl;
            hbDNNReleaseTask(task_handle);
            continue;
        }
        
        hbSysFlushMem(&(output[order[2]].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
        hbSysFlushMem(&(output[order[3]].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);

        auto *m_cls_raw = reinterpret_cast<float *>(output[order[2]].sysMem[0].virAddr);
        auto *m_bbox_raw = reinterpret_cast<int32_t *>(output[order[3]].sysMem[0].virAddr);
        auto *m_bbox_scale = reinterpret_cast<float *>(output[order[3]].properties.scale.scaleData);
        for (int h = 0; h < H_16; h++) {
            for (int w = 0; w < W_16; w++) {
                float *cur_m_cls_raw = m_cls_raw;
                int32_t *cur_m_bbox_raw = m_bbox_raw;

                int cls_id = 0;
                for (int i = 1; i < CLASSES_NUM; i++) {
                    if (cur_m_cls_raw[i] > cur_m_cls_raw[cls_id]) {
                        cls_id = i;
                    }
                }

                if (cur_m_cls_raw[cls_id] < CONF_THRES_RAW) {
                    m_cls_raw += CLASSES_NUM;
                    m_bbox_raw += REG * 4;
                    continue;
                }

                float score = 1 / (1 + std::exp(-cur_m_cls_raw[cls_id]));

                float ltrb[4], sum, dfl;
                for (int i = 0; i < 4; i++) {
                    ltrb[i] = 0.;
                    sum = 0.;
                    for (int j = 0; j < REG; j++) {
                        int index_id = REG * i + j;
                        dfl = std::exp(float(cur_m_bbox_raw[index_id]) * m_bbox_scale[index_id]);
                        ltrb[i] += dfl * j;
                        sum += dfl;
                    }
                    ltrb[i] /= sum;
                }

                if (ltrb[2] + ltrb[0] <= 0 || ltrb[3] + ltrb[1] <= 0) {
                    m_cls_raw += CLASSES_NUM;
                    m_bbox_raw += REG * 4;
                    continue;
                }

                float x1 = (w + 0.5 - ltrb[0]) * 16.0;
                float y1 = (h + 0.5 - ltrb[1]) * 16.0;
                float x2 = (w + 0.5 + ltrb[2]) * 16.0;
                float y2 = (h + 0.5 + ltrb[3]) * 16.0;

                bboxes[cls_id].push_back(cv::Rect2d(x1, y1, x2 - x1, y2 - y1));
                scores[cls_id].push_back(score);

                m_cls_raw += CLASSES_NUM;
                m_bbox_raw += REG * 4;
            }
        }

        // 大目标特征图
        if (output[order[4]].properties.quantiType != NONE || output[order[5]].properties.quantiType != SCALE) {
            std::cerr << "Invalid output quantization type" << std::endl;
            hbDNNReleaseTask(task_handle);
            continue;
        }
        
        hbSysFlushMem(&(output[order[4]].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
        hbSysFlushMem(&(output[order[5]].sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);

        auto *l_cls_raw = reinterpret_cast<float *>(output[order[4]].sysMem[0].virAddr);
        auto *l_bbox_raw = reinterpret_cast<int32_t *>(output[order[5]].sysMem[0].virAddr);
        auto *l_bbox_scale = reinterpret_cast<float *>(output[order[5]].properties.scale.scaleData);
        for (int h = 0; h < H_32; h++) {
            for (int w = 0; w < W_32; w++) {
                float *cur_l_cls_raw = l_cls_raw;
                int32_t *cur_l_bbox_raw = l_bbox_raw;

                int cls_id = 0;
                for (int i = 1; i < CLASSES_NUM; i++) {
                    if (cur_l_cls_raw[i] > cur_l_cls_raw[cls_id]) {
                        cls_id = i;
                    }
                }

                if (cur_l_cls_raw[cls_id] < CONF_THRES_RAW) {
                    l_cls_raw += CLASSES_NUM;
                    l_bbox_raw += REG * 4;
                    continue;
                }

                float score = 1 / (1 + std::exp(-cur_l_cls_raw[cls_id]));

                float ltrb[4], sum, dfl;
                for (int i = 0; i < 4; i++) {
                    ltrb[i] = 0.;
                    sum = 0.;
                    for (int j = 0; j < REG; j++) {
                        int index_id = REG * i + j;
                        dfl = std::exp(float(cur_l_bbox_raw[index_id]) * l_bbox_scale[index_id]);
                        ltrb[i] += dfl * j;
                        sum += dfl;
                    }
                    ltrb[i] /= sum;
                }

                if (ltrb[2] + ltrb[0] <= 0 || ltrb[3] + ltrb[1] <= 0) {
                    l_cls_raw += CLASSES_NUM;
                    l_bbox_raw += REG * 4;
                    continue;
                }

                float x1 = (w + 0.5 - ltrb[0]) * 32.0;
                float y1 = (h + 0.5 - ltrb[1]) * 32.0;
                float x2 = (w + 0.5 + ltrb[2]) * 32.0;
                float y2 = (h + 0.5 + ltrb[3]) * 32.0;

                bboxes[cls_id].push_back(cv::Rect2d(x1, y1, x2 - x1, y2 - y1));
                scores[cls_id].push_back(score);

                l_cls_raw += CLASSES_NUM;
                l_bbox_raw += REG * 4;
            }
        }

        // NMS
        std::vector<std::vector<int>> indices(CLASSES_NUM);
        for (int i = 0; i < CLASSES_NUM; i++) {
            cv::dnn::NMSBoxes(bboxes[i], scores[i], SCORE_THRESHOLD, NMS_THRESHOLD, indices[i], 1.f, NMS_TOP_K);
        }

        // 在原始图像上渲染结果
        for (int cls_id = 0; cls_id < CLASSES_NUM; cls_id++) {
            for (std::vector<int>::iterator it = indices[cls_id].begin(); it != indices[cls_id].end(); ++it) {
                float x1 = (bboxes[cls_id][*it].x - x_shift) / x_scale;
                float y1 = (bboxes[cls_id][*it].y - y_shift) / y_scale;
                float x2 = x1 + (bboxes[cls_id][*it].width) / x_scale;
                float y2 = y1 + (bboxes[cls_id][*it].height) / y_scale;
                float score = scores[cls_id][*it];
                std::string name = object_names[cls_id % CLASSES_NUM];

                // 绘制矩形
                cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0), LINE_SIZE);

                // 绘制字体
                std::string text = name + ": " + std::to_string(static_cast<int>(score * 100)) + "%";
                cv::putText(frame, text, cv::Point(x1, y1 - 5), cv::FONT_HERSHEY_SIMPLEX, FONT_SIZE, cv::Scalar(0, 0, 255), FONT_THICKNESS, cv::LINE_AA);
            }
        }

        // 计算并显示FPS
        float fps = fpsCalc.update();
        std::string fps_text = "FPS: " + std::to_string(static_cast<int>(fps));
        cv::putText(frame, fps_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);

        // 显示图像
        cv::imshow("YOLO11 Real-time Detection", frame);

        // 释放任务
        hbDNNReleaseTask(task_handle);

        // 检查是否按下'q'键退出
        if (cv::waitKey(1) == 'q') {
            std::cout << "Exiting..." << std::endl;
            break;
        }
    }

    // 释放资源
    hbSysFreeMem(&(input.sysMem[0]));
    for (int i = 0; i < 6; i++) {
        hbSysFreeMem(&(output[i].sysMem[0]));
    }
    delete[] output;

    // 释放模型
    hbDNNRelease(packed_dnn_handle);

    // 释放摄像头
    cap.release();
    cv::destroyAllWindows();

    std::cout << "Program ended successfully" << std::endl;
    return 0;
}
